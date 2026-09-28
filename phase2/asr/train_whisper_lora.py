"""Train a PEFT LoRA adapter for multilingual Whisper transcription."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from asr_pipeline.archives import stage_archived_audio
from asr_pipeline.artifacts import mark_complete
from asr_pipeline.checkpoints import restore_checkpoint, sync_checkpoint


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/voice-rag-phase2/stt")
DEFAULT_WORK_ROOT = Path("/content/voice-rag-asr")
DEFAULT_MODEL = "openai/whisper-large-v3-turbo"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a q_proj/v_proj PEFT LoRA adapter for Whisper ASR."
    )
    parser.add_argument("--config", type=Path, help="Optional smoke/V1 YAML configuration")
    parser.add_argument("--train-manifest", action="append", type=Path, default=[])
    parser.add_argument("--eval-manifest", action="append", type=Path, default=[])
    parser.add_argument("--drive-root", type=Path, default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--hinglish-language-mode", choices=("auto", "hi", "en"))
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--max-audio-seconds", type=float)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--precision", choices=("auto", "fp16", "bf16"), default="auto")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="auto, none, or a local checkpoint directory",
    )
    return parser


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by Colab setup
        raise SystemExit("PyYAML is required when --config is used") from exc
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise SystemExit("Training config must contain a YAML mapping")
    return value


def _setting(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    return value if value is not None else config.get(name, default)


def _manifest_path(value: Path) -> Path:
    value = value.resolve()
    return value / "manifest.jsonl" if value.is_dir() else value


def _load_manifest(path: Path, allowed_splits: set[str]) -> list[dict[str, Any]]:
    manifest = _manifest_path(path)
    rows: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {
                "id", "audio", "text", "language", "speaker_id", "duration", "source", "split"
            } - row.keys()
            if missing:
                raise ValueError(f"{manifest}:{line_number}: missing {sorted(missing)}")
            if row["language"] not in {"en", "hi", "hi-en"}:
                raise ValueError(f"{manifest}:{line_number}: invalid language")
            if row["split"] not in allowed_splits:
                raise ValueError(
                    f"{manifest}:{line_number}: split {row['split']!r} is not allowed here"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable rows in {manifest}")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_fingerprint(paths: Iterable[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(_manifest_path(path)), "sha256": _sha256_file(_manifest_path(path))}
        for path in paths
    ]


def _json_fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _resolve_model_revision(model: str, requested: str) -> str:
    from huggingface_hub import HfApi

    return str(HfApi().model_info(model, revision=requested).sha)


def _stage_rows(
    manifest_inputs: list[Path], rows_by_manifest: list[list[dict[str, Any]]], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    all_rows: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for index, (input_path, rows) in enumerate(zip(manifest_inputs, rows_by_manifest)):
        manifest = _manifest_path(input_path)
        artifact_dir = manifest.parent
        staged = stage_archived_audio(rows, artifact_dir, root / f"artifact-{index:02d}")
        for row in rows:
            sample_id = str(row["id"])
            if sample_id in paths:
                raise ValueError(f"Duplicate training/evaluation id: {sample_id}")
            paths[sample_id] = staged[sample_id]
            all_rows.append(row)
    return all_rows, paths


def _read_pcm16(path: Path) -> "Any":
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != 16000 or handle.getsampwidth() != 2:
            raise ValueError(f"Non-canonical audio reached trainer: {path}")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0


class ManifestDataset:
    def __init__(self, rows: list[dict[str, Any]], audio_paths: dict[str, Path], max_seconds: float):
        self.rows = [row for row in rows if float(row["duration"]) <= max_seconds]
        self.audio_paths = audio_paths
        if not self.rows:
            raise ValueError("All examples were filtered by --max-audio-seconds")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "id": row["id"],
            "audio": _read_pcm16(self.audio_paths[str(row["id"])]),
            "text": row["text"],
            "language": row["language"],
        }


@dataclass
class WhisperCollator:
    processor: Any
    hinglish_language_mode: str

    def _prefix_language(self, language: str) -> str | None:
        if language == "en":
            return "english"
        if language == "hi":
            return "hindi"
        if self.hinglish_language_mode == "hi":
            return "hindi"
        if self.hinglish_language_mode == "en":
            return "english"
        return None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        audio = [feature["audio"] for feature in features]
        batch = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
        )
        label_features = []
        for feature in features:
            #codec*change*+2026-09-17 Whisper has no hi-en token, so this
            # per-example prefix is an explicit experiment setting, not a fixed policy.
            self.processor.tokenizer.set_prefix_tokens(
                language=self._prefix_language(feature["language"]),
                task="transcribe",
                predict_timestamps=False,
            )
            label_features.append(
                {"input_ids": self.processor.tokenizer(feature["text"]).input_ids}
            )
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        decoder_start = self.processor.tokenizer.pad_token_id
        if labels.shape[1] and torch.all(labels[:, 0] == decoder_start):
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _select_precision(torch: Any, requested: str) -> tuple[Any, bool, bool]:
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("--precision bf16 requested but this GPU does not support BF16")
        return torch.bfloat16, True, False
    if requested == "fp16":
        return torch.float16, False, True
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def _promote_export(local_export: Path, persistent_export: Path, fingerprint: str) -> None:
    if persistent_export.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {persistent_export}")
    temporary = persistent_export.with_name(f"{persistent_export.name}.incomplete")
    if temporary.exists():
        raise FileExistsError(f"Incomplete export already exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_export, temporary)
    _atomic_json(temporary / "export-meta.json", {"run_fingerprint": fingerprint})
    mark_complete(temporary)
    os.replace(temporary, persistent_export)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    if not args.train_manifest:
        raise SystemExit("At least one --train-manifest is required")

    settings = {
        "model": _setting(args, config, "model", DEFAULT_MODEL),
        "model_revision_requested": _setting(args, config, "model_revision", "main"),
        "hinglish_language_mode": _setting(args, config, "hinglish_language_mode", "auto"),
        "epochs": float(_setting(args, config, "epochs", 2.0)),
        "max_steps": int(_setting(args, config, "max_steps", -1)),
        "per_device_train_batch_size": int(
            _setting(args, config, "per_device_train_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            _setting(args, config, "gradient_accumulation_steps", 16)
        ),
        "learning_rate": float(_setting(args, config, "learning_rate", 1e-4)),
        "warmup_steps": int(_setting(args, config, "warmup_steps", 0)),
        "warmup_ratio": float(_setting(args, config, "warmup_ratio", 0.0)),
        "save_steps": int(_setting(args, config, "save_steps", 100)),
        "logging_steps": int(_setting(args, config, "logging_steps", 5)),
        "seed": int(_setting(args, config, "seed", 17)),
        "lora_r": int(_setting(args, config, "lora_r", 16)),
        "lora_alpha": int(_setting(args, config, "lora_alpha", 32)),
        "lora_dropout": float(_setting(args, config, "lora_dropout", 0.05)),
        "max_audio_seconds": float(_setting(args, config, "max_audio_seconds", 30.0)),
        "gradient_checkpointing": bool(
            _setting(args, config, "gradient_checkpointing", True)
        ),
        "precision": args.precision,
        "load_in_8bit": args.load_in_8bit,
    }
    if settings["hinglish_language_mode"] not in {"auto", "hi", "en"}:
        raise SystemExit("hinglish_language_mode must be auto, hi, or en")

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            BitsAndBytesConfig,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            TrainerCallback,
            WhisperForConditionalGeneration,
            WhisperProcessor,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install phase2/asr/requirements-train.txt before training"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("Whisper LoRA training requires an NVIDIA CUDA runtime")

    dtype, use_bf16, use_fp16 = _select_precision(torch, settings["precision"])
    resolved_revision = _resolve_model_revision(
        settings["model"], settings["model_revision_requested"]
    )
    settings["model_revision_resolved"] = resolved_revision
    settings["train_manifests"] = _manifest_fingerprint(args.train_manifest)
    settings["eval_manifests"] = _manifest_fingerprint(args.eval_manifest)
    settings["run_id"] = args.run_id
    fingerprint = _json_fingerprint(settings)

    drive_root = args.drive_root.resolve()
    work_dir = (args.work_dir or DEFAULT_WORK_ROOT / args.run_id).resolve()
    local_checkpoints = work_dir / "checkpoints"
    persistent_checkpoints = drive_root / "checkpoints" / args.run_id
    experiment_dir = drive_root / "experiments" / args.run_id
    export_dir = drive_root / "exports" / args.run_id / "adapter"
    work_dir.mkdir(parents=True, exist_ok=True)
    local_checkpoints.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if (experiment_dir / "COMPLETE").exists():
        raise SystemExit(f"Run ID is already complete: {args.run_id}")
    run_metadata = {
        **settings,
        "run_fingerprint": fingerprint,
        "drive_root": str(drive_root),
        "work_dir": str(work_dir),
        "status": "running",
        "started_unix": time.time(),
        "argv": sys.argv if argv is None else argv,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    _atomic_json(experiment_dir / "run.json", run_metadata)

    train_rows_by_manifest = [
        _load_manifest(path, {"train"}) for path in args.train_manifest
    ]
    train_rows, train_audio = _stage_rows(
        args.train_manifest, train_rows_by_manifest, work_dir / "staged-data" / "train"
    )
    eval_rows: list[dict[str, Any]] = []
    eval_audio: dict[str, Path] = {}
    if args.eval_manifest:
        eval_rows_by_manifest = [
            _load_manifest(path, {"validation"}) for path in args.eval_manifest
        ]
        eval_rows, eval_audio = _stage_rows(
            args.eval_manifest, eval_rows_by_manifest, work_dir / "staged-data" / "eval"
        )

    processor = WhisperProcessor.from_pretrained(
        settings["model"], revision=resolved_revision
    )
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if settings["load_in_8bit"]:
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
        )
    model = WhisperForConditionalGeneration.from_pretrained(
        settings["model"], revision=resolved_revision, **model_kwargs
    )
    model.config.use_cache = False
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    if settings["load_in_8bit"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=settings["gradient_checkpointing"]
        )
    elif settings["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=settings["lora_r"],
            lora_alpha=settings["lora_alpha"],
            lora_dropout=settings["lora_dropout"],
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        ),
    )
    model.print_trainable_parameters()

    train_dataset = ManifestDataset(
        train_rows, train_audio, settings["max_audio_seconds"]
    )
    eval_dataset = (
        ManifestDataset(eval_rows, eval_audio, settings["max_audio_seconds"])
        if eval_rows
        else None
    )
    set_seed(settings["seed"])
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(local_checkpoints),
        num_train_epochs=settings["epochs"],
        max_steps=settings["max_steps"],
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        warmup_steps=settings["warmup_steps"],
        warmup_ratio=settings["warmup_ratio"],
        logging_steps=settings["logging_steps"],
        save_strategy="steps",
        save_steps=settings["save_steps"],
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=settings["save_steps"] if eval_dataset is not None else None,
        gradient_checkpointing=settings["gradient_checkpointing"],
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        seed=settings["seed"],
        data_seed=settings["seed"],
        dataloader_num_workers=0,
    )

    class DriveCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            sync_checkpoint(checkpoint, persistent_checkpoints, fingerprint)
            return control

    resume: str | bool | None = None
    if args.resume_from_checkpoint == "auto":
        try:
            resume = str(
                restore_checkpoint(
                    persistent_checkpoints,
                    local_checkpoints,
                    expected_fingerprint=fingerprint,
                )
            )
        except FileNotFoundError:
            resume = None
    elif args.resume_from_checkpoint.lower() != "none":
        resume = args.resume_from_checkpoint

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WhisperCollator(
            processor=processor,
            hinglish_language_mode=settings["hinglish_language_mode"],
        ),
        processing_class=processor,
        callbacks=[DriveCheckpointCallback()],
    )
    result = trainer.train(resume_from_checkpoint=resume)

    local_export = work_dir / "final-adapter"
    if local_export.exists():
        raise FileExistsError(f"Refusing to overwrite local export: {local_export}")
    trainer.save_model(str(local_export))
    processor.save_pretrained(local_export)
    _atomic_json(local_export / "run-fingerprint.json", {"run_fingerprint": fingerprint})
    _promote_export(local_export, export_dir, fingerprint)
    run_metadata.update(
        status="complete",
        completed_unix=time.time(),
        export_dir=str(export_dir),
        train_metrics=result.metrics,
    )
    _atomic_json(experiment_dir / "run.json", run_metadata)
    mark_complete(experiment_dir)
    print(json.dumps({"run_id": args.run_id, "adapter": str(export_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
