"""Create an immutable, speaker-disjoint custom ASR dataset release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from asr_pipeline.archives import TarShardWriter
from asr_pipeline.artifacts import mark_complete


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/voice-rag-phase2/stt")
DEFAULT_WORK_ROOT = Path("/content/voice-rag-asr")
LANGUAGES = {"en", "hi", "hi-en"}
SPLITS = ("train", "validation", "test")
REQUIRED_INPUT_FIELDS = {"id", "audio", "text", "language", "speaker_id", "source"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert verified custom recordings to canonical ASR training audio"
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--raw-audio-root", type=Path, required=True)
    parser.add_argument("--release-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--drive-root", type=Path, default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--verification-field", default="human_verified")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--split-ratios",
        default="0.8,0.1,0.1",
        help="train,validation,test speaker-level duration targets",
    )
    parser.add_argument("--shard-size-mb", type=int, default=128)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    return parser


def _read_input(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _is_verified(value: object) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _parse_ratios(value: str) -> dict[str, float]:
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("split ratios must be numeric") from exc
    if len(values) != 3 or any(item < 0 for item in values) or sum(values) <= 0:
        raise argparse.ArgumentTypeError("split ratios must contain three non-negative values")
    total = sum(values)
    return {name: item / total for name, item in zip(SPLITS, values)}


def _safe_component(value: str, label: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise SystemExit(f"{label} must be one safe path component")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_properties(path: Path) -> tuple[float, int, int, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        duration = handle.getnframes() / float(rate)
    return duration, channels, rate, width


def _convert_audio(source: Path, destination: Path, ffmpeg_bin: str) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.wav")
    if temporary.exists() or destination.exists():
        raise FileExistsError(f"Refusing to overwrite staged audio: {destination}")
    try:
        duration, channels, rate, width = _wav_properties(source)
        canonical = channels == 1 and rate == 16000 and width == 2
    except (wave.Error, EOFError):
        canonical = False
        duration = 0.0
    if canonical:
        shutil.copy2(source, temporary)
    else:
        command = [
            ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffmpeg executable not found: {ffmpeg_bin}") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg failed for {source}") from exc
    os.replace(temporary, destination)
    duration, channels, rate, width = _wav_properties(destination)
    if (channels, rate, width) != (1, 16000, 2) or duration <= 0:
        raise ValueError(f"Canonical conversion verification failed for {source}")
    return duration


def _stable_order(seed: int, speaker_id: str) -> str:
    return hashlib.sha256(f"{seed}:{speaker_id}".encode("utf-8")).hexdigest()


def _assign_speaker_splits(
    rows: list[dict[str, Any]], ratios: dict[str, float], seed: int
) -> None:
    speaker_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        speaker_rows[str(row["speaker_id"])].append(row)

    assignments: dict[str, str] = {}
    assigned_duration = {split: 0.0 for split in SPLITS}
    for speaker, group in speaker_rows.items():
        defined = {str(row.get("split", "")).strip() for row in group if row.get("split")}
        if len(defined) > 1:
            raise ValueError(f"Speaker {speaker!r} is preassigned to multiple splits")
        if defined:
            split = next(iter(defined))
            if split not in SPLITS:
                raise ValueError(f"Speaker {speaker!r} has invalid split {split!r}")
            assignments[speaker] = split
            assigned_duration[split] += sum(float(row["duration"]) for row in group)

    total_duration = sum(float(row["duration"]) for row in rows)
    targets = {split: total_duration * ratios[split] for split in SPLITS}
    unassigned = sorted(
        (speaker for speaker in speaker_rows if speaker not in assignments),
        key=lambda speaker: _stable_order(seed, speaker),
    )
    for speaker in unassigned:
        #codec*change*+2026-09-17 Whole speakers are greedily assigned by
        # remaining duration deficit; no utterance-level split can leak a speaker.
        split = max(
            SPLITS,
            key=lambda name: (
                targets[name] - assigned_duration[name],
                -SPLITS.index(name),
            ),
        )
        assignments[speaker] = split
        assigned_duration[split] += sum(
            float(row["duration"]) for row in speaker_rows[speaker]
        )
    for speaker, group in speaker_rows.items():
        for row in group:
            row["split"] = assignments[speaker]


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> str:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _sha256(path)


def _finalize_checksums(root: Path) -> None:
    mark_complete(root)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ratios = _parse_ratios(args.split_ratios)
    if args.shard_size_mb <= 0:
        raise SystemExit("--shard-size-mb must be positive")
    if args.output_dir:
        output_dir = args.output_dir.resolve()
        release_id = args.release_id or output_dir.name
    else:
        if not args.release_id:
            raise SystemExit("--release-id is required when --output-dir is omitted")
        release_id = args.release_id
        output_dir = (args.drive_root / "datasets" / "custom" / release_id).resolve()
    _safe_component(release_id, "--release-id")
    incomplete = output_dir.with_name(f"{output_dir.name}.incomplete")
    if output_dir.exists() or incomplete.exists():
        raise SystemExit(f"Refusing to overwrite an existing release: {output_dir}")

    raw_root = args.raw_audio_root.resolve()
    if not raw_root.is_dir():
        raise SystemExit(f"Raw audio root does not exist: {raw_root}")
    work_dir = (args.work_dir or DEFAULT_WORK_ROOT / f"custom-{release_id}").resolve()
    staged_audio = work_dir / "audio"
    staged_audio.mkdir(parents=True, exist_ok=True)

    input_rows = _read_input(args.input_manifest.resolve())
    if not input_rows:
        raise SystemExit("Input manifest is empty")
    ids: set[str] = set()
    processed: list[dict[str, Any]] = []
    for index, row in enumerate(input_rows, 1):
        missing = REQUIRED_INPUT_FIELDS - row.keys()
        if missing:
            raise ValueError(f"Input row {index} is missing {sorted(missing)}")
        sample_id = str(row["id"]).strip()
        if not sample_id or sample_id in ids:
            raise ValueError(f"Input row {index} has an empty or duplicate id")
        ids.add(sample_id)
        language = str(row["language"]).strip()
        if language not in LANGUAGES:
            raise ValueError(f"Input row {index} has invalid language {language!r}")
        if not str(row["text"]).strip() or not str(row["speaker_id"]).strip():
            raise ValueError(f"Input row {index} has empty text or speaker_id")
        if not str(row["source"]).strip():
            raise ValueError(f"Input row {index} has an empty source")
        if not _is_verified(row.get(args.verification_field)):
            raise ValueError(
                f"Input row {index} is not human verified via {args.verification_field!r}"
            )
        raw_relative = Path(str(row["audio"]))
        raw_path = (raw_root / raw_relative).resolve()
        if raw_root not in raw_path.parents or not raw_path.is_file():
            raise ValueError(f"Input row {index} has missing/unsafe raw audio path")
        raw_hash = _sha256(raw_path)
        filename = f"{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:24]}.wav"
        canonical_path = staged_audio / filename
        duration = _convert_audio(raw_path, canonical_path, args.ffmpeg_bin)
        canonical_hash = _sha256(canonical_path)
        processed.append(
            {
                "id": sample_id,
                "audio": f"audio/{filename}",
                "text": str(row["text"]),
                "language": language,
                "speaker_id": str(row["speaker_id"]).strip(),
                "duration": round(duration, 6),
                "source": str(row["source"]).strip(),
                "split": str(row.get("split", "")).strip(),
                "human_verified": True,
                "verified_by": row.get("verified_by"),
                "verified_at": row.get("verified_at"),
                "raw_audio_sha256": raw_hash,
                "audio_sha256": canonical_hash,
            }
        )

    _assign_speaker_splits(processed, ratios, args.seed)
    incomplete.mkdir(parents=True)
    archived_rows: list[dict[str, Any]] = []
    with TarShardWriter(
        incomplete / "shards", max_bytes=args.shard_size_mb * 1024 * 1024
    ) as writer:
        for row in processed:
            canonical_path = staged_audio / Path(str(row["audio"])).name
            archived = writer.add(canonical_path, str(row["audio"]))
            archived_rows.append({**row, "archive": archived.archive})
    manifest_sha = _write_manifest(incomplete / "manifest.jsonl", archived_rows)
    split_counts = Counter(str(row["split"]) for row in archived_rows)
    split_hours = {
        split: round(
            sum(float(row["duration"]) for row in archived_rows if row["split"] == split)
            / 3600.0,
            6,
        )
        for split in SPLITS
    }
    metadata = {
        "schema_version": 1,
        "artifact_type": "custom_asr_dataset",
        "release_id": release_id,
        "created_unix": time.time(),
        "manifest_sha256": manifest_sha,
        "records": len(archived_rows),
        "split_counts": dict(split_counts),
        "split_hours": split_hours,
        "seed": args.seed,
        "split_ratios": ratios,
        "audio_format": {"container": "wav", "sample_rate": 16000, "channels": 1, "pcm_bits": 16},
        "raw_audio_root": str(raw_root),
        "raw_audio_untouched": True,
    }
    _atomic_json(incomplete / "dataset.json", metadata)
    _finalize_checksums(incomplete)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(incomplete, output_dir)
    print(json.dumps({**metadata, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
