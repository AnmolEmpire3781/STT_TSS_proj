#!/usr/bin/env python3
"""Resumable, checksummed Whisper evaluation for immutable ASR suites."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from asr_pipeline.archives import stage_archived_audio
from asr_pipeline.artifacts import (
    ArtifactError,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    is_complete,
    mark_complete,
    sha256_file,
    sha256_json,
)
from asr_pipeline.metrics import (
    coerce_keyword_expectations,
    compute_evaluation_metrics,
)
from asr_pipeline.normalization import (
    DEFAULT_NORMALIZATION,
    contains_normalized_phrase,
)
from asr_pipeline.schema import CanonicalRecord, SchemaError
from asr_pipeline.whisper_runtime import (
    BASE_MODEL_ID,
    HINGLISH_MODES,
    DecodingConfig,
    ModelSpec,
    WhisperRuntime,
)


RUN_SCHEMA_VERSION = "asr-evaluation-run-v1"
PREDICTION_SCHEMA_VERSION = "asr-prediction-v1"
_PART_PATTERN = re.compile(r"^part-(\d{5})\.jsonl$")
_DEFAULT_KEYWORDS = Path(__file__).resolve().parent / "configs" / "domain_keywords.json"


class EvaluationError(RuntimeError):
    """Raised when a suite or resumable evaluation is unsafe/incompatible."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise EvaluationError(f"required JSON file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(
                        f"invalid JSON in {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise EvaluationError(
                        f"prediction/manifest row must be an object: {path}:{line_number}"
                    )
                rows.append(value)
    except FileNotFoundError as exc:
        raise EvaluationError(f"JSONL file is missing: {path}") from exc
    return rows


@dataclass(frozen=True)
class SuiteSelection:
    suite_id: str
    profile: str
    root: Path
    manifest_path: Path
    manifest_sha256: str
    suite_fingerprint: str
    rows: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "profile": self.profile,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "suite_fingerprint": self.suite_fingerprint,
            "sample_count": len(self.rows),
        }


def _resolve_suite_files(suite: Path) -> tuple[Path, Path | None, Path]:
    if suite.is_dir():
        root = suite
        manifest = root / "manifest.jsonl"
        metadata = root / "suite.json"
        return manifest, metadata if metadata.is_file() else None, root
    if not suite.is_file():
        raise EvaluationError(f"suite path does not exist: {suite}")
    if suite.suffix.lower() == ".jsonl":
        sibling = suite.parent / "suite.json"
        return suite, sibling if sibling.is_file() else None, suite.parent
    if suite.suffix.lower() == ".json":
        metadata_value = _read_json(suite)
        if not isinstance(metadata_value, Mapping):
            raise EvaluationError(f"suite metadata must be an object: {suite}")
        manifest_name = (
            metadata_value.get("manifest")
            or metadata_value.get("manifest_path")
            or "manifest.jsonl"
        )
        manifest = Path(str(manifest_name))
        if not manifest.is_absolute():
            manifest = suite.parent / manifest
        return manifest, suite, suite.parent
    raise EvaluationError("--suite must be a suite directory, suite.json, or JSONL manifest")


def load_suite(
    suite_path: str | os.PathLike[str],
    *,
    profile: str | None,
    require_complete: bool,
    allow_full: bool,
) -> SuiteSelection:
    suite = Path(suite_path).resolve()
    manifest_path, metadata_path, root = _resolve_suite_files(suite)
    if require_complete:
        if not is_complete(root, verify=True):
            raise EvaluationError(
                f"suite is missing a valid COMPLETE/checksum artifact: {root}"
            )
    metadata: Mapping[str, Any] = {}
    if metadata_path is not None:
        loaded_metadata = _read_json(metadata_path)
        if not isinstance(loaded_metadata, Mapping):
            raise EvaluationError(f"suite metadata must be an object: {metadata_path}")
        metadata = loaded_metadata
    if not manifest_path.is_file():
        raise EvaluationError(f"suite manifest is missing: {manifest_path}")

    manifest_hash = sha256_file(manifest_path)
    declared_hash = metadata.get("manifest_sha256")
    if declared_hash is not None and declared_hash != manifest_hash:
        raise EvaluationError(
            "suite manifest checksum does not match suite.json: "
            f"expected {declared_hash}, got {manifest_hash}"
        )
    raw_rows = _read_jsonl(manifest_path)
    by_id: dict[str, dict[str, Any]] = {}
    ordered_all: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        try:
            row = CanonicalRecord.from_mapping(raw).to_dict()
        except SchemaError as exc:
            raise EvaluationError(
                f"invalid canonical suite row at index {index}: {exc}"
            ) from exc
        record_id = row["id"]
        if record_id in by_id:
            raise EvaluationError(f"duplicate suite record id: {record_id}")
        if row["split"] == "train":
            raise EvaluationError(
                f"evaluation suite contains forbidden train row: {record_id}"
            )
        by_id[record_id] = row
        ordered_all.append(row)

    profiles = metadata.get("profiles", {})
    if profiles is None:
        profiles = {}
    if not isinstance(profiles, Mapping):
        raise EvaluationError("suite profiles must be an object")
    selected_profile = profile
    if selected_profile is None:
        if "benchmark500" in profiles:
            selected_profile = "benchmark500"
        elif "custom_all" in profiles:
            selected_profile = "custom_all"
        else:
            selected_profile = "all"
    if selected_profile == "all":
        selected_rows = ordered_all
    else:
        if selected_profile not in profiles:
            available = ", ".join(sorted(str(name) for name in profiles)) or "none"
            raise EvaluationError(
                f"suite profile {selected_profile!r} is unavailable; found: {available}"
            )
        profile_value = profiles[selected_profile]
        if isinstance(profile_value, Mapping):
            profile_value = profile_value.get("ids")
        if not isinstance(profile_value, Sequence) or isinstance(profile_value, str):
            raise EvaluationError(
                f"suite profile {selected_profile!r} must contain an ordered ID list"
            )
        selected_ids = [str(value) for value in profile_value]
        if len(set(selected_ids)) != len(selected_ids):
            raise EvaluationError(f"suite profile {selected_profile!r} has duplicate IDs")
        missing = [record_id for record_id in selected_ids if record_id not in by_id]
        if missing:
            raise EvaluationError(
                f"suite profile {selected_profile!r} references missing IDs: "
                + ", ".join(missing[:5])
            )
        selected_rows = [by_id[record_id] for record_id in selected_ids]
    if not selected_rows:
        raise EvaluationError("selected suite profile is empty")
    if len(selected_rows) > 500 and not allow_full:
        raise EvaluationError(
            f"selected profile has {len(selected_rows)} rows; pass --allow-full explicitly"
        )

    suite_id = str(metadata.get("suite_id") or root.name)
    selected_ids = [row["id"] for row in selected_rows]
    suite_fingerprint = sha256_json(
        {
            "schema_version": "asr-suite-selection-v1",
            "suite_id": suite_id,
            "profile": selected_profile,
            "manifest_sha256": manifest_hash,
            "ordered_ids": selected_ids,
        }
    )
    return SuiteSelection(
        suite_id=suite_id,
        profile=selected_profile,
        root=root,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=manifest_hash,
        suite_fingerprint=suite_fingerprint,
        rows=tuple(selected_rows),
        metadata=metadata,
    )


def _load_keyword_catalog(path: Path | None) -> tuple[list[Any], dict[str, Any]]:
    if path is None:
        return [], {"enabled": False, "sha256": None, "path": None}
    resolved = path.resolve()
    payload = _read_json(resolved)
    if isinstance(payload, Mapping) and "keywords" in payload:
        raw_keywords = payload["keywords"]
    else:
        raw_keywords = payload
    expectations = coerce_keyword_expectations(raw_keywords)
    serialized = [
        {"term": expectation.term, "aliases": list(expectation.aliases)}
        for expectation in expectations
    ]
    return serialized, {
        "enabled": True,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "entries": len(serialized),
    }


def _keywords_for_row(row: Mapping[str, Any], catalog: Sequence[Any]) -> list[Any]:
    explicit = coerce_keyword_expectations(
        row.get("keywords", row.get("domain_keywords"))
    )
    selected = [
        {"term": expectation.term, "aliases": list(expectation.aliases)}
        for expectation in explicit
    ]
    existing = {expectation.term.casefold() for expectation in explicit}
    reference = str(row["text"])
    for raw in catalog:
        expectation = coerce_keyword_expectations([raw])[0]
        if expectation.term.casefold() in existing:
            continue
        if any(
            contains_normalized_phrase(reference, candidate)
            for candidate in expectation.candidates
        ):
            selected.append(
                {"term": expectation.term, "aliases": list(expectation.aliases)}
            )
            existing.add(expectation.term.casefold())
    return selected


def resolve_audio_path(
    row: Mapping[str, Any], suite: SuiteSelection, audio_root: Path | None
) -> Path:
    audio_value = row["audio"]
    if isinstance(audio_value, Mapping):
        audio_value = audio_value.get("path")
    if not isinstance(audio_value, (str, os.PathLike)) or not str(audio_value).strip():
        raise EvaluationError(f"row {row['id']!r} has no persisted audio path")
    candidate = Path(audio_value)
    if candidate.is_absolute():
        return candidate
    root = (audio_root or suite.manifest_path.parent).resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvaluationError(
            f"row {row['id']!r} audio path escapes --audio-root: {audio_value}"
        ) from exc
    return resolved


def _part_sidecar(part_path: Path) -> Path:
    return part_path.with_name(part_path.name + ".sha256")


def _commit_part(
    parts_dir: Path,
    part_number: int,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    if not rows:
        raise EvaluationError("cannot commit an empty prediction part")
    part_path = parts_dir / f"part-{part_number:05d}.jsonl"
    atomic_write_jsonl(part_path, rows)
    digest = sha256_file(part_path)
    #codec*change*+2026-09-17: The checksum sidecar is committed after the
    # JSONL rename, so resume ignores a data file interrupted before commit.
    atomic_write_text(_part_sidecar(part_path), f"{digest}  {part_path.name}\n")
    return part_path


def _read_part_checksum(sidecar: Path, part_name: str) -> str:
    try:
        content = sidecar.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise EvaluationError(f"prediction part checksum is missing: {sidecar}") from exc
    try:
        digest, declared_name = content.split("  ", 1)
    except ValueError as exc:
        raise EvaluationError(f"invalid prediction checksum sidecar: {sidecar}") from exc
    if declared_name != part_name or len(digest) != 64:
        raise EvaluationError(f"invalid prediction checksum sidecar: {sidecar}")
    return digest


def _load_committed_parts(
    parts_dir: Path,
    *,
    evaluation_fingerprint: str,
) -> tuple[list[dict[str, Any]], int]:
    if not parts_dir.exists():
        return [], 0
    committed: list[tuple[int, Path]] = []
    for sidecar in parts_dir.glob("part-*.jsonl.sha256"):
        part_path = sidecar.with_name(sidecar.name[: -len(".sha256")])
        match = _PART_PATTERN.match(part_path.name)
        if match:
            committed.append((int(match.group(1)), part_path))
    committed.sort()
    numbers = [number for number, _ in committed]
    if numbers != list(range(len(numbers))):
        raise EvaluationError(f"committed prediction parts are not contiguous: {numbers}")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for _, part_path in committed:
        expected = _read_part_checksum(_part_sidecar(part_path), part_path.name)
        if not part_path.is_file() or sha256_file(part_path) != expected:
            raise EvaluationError(f"prediction part checksum mismatch: {part_path}")
        part_rows = _read_jsonl(part_path)
        if not part_rows:
            raise EvaluationError(f"committed prediction part is empty: {part_path}")
        for row in part_rows:
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise EvaluationError(f"prediction has invalid id in {part_path}")
            if record_id in seen_ids:
                raise EvaluationError(f"duplicate completed prediction id: {record_id}")
            if row.get("evaluation_fingerprint") != evaluation_fingerprint:
                raise EvaluationError(
                    f"prediction part is incompatible with this run: {part_path}"
                )
            seen_ids.add(record_id)
            rows.append(row)
    return rows, len(committed)


def _runtime_compatibility(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "resolved_device": metadata.get("resolved_device"),
        "resolved_dtype": metadata.get("resolved_dtype"),
        "actual_model_revision": metadata.get("actual_model_revision"),
        "torch_version": metadata.get("torch_version"),
        "cuda_device_name": metadata.get("cuda_device_name"),
    }


def _configuration(
    suite: SuiteSelection,
    model_spec: ModelSpec,
    decoding: DecodingConfig,
    *,
    requested_device: str,
    requested_dtype: str,
    seed: int,
    warmup_samples: int,
    part_size: int,
    keyword_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "suite_fingerprint": suite.suite_fingerprint,
        "model_fingerprint": model_spec.fingerprint,
        "decoding_fingerprint": decoding.fingerprint,
        "normalization_fingerprint": DEFAULT_NORMALIZATION.fingerprint,
        "keyword_catalog_sha256": keyword_metadata.get("sha256"),
        "requested_device": requested_device,
        "requested_dtype": requested_dtype,
        "seed": seed,
        "warmup_samples": warmup_samples,
        "part_size": part_size,
        "latency_scope": "audio_load_resample_feature_extraction_and_generation",
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.part_size < 1:
        raise EvaluationError("--part-size must be positive")
    if args.warmup_samples < 0:
        raise EvaluationError("--warmup-samples cannot be negative")
    suite = load_suite(
        args.suite,
        profile=args.profile,
        require_complete=not args.allow_incomplete_suite,
        allow_full=args.allow_full,
    )
    audio_root = Path(args.audio_root).resolve() if args.audio_root else None
    keyword_path = None if args.no_domain_keywords else Path(args.keywords_file)
    keyword_catalog, keyword_metadata = _load_keyword_catalog(keyword_path)
    model_spec = ModelSpec(
        model_id=args.model_id,
        model_revision=args.model_revision,
        adapter_path=args.adapter_path,
        adapter_revision=args.adapter_revision,
        processor_id=args.processor_id,
        local_files_only=args.local_files_only,
    )
    if args.decoding_config:
        decoding_config_path = Path(args.decoding_config).resolve()
        if not is_complete(decoding_config_path.parent, verify=True):
            raise EvaluationError(
                "--decoding-config must belong to a complete checksummed artifact"
            )
        decoding_document = _read_json(decoding_config_path)
        if not isinstance(decoding_document, Mapping):
            raise EvaluationError("--decoding-config must contain a JSON object")
        decoding_value = decoding_document.get("decoding")
        if not isinstance(decoding_value, Mapping):
            raise EvaluationError("--decoding-config must contain a decoding object")
        decoding = DecodingConfig(
            hinglish_mode=str(decoding_value["hinglish_mode"]),
            num_beams=int(decoding_value["num_beams"]),
            max_new_tokens=int(decoding_value["max_new_tokens"]),
            chunk_length_seconds=float(decoding_value["chunk_length_seconds"]),
        )
    else:
        decoding = DecodingConfig(
            hinglish_mode=args.hinglish_mode,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            chunk_length_seconds=args.chunk_length_seconds,
        )
    configuration = _configuration(
        suite,
        model_spec,
        decoding,
        requested_device=args.device,
        requested_dtype=args.dtype,
        seed=args.seed,
        warmup_samples=args.warmup_samples,
        part_size=args.part_size,
        keyword_metadata=keyword_metadata,
    )
    configuration_fingerprint = sha256_json(configuration)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    complete_path = output_dir / "COMPLETE"
    existing_run = _read_json(run_path) if run_path.is_file() else None
    if complete_path.exists():
        if not is_complete(output_dir, verify=True):
            raise EvaluationError(f"completed evaluation is corrupt: {output_dir}")
        if not isinstance(existing_run, Mapping) or existing_run.get(
            "configuration_fingerprint"
        ) != configuration_fingerprint:
            raise EvaluationError(
                "output directory already contains an incompatible completed evaluation"
            )
        metrics = _read_json(output_dir / "metrics.json")
        if not isinstance(metrics, dict):
            raise EvaluationError("metrics.json must contain an object")
        return metrics
    if existing_run is not None:
        if not args.resume:
            raise EvaluationError(
                "output directory contains a partial run and --no-resume was requested"
            )
        if not isinstance(existing_run, Mapping) or existing_run.get(
            "configuration_fingerprint"
        ) != configuration_fingerprint:
            raise EvaluationError(
                "partial evaluation configuration differs; use a new --output-dir"
            )
    elif any(output_dir.iterdir()):
        raise EvaluationError(
            "non-empty output directory has no run.json; use a new --output-dir"
        )

    runtime = WhisperRuntime(
        model_spec,
        decoding,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    ).load()
    runtime_metadata = runtime.metadata()
    runtime_compatibility = _runtime_compatibility(runtime_metadata)
    runtime_fingerprint = sha256_json(runtime_compatibility)
    evaluation_fingerprint = sha256_json(
        {
            "configuration_fingerprint": configuration_fingerprint,
            "runtime_fingerprint": runtime_fingerprint,
        }
    )
    if existing_run is not None:
        if existing_run.get("evaluation_fingerprint") != evaluation_fingerprint:
            raise EvaluationError(
                "runtime/model environment differs from the partial evaluation; "
                "use a new --output-dir"
            )
        run_document = dict(existing_run)
    else:
        run_document = {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": "running",
            "created_at": _utc_now(),
            "suite": suite.to_dict(),
            "model": model_spec.to_dict(),
            "model_fingerprint": model_spec.fingerprint,
            "decoding": decoding.to_dict(),
            "decoding_fingerprint": decoding.fingerprint,
            "normalization": DEFAULT_NORMALIZATION.to_dict(),
            "normalization_fingerprint": DEFAULT_NORMALIZATION.fingerprint,
            "keyword_catalog": keyword_metadata,
            "runtime": runtime_metadata,
            "runtime_fingerprint": runtime_fingerprint,
            "configuration_fingerprint": configuration_fingerprint,
            "evaluation_fingerprint": evaluation_fingerprint,
            "warmup_samples": args.warmup_samples,
            "part_size": args.part_size,
        }
        atomic_write_json(run_path, run_document)

    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    committed_rows, next_part = _load_committed_parts(
        parts_dir, evaluation_fingerprint=evaluation_fingerprint
    )
    selected_ids = [row["id"] for row in suite.rows]
    selected_id_set = set(selected_ids)
    completed_ids = {row["id"] for row in committed_rows}
    unexpected = completed_ids - selected_id_set
    if unexpected:
        raise EvaluationError(
            "partial evaluation contains IDs outside the selected suite: "
            + ", ".join(sorted(unexpected)[:5])
        )

    pending_rows = [row for row in suite.rows if row["id"] not in completed_ids]
    staged_audio: dict[str, Path] = {}
    if any(row.get("archive") for row in pending_rows) and audio_root is None:
        if not args.staging_dir:
            raise EvaluationError(
                "suite uses tar shards; provide disposable local --staging-dir "
                "or an already-extracted --audio-root"
            )
        try:
            staged_audio = stage_archived_audio(
                pending_rows, suite.root, Path(args.staging_dir)
            )
        except (OSError, ValueError) as exc:
            raise EvaluationError(f"failed to stage archived suite audio: {exc}") from exc

    buffered: list[dict[str, Any]] = []

    def commit_buffer() -> None:
        nonlocal buffered, next_part
        if not buffered:
            return
        _commit_part(parts_dir, next_part, buffered)
        committed_rows.extend(buffered)
        completed_ids.update(row["id"] for row in buffered)
        next_part += 1
        buffered = []
        atomic_write_json(
            output_dir / "progress.json",
            {
                "schema_version": "asr-evaluation-progress-v1",
                "evaluation_fingerprint": evaluation_fingerprint,
                "completed": len(completed_ids),
                "total": len(suite.rows),
                "updated_at": _utc_now(),
            },
        )

    try:
        for ordinal, row in enumerate(suite.rows):
            if row["id"] in completed_ids:
                continue
            audio_path = staged_audio.get(row["id"])
            if audio_path is None:
                audio_path = resolve_audio_path(row, suite, audio_root)
            result = runtime.transcribe(audio_path, row["language"])
            keywords = _keywords_for_row(row, keyword_catalog)
            prediction = {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "id": row["id"],
                "reference": row["text"],
                "hypothesis": result.text,
                "language": row["language"],
                "source": row["source"],
                "split": row["split"],
                "speaker_id": row["speaker_id"],
                "manifest_duration_seconds": row["duration"],
                "decoded_audio_duration_seconds": result.audio_duration_seconds,
                "latency_seconds": result.latency_seconds,
                "latency_included": ordinal >= args.warmup_samples,
                "chunks": result.chunks,
                "keywords": keywords,
                "suite_fingerprint": suite.suite_fingerprint,
                "model_fingerprint": model_spec.fingerprint,
                "decoding_fingerprint": decoding.fingerprint,
                "evaluation_fingerprint": evaluation_fingerprint,
            }
            buffered.append(prediction)
            if len(buffered) >= args.part_size:
                commit_buffer()
    except BaseException:
        #codec*change*+2026-09-17: Already completed predictions are committed
        # even when Colab disconnects or a later sample fails.
        commit_buffer()
        raise
    commit_buffer()

    all_rows, _ = _load_committed_parts(
        parts_dir, evaluation_fingerprint=evaluation_fingerprint
    )
    by_id = {row["id"]: row for row in all_rows}
    missing = [record_id for record_id in selected_ids if record_id not in by_id]
    if missing:
        raise EvaluationError(
            f"evaluation ended with {len(missing)} missing predictions"
        )
    ordered_predictions = [by_id[record_id] for record_id in selected_ids]
    atomic_write_jsonl(output_dir / "predictions.jsonl", ordered_predictions)
    metrics = compute_evaluation_metrics(ordered_predictions)
    metrics.update(
        {
            "suite": suite.to_dict(),
            "model": model_spec.to_dict(),
            "model_fingerprint": model_spec.fingerprint,
            "decoding": decoding.to_dict(),
            "decoding_fingerprint": decoding.fingerprint,
            "evaluation_fingerprint": evaluation_fingerprint,
            "keyword_catalog": keyword_metadata,
            "runtime": runtime_metadata,
        }
    )
    atomic_write_json(output_dir / "metrics.json", metrics)
    run_document.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "completed_samples": len(ordered_predictions),
            "runtime": runtime_metadata,
        }
    )
    atomic_write_json(run_path, run_document)
    mark_complete(output_dir)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate base Whisper or a PEFT adapter with resumable, "
            "checksummed prediction parts."
        )
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Immutable suite directory, suite.json, or manifest.jsonl",
    )
    parser.add_argument(
        "--profile",
        help="Suite profile (defaults to benchmark500 when present, otherwise all)",
    )
    parser.add_argument(
        "--audio-root",
        help="Root containing materialized suite audio (defaults to manifest directory)",
    )
    parser.add_argument(
        "--staging-dir",
        help=(
            "Disposable local directory used to extract referenced tar shards; "
            "required for archived suites unless --audio-root is supplied"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Persistent evaluation artifact directory (Drive-friendly)",
    )
    parser.add_argument("--model-id", default=BASE_MODEL_ID)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--processor-id")
    parser.add_argument("--adapter-path", help="Local path or Hub ID for a PEFT adapter")
    parser.add_argument("--adapter-revision")
    parser.add_argument(
        "--decoding-config",
        help="selected-decoding.json produced by compare_models.py",
    )
    parser.add_argument(
        "--hinglish-mode",
        choices=HINGLISH_MODES,
        default="auto",
        help="hi-en prefix policy: auto has no forced language; hi/en force that prefix",
    )
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=440)
    parser.add_argument("--chunk-length-seconds", type=float, default=30.0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--part-size", type=int, default=10)
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=1,
        help="Initial ordered samples excluded from latency percentiles",
    )
    parser.add_argument(
        "--keywords-file",
        default=str(_DEFAULT_KEYWORDS),
        help="JSON domain keyword catalog",
    )
    parser.add_argument(
        "--no-domain-keywords",
        action="store_true",
        help="Use only per-row keyword annotations",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume compatible committed parts (default: enabled)",
    )
    parser.add_argument(
        "--allow-incomplete-suite",
        action="store_true",
        help="Permit a suite without verified COMPLETE/SHA256SUMS (development only)",
    )
    parser.add_argument(
        "--allow-full",
        action="store_true",
        help="Explicitly permit profiles larger than 500 samples",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metrics = run_evaluation(args)
    except (EvaluationError, ArtifactError, ValueError) as exc:
        parser.error(str(exc))
    overall = metrics["overall"]
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "samples": overall["sample_count"],
                "wer": overall["wer"],
                "cer": overall["cer"],
                "domain_keyword_accuracy": overall["domain_keyword_accuracy"],
                "p50_latency_seconds": overall["p50_latency_seconds"],
                "p95_latency_seconds": overall["p95_latency_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
