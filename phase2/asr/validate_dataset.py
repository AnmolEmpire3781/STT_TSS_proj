"""Validate canonical ASR manifests and their materialized audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

from asr_pipeline.archives import stage_archived_audio


REQUIRED_FIELDS = {
    "id",
    "audio",
    "text",
    "language",
    "speaker_id",
    "duration",
    "source",
    "split",
}
LANGUAGES = {"en", "hi", "hi-en"}
SPLITS = {"train", "validation", "test"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a canonical ASR dataset release")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=("training", "evaluation", "custom", "snapshot"),
        required=True,
    )
    parser.add_argument("--against-eval-manifest", action="append", type=Path, default=[])
    parser.add_argument(
        "--check-audio", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--require-human-verification", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-errors", type=int, default=100)
    return parser


def _manifest_path(path: Path) -> Path:
    path = path.resolve()
    return path / "manifest.jsonl" if path.is_dir() else path


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    manifest = _manifest_path(path)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: row must be an object")
                continue
            value["_line"] = line_number
            rows.append(value)
    return rows, errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_info(path: Path) -> tuple[float, int, int, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.getnframes()
    return frames / float(rate), channels, rate, width


def validate(
    manifest: Path,
    purpose: str,
    against_eval: list[Path],
    check_audio: bool,
    require_human_verification: bool,
    max_errors: int,
) -> dict[str, Any]:
    rows, errors = _read_rows(manifest)
    warnings: list[str] = []
    ids: set[str] = set()
    audio_hashes: dict[str, str] = {}
    speakers: dict[str, set[str]] = defaultdict(set)
    source_ids: set[tuple[str, str]] = set()

    for row in rows:
        line = row.pop("_line")
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"line {line}: missing fields {sorted(missing)}")
            if len(errors) >= max_errors:
                break
            continue
        sample_id = str(row["id"]).strip()
        if not sample_id:
            errors.append(f"line {line}: id is empty")
        elif sample_id in ids:
            errors.append(f"line {line}: duplicate id {sample_id!r}")
        ids.add(sample_id)
        if row["language"] not in LANGUAGES:
            errors.append(f"line {line}: invalid language {row['language']!r}")
        if row["split"] not in SPLITS:
            errors.append(f"line {line}: invalid split {row['split']!r}")
        if purpose == "training" and row["split"] != "train":
            errors.append(f"line {line}: training manifest contains {row['split']!r}")
        if purpose == "evaluation" and row["split"] == "train":
            errors.append(f"line {line}: evaluation manifest contains train data")
        text = str(row["text"])
        if not text.strip():
            errors.append(f"line {line}: transcript is empty")
        elif unicodedata.normalize("NFC", text) != text:
            warnings.append(f"line {line}: transcript is not Unicode NFC; text was not changed")
        speaker = str(row["speaker_id"]).strip()
        if not speaker:
            errors.append(f"line {line}: speaker_id is empty")
        speakers[speaker].add(str(row["split"]))
        try:
            if float(row["duration"]) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"line {line}: duration must be positive")
        if (purpose == "custom" or require_human_verification) and row.get(
            "human_verified"
        ) is not True:
            errors.append(f"line {line}: human_verified must be true")
        source_id = str(row.get("source_example_id", sample_id))
        source_key = (str(row["source"]), source_id)
        if source_key in source_ids:
            errors.append(f"line {line}: duplicate source identity {source_key}")
        source_ids.add(source_key)
        if row.get("audio_sha256"):
            audio_hash = str(row["audio_sha256"])
            if audio_hash in audio_hashes:
                errors.append(
                    f"line {line}: duplicate audio hash also used by {audio_hashes[audio_hash]}"
                )
            audio_hashes[audio_hash] = sample_id

    if purpose == "custom":
        for speaker, split_values in speakers.items():
            if len(split_values) > 1:
                errors.append(
                    f"speaker {speaker!r} occurs in multiple splits: {sorted(split_values)}"
                )

    protected_ids: set[str] = set()
    protected_hashes: set[str] = set()
    protected_sources: set[tuple[str, str]] = set()
    for protected in against_eval:
        protected_rows, protected_errors = _read_rows(protected)
        errors.extend(f"protected {protected}: {message}" for message in protected_errors)
        for row in protected_rows:
            protected_ids.add(str(row.get("id", "")))
            if row.get("audio_sha256"):
                protected_hashes.add(str(row["audio_sha256"]))
            protected_sources.add(
                (
                    str(row.get("source", "")),
                    str(row.get("source_example_id", row.get("id", ""))),
                )
            )
    for row in rows:
        source_key = (
            str(row.get("source", "")),
            str(row.get("source_example_id", row.get("id", ""))),
        )
        if str(row.get("id")) in protected_ids or source_key in protected_sources:
            errors.append(f"evaluation overlap for id {row.get('id')!r}")
        if row.get("audio_sha256") in protected_hashes:
            errors.append(f"evaluation audio overlap for id {row.get('id')!r}")

    checked_audio = 0
    if check_audio and rows and len(errors) < max_errors:
        artifact_dir = _manifest_path(manifest).parent
        with tempfile.TemporaryDirectory(prefix="asr-validate-") as temporary:
            try:
                paths = stage_archived_audio(rows, artifact_dir, Path(temporary))
            except Exception as exc:
                errors.append(f"audio staging failed: {exc}")
                paths = {}
            for row in rows:
                path = paths.get(str(row["id"]))
                if path is None:
                    continue
                try:
                    duration, channels, rate, width = _wav_info(path)
                    if (channels, rate, width) != (1, 16000, 2):
                        errors.append(
                            f"{row['id']}: expected mono/16000Hz/PCM16, got "
                            f"{channels}ch/{rate}Hz/{width * 8}-bit"
                        )
                    expected = float(row["duration"])
                    if abs(duration - expected) > max(0.05, expected * 0.01):
                        errors.append(
                            f"{row['id']}: duration {duration:.3f}s differs from manifest "
                            f"{expected:.3f}s"
                        )
                    if row.get("audio_sha256") and _sha256(path) != row["audio_sha256"]:
                        errors.append(f"{row['id']}: audio SHA256 mismatch")
                    checked_audio += 1
                except Exception as exc:
                    errors.append(f"{row['id']}: invalid WAV: {exc}")
                if len(errors) >= max_errors:
                    break

    return {
        "ok": not errors,
        "manifest": str(_manifest_path(manifest)),
        "purpose": purpose,
        "records": len(rows),
        "audio_checked": checked_audio,
        "errors": errors[:max_errors],
        "warnings": warnings,
        "language_counts": {
            language: sum(row.get("language") == language for row in rows)
            for language in sorted(LANGUAGES)
        },
        "split_counts": {
            split: sum(row.get("split") == split for row in rows) for split in sorted(SPLITS)
        },
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate(
        args.manifest,
        args.purpose,
        args.against_eval_manifest,
        args.check_audio,
        args.require_human_verification,
        args.max_errors,
    )
    if args.report:
        _write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
