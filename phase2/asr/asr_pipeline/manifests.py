"""Streaming JSON/JSONL manifest I/O for canonical ASR records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping

from .schema import CanonicalRecord, SchemaError, validate_record


class ManifestError(ValueError):
    """Raised for malformed, ambiguous, or unsafe manifest operations."""


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    """Result of writing a manifest."""

    path: Path
    rows: int
    duration_seconds: float
    sha256: str


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without reading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "manifest.jsonl"
    return candidate


def _parse_row(raw: Any, *, location: str, validate: bool) -> CanonicalRecord | dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{location}: manifest row must be a JSON object")
    if not validate:
        return dict(raw)
    try:
        return CanonicalRecord.from_mapping(raw)
    except SchemaError as exc:
        raise ManifestError(f"{location}: {exc}") from exc


def iter_manifest(
    path: str | os.PathLike[str], *, validate: bool = True
) -> Iterator[CanonicalRecord | dict[str, Any]]:
    """Yield records from JSONL or a JSON array/object without bulk loading JSONL."""

    manifest_path = _manifest_path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"manifest does not exist: {manifest_path}")

    if manifest_path.suffix.lower() == ".jsonl":
        with manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(
                        f"{manifest_path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                yield _parse_row(
                    raw,
                    location=f"{manifest_path}:{line_number}",
                    validate=validate,
                )
        return

    if manifest_path.suffix.lower() != ".json":
        raise ManifestError("manifest filename must end in .jsonl or .json")
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{manifest_path}: invalid JSON: {exc.msg}") from exc
    rows = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ManifestError(f"{manifest_path}: JSON manifest must be an array or records object")
    for index, raw in enumerate(rows):
        yield _parse_row(raw, location=f"{manifest_path}:records[{index}]", validate=validate)


def read_manifest(
    path: str | os.PathLike[str], *, validate: bool = True
) -> list[CanonicalRecord] | list[dict[str, Any]]:
    """Read a complete manifest; prefer :func:`iter_manifest` for large files."""

    return list(iter_manifest(path, validate=validate))


def _jsonable_record(value: CanonicalRecord | Mapping[str, Any]) -> dict[str, Any]:
    record = validate_record(value)
    result = record.to_dict()
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ManifestError(
            f"record {record.id!r} is not JSON serializable; freeze audio to a path first"
        ) from exc
    return result


def write_manifest(
    path: str | os.PathLike[str],
    records: Iterable[CanonicalRecord | Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> ManifestSummary:
    """Atomically write canonical JSONL, rejecting duplicate IDs.

    Existing manifests are immutable by default; callers must opt into an
    overwrite explicitly for disposable intermediate artifacts.
    """

    manifest_path = Path(path)
    if manifest_path.suffix.lower() != ".jsonl":
        raise ManifestError("write_manifest requires a .jsonl destination")
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
    )
    temporary_path = Path(temporary_name)
    seen_ids: set[str] = set()
    rows = 0
    duration_seconds = 0.0
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for value in records:
                item = _jsonable_record(value)
                record_id = item["id"]
                if record_id in seen_ids:
                    raise ManifestError(f"duplicate manifest id: {record_id}")
                seen_ids.add(record_id)
                stream.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        sort_keys=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                stream.write("\n")
                rows += 1
                duration_seconds += float(item["duration"])
            stream.flush()
            os.fsync(stream.fileno())
        #codec*change*+2026-09-17: A manifest is exposed only after every row
        # validates and the complete temporary file has been flushed.
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return ManifestSummary(
        path=manifest_path,
        rows=rows,
        duration_seconds=duration_seconds,
        sha256=sha256_file(manifest_path),
    )


def resolve_audio_path(
    record: CanonicalRecord | Mapping[str, Any], manifest_path: str | os.PathLike[str]
) -> Path:
    """Resolve a manifest audio path relative to its manifest directory."""

    canonical = validate_record(record)
    if not isinstance(canonical.audio, (str, os.PathLike)):
        raise ManifestError(f"record {canonical.id!r} audio is not a persisted path")
    audio_path = Path(canonical.audio)
    if not audio_path.is_absolute():
        audio_path = Path(manifest_path).parent / audio_path
    return audio_path.resolve()

