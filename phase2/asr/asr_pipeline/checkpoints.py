"""Crash-safe checkpoint persistence and resume helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .safety import (
    COMPLETE_MARKER,
    SafetyError,
    seal_artifact,
    verify_complete_artifact,
)


CHECKPOINT_METADATA = "checkpoint-meta.json"
LATEST_CHECKPOINT = "latest_checkpoint.json"
_CHECKPOINT_NAME = re.compile(r"^checkpoint-(?P<step>[0-9]+)$")
_RESERVED_ROOT_FILES = frozenset(
    {"SHA256SUMS", COMPLETE_MARKER, CHECKPOINT_METADATA}
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be safely synced or restored."""


class IncompatibleCheckpointError(CheckpointError):
    """Raised when a checkpoint belongs to a different run fingerprint."""


@dataclass(frozen=True)
class CheckpointInfo:
    """Verified checkpoint metadata."""

    path: Path
    name: str
    step: int
    fingerprint_sha256: str
    fingerprint: Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_fingerprint(fingerprint: Any) -> str:
    try:
        return json.dumps(
            fingerprint,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"Run fingerprint must be JSON serializable: {exc}") from exc


def fingerprint_sha256(fingerprint: Any) -> str:
    """Return the canonical digest used for resume compatibility checks."""

    import hashlib

    return hashlib.sha256(_canonical_fingerprint(fingerprint).encode("utf-8")).hexdigest()


def _parse_checkpoint_name(name: str) -> int:
    match = _CHECKPOINT_NAME.fullmatch(name)
    if match is None:
        raise CheckpointError(
            f"Checkpoint name must use the checkpoint-N form, got: {name!r}"
        )
    return int(match.group("step"))


def _safe_checkpoint_name(name: str) -> str:
    if name != Path(name).name or name in {".", ".."}:
        raise CheckpointError(f"Checkpoint name must be one path component: {name!r}")
    _parse_checkpoint_name(name)
    return name


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise CheckpointError(f"Checkpoint root cannot be a symlink: {root}")
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise CheckpointError(f"Checkpoint contains a symlink: {candidate}")


def _remove_incomplete(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir() or not path.name.endswith(".incomplete"):
        raise CheckpointError(f"Unsafe incomplete checkpoint path: {path}")
    shutil.rmtree(path)


def _copy_source_checkpoint(source: Path, incomplete: Path) -> None:
    _reject_symlinks(source)
    shutil.copytree(source, incomplete)
    for reserved_name in _RESERVED_ROOT_FILES:
        reserved = incomplete / reserved_name
        if reserved.is_dir():
            raise CheckpointError(f"Reserved checkpoint path is a directory: {reserved}")
        if reserved.exists():
            reserved.unlink()


def _metadata_payload(name: str, fingerprint: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_name": name,
        "step": _parse_checkpoint_name(name),
        "fingerprint": fingerprint,
        "fingerprint_sha256": fingerprint_sha256(fingerprint),
        "created_at": _utc_now(),
    }


def _read_metadata(checkpoint: Path) -> dict[str, Any]:
    metadata_path = checkpoint / CHECKPOINT_METADATA
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise CheckpointError(f"Missing regular {CHECKPOINT_METADATA}: {checkpoint}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Invalid checkpoint metadata at {metadata_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"Checkpoint metadata must be a JSON object: {metadata_path}")
    return payload


def _verify_checkpoint_artifact(
    checkpoint: str | os.PathLike[str],
    expected_fingerprint: Any | None,
    logical_name: str,
) -> CheckpointInfo:
    raw_path = Path(checkpoint).expanduser()
    if raw_path.is_symlink():
        raise CheckpointError(f"Checkpoint cannot be a symlink: {raw_path}")
    try:
        path = raw_path.resolve(strict=True)
        verify_complete_artifact(path)
    except (OSError, SafetyError) as exc:
        raise CheckpointError(f"Invalid checkpoint artifact {raw_path}: {exc}") from exc
    safe_logical_name = _safe_checkpoint_name(logical_name)
    step = _parse_checkpoint_name(safe_logical_name)
    payload = _read_metadata(path)
    metadata_name = payload.get("checkpoint_name")
    metadata_step = payload.get("step")
    fingerprint = payload.get("fingerprint")
    stored_digest = payload.get("fingerprint_sha256")
    if metadata_name != safe_logical_name or metadata_step != step:
        raise CheckpointError(f"Checkpoint metadata does not match directory name: {path}")
    actual_digest = fingerprint_sha256(fingerprint)
    if stored_digest != actual_digest:
        raise CheckpointError(f"Checkpoint fingerprint digest is invalid: {path}")
    if expected_fingerprint is not None:
        expected_digest = fingerprint_sha256(expected_fingerprint)
        if actual_digest != expected_digest:
            raise IncompatibleCheckpointError(
                f"Checkpoint {safe_logical_name} has fingerprint {actual_digest}, expected {expected_digest}"
            )
    return CheckpointInfo(
        path=path,
        name=safe_logical_name,
        step=step,
        fingerprint_sha256=actual_digest,
        fingerprint=fingerprint,
    )


def verify_checkpoint(
    checkpoint: str | os.PathLike[str], expected_fingerprint: Any | None = None
) -> CheckpointInfo:
    """Verify completion, checksums, metadata, and optional compatibility."""

    path = Path(checkpoint).expanduser()
    return _verify_checkpoint_artifact(path, expected_fingerprint, path.name)


def _write_latest_pointer(root: Path, info: CheckpointInfo) -> None:
    _atomic_write_json(
        root / LATEST_CHECKPOINT,
        {
            "schema_version": 1,
            "checkpoint_name": info.name,
            "step": info.step,
            "fingerprint_sha256": info.fingerprint_sha256,
            "updated_at": _utc_now(),
        },
    )


def sync_checkpoint(
    local_checkpoint: str | os.PathLike[str],
    persistent_root: str | os.PathLike[str],
    fingerprint: Any,
    checkpoint_name: str | None = None,
) -> CheckpointInfo:
    """Seal and promote a local Trainer checkpoint into persistent storage."""

    source_raw = Path(local_checkpoint).expanduser()
    if source_raw.is_symlink():
        raise CheckpointError(f"Local checkpoint cannot be a symlink: {source_raw}")
    try:
        source = source_raw.resolve(strict=True)
    except OSError as exc:
        raise CheckpointError(f"Local checkpoint does not exist: {source_raw}") from exc
    if not source.is_dir():
        raise CheckpointError(f"Local checkpoint must be a directory: {source}")
    name = _safe_checkpoint_name(checkpoint_name or source.name)
    root = Path(persistent_root).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    if root.is_symlink():
        raise CheckpointError(f"Persistent checkpoint root cannot be a symlink: {root}")
    if _is_relative_to(root, source) or _is_relative_to(source, root):
        raise CheckpointError("Local and persistent checkpoint paths must not overlap")

    final = root / name
    incomplete = root / f"{name}.incomplete"
    if final.exists():
        info = verify_checkpoint(final, fingerprint)
        _write_latest_pointer(root, info)
        return info

    _remove_incomplete(incomplete)
    try:
        _copy_source_checkpoint(source, incomplete)
        _atomic_write_json(incomplete / CHECKPOINT_METADATA, _metadata_payload(name, fingerprint))
        seal_artifact(incomplete)
        promoted_info = _verify_checkpoint_artifact(incomplete, fingerprint, name)
        os.replace(incomplete, final)
        info = CheckpointInfo(
            path=final.resolve(strict=True),
            name=promoted_info.name,
            step=_parse_checkpoint_name(name),
            fingerprint_sha256=promoted_info.fingerprint_sha256,
            fingerprint=promoted_info.fingerprint,
        )
        info = verify_checkpoint(final, fingerprint)
        _write_latest_pointer(root, info)
        return info
    except Exception as exc:
        if incomplete.exists():
            _remove_incomplete(incomplete)
        if isinstance(exc, CheckpointError):
            raise
        if isinstance(exc, SafetyError):
            raise CheckpointError(f"Could not seal checkpoint {name}: {exc}") from exc
        raise CheckpointError(f"Could not sync checkpoint {name}: {exc}") from exc


def discover_latest_checkpoint(
    persistent_root: str | os.PathLike[str], expected_fingerprint: Any
) -> CheckpointInfo | None:
    """Return the newest complete compatible checkpoint, ignoring partial copies."""

    root = Path(persistent_root).expanduser().resolve(strict=False)
    if not root.exists():
        return None
    if not root.is_dir() or root.is_symlink():
        raise CheckpointError(f"Persistent checkpoint root must be a directory: {root}")
    candidates: list[tuple[int, Path]] = []
    for candidate in root.iterdir():
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        match = _CHECKPOINT_NAME.fullmatch(candidate.name)
        if match is not None and (candidate / COMPLETE_MARKER).is_file():
            candidates.append((int(match.group("step")), candidate))
    for _, candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        try:
            return verify_checkpoint(candidate, expected_fingerprint)
        except IncompatibleCheckpointError:
            continue
    return None


def restore_checkpoint(
    checkpoint_or_root: str | os.PathLike[str],
    destination_root: str | os.PathLike[str],
    expected_fingerprint: Any,
    checkpoint_name: str | None = None,
) -> Path:
    """Restore one verified checkpoint locally through an incomplete directory."""

    source_or_root = Path(checkpoint_or_root).expanduser().resolve(strict=False)
    if checkpoint_name is not None:
        source_info = verify_checkpoint(
            source_or_root / _safe_checkpoint_name(checkpoint_name), expected_fingerprint
        )
    elif _CHECKPOINT_NAME.fullmatch(source_or_root.name):
        source_info = verify_checkpoint(source_or_root, expected_fingerprint)
    else:
        source_info = discover_latest_checkpoint(source_or_root, expected_fingerprint)
        if source_info is None:
            raise FileNotFoundError(
                f"No complete compatible checkpoint found below {source_or_root}"
            )

    destination = Path(destination_root).expanduser().resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve(strict=True)
    if destination.is_symlink():
        raise CheckpointError(f"Restore destination cannot be a symlink: {destination}")
    if _is_relative_to(destination, source_info.path) or _is_relative_to(
        source_info.path, destination
    ):
        raise CheckpointError("Persistent and restore checkpoint paths must not overlap")
    final = destination / source_info.name
    incomplete = destination / f"{source_info.name}.incomplete"
    if final.exists():
        return verify_checkpoint(final, expected_fingerprint).path
    _remove_incomplete(incomplete)
    try:
        shutil.copytree(source_info.path, incomplete)
        _verify_checkpoint_artifact(
            incomplete, expected_fingerprint, source_info.name
        )
        os.replace(incomplete, final)
        return verify_checkpoint(final, expected_fingerprint).path
    except Exception as exc:
        if incomplete.exists():
            _remove_incomplete(incomplete)
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"Could not restore {source_info.name}: {exc}") from exc


def restore_latest_checkpoint(
    persistent_root: str | os.PathLike[str],
    destination_root: str | os.PathLike[str],
    expected_fingerprint: Any,
) -> Path:
    """Convenience wrapper for automatic latest-compatible resume."""

    return restore_checkpoint(persistent_root, destination_root, expected_fingerprint)


find_latest_compatible_checkpoint = discover_latest_checkpoint
