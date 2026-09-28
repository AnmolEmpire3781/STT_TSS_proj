"""Filesystem integrity and deletion guards for offline ASR artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CHECKSUMS_FILE = "SHA256SUMS"
COMPLETE_MARKER = "COMPLETE"

_SHA256_LINE = re.compile(r"^(?P<digest>[0-9a-f]{64})  (?P<path>.+)$")
_CONTROL_FILES = frozenset({CHECKSUMS_FILE, COMPLETE_MARKER})


class SafetyError(RuntimeError):
    """Raised when an artifact or cleanup path fails a safety invariant."""


@dataclass(frozen=True)
class ArtifactVerification:
    """Result of validating a sealed persistent artifact."""

    path: Path
    checked_files: int


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a regular file."""

    file_path = Path(path)
    if not file_path.is_file() or file_path.is_symlink():
        raise SafetyError(f"Checksum input must be a regular non-symlink file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(path: str | os.PathLike[str], *, strict: bool = False) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=strict)
    except (OSError, RuntimeError) as exc:
        raise SafetyError(f"Could not resolve path {path!s}: {exc}") from exc


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _strict_descendant(path: Path, root: Path) -> bool:
    return path != root and _is_relative_to(path, root)


def _payload_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise SafetyError(f"Artifact contains a symlink: {candidate}")
        if candidate.is_file() and candidate.relative_to(root).as_posix() not in _CONTROL_FILES:
            files.append(candidate)
    return files


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_sha256sums(artifact_dir: str | os.PathLike[str]) -> Path:
    """Create deterministic checksums for every payload file in an artifact."""

    root = _resolved(artifact_dir, strict=True)
    if not root.is_dir() or root.is_symlink():
        raise SafetyError(f"Artifact root must be a regular directory: {root}")
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in _payload_files(root)
    ]
    output = root / CHECKSUMS_FILE
    _atomic_write_text(output, "\n".join(lines) + ("\n" if lines else ""))
    return output


def mark_complete(artifact_dir: str | os.PathLike[str]) -> Path:
    """Add the completion marker after a checksum manifest exists."""

    root = _resolved(artifact_dir, strict=True)
    checksums = root / CHECKSUMS_FILE
    if not checksums.is_file() or checksums.is_symlink():
        raise SafetyError(f"Cannot mark artifact complete without {CHECKSUMS_FILE}: {root}")
    marker = root / COMPLETE_MARKER
    marker_payload = {
        "schema_version": "asr-artifact-complete-v1",
        "sha256sums": CHECKSUMS_FILE,
        "sha256sums_sha256": sha256_file(checksums),
    }
    _atomic_write_text(
        marker,
        json.dumps(marker_payload, sort_keys=True, indent=2) + "\n",
    )
    return marker


def _read_checksum_manifest(root: Path) -> dict[str, str]:
    manifest_path = root / CHECKSUMS_FILE
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise SafetyError(f"Missing regular {CHECKSUMS_FILE}: {root}")
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = _SHA256_LINE.fullmatch(raw_line)
        if match is None:
            raise SafetyError(
                f"Malformed {CHECKSUMS_FILE} line {line_number} in {root}"
            )
        relative_text = match.group("path")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative_text in _CONTROL_FILES:
            raise SafetyError(f"Unsafe checksum path in {root}: {relative_text}")
        normalized = relative.as_posix()
        if normalized in entries:
            raise SafetyError(f"Duplicate checksum path in {root}: {normalized}")
        entries[normalized] = match.group("digest")
    return entries


def verify_sha256sums(
    artifact_dir: str | os.PathLike[str], *, require_all_files: bool = True
) -> int:
    """Verify an artifact checksum manifest and return its file count."""

    root = _resolved(artifact_dir, strict=True)
    if not root.is_dir() or root.is_symlink():
        raise SafetyError(f"Artifact root must be a regular directory: {root}")
    entries = _read_checksum_manifest(root)
    for relative_text, expected in entries.items():
        candidate = root / Path(relative_text)
        resolved_candidate = _resolved(candidate, strict=True)
        if not _strict_descendant(resolved_candidate, root):
            raise SafetyError(f"Checksum path escapes artifact root: {relative_text}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise SafetyError(
                f"Checksum mismatch for {relative_text}: expected {expected}, got {actual}"
            )
    if require_all_files:
        actual_files = {
            path.relative_to(root).as_posix() for path in _payload_files(root)
        }
        listed_files = set(entries)
        if actual_files != listed_files:
            missing = sorted(actual_files - listed_files)
            stale = sorted(listed_files - actual_files)
            raise SafetyError(
                f"Checksum manifest does not match payload files; unlisted={missing}, missing={stale}"
            )
    return len(entries)


def verify_complete_artifact(
    artifact_dir: str | os.PathLike[str],
) -> ArtifactVerification:
    """Require a completion marker and a complete, valid checksum manifest."""

    root = _resolved(artifact_dir, strict=True)
    marker = root / COMPLETE_MARKER
    if not marker.is_file() or marker.is_symlink():
        raise SafetyError(f"Missing regular {COMPLETE_MARKER} marker: {root}")
    try:
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"Invalid {COMPLETE_MARKER} marker in {root}: {exc}") from exc
    expected_marker = {
        "schema_version": "asr-artifact-complete-v1",
        "sha256sums": CHECKSUMS_FILE,
        "sha256sums_sha256": sha256_file(root / CHECKSUMS_FILE),
    }
    if marker_payload != expected_marker:
        raise SafetyError(f"{COMPLETE_MARKER} does not match {CHECKSUMS_FILE}: {root}")
    checked_files = verify_sha256sums(root, require_all_files=True)
    return ArtifactVerification(path=root, checked_files=checked_files)


def seal_artifact(artifact_dir: str | os.PathLike[str]) -> ArtifactVerification:
    """Write checksums, mark an artifact complete, and verify the result."""

    root = _resolved(artifact_dir, strict=True)
    marker = root / COMPLETE_MARKER
    if marker.exists():
        raise SafetyError(f"Refusing to reseal an already complete artifact: {root}")
    write_sha256sums(root)
    mark_complete(root)
    return verify_complete_artifact(root)


def _contains_git_marker(path: Path) -> bool:
    return (path / ".git").exists() or any(
        candidate.exists() for candidate in path.rglob(".git")
    )


def _has_nested_mount_ancestor(path: Path) -> bool:
    current = path
    while current.parent != current:
        if os.path.ismount(current):
            return True
        current = current.parent
    return False


def _validate_root_shape(root: Path, label: str) -> None:
    anchor = Path(root.anchor) if root.anchor else None
    if anchor is not None and root == anchor:
        raise SafetyError(f"{label} cannot be a filesystem root: {root}")
    if root == _resolved(Path.home()):
        raise SafetyError(f"{label} cannot be the user home directory: {root}")
    if os.path.ismount(root):
        raise SafetyError(f"{label} cannot be a mount point: {root}")


def validate_cleanup_target(
    target: str | os.PathLike[str],
    *,
    work_root: str | os.PathLike[str],
    persistent_root: str | os.PathLike[str],
    repository_roots: Iterable[str | os.PathLike[str]] = (),
) -> Path:
    """Resolve and validate one disposable staging directory for deletion."""

    raw_target = Path(target).expanduser().absolute()
    raw_work_root = Path(work_root).expanduser().absolute()
    if raw_target.is_symlink() or raw_work_root.is_symlink():
        raise SafetyError("Cleanup target and work root cannot be symlinks")
    resolved_target = _resolved(raw_target, strict=True)
    resolved_work_root = _resolved(raw_work_root, strict=True)
    resolved_persistent_root = _resolved(persistent_root, strict=False)
    _validate_root_shape(resolved_work_root, "Work root")
    if not resolved_target.is_dir():
        raise SafetyError(f"Cleanup target must be an existing directory: {resolved_target}")
    if not _strict_descendant(resolved_target, resolved_work_root):
        raise SafetyError(
            f"Cleanup target must be a strict descendant of {resolved_work_root}: {resolved_target}"
        )
    if _is_relative_to(resolved_target, resolved_persistent_root) or _is_relative_to(
        resolved_persistent_root, resolved_target
    ):
        raise SafetyError(f"Cleanup target overlaps persistent storage: {resolved_target}")
    if os.path.ismount(resolved_target):
        raise SafetyError(f"Cleanup target cannot be a mount point: {resolved_target}")
    if _has_nested_mount_ancestor(resolved_target):
        raise SafetyError(
            f"Cleanup target cannot be inside a nested mount: {resolved_target}"
        )
    for repository_root in repository_roots:
        resolved_repository = _resolved(repository_root, strict=False)
        if _is_relative_to(resolved_target, resolved_repository) or _is_relative_to(
            resolved_repository, resolved_target
        ):
            raise SafetyError(
                f"Cleanup target overlaps a repository: {resolved_target}"
            )
    if _contains_git_marker(resolved_target):
        raise SafetyError(f"Cleanup target is or contains the active repository boundary: {resolved_target}")
    return resolved_target


def validate_persistent_artifacts(
    artifact_paths: Iterable[str | os.PathLike[str]],
    *,
    persistent_root: str | os.PathLike[str],
) -> tuple[ArtifactVerification, ...]:
    """Verify required sealed artifacts are strict descendants of persistent storage."""

    root = _resolved(persistent_root, strict=True)
    if not root.is_dir() or root.is_symlink():
        raise SafetyError(f"Persistent root must be a regular directory: {root}")
    verifications: list[ArtifactVerification] = []
    for raw_path in artifact_paths:
        raw = Path(raw_path)
        candidate = raw if raw.is_absolute() else root / raw
        if candidate.is_symlink():
            raise SafetyError(f"Persistent artifact cannot be a symlink: {candidate}")
        resolved_candidate = _resolved(candidate, strict=True)
        if not _strict_descendant(resolved_candidate, root):
            raise SafetyError(
                f"Persistent artifact must be a strict descendant of {root}: {resolved_candidate}"
            )
        verifications.append(verify_complete_artifact(resolved_candidate))
    if not verifications:
        raise SafetyError("At least one verified persistent artifact is required before cleanup")
    return tuple(verifications)
