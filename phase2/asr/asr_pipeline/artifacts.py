"""Atomic, checksummed artifact helpers used by the offline ASR tools."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SHA256SUMS_NAME = "SHA256SUMS"
COMPLETE_MARKER_NAME = "COMPLETE"
_IO_CHUNK_SIZE = 1024 * 1024


class ArtifactError(RuntimeError):
    """Raised when an artifact is incomplete, unsafe, or corrupt."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes suitable for fingerprints."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | os.PathLike[str], chunk_size: int = _IO_CHUNK_SIZE) -> str:
    """Hash a regular file without loading it into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        #codec*change*+2026-09-17: Keep the temporary file beside its target so
        # os.replace remains atomic on local Colab storage and mounted Drive.
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-", dir=str(path.parent)
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        return path
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> Path:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return _atomic_replace(Path(path), payload)


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    return _atomic_replace(Path(path), text.encode(encoding))


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    indent: int = 2,
) -> Path:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )
        + "\n"
    )
    return atomic_write_text(path, payload)


def atomic_write_jsonl(
    path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]
) -> Path:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )
    return atomic_write_text(path, payload)


def _relative_artifact_path(root: Path, path: Path) -> str:
    if path.is_symlink():
        raise ArtifactError(f"artifact entry must not be a symlink: {path}")
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactError(f"artifact file is outside root: {path}") from exc
    if resolved_path == resolved_root or not resolved_path.is_file():
        raise ArtifactError(f"artifact entry is not a regular file: {path}")
    return relative.as_posix()


def _default_checksum_files(root: Path) -> list[Path]:
    ignored_root_paths = {SHA256SUMS_NAME, COMPLETE_MARKER_NAME}
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ArtifactError(f"artifact contains a symlink: {candidate}")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative in ignored_root_paths:
            continue
        if ".tmp-" in candidate.name:
            continue
        files.append(candidate)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def write_sha256sums(
    root: str | os.PathLike[str],
    files: Sequence[str | os.PathLike[str]] | None = None,
) -> Path:
    """Atomically write conventional SHA256SUMS for files beneath ``root``."""

    artifact_root = Path(root)
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise ArtifactError(f"artifact root is not a directory: {artifact_root}")
    selected = _default_checksum_files(artifact_root) if files is None else [Path(p) for p in files]
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate in selected:
        if not candidate.is_absolute():
            candidate = artifact_root / candidate
        relative = _relative_artifact_path(artifact_root, candidate)
        if relative in seen:
            raise ArtifactError(f"duplicate checksum entry: {relative}")
        if relative in {SHA256SUMS_NAME, COMPLETE_MARKER_NAME}:
            continue
        seen.add(relative)
        entries.append((relative, sha256_file(candidate)))
    entries.sort()
    contents = "".join(f"{digest}  {relative}\n" for relative, digest in entries)
    return atomic_write_text(artifact_root / SHA256SUMS_NAME, contents)


def read_sha256sums(path: str | os.PathLike[str]) -> dict[str, str]:
    sums_path = Path(path)
    entries: dict[str, str] = {}
    try:
        lines = sums_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ArtifactError(f"checksum manifest is missing: {sums_path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ArtifactError(
                f"invalid checksum line {line_number} in {sums_path}"
            ) from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ArtifactError(
                f"invalid SHA-256 on line {line_number} in {sums_path}"
            )
        normalized = Path(relative).as_posix()
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ArtifactError(f"unsafe checksum path on line {line_number}: {relative}")
        if normalized in entries:
            raise ArtifactError(f"duplicate checksum path: {normalized}")
        entries[normalized] = digest
    return entries


def verify_sha256sums(
    root: str | os.PathLike[str],
    sums_path: str | os.PathLike[str] | None = None,
    *,
    require_all_files: bool = True,
) -> bool:
    """Verify every listed checksum, raising ``ArtifactError`` on any mismatch."""

    artifact_root = Path(root).resolve()
    checksum_path = Path(sums_path) if sums_path is not None else artifact_root / SHA256SUMS_NAME
    if not checksum_path.is_absolute():
        checksum_path = artifact_root / checksum_path
    try:
        checksum_path.resolve().relative_to(artifact_root)
    except ValueError as exc:
        raise ArtifactError(
            f"checksum manifest is outside artifact root: {checksum_path}"
        ) from exc
    entries = read_sha256sums(checksum_path)
    for relative, expected in entries.items():
        candidate = (artifact_root / relative).resolve()
        try:
            candidate.relative_to(artifact_root)
        except ValueError as exc:
            raise ArtifactError(f"checksum path escapes artifact root: {relative}") from exc
        if not candidate.is_file():
            raise ArtifactError(f"checksummed file is missing: {relative}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ArtifactError(
                f"checksum mismatch for {relative}: expected {expected}, got {actual}"
            )
    if require_all_files:
        actual_paths = {
            candidate.relative_to(artifact_root).as_posix()
            for candidate in _default_checksum_files(artifact_root)
        }
        try:
            checksum_relative = checksum_path.resolve().relative_to(
                artifact_root
            ).as_posix()
            actual_paths.discard(checksum_relative)
        except ValueError:
            pass
        listed_paths = set(entries)
        if actual_paths != listed_paths:
            unlisted = sorted(actual_paths - listed_paths)
            missing = sorted(listed_paths - actual_paths)
            raise ArtifactError(
                "checksum manifest does not match artifact files; "
                f"unlisted={unlisted}, missing={missing}"
            )
    return True


def mark_complete(
    root: str | os.PathLike[str],
    files: Sequence[str | os.PathLike[str]] | None = None,
) -> Path:
    """Checksum an artifact and atomically add its COMPLETE marker last."""

    artifact_root = Path(root)
    if (artifact_root / COMPLETE_MARKER_NAME).exists():
        raise ArtifactError(f"refusing to reseal complete artifact: {artifact_root}")
    sums_path = write_sha256sums(artifact_root, files=files)
    verify_sha256sums(artifact_root, sums_path)
    marker = {
        "schema_version": "asr-artifact-complete-v1",
        "sha256sums": SHA256SUMS_NAME,
        "sha256sums_sha256": sha256_file(sums_path),
    }
    return atomic_write_json(artifact_root / COMPLETE_MARKER_NAME, marker)


def is_complete(root: str | os.PathLike[str], *, verify: bool = False) -> bool:
    artifact_root = Path(root)
    marker_path = artifact_root / COMPLETE_MARKER_NAME
    sums_path = artifact_root / SHA256SUMS_NAME
    if not marker_path.is_file() or not sums_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("schema_version") != "asr-artifact-complete-v1":
            return False
        if marker.get("sha256sums_sha256") != sha256_file(sums_path):
            return False
        if verify:
            verify_sha256sums(artifact_root, sums_path)
    except (ArtifactError, json.JSONDecodeError, OSError, TypeError):
        return False
    return True


def require_complete(root: str | os.PathLike[str], *, verify: bool = True) -> Path:
    artifact_root = Path(root)
    if not is_complete(artifact_root, verify=verify):
        raise ArtifactError(f"artifact is not complete and verified: {artifact_root}")
    return artifact_root
