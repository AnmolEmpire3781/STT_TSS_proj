"""Resolve mutable Hugging Face refs to immutable dataset revisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


_FULL_GIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


class RevisionError(RuntimeError):
    """Raised when a dataset revision cannot be pinned or verified."""


@dataclass(frozen=True, slots=True)
class ResolvedRevision:
    """Immutable source revision and its requested human-facing ref."""

    dataset_id: str
    requested_revision: str
    resolved_sha: str
    resolved_at_utc: str

    def to_dict(self) -> dict[str, str]:
        """Return safe lock metadata; authentication is intentionally absent."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResolvedRevision":
        """Validate a serialized revision lock."""

        try:
            result = cls(
                dataset_id=str(value["dataset_id"]),
                requested_revision=str(value["requested_revision"]),
                resolved_sha=str(value["resolved_sha"]),
                resolved_at_utc=str(value["resolved_at_utc"]),
            )
        except KeyError as exc:
            raise RevisionError(f"revision lock is missing {exc.args[0]!r}") from exc
        require_immutable_revision(result.resolved_sha)
        if not result.dataset_id.strip():
            raise RevisionError("revision lock dataset_id is empty")
        return result


def is_immutable_revision(revision: str) -> bool:
    """Return whether ``revision`` is a full Git commit SHA."""

    return bool(_FULL_GIT_SHA.fullmatch(revision or ""))


def require_immutable_revision(revision: str) -> str:
    """Reject branches, tags, and abbreviated SHAs for source iteration."""

    if not is_immutable_revision(revision):
        raise RevisionError(
            "dataset revision must be a resolved 40-character commit SHA; "
            "call resolve_dataset_revision first"
        )
    return revision.lower()


def resolve_dataset_revision(
    dataset_id: str,
    requested_revision: str = "main",
    *,
    token: str | None = None,
    api: Any | None = None,
) -> ResolvedRevision:
    """Resolve a dataset branch/tag/SHA through the Hugging Face Hub API.

    ``api`` is injectable for offline tests.  ``huggingface_hub`` is imported
    lazily so the rest of the data tooling stays importable without it.
    """

    if not dataset_id.strip():
        raise RevisionError("dataset_id must not be empty")
    if not requested_revision.strip():
        raise RevisionError("requested_revision must not be empty")
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - dependency environment
            raise RevisionError(
                "huggingface_hub is required to resolve public dataset revisions"
            ) from exc
        api = HfApi(token=token)
    try:
        info = api.dataset_info(
            repo_id=dataset_id, revision=requested_revision, token=token
        )
    except Exception as exc:  # Hub clients expose several transport exceptions.
        raise RevisionError(
            f"failed to resolve {dataset_id}@{requested_revision}: {exc}"
        ) from exc
    resolved_sha = getattr(info, "sha", None)
    if resolved_sha is None and isinstance(info, Mapping):
        resolved_sha = info.get("sha")
    if not isinstance(resolved_sha, str) or not is_immutable_revision(resolved_sha):
        raise RevisionError(
            f"Hub returned a non-immutable revision for {dataset_id}: {resolved_sha!r}"
        )
    #codec*change*+2026-09-17: Network resolution happens once; every stream
    # and resume operation consumes only this full commit SHA.
    return ResolvedRevision(
        dataset_id=dataset_id,
        requested_revision=requested_revision,
        resolved_sha=resolved_sha.lower(),
        resolved_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def write_revision_lock(
    path: str | os.PathLike[str],
    revision: ResolvedRevision,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically persist source revision metadata without credentials."""

    lock_path = Path(path)
    if lock_path.exists() and not overwrite:
        raise FileExistsError(f"revision lock already exists: {lock_path}")
    payload: dict[str, Any] = revision.to_dict()
    if metadata:
        forbidden = _credential_paths(metadata)
        if forbidden:
            raise RevisionError(
                f"credentials must not be stored in revision metadata: {sorted(forbidden)}"
            )
        payload["metadata"] = dict(metadata)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{lock_path.name}.", suffix=".tmp", dir=lock_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, lock_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return lock_path


def _credential_paths(value: Any, prefix: str = "metadata") -> set[str]:
    """Find credential-like keys recursively before serializing metadata."""

    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}"
            normalized = str(key).lower().replace("-", "_")
            if any(
                marker in normalized
                for marker in ("token", "secret", "password", "api_key", "authorization")
            ):
                found.add(path)
            found.update(_credential_paths(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.update(_credential_paths(nested, f"{prefix}[{index}]"))
    return found


def read_revision_lock(path: str | os.PathLike[str]) -> ResolvedRevision:
    """Read and validate a revision lock."""

    lock_path = Path(path)
    try:
        with lock_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RevisionError(f"cannot read revision lock {lock_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RevisionError(f"revision lock must contain a JSON object: {lock_path}")
    return ResolvedRevision.from_mapping(payload)
