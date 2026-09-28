"""Base types and helpers for lazily streamed dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from ..audio import AudioError, duration_from_audio_payload
from ..revisions import require_immutable_revision
from ..schema import CanonicalRecord, SchemaError


class AdapterError(ValueError):
    """Raised when a source row or requested source role is invalid."""


@dataclass(frozen=True, slots=True)
class AdapterExample:
    """Canonical metadata plus the untouched source audio payload.

    ``source_audio`` deliberately stays outside the serializable manifest
    schema.  Snapshot creation converts it to WAV and then calls
    :meth:`to_canonical` with the final relative audio path.
    """

    id: str
    text: str
    language: str
    speaker_id: str
    duration: float
    source: str
    split: str
    source_audio: Any = field(repr=False, compare=False)
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        try:
            CanonicalRecord(
                id=self.id,
                audio="__unfrozen_source_audio__.wav",
                text=self.text,
                language=self.language,
                speaker_id=self.speaker_id,
                duration=self.duration,
                source=self.source,
                split=self.split,
                extensions=self.extensions,
            )
        except SchemaError as exc:
            raise AdapterError(str(exc)) from exc

    def to_canonical(
        self,
        audio: str | Path,
        *,
        extensions: Mapping[str, Any] | None = None,
    ) -> CanonicalRecord:
        """Attach a persisted relative WAV path and return a canonical row."""

        combined = dict(self.extensions)
        if extensions:
            combined.update(extensions)
        return CanonicalRecord(
            id=self.id,
            audio=audio,
            text=self.text,
            language=self.language,
            speaker_id=self.speaker_id,
            duration=self.duration,
            source=self.source,
            split=self.split,
            extensions=combined,
        )

    def metadata(self) -> dict[str, Any]:
        """Return canonical metadata before an audio path is assigned."""

        result: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "duration": self.duration,
            "source": self.source,
            "split": self.split,
        }
        result.update(self.extensions)
        return result


DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]


class DatasetAdapter(ABC):
    """Map one pinned public dataset into :class:`AdapterExample` objects."""

    key: str
    dataset_id: str
    config_name: str | None = None
    split_map: Mapping[str, str]

    @property
    def allowed_source_splits(self) -> tuple[str, ...]:
        """Source split names this adapter is permitted to expose."""

        return tuple(self.split_map)

    def canonical_split(self, source_split: str) -> str:
        """Map a source split to its protected canonical role."""

        try:
            return self.split_map[source_split]
        except KeyError as exc:
            allowed = ", ".join(self.allowed_source_splits)
            raise AdapterError(
                f"{self.key} source split {source_split!r} is not allowed; choose {allowed}"
            ) from exc

    def iter_source(
        self,
        source_split: str,
        revision: str,
        *,
        token: str | None = None,
        streaming: bool = True,
        dataset_loader: DatasetLoader | None = None,
        load_kwargs: Mapping[str, Any] | None = None,
    ) -> Iterator[AdapterExample]:
        """Stream and map a protected split from an immutable revision.

        ``dataset_loader`` is injectable for offline unit tests.  Public
        adapters reject non-streaming access by design.
        """

        canonical_split = self.canonical_split(source_split)
        pinned_revision = require_immutable_revision(revision)
        if not streaming:
            raise AdapterError("public dataset adapters require streaming=True")
        if dataset_loader is None:
            try:
                from datasets import load_dataset
            except ImportError as exc:  # pragma: no cover - dependency environment
                raise AdapterError("datasets is required to stream public sources") from exc
            dataset_loader = load_dataset
        kwargs: dict[str, Any] = {
            "split": source_split,
            "revision": pinned_revision,
            "streaming": True,
        }
        if token is not None:
            kwargs["token"] = token
        if load_kwargs:
            protected = {"split", "revision", "streaming", "token"}.intersection(load_kwargs)
            if protected:
                raise AdapterError(
                    f"load_kwargs cannot override protected options: {sorted(protected)}"
                )
            kwargs.update(load_kwargs)
        try:
            if self.config_name is None:
                rows = dataset_loader(self.dataset_id, **kwargs)
            else:
                rows = dataset_loader(self.dataset_id, self.config_name, **kwargs)
        except Exception as exc:
            raise AdapterError(
                f"failed to open {self.dataset_id}/{source_split}@{pinned_revision}: {exc}"
            ) from exc

        #codec*change*+2026-09-17: Mapping stays lazy and the source audio
        # object is passed through verbatim until a selected snapshot is frozen.
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise AdapterError(f"{self.key} row {index} is not a mapping")
            try:
                yield self.map_row(
                    row,
                    source_split=source_split,
                    canonical_split=canonical_split,
                    revision=pinned_revision,
                    index=index,
                )
            except (AdapterError, AudioError, SchemaError) as exc:
                raise AdapterError(
                    f"{self.key}/{source_split} row {index}: {exc}"
                ) from exc

    @abstractmethod
    def map_row(
        self,
        row: Mapping[str, Any],
        *,
        source_split: str,
        canonical_split: str,
        revision: str,
        index: int,
    ) -> AdapterExample:
        """Map one source row without mutating its audio payload."""

    def provenance(
        self,
        *,
        source_split: str,
        revision: str,
        source_example_id: str,
    ) -> dict[str, Any]:
        """Build common immutable source lineage fields."""

        result: dict[str, Any] = {
            "source_dataset": self.dataset_id,
            "source_split": source_split,
            "source_revision": revision,
            "source_example_id": source_example_id,
        }
        if self.config_name is not None:
            result["source_config"] = self.config_name
        return result


def first_value(
    row: Mapping[str, Any], keys: tuple[str, ...], *, required: bool = True
) -> Any:
    """Return the first present, non-null source value."""

    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    if required:
        raise AdapterError(f"missing required source field; expected one of {keys}")
    return None


def exact_text(row: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Read a non-empty transcript without changing its contents."""

    value = first_value(row, keys)
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"transcript field {keys} must be a non-empty string")
    return value


def stable_example_identity(
    adapter_key: str,
    row: Mapping[str, Any],
    *,
    source_split: str,
    index: int,
    identity_keys: tuple[str, ...],
) -> tuple[str, str]:
    """Create a portable stable ID and retain the original source identity."""

    source_value = first_value(row, identity_keys, required=False)
    if source_value is None:
        source_value = f"row-{index}"
    source_example_id = str(source_value)
    digest = hashlib.sha256(
        f"{adapter_key}\0{source_split}\0{source_example_id}".encode("utf-8")
    ).hexdigest()[:20]
    return f"{adapter_key}-{digest}", source_example_id


def source_audio(row: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    """Return an audio payload exactly as supplied by the source dataset."""

    value = first_value(row, keys)
    if isinstance(value, str) and not value.strip():
        raise AdapterError("source audio path is empty")
    return value


def source_duration(
    row: Mapping[str, Any],
    audio_value: Any,
    *,
    keys: tuple[str, ...] = ("duration", "duration_seconds", "audio_duration"),
) -> float:
    """Read source duration or infer it from decoded audio."""

    value = first_value(row, keys, required=False)
    if value is not None and not isinstance(value, bool):
        try:
            duration = float(value)
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"invalid source duration: {value!r}") from exc
        if math.isfinite(duration) and duration > 0:
            return duration
    try:
        return duration_from_audio_payload(audio_value)
    except AudioError as exc:
        raise AdapterError(
            "source row has no usable duration; provide duration or decoded audio"
        ) from exc


def source_speaker(
    row: Mapping[str, Any],
    *,
    source: str,
    keys: tuple[str, ...] = ("speaker_id", "speaker", "speaker_name"),
    fallback_id: str | None = None,
) -> str:
    """Read and namespace a speaker ID, or use an explicit safe fallback."""

    value = first_value(row, keys, required=False)
    if value is None or not str(value).strip():
        if fallback_id is None:
            raise AdapterError(f"missing speaker identity; expected one of {keys}")
        value = fallback_id
    return f"{source}:{value}"

