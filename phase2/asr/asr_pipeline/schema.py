"""Canonical dataset schema shared by training and evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from os import PathLike
from typing import Any, Mapping


CANONICAL_FIELDS: tuple[str, ...] = (
    "id",
    "audio",
    "text",
    "language",
    "speaker_id",
    "duration",
    "source",
    "split",
)
ALLOWED_LANGUAGES: frozenset[str] = frozenset({"en", "hi", "hi-en"})
ALLOWED_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})


class SchemaError(ValueError):
    """Raised when a row cannot satisfy the canonical ASR schema."""


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name!r} must be a non-empty string")
    return value


def _validate_audio(value: Any) -> Any:
    """Validate an audio reference without decoding or mutating it."""

    if isinstance(value, (str, PathLike)):
        if not str(value).strip():
            raise SchemaError("'audio' must not be an empty path")
        return value
    #codec*change*+2026-09-17: Raw HF audio belongs on AdapterExample;
    # canonical rows are persistable and therefore always point to frozen WAV.
    raise SchemaError("'audio' must be a non-empty persisted path")


@dataclass(frozen=True, slots=True)
class CanonicalRecord:
    """One validated ASR manifest row.

    The eight named attributes are the required canonical fields.  Additional
    provenance fields live in ``extensions`` and are round-tripped unchanged.
    Transcript text is validated but never stripped, normalized, or
    transliterated.
    """

    id: str
    audio: str | PathLike[str]
    text: str
    language: str
    speaker_id: str
    duration: float
    source: str
    split: str
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "id"))
        _validate_audio(self.audio)
        _required_text(self.text, "text")
        object.__setattr__(
            self, "speaker_id", _required_text(self.speaker_id, "speaker_id")
        )
        object.__setattr__(self, "source", _required_text(self.source, "source"))

        if self.language not in ALLOWED_LANGUAGES:
            allowed = ", ".join(sorted(ALLOWED_LANGUAGES))
            raise SchemaError(f"'language' must be one of {allowed}; got {self.language!r}")
        if self.split not in ALLOWED_SPLITS:
            allowed = ", ".join(sorted(ALLOWED_SPLITS))
            raise SchemaError(f"'split' must be one of {allowed}; got {self.split!r}")

        if isinstance(self.duration, bool):
            raise SchemaError("'duration' must be a positive finite number")
        try:
            duration = float(self.duration)
        except (TypeError, ValueError) as exc:
            raise SchemaError("'duration' must be a positive finite number") from exc
        if not math.isfinite(duration) or duration <= 0.0:
            raise SchemaError("'duration' must be a positive finite number")
        object.__setattr__(self, "duration", duration)

        if not isinstance(self.extensions, Mapping):
            raise SchemaError("extensions must be a mapping")
        collisions = set(self.extensions).intersection(CANONICAL_FIELDS)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise SchemaError(f"extension fields collide with canonical fields: {names}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CanonicalRecord":
        """Build and validate a record while preserving optional fields."""

        if not isinstance(value, Mapping):
            raise SchemaError("manifest row must be a mapping")
        missing = [name for name in CANONICAL_FIELDS if name not in value]
        if missing:
            raise SchemaError(f"missing canonical fields: {', '.join(missing)}")
        extensions = {key: item for key, item in value.items() if key not in CANONICAL_FIELDS}
        return cls(
            id=value["id"],
            audio=value["audio"],
            text=value["text"],
            language=value["language"],
            speaker_id=value["speaker_id"],
            duration=value["duration"],
            source=value["source"],
            split=value["split"],
            extensions=extensions,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return canonical fields first, followed by optional provenance."""

        result: dict[str, Any] = {
            "id": self.id,
            "audio": str(self.audio) if isinstance(self.audio, PathLike) else self.audio,
            "text": self.text,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "duration": self.duration,
            "source": self.source,
            "split": self.split,
        }
        result.update(self.extensions)
        return result


def validate_record(value: CanonicalRecord | Mapping[str, Any]) -> CanonicalRecord:
    """Return ``value`` as a validated :class:`CanonicalRecord`."""

    if isinstance(value, CanonicalRecord):
        return value
    return CanonicalRecord.from_mapping(value)
