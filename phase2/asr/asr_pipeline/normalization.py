"""Versioned, script-preserving text normalization for ASR metrics."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable

from .artifacts import sha256_json


NORMALIZATION_VERSION = "asr-unicode-nfkc-v1"
_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


@dataclass(frozen=True)
class NormalizationConfig:
    """A serialized policy; changing any field changes metric fingerprints."""

    version: str = NORMALIZATION_VERSION
    unicode_form: str = "NFKC"
    casefold: bool = True
    punctuation: str = "space"
    collapse_whitespace: bool = True
    strip_whitespace: bool = True
    cer_remove_whitespace: bool = True

    def __post_init__(self) -> None:
        if self.version != NORMALIZATION_VERSION:
            raise ValueError(f"unsupported normalization version: {self.version}")
        if self.unicode_form not in {"NFC", "NFKC"}:
            raise ValueError("unicode_form must be NFC or NFKC")
        if self.punctuation not in {"keep", "space"}:
            raise ValueError("punctuation must be keep or space")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


DEFAULT_NORMALIZATION = NormalizationConfig()


def _space_punctuation(text: str) -> str:
    #codec*change*+2026-09-17: Unicode category checks handle both Latin
    # punctuation and Hindi danda without transliterating either script.
    return "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)


def normalize_text(
    text: str,
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> str:
    """Normalize metric text while preserving the original writing system."""

    if not isinstance(text, str):
        raise TypeError("ASR metric text must be a string")
    normalized = unicodedata.normalize(config.unicode_form, text)
    if config.casefold:
        normalized = normalized.casefold()
    if config.punctuation == "space":
        normalized = _space_punctuation(normalized)
    if config.collapse_whitespace:
        normalized = _WHITESPACE_RE.sub(" ", normalized)
    if config.strip_whitespace:
        normalized = normalized.strip()
    return normalized


def word_tokens(
    text: str,
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> list[str]:
    normalized = normalize_text(text, config)
    return normalized.split() if normalized else []


def character_tokens(
    text: str,
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> list[str]:
    normalized = normalize_text(text, config)
    if config.cer_remove_whitespace:
        normalized = "".join(normalized.split())
    return list(normalized)


def normalized_phrase_tokens(
    phrase: str,
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> tuple[str, ...]:
    return tuple(word_tokens(phrase, config))


def contains_normalized_phrase(
    text: str,
    phrase: str,
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> bool:
    """Check a normalized token phrase without changing scripts or spelling."""

    haystack = word_tokens(text, config)
    needle = list(normalized_phrase_tokens(phrase, config))
    if not needle:
        return False
    width = len(needle)
    return any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1))


def normalize_many(
    texts: Iterable[str],
    config: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> list[str]:
    return [normalize_text(text, config) for text in texts]

