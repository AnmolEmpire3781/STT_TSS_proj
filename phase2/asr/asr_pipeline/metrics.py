"""Dependency-light ASR metrics with reproducible normalization and slices."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .normalization import (
    DEFAULT_NORMALIZATION,
    NormalizationConfig,
    character_tokens,
    contains_normalized_phrase,
    word_tokens,
)


METRICS_SCHEMA_VERSION = "asr-metrics-v1"


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Memory-bounded Levenshtein distance."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class KeywordExpectation:
    term: str
    aliases: tuple[str, ...] = ()

    @property
    def candidates(self) -> tuple[str, ...]:
        return (self.term, *self.aliases)


def coerce_keyword_expectations(value: Any) -> list[KeywordExpectation]:
    """Accept simple strings and ``{term, aliases}`` suite annotations."""

    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, Mapping):
        if any(key in value for key in ("term", "keyword", "canonical")):
            value = [value]
        else:
            value = [
                {"term": term, "aliases": aliases}
                for term, aliases in value.items()
            ]
    if not isinstance(value, Sequence):
        raise TypeError("keywords must be a string, mapping, or sequence")

    expectations: list[KeywordExpectation] = []
    for item in value:
        if isinstance(item, str):
            term, aliases = item, ()
        elif isinstance(item, Mapping):
            term = item.get("term") or item.get("keyword") or item.get("canonical")
            alias_value = item.get("aliases", ())
            if isinstance(alias_value, str):
                alias_value = [alias_value]
            if not isinstance(alias_value, Sequence):
                raise TypeError("keyword aliases must be a string or sequence")
            aliases = tuple(str(alias) for alias in alias_value if str(alias).strip())
        else:
            raise TypeError(f"invalid keyword entry: {item!r}")
        if not isinstance(term, str) or not term.strip():
            raise ValueError(f"keyword term must be non-empty: {term!r}")
        expectations.append(KeywordExpectation(term=term, aliases=aliases))
    return expectations


@dataclass
class MetricAccumulator:
    normalization: NormalizationConfig = DEFAULT_NORMALIZATION
    sample_count: int = 0
    word_errors: int = 0
    reference_words: int = 0
    character_errors: int = 0
    reference_characters: int = 0
    keyword_matches: int = 0
    reference_keywords: int = 0
    latencies: list[float] = field(default_factory=list)
    keyword_terms: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )

    def add(
        self,
        reference: str,
        hypothesis: str,
        *,
        keywords: Any = None,
        latency_seconds: float | None = None,
    ) -> None:
        reference_word_tokens = word_tokens(reference, self.normalization)
        hypothesis_word_tokens = word_tokens(hypothesis, self.normalization)
        reference_character_tokens = character_tokens(reference, self.normalization)
        hypothesis_character_tokens = character_tokens(hypothesis, self.normalization)

        self.sample_count += 1
        self.word_errors += edit_distance(reference_word_tokens, hypothesis_word_tokens)
        self.reference_words += len(reference_word_tokens)
        self.character_errors += edit_distance(
            reference_character_tokens, hypothesis_character_tokens
        )
        self.reference_characters += len(reference_character_tokens)

        for expectation in coerce_keyword_expectations(keywords):
            matched = any(
                contains_normalized_phrase(hypothesis, candidate, self.normalization)
                for candidate in expectation.candidates
            )
            self.reference_keywords += 1
            self.keyword_matches += int(matched)
            per_term = self.keyword_terms[expectation.term]
            per_term[0] += int(matched)
            per_term[1] += 1

        if latency_seconds is not None:
            latency = float(latency_seconds)
            if not math.isfinite(latency) or latency < 0:
                raise ValueError(f"invalid latency: {latency_seconds!r}")
            self.latencies.append(latency)

    def result(self) -> dict[str, Any]:
        per_term = {
            term: {
                "accuracy": _ratio(counts[0], counts[1]),
                "matches": counts[0],
                "references": counts[1],
            }
            for term, counts in sorted(self.keyword_terms.items())
        }
        return {
            "sample_count": self.sample_count,
            "wer": _ratio(self.word_errors, self.reference_words),
            "word_errors": self.word_errors,
            "reference_words": self.reference_words,
            "cer": _ratio(self.character_errors, self.reference_characters),
            "character_errors": self.character_errors,
            "reference_characters": self.reference_characters,
            "domain_keyword_accuracy": _ratio(
                self.keyword_matches, self.reference_keywords
            ),
            "keyword_matches": self.keyword_matches,
            "reference_keywords": self.reference_keywords,
            "domain_keywords": per_term,
            "latency_sample_count": len(self.latencies),
            "p50_latency_seconds": _percentile(self.latencies, 0.50),
            "p95_latency_seconds": _percentile(self.latencies, 0.95),
        }


def _row_latency(row: Mapping[str, Any]) -> float | None:
    if row.get("latency_included") is False:
        return None
    value = row.get("latency_seconds", row.get("inference_seconds"))
    return None if value is None else float(value)


def _add_row(accumulator: MetricAccumulator, row: Mapping[str, Any]) -> None:
    try:
        reference = row["reference"]
        hypothesis = row["hypothesis"]
    except KeyError as exc:
        raise ValueError(f"prediction row is missing {exc.args[0]!r}") from exc
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("prediction reference and hypothesis must be strings")
    accumulator.add(
        reference,
        hypothesis,
        keywords=row.get("keywords", row.get("domain_keywords")),
        latency_seconds=_row_latency(row),
    )


def compute_evaluation_metrics(
    predictions: Iterable[Mapping[str, Any]],
    *,
    normalization: NormalizationConfig = DEFAULT_NORMALIZATION,
) -> dict[str, Any]:
    """Compute micro-averaged global, language, source, and joint slices."""

    overall = MetricAccumulator(normalization)
    by_language: dict[str, MetricAccumulator] = {}
    by_source: dict[str, MetricAccumulator] = {}
    by_language_source: dict[str, MetricAccumulator] = {}

    for row in predictions:
        _add_row(overall, row)
        language = str(row.get("language", "<missing>"))
        source = str(row.get("source", "<missing>"))
        language_accumulator = by_language.setdefault(
            language, MetricAccumulator(normalization)
        )
        source_accumulator = by_source.setdefault(source, MetricAccumulator(normalization))
        joint_key = f"{language}|{source}"
        joint_accumulator = by_language_source.setdefault(
            joint_key, MetricAccumulator(normalization)
        )
        _add_row(language_accumulator, row)
        _add_row(source_accumulator, row)
        _add_row(joint_accumulator, row)

    #codec*change*+2026-09-17: Keep named slice maps in the stable output so
    # comparisons cannot accidentally hide a regression in one language/source.
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "normalization": normalization.to_dict(),
        "normalization_fingerprint": normalization.fingerprint,
        "overall": overall.result(),
        "by_language": {
            key: value.result() for key, value in sorted(by_language.items())
        },
        "by_source": {
            key: value.result() for key, value in sorted(by_source.items())
        },
        "by_language_source": {
            key: value.result()
            for key, value in sorted(by_language_source.items())
        },
    }


#codec*change*+2026-09-17: Retain a concise public alias for scripts/tests.
compute_metrics = compute_evaluation_metrics
