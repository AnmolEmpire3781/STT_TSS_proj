"""Deterministic bounded-memory sampling for streamed ASR sources."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, TypeVar


T = TypeVar("T")


class SamplingError(ValueError):
    """Raised when sampling inputs cannot meet the requested policy."""


@dataclass(frozen=True, slots=True)
class HourSelection(Generic[T]):
    """Selected items and accounting for an hour-based snapshot."""

    items: tuple[T, ...]
    target_seconds: float
    actual_seconds: float
    exhausted: bool
    seed: int | str
    buffer_size: int

    @property
    def actual_hours(self) -> float:
        """Selected duration in hours."""

        return self.actual_seconds / 3600.0


def _seed_integer(seed: int | str) -> int:
    encoded = str(seed).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big")


def bounded_shuffle(
    items: Iterable[T], *, seed: int | str, buffer_size: int = 1_000
) -> Iterator[T]:
    """Yield a deterministic approximate shuffle using fixed memory.

    At most ``buffer_size`` input objects are retained, so this can operate on
    Hugging Face streaming datasets without materializing their full split.
    """

    if buffer_size <= 0:
        raise SamplingError("buffer_size must be positive")
    generator = random.Random(_seed_integer(seed))
    iterator = iter(items)
    buffer: list[T] = []
    for _ in range(buffer_size):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break
    if not buffer:
        return
    for item in iterator:
        index = generator.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    generator.shuffle(buffer)
    yield from buffer


def _default_duration(item: Any) -> float:
    if isinstance(item, Mapping):
        value = item.get("duration")
    else:
        value = getattr(item, "duration", None)
    if isinstance(value, bool):
        raise SamplingError("duration must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SamplingError("item is missing a numeric duration") from exc
    if not math.isfinite(result) or result <= 0:
        raise SamplingError(f"duration must be positive and finite; got {value!r}")
    return result


def deterministic_hour_sample(
    items: Iterable[T],
    *,
    target_hours: float,
    seed: int | str,
    buffer_size: int = 1_000,
    duration_getter: Callable[[T], float] | None = None,
) -> HourSelection[T]:
    """Select a seeded streamed prefix until it reaches the requested hours."""

    if not math.isfinite(target_hours) or target_hours <= 0:
        raise SamplingError("target_hours must be positive and finite")
    target_seconds = target_hours * 3600.0
    duration_of = duration_getter or _default_duration
    selected: list[T] = []
    actual_seconds = 0.0
    exhausted = True
    for item in bounded_shuffle(items, seed=seed, buffer_size=buffer_size):
        try:
            duration = float(duration_of(item))
        except (TypeError, ValueError) as exc:
            raise SamplingError("duration_getter returned a non-numeric value") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise SamplingError(f"duration must be positive and finite; got {duration!r}")
        selected.append(item)
        actual_seconds += duration
        if actual_seconds >= target_seconds:
            exhausted = False
            break
    #codec*change*+2026-09-17: Only the bounded shuffle buffer and selected
    # snapshot are retained; complete public splits are never materialized.
    return HourSelection(
        items=tuple(selected),
        target_seconds=target_seconds,
        actual_seconds=actual_seconds,
        exhausted=exhausted,
        seed=seed,
        buffer_size=buffer_size,
    )


def _default_id(item: Any) -> str:
    value = item.get("id") if isinstance(item, Mapping) else getattr(item, "id", None)
    if not isinstance(value, str) or not value:
        raise SamplingError("every quota-sampled item needs a non-empty string id")
    return value


def _default_stratum(item: Any) -> str:
    value = item.get("source") if isinstance(item, Mapping) else getattr(item, "source", None)
    if not isinstance(value, str) or not value:
        raise SamplingError("every quota-sampled item needs a non-empty source")
    return value


def _stable_rank(seed: int | str, stratum: str, item_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{stratum}\0{item_id}".encode("utf-8")).digest()


def deterministic_nested_quota_sample(
    items: Iterable[T],
    quotas_by_view: Mapping[str, Mapping[str, int]],
    *,
    seed: int | str,
    id_getter: Callable[[T], str] | None = None,
    stratum_getter: Callable[[T], str] | None = None,
) -> dict[str, tuple[T, ...]]:
    """Create exact, deterministic, nested stratified views.

    Views are ordered by total quota.  For every stratum, each larger view must
    request at least as many examples as the preceding smaller view.  Stable
    hash ranking then makes each smaller selection a prefix of the larger one.
    Evaluation pools are intentionally finite, so grouping them in memory is
    acceptable and keeps nesting independently reproducible.
    """

    if not quotas_by_view:
        raise SamplingError("at least one quota view is required")
    identify = id_getter or _default_id
    stratify = stratum_getter or _default_stratum
    groups: dict[str, list[tuple[str, T]]] = {}
    seen_ids: set[str] = set()
    for item in items:
        item_id = identify(item)
        if item_id in seen_ids:
            raise SamplingError(f"duplicate sample id: {item_id}")
        seen_ids.add(item_id)
        stratum = stratify(item)
        groups.setdefault(stratum, []).append((item_id, item))

    views = sorted(quotas_by_view, key=lambda name: (sum(quotas_by_view[name].values()), name))
    strata = set().union(*(set(quotas_by_view[name]) for name in views))
    previous = {stratum: 0 for stratum in strata}
    for view in views:
        unexpected = set(quotas_by_view[view]) - strata
        if unexpected:  # Defensive; strata is the union above.
            raise SamplingError(f"unexpected strata for {view}: {sorted(unexpected)}")
        for stratum in strata:
            quota = quotas_by_view[view].get(stratum, 0)
            if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
                raise SamplingError(f"quota for {view}/{stratum} must be a non-negative integer")
            if quota < previous[stratum]:
                raise SamplingError(
                    f"views are not nested for {stratum}: {view} requests {quota} "
                    f"after {previous[stratum]}"
                )
            available = len(groups.get(stratum, ()))
            if quota > available:
                raise SamplingError(
                    f"{view}/{stratum} requests {quota}, but only {available} are available"
                )
            previous[stratum] = quota

    ranked: dict[str, list[T]] = {}
    for stratum, values in groups.items():
        ordered = sorted(values, key=lambda pair: (_stable_rank(seed, stratum, pair[0]), pair[0]))
        ranked[stratum] = [item for _, item in ordered]

    result: dict[str, tuple[T, ...]] = {}
    for view in views:
        selection: list[T] = []
        for stratum in sorted(strata):
            selection.extend(ranked.get(stratum, ())[: quotas_by_view[view].get(stratum, 0)])
        selection.sort(key=lambda item: (_stable_rank(seed, stratify(item), identify(item)), identify(item)))
        result[view] = tuple(selection)
    return result


def deterministic_quota_sample(
    items: Iterable[T],
    quotas: Mapping[str, int],
    *,
    seed: int | str,
    id_getter: Callable[[T], str] | None = None,
    stratum_getter: Callable[[T], str] | None = None,
) -> tuple[T, ...]:
    """Convenience wrapper for one exact stratified view."""

    return deterministic_nested_quota_sample(
        items,
        {"selection": quotas},
        seed=seed,
        id_getter=id_getter,
        stratum_getter=stratum_getter,
    )["selection"]
