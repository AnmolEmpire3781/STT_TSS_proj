"""Small explicit registry for supported ASR dataset adapters."""

from __future__ import annotations

from typing import TypeAlias

from .base import AdapterError, DatasetAdapter
from .custom import CustomManifestAdapter
from .indicvoices import IndicVoicesAdapter
from .mucs_hinglish import MUCSHinglishAdapter
from .svarah import SvarahAdapter


AdapterType: TypeAlias = type[DatasetAdapter]

_ADAPTERS: dict[str, AdapterType] = {
    "indicvoices_hi": IndicVoicesAdapter,
    "mucs_hinglish": MUCSHinglishAdapter,
    "svarah": SvarahAdapter,
    "custom": CustomManifestAdapter,
}
_ALIASES = {
    "indicvoices": "indicvoices_hi",
    "indic-voices": "indicvoices_hi",
    "mucs": "mucs_hinglish",
    "mucs-hinglish": "mucs_hinglish",
}


def adapter_names() -> tuple[str, ...]:
    """Return stable canonical adapter names."""

    return tuple(sorted(_ADAPTERS))


def get_adapter_class(name: str) -> AdapterType:
    """Resolve a canonical name or documented alias."""

    normalized = name.strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    try:
        return _ADAPTERS[normalized]
    except KeyError as exc:
        raise AdapterError(
            f"unknown dataset adapter {name!r}; choose {', '.join(adapter_names())}"
        ) from exc


def create_adapter(name: str) -> DatasetAdapter:
    """Instantiate a registered stateless adapter."""

    return get_adapter_class(name)()


def get_adapter(name: str) -> DatasetAdapter:
    """Instantiate an adapter (preferred public registry entry point)."""

    return create_adapter(name)
