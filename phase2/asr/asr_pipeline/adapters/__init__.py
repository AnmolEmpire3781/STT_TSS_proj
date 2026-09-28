"""Public and custom source adapters for canonical ASR data."""

from .base import AdapterError, AdapterExample, DatasetAdapter
from .custom import CustomManifestAdapter
from .indicvoices import IndicVoicesAdapter
from .mucs_hinglish import MUCSHinglishAdapter
from .registry import adapter_names, create_adapter, get_adapter, get_adapter_class
from .svarah import SvarahAdapter

__all__ = [
    "AdapterError",
    "AdapterExample",
    "CustomManifestAdapter",
    "DatasetAdapter",
    "IndicVoicesAdapter",
    "MUCSHinglishAdapter",
    "SvarahAdapter",
    "adapter_names",
    "create_adapter",
    "get_adapter",
    "get_adapter_class",
]
