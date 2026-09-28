"""Reusable, offline-first building blocks for the Phase-2 ASR workflow.

Heavy optional dependencies such as ``datasets`` and ``huggingface_hub`` are
imported only by the functions that use them.  Importing this package is safe
in the lightweight validation/test environment.
"""

from .schema import (
    ALLOWED_LANGUAGES,
    ALLOWED_SPLITS,
    CANONICAL_FIELDS,
    CanonicalRecord,
    SchemaError,
)

__all__ = [
    "ALLOWED_LANGUAGES",
    "ALLOWED_SPLITS",
    "CANONICAL_FIELDS",
    "CanonicalRecord",
    "SchemaError",
]

