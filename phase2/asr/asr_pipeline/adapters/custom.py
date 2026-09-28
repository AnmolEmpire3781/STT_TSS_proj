"""Adapter for immutable, human-verified custom canonical manifests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from .base import AdapterError, AdapterExample, DatasetAdapter
from ..manifests import iter_manifest as iter_manifest_rows
from ..schema import CanonicalRecord


class CustomManifestAdapter(DatasetAdapter):
    """Read processed custom data while enforcing human verification."""

    key = "custom"
    dataset_id = "local/custom"
    config_name = None
    split_map = {"train": "train", "validation": "validation", "test": "test"}

    def iter_source(self, *args: Any, **kwargs: Any) -> Iterator[AdapterExample]:
        """Reject Hub-style access; use :meth:`iter_manifest` for local data."""

        del args, kwargs
        raise AdapterError("custom data is local; call iter_manifest(path) instead")
        yield  # pragma: no cover - keeps this an iterator for a clear API

    def iter_manifest(
        self,
        path: str | os.PathLike[str],
        *,
        splits: set[str] | frozenset[str] | None = None,
        require_human_verified: bool = True,
    ) -> Iterator[AdapterExample]:
        """Yield local canonical rows and detect cross-split speaker leakage."""

        manifest_path = Path(path)
        speaker_splits: dict[str, str] = {}
        for raw_record in iter_manifest_rows(manifest_path, validate=True):
            assert isinstance(raw_record, CanonicalRecord)
            record = raw_record
            if require_human_verified and record.extensions.get("human_verified") is not True:
                raise AdapterError(
                    f"custom record {record.id!r} lacks human_verified=true"
                )
            previous = speaker_splits.setdefault(record.speaker_id, record.split)
            if previous != record.split:
                raise AdapterError(
                    f"speaker {record.speaker_id!r} leaks across {previous!r} and {record.split!r}"
                )
            if splits is not None and record.split not in splits:
                continue
            if not isinstance(record.audio, (str, os.PathLike)):
                raise AdapterError(f"custom record {record.id!r} audio must be a path")
            audio_path = Path(record.audio)
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            yield AdapterExample(
                id=record.id,
                text=record.text,
                language=record.language,
                speaker_id=record.speaker_id,
                duration=record.duration,
                source=record.source,
                split=record.split,
                source_audio=audio_path.resolve(),
                extensions=record.extensions,
            )

    def map_row(
        self,
        row: Mapping[str, Any],
        *,
        source_split: str,
        canonical_split: str,
        revision: str,
        index: int,
    ) -> AdapterExample:
        """Custom rows are already canonical and never mapped from Hub data."""

        del row, source_split, canonical_split, revision, index
        raise AdapterError("custom rows must be read with iter_manifest")
