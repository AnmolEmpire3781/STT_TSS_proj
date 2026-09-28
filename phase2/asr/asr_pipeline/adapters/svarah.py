"""Adapter for the Svarah Indian-English evaluation corpus."""

from __future__ import annotations

from typing import Any, Mapping

from .base import (
    AdapterExample,
    DatasetAdapter,
    exact_text,
    source_audio,
    source_duration,
    source_speaker,
    stable_example_identity,
)


class SvarahAdapter(DatasetAdapter):
    """Expose Svarah's public split as evaluation-only test data."""

    key = "svarah"
    dataset_id = "ai4bharat/Svarah"
    config_name = None
    split_map = {"test": "test"}

    def map_row(
        self,
        row: Mapping[str, Any],
        *,
        source_split: str,
        canonical_split: str,
        revision: str,
        index: int,
    ) -> AdapterExample:
        audio_value = source_audio(row, ("audio_filepath", "audio", "path"))
        record_id, source_example_id = stable_example_identity(
            self.key,
            row,
            source_split=source_split,
            index=index,
            identity_keys=("id", "utt_id", "utterance_id", "audio_filepath"),
        )
        return AdapterExample(
            id=record_id,
            text=exact_text(row, ("text", "transcript", "transcription")),
            language="en",
            # Svarah's published schema does not guarantee a speaker column.
            # A per-example fallback is conservative for this test-only source.
            speaker_id=source_speaker(
                row,
                source=self.key,
                keys=("speaker_id", "speaker", "speaker_name"),
                fallback_id=f"unknown-{record_id}",
            ),
            duration=source_duration(row, audio_value),
            source=self.key,
            split=canonical_split,
            source_audio=audio_value,
            extensions=self.provenance(
                source_split=source_split,
                revision=revision,
                source_example_id=source_example_id,
            ),
        )

