"""Adapter for the MUCS Hindi-English code-switching corpus."""

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


class MUCSHinglishAdapter(DatasetAdapter):
    """Expose MUCS train for training and MUCS test only as test."""

    key = "mucs_hinglish"
    dataset_id = "dianavdavidson/MUCS-Hinglish"
    config_name = None
    split_map = {"train": "train", "test": "test"}

    def map_row(
        self,
        row: Mapping[str, Any],
        *,
        source_split: str,
        canonical_split: str,
        revision: str,
        index: int,
    ) -> AdapterExample:
        audio_value = source_audio(row, ("audio", "audio_filepath", "path"))
        record_id, source_example_id = stable_example_identity(
            self.key,
            row,
            source_split=source_split,
            index=index,
            identity_keys=("segment_id", "id", "utt_id", "audio_filepath"),
        )
        return AdapterExample(
            id=record_id,
            text=exact_text(row, ("transcript", "text", "transcription")),
            language="hi-en",
            speaker_id=source_speaker(
                row,
                source=self.key,
                keys=("speaker_id", "speaker", "speaker_name"),
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

