"""Adapter for the Hindi configuration of AI4Bharat IndicVoices."""

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


class IndicVoicesAdapter(DatasetAdapter):
    """Expose only Hindi train and valid with fixed canonical roles."""

    key = "indicvoices_hi"
    dataset_id = "ai4bharat/IndicVoices"
    config_name = "hindi"
    split_map = {"train": "train", "valid": "validation"}

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
            language="hi",
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
