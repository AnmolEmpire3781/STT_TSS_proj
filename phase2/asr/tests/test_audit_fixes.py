from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import asr_pipeline.audio as audio_module
from asr_pipeline.artifacts import is_complete, mark_complete
from asr_pipeline.audio import duration_from_audio_payload, materialize_source_audio
from asr_pipeline.whisper_runtime import DecodingConfig, ModelSpec
from compare_models import select_hinglish_decoding, write_selected_decoding
from freeze_eval_suite import _freeze_custom
from train_whisper_lora import (
    WhisperCollator,
    _assert_train_eval_isolation,
    _load_manifest,
    _safe_component,
)


def _write_custom_artifact(root: Path, rows: list[dict[str, object]]) -> Path:
    root.mkdir()
    (root / "dataset.json").write_text(
        json.dumps({"artifact_type": "custom_asr_dataset"}), encoding="utf-8"
    )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    mark_complete(root)
    return root


def _row(sample_id: str, split: str, speaker: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "audio": f"audio/{sample_id}.wav",
        "text": "verified transcript",
        "language": "hi-en",
        "speaker_id": speaker,
        "duration": 1.0,
        "source": "custom-support",
        "split": split,
        "human_verified": True,
        "audio_sha256": sample_id * 64,
    }


def test_mixed_custom_manifest_filters_splits_and_requires_verification(
    tmp_path: Path,
) -> None:
    artifact = _write_custom_artifact(
        tmp_path / "custom",
        [_row("a", "train", "speaker-a"), _row("b", "test", "speaker-b")],
    )

    selected = _load_manifest(artifact, {"train"})

    assert [row["id"] for row in selected] == ["a"]
    assert selected[0]["_artifact_type"] == "custom_asr_dataset"


def test_protected_public_source_rejects_relabeled_test_row(tmp_path: Path) -> None:
    row = _row("a", "train", "speaker-a")
    row.update(source="mucs_hinglish", source_split="test")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protected source split provenance"):
        _load_manifest(manifest, {"train"}, allow_incomplete=True)


def test_custom_speaker_overlap_is_rejected() -> None:
    train = [{**_row("a", "train", "speaker-a"), "_artifact_type": "custom_asr_dataset"}]
    evaluation = [
        {**_row("b", "validation", "speaker-a"), "_artifact_type": "custom_asr_dataset"}
    ]

    with pytest.raises(ValueError, match="speaker leakage"):
        _assert_train_eval_isolation(train, evaluation)


def test_custom_suite_freeze_rejects_source_speaker_leakage(tmp_path: Path) -> None:
    artifact = _write_custom_artifact(
        tmp_path / "custom",
        [_row("a", "train", "speaker-a"), _row("b", "test", "speaker-a")],
    )
    args = types.SimpleNamespace(
        custom_manifest=artifact,
        custom_limit=0,
        seed=17,
        shard_size_mb=128,
    )

    with pytest.raises(RuntimeError, match="speakers in multiple splits"):
        _freeze_custom(args, tmp_path / "incomplete", tmp_path / "work", [])


def test_run_id_must_be_one_component() -> None:
    assert _safe_component("run-001", "--run-id") == "run-001"
    with pytest.raises(SystemExit, match="safe path component"):
        _safe_component("../outside", "--run-id")


class _FakeSamples:
    sample_rate = 16_000

    @property
    def data(self) -> object:
        import numpy as np

        return np.zeros((1, 32_000), dtype=np.float32)


class _FakeAudioDecoder:
    def get_all_samples(self) -> _FakeSamples:
        return _FakeSamples()


def test_torchcodec_audio_duration_is_supported() -> None:
    assert duration_from_audio_payload(_FakeAudioDecoder()) == 2.0


def test_torchcodec_audio_materialization_is_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wave

    def fake_ffmpeg(
        _input_arguments: object,
        temporary_output: Path,
        **_kwargs: object,
    ) -> None:
        with wave.open(str(temporary_output), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * 32_000)

    monkeypatch.setattr(audio_module, "_run_ffmpeg", fake_ffmpeg)
    destination = tmp_path / "selected.wav"

    info = materialize_source_audio(_FakeAudioDecoder(), destination)

    assert info.duration == 2.0
    assert destination.is_file()


def test_remote_adapter_revision_must_be_pinned() -> None:
    with pytest.raises(ValueError, match="full commit SHA"):
        ModelSpec(adapter_path="organization/adapter")
    spec = ModelSpec(adapter_path="organization/adapter", adapter_revision="a" * 40)
    assert spec.adapter_revision == "a" * 40


def _evaluation(tmp_path: Path, mode: str, wer: float) -> dict[str, object]:
    decoding = DecodingConfig(hinglish_mode=mode).to_dict()
    return {
        "root": tmp_path / mode,
        "prediction_ids": ["one", "two"],
        "run": {
            "suite": {"suite_fingerprint": "suite"},
            "normalization_fingerprint": "normalization",
            "model_fingerprint": "model",
            "decoding": decoding,
            "decoding_fingerprint": f"decoding-{mode}",
            "keyword_catalog": {"sha256": "keywords"},
            "runtime": {
                "resolved_device": "cuda",
                "resolved_dtype": "float16",
                "cuda_device_name": "test-gpu",
                "latency_scope": "full",
            },
            "warmup_samples": 1,
        },
        "metrics": {"by_language": {"hi-en": {"wer": wer}}},
    }


def test_hinglish_selection_records_empirical_winner(tmp_path: Path) -> None:
    selection = select_hinglish_decoding(
        [
            _evaluation(tmp_path, "auto", 0.30),
            _evaluation(tmp_path, "hi", 0.20),
            _evaluation(tmp_path, "en", 0.25),
        ]
    )

    assert selection["selected_mode"] == "hi"
    assert selection["decoding"]["task"] == "transcribe"
    selected_path = write_selected_decoding(selection, tmp_path / "selection")
    assert selected_path.name == "selected-decoding.json"
    assert is_complete(selected_path.parent, verify=True)


class _FakeLabels:
    shape = (1, 2)

    def masked_fill(self, *_args: object) -> "_FakeLabels":
        return self

    def __getitem__(self, key: object) -> object:
        if key == (slice(None), 0):
            return 99
        if key == (slice(None), slice(1, None)):
            return "trimmed"
        raise AssertionError(key)


class _FakePadded(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=_FakeLabels())
        self.attention_mask = types.SimpleNamespace(ne=lambda _value: object())


class _FakeTokenizer:
    def set_prefix_tokens(self, **_kwargs: object) -> None:
        return None

    def __call__(self, _text: str) -> object:
        return types.SimpleNamespace(input_ids=[99, 10])

    def pad(self, *_args: object, **_kwargs: object) -> _FakePadded:
        return _FakePadded()


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def feature_extractor(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {}


def test_collator_strips_model_decoder_start_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(all=bool))
    collator = WhisperCollator(
        processor=_FakeProcessor(),
        hinglish_language_mode="auto",
        decoder_start_token_id=99,
    )

    batch = collator([{"audio": [0.0], "text": "hello", "language": "en"}])

    assert batch["labels"] == "trimmed"
