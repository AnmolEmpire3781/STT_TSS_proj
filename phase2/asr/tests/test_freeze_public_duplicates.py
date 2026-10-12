from dataclasses import replace
from itertools import repeat
from types import SimpleNamespace

import pytest

import freeze_eval_suite as freezer
from asr_pipeline.adapters.base import AdapterExample
from asr_pipeline.revisions import ResolvedRevision


def example(source, index):
    return AdapterExample(
        id=f"{source}-{index}", text=f"first {index}", language="hi",
        speaker_id=f"{source}:speaker", duration=1.0, source=source,
        split="test", source_audio="unused.wav",
    )


def test_duplicates_preserve_first_occurrence_and_hash_selection():
    rows = [example("indicvoices_hi", index) for index in range(20)]
    expected = freezer._bounded_hash_sample(rows, source="indicvoices_hi", quota=10, seed=17)
    duplicated = [value for row in rows for value in (row, replace(row, text="later"))]
    for _ in range(2):
        stats = {}
        actual = freezer._bounded_hash_sample(
            iter(duplicated), source="indicvoices_hi", quota=10, seed=17, stats=stats,
        )
        assert actual == expected
        assert len({row.id for row in actual}) == 10
        assert all(row.text.startswith("first") for row in actual)
        assert stats == {"rows_scanned": 40, "unique_rows": 20, "duplicate_rows": 20}


def test_insufficient_unique_rows_fail():
    with pytest.raises(RuntimeError, match="only 1 unique rows; need 2"):
        freezer._bounded_hash_sample([example("s", 0)] * 5, source="s", quota=2, seed=17)


@pytest.mark.parametrize("quota", [1, 2])
def test_infinite_duplicate_stream_fails_even_when_quota_met(quota):
    with pytest.raises(RuntimeError, match="scan limit of 5"):
        freezer._bounded_hash_sample(
            repeat(example("s", 0)), source="s", quota=quota, seed=17, max_rows=5,
        )


def test_public_freeze_preserves_500_unique_rows_and_reports_duplicates(monkeypatch, tmp_path):
    def adapter(source):
        rows = [example(source, index) for index in range(250)]
        return SimpleNamespace(
            dataset_id=source,
            iter_source=lambda *args, **kwargs: iter([rows[0], *rows]),
        )

    monkeypatch.setattr(freezer, "get_adapter", adapter)
    monkeypatch.setattr(
        freezer, "resolve_dataset_revision",
        lambda dataset, revision, **kwargs: ResolvedRevision(dataset, revision, "a" * 40, "now"),
    )
    monkeypatch.setattr(freezer, "_materialize_adapter_examples", lambda examples, *args: examples)
    args = SimpleNamespace(hf_token_env="UNUSED_TEST_TOKEN", revision=[], seed=17,
                           shard_size_mb=128, ffmpeg_bin="ffmpeg")
    rows, metadata = freezer._freeze_public(args, tmp_path, tmp_path, [])
    assert len(rows) == len({row.id for row in rows}) == 500
    assert metadata["duplicate_rows"] == 3
    for source, config in freezer.PUBLIC_SOURCES.items():
        assert sum(row.source == source for row in rows) == config["quota"]
        assert metadata["source_selection_stats"][source] == {
            "rows_scanned": 251, "unique_rows": 250, "duplicate_rows": 1,
        }
    for profile, quotas in freezer.PUBLIC_PROFILES.items():
        ids = metadata["profiles"][profile]
        assert len(ids) == len(set(ids)) == sum(quotas.values())
