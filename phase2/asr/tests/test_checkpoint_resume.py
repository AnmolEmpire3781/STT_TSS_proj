from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_pipeline.artifacts import is_complete
from asr_pipeline.checkpoints import (
    CheckpointError,
    discover_latest_checkpoint,
    restore_checkpoint,
    prune_checkpoints,
    sync_checkpoint,
    verify_checkpoint,
)


def _local_checkpoint(root: Path, step: int, value: str) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(value.encode("utf-8"))
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    return checkpoint


def _fingerprint(run: str = "run-a") -> dict[str, object]:
    return {
        "run_id": run,
        "model_revision": "model-commit",
        "dataset_hash": "dataset-hash",
        "lora": {"r": 16, "targets": ["q_proj", "v_proj"]},
        "seed": 17,
    }


def test_sync_promotes_complete_checkpoint_and_updates_pointer(tmp_path: Path) -> None:
    local = _local_checkpoint(tmp_path / "local", 20, "weights")
    persistent = tmp_path / "drive" / "checkpoints" / "run-a"

    info = sync_checkpoint(local, persistent, _fingerprint())

    assert info.path == (persistent / "checkpoint-20").resolve()
    assert info.step == 20
    assert not (persistent / "checkpoint-20.incomplete").exists()
    assert is_complete(info.path, verify=True)
    assert (local / "adapter_model.safetensors").read_bytes() == b"weights"
    pointer = json.loads((persistent / "latest_checkpoint.json").read_text("utf-8"))
    assert pointer["checkpoint_name"] == "checkpoint-20"
    assert pointer["fingerprint_sha256"] == info.fingerprint_sha256


def test_sync_is_idempotent_and_replaces_stale_incomplete_copy(tmp_path: Path) -> None:
    local = _local_checkpoint(tmp_path / "local", 3, "good")
    persistent = tmp_path / "persistent"
    stale = persistent / "checkpoint-3.incomplete"
    stale.mkdir(parents=True)
    (stale / "partial").write_text("partial", encoding="utf-8")

    first = sync_checkpoint(local, persistent, _fingerprint())
    second = sync_checkpoint(local, persistent, _fingerprint())

    assert first == second
    assert not stale.exists()


def test_discovery_uses_newest_compatible_complete_checkpoint(tmp_path: Path) -> None:
    persistent = tmp_path / "persistent"
    sync_checkpoint(
        _local_checkpoint(tmp_path / "a", 5, "a"), persistent, _fingerprint("run-a")
    )
    sync_checkpoint(
        _local_checkpoint(tmp_path / "b", 12, "b"), persistent, _fingerprint("run-b")
    )
    (persistent / "checkpoint-999.incomplete").mkdir()

    latest_a = discover_latest_checkpoint(persistent, _fingerprint("run-a"))
    latest_b = discover_latest_checkpoint(persistent, _fingerprint("run-b"))

    assert latest_a is not None and latest_a.step == 5
    assert latest_b is not None and latest_b.step == 12
    assert discover_latest_checkpoint(persistent, _fingerprint("missing")) is None


def test_corrupt_complete_checkpoint_fails_closed(tmp_path: Path) -> None:
    persistent = tmp_path / "persistent"
    info = sync_checkpoint(
        _local_checkpoint(tmp_path / "local", 8, "before"),
        persistent,
        _fingerprint(),
    )
    (info.path / "adapter_model.safetensors").write_bytes(b"after")

    with pytest.raises(CheckpointError, match="Checksum mismatch"):
        discover_latest_checkpoint(persistent, _fingerprint())


def test_restore_latest_verifies_source_and_restored_copy(tmp_path: Path) -> None:
    persistent = tmp_path / "persistent"
    sync_checkpoint(
        _local_checkpoint(tmp_path / "first", 2, "old"), persistent, _fingerprint()
    )
    sync_checkpoint(
        _local_checkpoint(tmp_path / "second", 9, "new"), persistent, _fingerprint()
    )

    restored = restore_checkpoint(persistent, tmp_path / "staging", _fingerprint())

    assert restored.name == "checkpoint-9"
    assert (restored / "adapter_model.safetensors").read_bytes() == b"new"
    assert verify_checkpoint(restored, _fingerprint()).step == 9
    assert not (tmp_path / "staging" / "checkpoint-9.incomplete").exists()


def test_restore_without_compatible_checkpoint_raises_file_not_found(
    tmp_path: Path,
) -> None:
    persistent = tmp_path / "persistent"
    persistent.mkdir()

    with pytest.raises(FileNotFoundError, match="No complete compatible checkpoint"):
        restore_checkpoint(persistent, tmp_path / "staging", _fingerprint())


def test_checkpoint_name_and_source_symlinks_are_rejected(tmp_path: Path) -> None:
    local = _local_checkpoint(tmp_path / "local", 1, "weights")
    with pytest.raises(CheckpointError, match="checkpoint-N"):
        sync_checkpoint(local, tmp_path / "persistent", _fingerprint(), "latest")

    link = local / "linked-weights"
    try:
        link.symlink_to(local / "adapter_model.safetensors")
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")
    with pytest.raises(CheckpointError, match="symlink"):
        sync_checkpoint(local, tmp_path / "other-persistent", _fingerprint())


def test_prune_removes_only_old_compatible_checkpoints(tmp_path: Path) -> None:
    persistent = tmp_path / "persistent"
    for step in (1, 2, 3):
        sync_checkpoint(
            _local_checkpoint(tmp_path / f"local-{step}", step, str(step)),
            persistent,
            _fingerprint(),
        )
    sync_checkpoint(
        _local_checkpoint(tmp_path / "other", 4, "other"),
        persistent,
        _fingerprint("other"),
    )

    removed = prune_checkpoints(
        persistent, expected_fingerprint=_fingerprint(), keep=2
    )

    assert [path.name for path in removed] == ["checkpoint-1"]
    assert (persistent / "checkpoint-2").is_dir()
    assert (persistent / "checkpoint-3").is_dir()
    assert (persistent / "checkpoint-4").is_dir()
