from __future__ import annotations

from pathlib import Path

import pytest

from asr_pipeline.artifacts import mark_complete
from asr_pipeline.safety import SafetyError, validate_cleanup_target
from cleanup_colab import cleanup_staging, main


def _sealed_artifact(drive_root: Path, relative: str = "experiments/run-a") -> Path:
    artifact = drive_root / relative
    artifact.mkdir(parents=True)
    (artifact / "run.json").write_text('{"status":"saved"}\n', encoding="utf-8")
    mark_complete(artifact)
    return artifact


def _staging_run(work_root: Path, run_id: str = "run-a") -> Path:
    target = work_root / run_id
    target.mkdir(parents=True)
    (target / "temporary.bin").write_bytes(b"temporary")
    return target


def test_cleanup_is_dry_run_by_default(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    drive_root = tmp_path / "drive"
    target = _staging_run(work_root)
    artifact = _sealed_artifact(drive_root)

    result = cleanup_staging(
        target=target,
        work_root=work_root,
        persistent_root=drive_root,
        required_artifacts=[artifact],
    )

    assert result.executed is False
    assert target.exists()
    assert result.verified_artifacts[0].checked_files == 1


def test_execute_removes_only_staging_and_preserves_drive(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    drive_root = tmp_path / "drive"
    target = _staging_run(work_root)
    artifact = _sealed_artifact(drive_root)

    result = cleanup_staging(
        target=target,
        work_root=work_root,
        persistent_root=drive_root,
        required_artifacts=["experiments/run-a"],
        execute=True,
    )

    assert result.executed is True
    assert not target.exists()
    assert artifact.exists()
    assert drive_root.exists()


def test_cleanup_refuses_corrupt_or_incomplete_persistence(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    drive_root = tmp_path / "drive"
    target = _staging_run(work_root)
    artifact = _sealed_artifact(drive_root)
    (artifact / "run.json").write_text("tampered", encoding="utf-8")

    with pytest.raises(SafetyError, match="Checksum mismatch"):
        cleanup_staging(
            target=target,
            work_root=work_root,
            persistent_root=drive_root,
            required_artifacts=[artifact],
            execute=True,
        )
    assert target.exists()

    incomplete = drive_root / "experiments" / "incomplete"
    incomplete.mkdir()
    (incomplete / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SafetyError, match="COMPLETE"):
        cleanup_staging(
            target=target,
            work_root=work_root,
            persistent_root=drive_root,
            required_artifacts=[incomplete],
        )


def test_cleanup_requires_at_least_one_persistent_artifact(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    drive_root = tmp_path / "drive"
    target = _staging_run(work_root)
    drive_root.mkdir()

    with pytest.raises(SafetyError, match="At least one"):
        cleanup_staging(
            target=target,
            work_root=work_root,
            persistent_root=drive_root,
            required_artifacts=[],
        )


def test_cleanup_refuses_root_outside_repository_and_persistent_paths(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "work"
    drive_root = tmp_path / "drive"
    target = _staging_run(work_root)
    _sealed_artifact(drive_root)
    outside = tmp_path / "outside"
    outside.mkdir()

    for unsafe in (work_root, outside, drive_root):
        with pytest.raises(SafetyError):
            validate_cleanup_target(
                unsafe, work_root=work_root, persistent_root=drive_root
            )

    (target / ".git").mkdir()
    with pytest.raises(SafetyError, match="repository"):
        validate_cleanup_target(
            target,
            work_root=work_root,
            persistent_root=drive_root,
        )


def test_cleanup_refuses_symlink_escape(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    drive_root = tmp_path / "drive"
    outside = tmp_path / "outside"
    work_root.mkdir()
    drive_root.mkdir()
    outside.mkdir()
    link = work_root / "run-a"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")

    with pytest.raises(SafetyError, match="symlink|strict descendant"):
        validate_cleanup_target(
            link, work_root=work_root, persistent_root=drive_root
        )


def test_cli_reports_dry_run_and_requires_execute(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    work_root = tmp_path / "work"
    drive_root = tmp_path / "drive"
    target = _staging_run(work_root)
    _sealed_artifact(drive_root)

    exit_code = main(
        [
            "--run-id",
            "run-a",
            "--work-root",
            str(work_root),
            "--drive-root",
            str(drive_root),
            "--required-artifact",
            "experiments/run-a",
        ]
    )

    assert exit_code == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert target.exists()


def test_cleanup_detects_repository_above_nested_work_root(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    work_root = repository / "nested" / "work"
    target = _staging_run(work_root)
    drive_root = tmp_path / "drive"
    drive_root.mkdir()

    with pytest.raises(SafetyError, match="repository"):
        validate_cleanup_target(
            target,
            work_root=work_root,
            persistent_root=drive_root,
            repository_roots=[repository],
        )
