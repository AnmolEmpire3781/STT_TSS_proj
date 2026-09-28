"""Safely remove disposable Colab staging after persistent artifacts verify."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from asr_pipeline.safety import (
    ArtifactVerification,
    SafetyError,
    validate_cleanup_target,
    validate_persistent_artifacts,
)


#codec*change*+2026-09-17 Defaults belong to this Colab CLI; reusable safety logic takes injected roots.
DEFAULT_WORK_ROOT = Path("/content/voice-rag-asr")
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/voice-rag-phase2/stt")
DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CleanupResult:
    """Outcome of a dry-run or executed cleanup."""

    target: Path
    executed: bool
    verified_artifacts: tuple[ArtifactVerification, ...]


def cleanup_staging(
    *,
    target: str | os.PathLike[str],
    work_root: str | os.PathLike[str],
    persistent_root: str | os.PathLike[str],
    required_artifacts: Sequence[str | os.PathLike[str]],
    execute: bool = False,
    repository_roots: Sequence[str | os.PathLike[str]] = (DEFAULT_REPOSITORY_ROOT,),
) -> CleanupResult:
    """Verify persistence and optionally remove one guarded staging directory."""

    safe_target = validate_cleanup_target(
        target,
        work_root=work_root,
        persistent_root=persistent_root,
        repository_roots=repository_roots,
    )
    verified = validate_persistent_artifacts(
        required_artifacts, persistent_root=persistent_root
    )
    if execute:
        #codec*change*+2026-09-17 Revalidate immediately before deletion to narrow path-swap risk.
        second_resolution = validate_cleanup_target(
            safe_target,
            work_root=work_root,
            persistent_root=persistent_root,
            repository_roots=repository_roots,
        )
        if second_resolution != safe_target:
            raise SafetyError(f"Cleanup target changed during validation: {safe_target}")
        shutil.rmtree(safe_target)
    return CleanupResult(
        target=safe_target,
        executed=execute,
        verified_artifacts=verified,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify persistent artifacts, then safely remove one disposable Colab run directory. "
            "The command is a dry run unless --execute is supplied."
        )
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run directory name directly below --work-root.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=DEFAULT_WORK_ROOT,
        help=f"Disposable staging root (default: {DEFAULT_WORK_ROOT}).",
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=DEFAULT_DRIVE_ROOT,
        help=f"Persistent artifact root (default: {DEFAULT_DRIVE_ROOT}).",
    )
    parser.add_argument(
        "--required-artifact",
        action="append",
        required=True,
        metavar="PATH",
        help=(
            "Completed artifact to verify before cleanup; relative paths resolve below "
            "--drive-root. Repeat for every required artifact."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform deletion after validation; without this flag only print the plan.",
    )
    return parser


def _validate_run_id(parser: argparse.ArgumentParser, run_id: str) -> str:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("--run-id must be one non-empty path component")
    return run_id


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_id = _validate_run_id(parser, args.run_id)
    target = args.work_root / run_id
    try:
        result = cleanup_staging(
            target=target,
            work_root=args.work_root,
            persistent_root=args.drive_root,
            required_artifacts=args.required_artifact,
            execute=args.execute,
        )
    except SafetyError as exc:
        parser.error(str(exc))
    for artifact in result.verified_artifacts:
        print(f"VERIFIED: {artifact.path} ({artifact.checked_files} files)")
    if result.executed:
        print(f"REMOVED: {result.target}")
    else:
        print(f"DRY RUN: would remove {result.target}")
        print("Re-run with --execute to perform deletion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
