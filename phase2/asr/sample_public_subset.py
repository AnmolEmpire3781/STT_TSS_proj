"""Freeze a deterministic hour-based public training snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from asr_pipeline.adapters.registry import get_adapter
from asr_pipeline.archives import TarShardWriter
from asr_pipeline.artifacts import mark_complete, sha256_file
from asr_pipeline.audio import materialize_source_audio
from asr_pipeline.manifests import write_manifest
from asr_pipeline.revisions import resolve_dataset_revision, write_revision_lock
from asr_pipeline.sampling import bounded_shuffle
from asr_pipeline.schema import CanonicalRecord


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/voice-rag-phase2/stt")
DEFAULT_WORK_ROOT = Path("/content/voice-rag-asr")
TRAINING_SOURCES = ("indicvoices_hi", "mucs_hinglish")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream and persist only a deterministic hour-based public ASR subset"
    )
    parser.add_argument("--source", required=True, choices=TRAINING_SOURCES)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--target-hours", type=float, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1000)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--min-duration-seconds", type=float, default=0.1)
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    parser.add_argument("--drive-root", type=Path, default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    return parser


def _safe_component(value: str, label: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise SystemExit(f"{label} must be one safe path component")
    return value


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _safe_component(args.snapshot_id, "--snapshot-id")
    if args.target_hours <= 0 or args.shuffle_buffer_size <= 0 or args.shard_size_mb <= 0:
        raise SystemExit("hours, buffer size, and shard size must be positive")
    if not 0 < args.min_duration_seconds <= args.max_duration_seconds:
        raise SystemExit("duration bounds must satisfy 0 < min <= max")

    adapter = get_adapter(args.source)
    if adapter.canonical_split(args.source_split) != "train":
        raise SystemExit("Public training snapshots may only use a protected train split")
    token = os.getenv(args.hf_token_env) or None
    revision = resolve_dataset_revision(
        adapter.dataset_id, args.revision, token=token
    )
    output = (
        args.output_dir
        or args.drive_root / "datasets" / "public_snapshots" / args.source / args.snapshot_id
    ).resolve()
    incomplete = output.with_name(f"{output.name}.incomplete")
    if output.exists() or incomplete.exists():
        raise SystemExit(f"Refusing to overwrite existing snapshot: {output}")
    work = (
        args.work_dir
        or DEFAULT_WORK_ROOT / f"snapshot-{args.source}-{args.snapshot_id}"
    ).resolve()
    temp_audio = work / "selected-audio"
    temp_audio.mkdir(parents=True, exist_ok=True)
    incomplete.mkdir(parents=True)

    write_revision_lock(
        incomplete / "source.lock.json",
        revision,
        metadata={"adapter": args.source, "source_split": args.source_split},
    )
    rows: list[CanonicalRecord] = []
    elapsed = 0.0
    target_seconds = args.target_hours * 3600.0
    seen: set[str] = set()
    skipped_duration = 0
    stream = adapter.iter_source(
        args.source_split,
        revision.resolved_sha,
        token=token,
        streaming=True,
    )
    shuffled = bounded_shuffle(
        stream,
        seed=f"{args.seed}:{args.source}:{revision.resolved_sha}",
        buffer_size=args.shuffle_buffer_size,
    )
    with TarShardWriter(
        incomplete / "shards", max_bytes=args.shard_size_mb * 1024 * 1024
    ) as writer:
        for selected_index, example in enumerate(shuffled):
            if example.id in seen:
                raise RuntimeError(f"Source produced duplicate id: {example.id}")
            seen.add(example.id)
            if not args.min_duration_seconds <= example.duration <= args.max_duration_seconds:
                skipped_duration += 1
                continue
            filename = f"{hashlib.sha256(example.id.encode('utf-8')).hexdigest()[:24]}.wav"
            member = f"audio/{filename}"
            local_wav = temp_audio / filename
            info = materialize_source_audio(
                example.source_audio, local_wav, ffmpeg_bin=args.ffmpeg_bin
            )
            if not args.min_duration_seconds <= info.duration <= args.max_duration_seconds:
                local_wav.unlink()
                skipped_duration += 1
                continue
            audio_hash = sha256_file(local_wav)
            archived = writer.add(local_wav, member)
            local_wav.unlink()
            mapping = example.metadata()
            mapping.update(
                audio=member,
                duration=round(info.duration, 6),
                archive=archived.archive,
                audio_sha256=audio_hash,
                selection_index=selected_index,
            )
            rows.append(CanonicalRecord.from_mapping(mapping))
            elapsed += info.duration
            if elapsed >= target_seconds:
                break
    if elapsed < target_seconds:
        raise RuntimeError(
            f"Source exhausted at {elapsed / 3600.0:.3f}h before {args.target_hours:.3f}h"
        )

    summary = write_manifest(incomplete / "manifest.jsonl", rows)
    metadata = {
        "schema_version": 1,
        "artifact_type": "public_training_snapshot",
        "snapshot_id": args.snapshot_id,
        "source": args.source,
        "source_dataset": adapter.dataset_id,
        "source_config": adapter.config_name,
        "source_split": args.source_split,
        "requested_revision": revision.requested_revision,
        "resolved_revision": revision.resolved_sha,
        "seed": args.seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "selection_algorithm": "bounded-shuffle-prefix-v1",
        "duration_bounds_seconds": [
            args.min_duration_seconds,
            args.max_duration_seconds,
        ],
        "skipped_duration": skipped_duration,
        "target_hours": args.target_hours,
        "actual_hours": elapsed / 3600.0,
        "records": len(rows),
        "manifest_sha256": summary.sha256,
        "created_unix": time.time(),
        "audio_format": {"container": "wav", "sample_rate": 16000, "channels": 1, "pcm_bits": 16},
    }
    _atomic_json(incomplete / "snapshot.json", metadata)
    mark_complete(incomplete)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(incomplete, output)
    print(json.dumps({**metadata, "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
