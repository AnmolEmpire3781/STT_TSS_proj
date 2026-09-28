"""Freeze immutable public or custom-domain ASR evaluation suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

from asr_pipeline.adapters.base import AdapterExample
from asr_pipeline.adapters.registry import get_adapter
from asr_pipeline.archives import TarShardWriter, stage_archived_audio
from asr_pipeline.artifacts import mark_complete, require_complete, sha256_file
from asr_pipeline.audio import materialize_source_audio, validate_wav
from asr_pipeline.manifests import read_manifest, write_manifest
from asr_pipeline.revisions import ResolvedRevision, resolve_dataset_revision
from asr_pipeline.schema import CanonicalRecord


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/voice-rag-phase2/stt")
DEFAULT_WORK_ROOT = Path("/content/voice-rag-asr")
DEFAULT_KEYWORDS = Path(__file__).resolve().parent / "configs" / "domain_keywords.json"
PUBLIC_SOURCES = {
    "indicvoices_hi": {"split": "valid", "quota": 200},
    "mucs_hinglish": {"split": "test", "quota": 200},
    "svarah": {"split": "test", "quota": 100},
}
PUBLIC_PROFILES = {
    "smoke25": {"indicvoices_hi": 10, "mucs_hinglish": 10, "svarah": 5},
    "smoke100": {"indicvoices_hi": 40, "mucs_hinglish": 40, "svarah": 20},
    "benchmark500": {"indicvoices_hi": 200, "mucs_hinglish": 200, "svarah": 100},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a checksummed benchmark500 or custom-domain ASR suite"
    )
    parser.add_argument("--suite-id", required=True)
    parser.add_argument(
        "--kind", choices=("benchmark500", "custom-domain"), default="benchmark500"
    )
    parser.add_argument("--custom-manifest", type=Path)
    parser.add_argument("--custom-limit", type=int, default=0, help="0 keeps all custom test rows")
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="SOURCE=REVISION",
        help="Override a public source revision; repeat per source",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--shard-size-mb", type=int, default=128)
    parser.add_argument("--drive-root", type=Path, default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--keywords-file", type=Path, default=DEFAULT_KEYWORDS)
    return parser


def _safe_component(value: str, label: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise SystemExit(f"{label} must be one safe path component")
    return value


def _revision_overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--revision must use SOURCE=REVISION")
        source, revision = (part.strip() for part in value.split("=", 1))
        if source not in PUBLIC_SOURCES or not revision:
            raise SystemExit(f"Invalid public revision override: {value}")
        result[source] = revision
    return result


def _rank(seed: int, source: str, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{source}\0{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def _bounded_hash_sample(
    rows: Iterable[AdapterExample], *, source: str, quota: int, seed: int
) -> list[AdapterExample]:
    """Keep only the lowest deterministic hash ranks while streaming a split."""

    selected: list[tuple[int, str, AdapterExample]] = []
    seen: set[str] = set()
    for row in rows:
        if row.id in seen:
            raise RuntimeError(f"Duplicate source id in {source}: {row.id}")
        seen.add(row.id)
        candidate = (_rank(seed, source, row.id), row.id, row)
        if len(selected) < quota:
            selected.append(candidate)
            continue
        worst_index = max(range(len(selected)), key=lambda index: selected[index][:2])
        if candidate[:2] < selected[worst_index][:2]:
            selected[worst_index] = candidate
    if len(selected) < quota:
        raise RuntimeError(f"{source} contains only {len(selected)} rows; need {quota}")
    selected.sort(key=lambda item: item[:2])
    return [item[2] for item in selected]


def _load_keywords(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("keywords") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise ValueError("Keyword file must contain a keywords list")
    return values, sha256_file(path)


def _keywords_in_text(text: str, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in catalog:
        canonical = str(item.get("canonical", "")).strip()
        aliases = [str(value) for value in item.get("aliases", [])]
        if not canonical:
            continue
        candidates = [canonical, *aliases]
        if any(
            re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", text, flags=re.IGNORECASE)
            for candidate in candidates
            if candidate
        ):
            selected.append({"term": canonical, "aliases": aliases})
    return selected


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _public_selection(
    revisions: dict[str, str], token: str | None, seed: int
) -> tuple[dict[str, list[AdapterExample]], dict[str, ResolvedRevision]]:
    pools: dict[str, list[AdapterExample]] = {}
    locks: dict[str, ResolvedRevision] = {}
    for source, source_config in PUBLIC_SOURCES.items():
        adapter = get_adapter(source)
        lock = resolve_dataset_revision(
            adapter.dataset_id, revisions.get(source, "main"), token=token
        )
        stream = adapter.iter_source(
            str(source_config["split"]), lock.resolved_sha, token=token, streaming=True
        )
        pools[source] = _bounded_hash_sample(
            stream, source=source, quota=int(source_config["quota"]), seed=seed
        )
        locks[source] = lock
    return pools, locks


def _profile_ids(pools: dict[str, list[AdapterExample]], seed: int) -> dict[str, list[str]]:
    profiles: dict[str, list[str]] = {}
    for profile, quotas in PUBLIC_PROFILES.items():
        ids = [
            example.id
            for source in PUBLIC_SOURCES
            for example in pools[source][: quotas[source]]
        ]
        ids.sort(key=lambda sample_id: _rank(seed, profile, sample_id))
        profiles[profile] = ids
    return profiles


def _freeze_public(
    args: argparse.Namespace,
    incomplete: Path,
    work: Path,
    keyword_catalog: list[dict[str, Any]],
) -> tuple[list[CanonicalRecord], dict[str, Any]]:
    token = os.getenv(args.hf_token_env) or None
    pools, locks = _public_selection(_revision_overrides(args.revision), token, args.seed)
    profiles = _profile_ids(pools, args.seed)
    by_id = {example.id: example for values in pools.values() for example in values}
    ordered = [by_id[sample_id] for sample_id in profiles["benchmark500"]]
    _atomic_json(
        incomplete / "source-locks.json",
        {source: lock.to_dict() for source, lock in locks.items()},
    )
    rows = _materialize_adapter_examples(
        ordered, incomplete, work, keyword_catalog, args.shard_size_mb, args.ffmpeg_bin
    )
    metadata = {
        "suite_type": "public_benchmark",
        "profiles": profiles,
        "selection_algorithm": "streaming-lowest-sha256-rank-v1",
        "source_revisions": {
            source: lock.resolved_sha for source, lock in locks.items()
        },
        "source_splits": {
            source: config["split"] for source, config in PUBLIC_SOURCES.items()
        },
        "quotas": {source: config["quota"] for source, config in PUBLIC_SOURCES.items()},
    }
    return rows, metadata


def _materialize_adapter_examples(
    examples: list[AdapterExample],
    incomplete: Path,
    work: Path,
    keyword_catalog: list[dict[str, Any]],
    shard_size_mb: int,
    ffmpeg_bin: str,
) -> list[CanonicalRecord]:
    audio_dir = work / "selected-audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[CanonicalRecord] = []
    with TarShardWriter(
        incomplete / "shards", max_bytes=shard_size_mb * 1024 * 1024
    ) as writer:
        for example in examples:
            filename = f"{hashlib.sha256(example.id.encode('utf-8')).hexdigest()[:24]}.wav"
            member = f"audio/{filename}"
            local_wav = audio_dir / filename
            info = materialize_source_audio(
                example.source_audio, local_wav, ffmpeg_bin=ffmpeg_bin
            )
            audio_hash = sha256_file(local_wav)
            archived = writer.add(local_wav, member)
            local_wav.unlink()
            mapping = example.metadata()
            mapping.update(
                audio=member,
                duration=round(info.duration, 6),
                archive=archived.archive,
                audio_sha256=audio_hash,
                keywords=_keywords_in_text(example.text, keyword_catalog),
            )
            records.append(CanonicalRecord.from_mapping(mapping))
    return records


def _freeze_custom(
    args: argparse.Namespace,
    incomplete: Path,
    work: Path,
    keyword_catalog: list[dict[str, Any]],
) -> tuple[list[CanonicalRecord], dict[str, Any]]:
    if args.custom_manifest is None:
        raise SystemExit("--custom-manifest is required for --kind custom-domain")
    source_manifest = (
        args.custom_manifest / "manifest.jsonl"
        if args.custom_manifest.is_dir()
        else args.custom_manifest
    ).resolve()
    require_complete(source_manifest.parent, verify=True)
    all_source_rows = read_manifest(source_manifest)
    speaker_splits: dict[str, set[str]] = {}
    for row in all_source_rows:
        if row.extensions.get("human_verified") is not True:
            raise RuntimeError(f"Custom row is not human verified: {row.id}")
        speaker_splits.setdefault(row.speaker_id, set()).add(row.split)
    leaking = sorted(
        speaker for speaker, splits in speaker_splits.items() if len(splits) > 1
    )
    if leaking:
        raise RuntimeError(
            f"Custom source has speakers in multiple splits: {leaking[:5]}"
        )
    source_rows = [row for row in all_source_rows if row.split == "test"]
    if not source_rows:
        raise RuntimeError("Custom manifest contains no test rows")
    source_rows.sort(key=lambda row: (_rank(args.seed, "custom-domain", row.id), row.id))
    if args.custom_limit:
        if args.custom_limit < 1:
            raise SystemExit("--custom-limit must be positive or zero for all")
        source_rows = source_rows[: args.custom_limit]
    source_dicts = [row.to_dict() for row in source_rows]
    staged = stage_archived_audio(
        source_dicts, source_manifest.parent, work / "custom-source-audio"
    )
    records: list[CanonicalRecord] = []
    with TarShardWriter(
        incomplete / "shards", max_bytes=args.shard_size_mb * 1024 * 1024
    ) as writer:
        for row in source_rows:
            local_wav = staged[row.id]
            info = validate_wav(local_wav)
            filename = f"{hashlib.sha256(row.id.encode('utf-8')).hexdigest()[:24]}.wav"
            member = f"audio/{filename}"
            archived = writer.add(local_wav, member)
            value = row.to_dict()
            value.update(
                audio=member,
                archive=archived.archive,
                duration=round(info.duration, 6),
                audio_sha256=sha256_file(local_wav),
                keywords=row.extensions.get("keywords")
                or _keywords_in_text(row.text, keyword_catalog),
            )
            records.append(CanonicalRecord.from_mapping(value))
    ids = [row.id for row in records]
    profiles: dict[str, list[str]] = {"custom_all": ids}
    if len(ids) >= 25:
        profiles["smoke25"] = ids[:25]
    if len(ids) >= 100:
        profiles["smoke100"] = ids[:100]
    return records, {
        "suite_type": "custom_domain",
        "profiles": profiles,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "selection_algorithm": "stable-sha256-rank-v1",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _safe_component(args.suite_id, "--suite-id")
    if args.shard_size_mb <= 0:
        raise SystemExit("--shard-size-mb must be positive")
    if args.kind == "benchmark500" and args.custom_manifest is not None:
        raise SystemExit("--custom-manifest is only valid for --kind custom-domain")
    output = (
        args.output_dir or args.drive_root / "evaluation" / "suites" / args.suite_id
    ).resolve()
    incomplete = output.with_name(f"{output.name}.incomplete")
    if output.exists() or incomplete.exists():
        raise SystemExit(f"Refusing to overwrite existing suite: {output}")
    work = (args.work_dir or DEFAULT_WORK_ROOT / f"suite-{args.suite_id}").resolve()
    work.mkdir(parents=True, exist_ok=True)
    incomplete.mkdir(parents=True)
    keyword_catalog, keyword_hash = _load_keywords(args.keywords_file.resolve())

    if args.kind == "benchmark500":
        records, kind_metadata = _freeze_public(args, incomplete, work, keyword_catalog)
    else:
        records, kind_metadata = _freeze_custom(args, incomplete, work, keyword_catalog)
    manifest = write_manifest(incomplete / "manifest.jsonl", records)
    suite_metadata = {
        "schema_version": 1,
        "artifact_type": "asr_evaluation_suite",
        "suite_id": args.suite_id,
        "kind": args.kind,
        "manifest": "manifest.jsonl",
        "manifest_sha256": manifest.sha256,
        "records": manifest.rows,
        "duration_hours": manifest.duration_seconds / 3600.0,
        "seed": args.seed,
        "keyword_catalog_sha256": keyword_hash,
        "created_unix": time.time(),
        "audio_format": {"container": "wav", "sample_rate": 16000, "channels": 1, "pcm_bits": 16},
        **kind_metadata,
    }
    _atomic_json(incomplete / "suite.json", suite_metadata)
    mark_complete(incomplete)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(incomplete, output)
    print(json.dumps({**suite_metadata, "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
