#!/usr/bin/env python3
"""Strictly compare compatible base and adapter ASR evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from asr_pipeline.artifacts import (
    ArtifactError,
    atomic_write_json,
    atomic_write_text,
    is_complete,
    mark_complete,
    sha256_json,
)


COMPARISON_SCHEMA_VERSION = "asr-model-comparison-v1"
_METRIC_FIELDS = (
    "wer",
    "cer",
    "domain_keyword_accuracy",
    "p50_latency_seconds",
    "p95_latency_seconds",
)


class ComparisonError(RuntimeError):
    """Raised when evaluation artifacts cannot be compared safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ComparisonError(f"required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"JSON file must contain an object: {path}")
    return value


def _prediction_ids(path: Path) -> list[str]:
    ids: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ComparisonError(
                        f"invalid prediction JSON at {path}:{line_number}"
                    ) from exc
                record_id = value.get("id") if isinstance(value, Mapping) else None
                if not isinstance(record_id, str) or not record_id:
                    raise ComparisonError(
                        f"invalid prediction ID at {path}:{line_number}"
                    )
                ids.append(record_id)
    except FileNotFoundError as exc:
        raise ComparisonError(f"predictions are missing: {path}") from exc
    if len(set(ids)) != len(ids):
        raise ComparisonError(f"predictions contain duplicate IDs: {path}")
    return ids


def load_evaluation(
    path: str | Path, *, verify_artifact: bool = True
) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ComparisonError(f"evaluation path is not a directory: {root}")
    if verify_artifact and not is_complete(root, verify=True):
        raise ComparisonError(
            f"evaluation lacks valid COMPLETE/checksums: {root}"
        )
    run = _read_json(root / "run.json")
    metrics = _read_json(root / "metrics.json")
    if run.get("status") != "complete":
        raise ComparisonError(f"evaluation run is not complete: {root}")
    prediction_ids = _prediction_ids(root / "predictions.jsonl")
    if metrics.get("overall", {}).get("sample_count") != len(prediction_ids):
        raise ComparisonError(
            f"metrics/prediction sample counts differ in {root}"
        )
    return {
        "root": root,
        "run": run,
        "metrics": metrics,
        "prediction_ids": prediction_ids,
    }


def _required_fingerprint(
    evaluation: Mapping[str, Any], field: str, nested: str | None = None
) -> str:
    run = evaluation["run"]
    value = run.get(field)
    if value is None and nested:
        section = run.get(nested, {})
        value = section.get(field) if isinstance(section, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ComparisonError(
            f"evaluation {evaluation['root']} is missing {field}"
        )
    return value


def _hardware_signature(run: Mapping[str, Any]) -> dict[str, Any]:
    runtime = run.get("runtime", {})
    if not isinstance(runtime, Mapping):
        runtime = {}
    return {
        "resolved_device": runtime.get("resolved_device"),
        "resolved_dtype": runtime.get("resolved_dtype"),
        "cuda_device_name": runtime.get("cuda_device_name"),
        "latency_scope": runtime.get("latency_scope"),
        "warmup_samples": run.get("warmup_samples"),
    }


def assert_compatible(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "suite_fingerprint": (
            _required_fingerprint(base, "suite_fingerprint", "suite"),
            _required_fingerprint(candidate, "suite_fingerprint", "suite"),
        ),
        "decoding_fingerprint": (
            _required_fingerprint(base, "decoding_fingerprint"),
            _required_fingerprint(candidate, "decoding_fingerprint"),
        ),
        "normalization_fingerprint": (
            _required_fingerprint(base, "normalization_fingerprint"),
            _required_fingerprint(candidate, "normalization_fingerprint"),
        ),
    }
    for name, (base_value, candidate_value) in checks.items():
        if base_value != candidate_value:
            raise ComparisonError(
                f"incompatible {name}: base={base_value}, candidate={candidate_value}"
            )
    if base["prediction_ids"] != candidate["prediction_ids"]:
        raise ComparisonError(
            "evaluations contain different prediction IDs or ordering"
        )
    base_keyword_hash = base["run"].get("keyword_catalog", {}).get("sha256")
    candidate_keyword_hash = candidate["run"].get("keyword_catalog", {}).get(
        "sha256"
    )
    if base_keyword_hash != candidate_keyword_hash:
        raise ComparisonError("evaluations used different domain keyword catalogs")
    base_hardware = _hardware_signature(base["run"])
    candidate_hardware = _hardware_signature(candidate["run"])
    if base_hardware != candidate_hardware:
        raise ComparisonError(
            "evaluations used different latency hardware/settings; rerun on a "
            "matching runtime before comparison"
        )
    return {
        name: base_value for name, (base_value, _) in checks.items()
    } | {
        "prediction_count": len(base["prediction_ids"]),
        "keyword_catalog_sha256": base_keyword_hash,
        "hardware": base_hardware,
    }


def _delta(base_value: Any, candidate_value: Any) -> dict[str, Any]:
    if base_value is None or candidate_value is None:
        return {
            "base": base_value,
            "candidate": candidate_value,
            "absolute_delta": None,
            "relative_change": None,
        }
    base_number = float(base_value)
    candidate_number = float(candidate_value)
    absolute = candidate_number - base_number
    return {
        "base": base_number,
        "candidate": candidate_number,
        "absolute_delta": absolute,
        "relative_change": absolute / base_number if base_number else None,
    }


def _compare_metric_block(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    if base.get("sample_count") != candidate.get("sample_count"):
        raise ComparisonError("metric slices have different sample counts")
    result: dict[str, Any] = {"sample_count": base.get("sample_count")}
    for field in _METRIC_FIELDS:
        result[field] = _delta(base.get(field), candidate.get(field))
    return result


def _compare_slice_map(
    base: Mapping[str, Any], candidate: Mapping[str, Any], name: str
) -> dict[str, Any]:
    base_keys = set(base)
    candidate_keys = set(candidate)
    if base_keys != candidate_keys:
        raise ComparisonError(f"{name} slices differ between evaluations")
    return {
        key: _compare_metric_block(base[key], candidate[key])
        for key in sorted(base_keys)
    }


def compare_evaluations(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    base_label: str = "base",
    candidate_label: str = "candidate",
) -> dict[str, Any]:
    compatibility = assert_compatible(base, candidate)
    base_metrics = base["metrics"]
    candidate_metrics = candidate["metrics"]
    comparison = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "labels": {"base": base_label, "candidate": candidate_label},
        "inputs": {
            "base": str(base["root"]),
            "candidate": str(candidate["root"]),
            "base_evaluation_fingerprint": base["run"].get(
                "evaluation_fingerprint"
            ),
            "candidate_evaluation_fingerprint": candidate["run"].get(
                "evaluation_fingerprint"
            ),
            "base_model_fingerprint": base["run"].get("model_fingerprint"),
            "candidate_model_fingerprint": candidate["run"].get(
                "model_fingerprint"
            ),
        },
        "compatibility": compatibility,
        "overall": _compare_metric_block(
            base_metrics["overall"], candidate_metrics["overall"]
        ),
    }
    for slice_name in ("by_language", "by_source", "by_language_source"):
        comparison[slice_name] = _compare_slice_map(
            base_metrics.get(slice_name, {}),
            candidate_metrics.get(slice_name, {}),
            slice_name,
        )
    comparison["comparison_fingerprint"] = sha256_json(
        {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "labels": comparison["labels"],
            "inputs": comparison["inputs"],
            "compatibility": compatibility,
        }
    )
    return comparison


def _format_value(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(comparison: Mapping[str, Any]) -> str:
    labels = comparison["labels"]
    lines = [
        "# ASR model comparison",
        "",
        f"Base: `{labels['base']}`  ",
        f"Candidate: `{labels['candidate']}`",
        "",
        "## Overall",
        "",
        "| Metric | Base | Candidate | Delta (candidate - base) |",
        "|---|---:|---:|---:|",
    ]
    for field in _METRIC_FIELDS:
        values = comparison["overall"][field]
        lines.append(
            f"| {field} | {_format_value(values['base'])} | "
            f"{_format_value(values['candidate'])} | "
            f"{_format_value(values['absolute_delta'])} |"
        )
    for slice_name in ("by_language", "by_source", "by_language_source"):
        slices = comparison.get(slice_name, {})
        if not slices:
            continue
        lines.extend(
            [
                "",
                f"## {slice_name}",
                "",
                "| Slice | Samples | WER base | WER candidate | WER delta | "
                "CER base | CER candidate | CER delta |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key, block in slices.items():
            wer = block["wer"]
            cer = block["cer"]
            lines.append(
                f"| {_markdown_cell(key)} | {block['sample_count']} | {_format_value(wer['base'])} | "
                f"{_format_value(wer['candidate'])} | "
                f"{_format_value(wer['absolute_delta'])} | "
                f"{_format_value(cer['base'])} | "
                f"{_format_value(cer['candidate'])} | "
                f"{_format_value(cer['absolute_delta'])} |"
            )
    return "\n".join(lines) + "\n"


def write_comparison(
    comparison: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    complete_path = root / "COMPLETE"
    if complete_path.exists():
        if not is_complete(root, verify=True):
            raise ComparisonError(f"existing comparison artifact is corrupt: {root}")
        previous = _read_json(root / "comparison.json")
        if previous.get("comparison_fingerprint") != comparison.get(
            "comparison_fingerprint"
        ):
            raise ComparisonError(
                "output directory has a different completed comparison"
            )
        return root / "comparison.json", root / "comparison.md"
    if any(root.iterdir()):
        raise ComparisonError(
            "comparison output directory is non-empty; use a new --output-dir"
        )
    json_path = atomic_write_json(root / "comparison.json", comparison)
    markdown_path = atomic_write_text(
        root / "comparison.md", render_markdown(comparison)
    )
    mark_complete(root)
    return json_path, markdown_path


def _decoding_without_hinglish(run: Mapping[str, Any]) -> dict[str, Any]:
    decoding = dict(run.get("decoding", {}))
    prefixes = dict(decoding.get("language_prefix_by_canonical_language", {}))
    prefixes["hi-en"] = "<selected>"
    decoding["language_prefix_by_canonical_language"] = prefixes
    decoding.pop("hinglish_mode", None)
    return decoding


def select_hinglish_decoding(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the lowest Hinglish WER across compatible auto/hi/en baselines."""

    if len(evaluations) != 3:
        raise ComparisonError("exactly three baseline evaluations are required")
    first = evaluations[0]
    first_run = first["run"]
    expected = {
        "suite": _required_fingerprint(first, "suite_fingerprint", "suite"),
        "normalization": _required_fingerprint(first, "normalization_fingerprint"),
        "model": _required_fingerprint(first, "model_fingerprint"),
        "keywords": first_run.get("keyword_catalog", {}).get("sha256"),
        "hardware": _hardware_signature(first_run),
        "decoding": _decoding_without_hinglish(first_run),
        "prediction_ids": first["prediction_ids"],
    }
    by_mode: dict[str, dict[str, Any]] = {}
    for evaluation in evaluations:
        run = evaluation["run"]
        if _required_fingerprint(evaluation, "suite_fingerprint", "suite") != expected["suite"]:
            raise ComparisonError("Hinglish baselines use different suites")
        if _required_fingerprint(evaluation, "normalization_fingerprint") != expected["normalization"]:
            raise ComparisonError("Hinglish baselines use different normalization")
        if _required_fingerprint(evaluation, "model_fingerprint") != expected["model"]:
            raise ComparisonError("Hinglish baselines use different models")
        if run.get("keyword_catalog", {}).get("sha256") != expected["keywords"]:
            raise ComparisonError("Hinglish baselines use different keyword catalogs")
        if _hardware_signature(run) != expected["hardware"]:
            raise ComparisonError("Hinglish baselines use different hardware/settings")
        if _decoding_without_hinglish(run) != expected["decoding"]:
            raise ComparisonError("Hinglish baselines differ beyond language-prefix mode")
        if evaluation["prediction_ids"] != expected["prediction_ids"]:
            raise ComparisonError("Hinglish baselines use different prediction rows")
        decoding = run.get("decoding", {})
        mode = decoding.get("hinglish_mode")
        if mode not in {"auto", "hi", "en"} or mode in by_mode:
            raise ComparisonError("baselines must contain auto, hi, and en exactly once")
        metric = evaluation["metrics"].get("by_language", {}).get("hi-en", {})
        wer = metric.get("wer")
        if wer is None:
            raise ComparisonError(f"baseline {mode} has no hi-en WER")
        by_mode[str(mode)] = {
            "wer": float(wer),
            "evaluation": str(evaluation["root"]),
            "decoding_fingerprint": run.get("decoding_fingerprint"),
            "decoding": decoding,
        }
    if set(by_mode) != {"auto", "hi", "en"}:
        raise ComparisonError("baselines must contain auto, hi, and en exactly once")
    tie_order = {"auto": 0, "hi": 1, "en": 2}
    selected_mode = min(by_mode, key=lambda mode: (by_mode[mode]["wer"], tie_order[mode]))
    return {
        "schema_version": "asr-selected-decoding-v1",
        "created_at": _utc_now(),
        "selection_metric": "hi-en.wer",
        "tie_break_order": ["auto", "hi", "en"],
        "selected_mode": selected_mode,
        "decoding": by_mode[selected_mode]["decoding"],
        "decoding_fingerprint": by_mode[selected_mode]["decoding_fingerprint"],
        "suite_fingerprint": expected["suite"],
        "normalization_fingerprint": expected["normalization"],
        "model_fingerprint": expected["model"],
        "results": by_mode,
    }


def write_selected_decoding(selection: Mapping[str, Any], output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ComparisonError("selection output directory must be empty")
    path = atomic_write_json(root / "selected-decoding.json", selection)
    mark_complete(root)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two complete ASR evaluations; suite and decoding "
            "fingerprints must match."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", help="Base evaluation directory")
    mode.add_argument(
        "--select-hinglish-from",
        nargs=3,
        metavar=("AUTO_DIR", "HI_DIR", "EN_DIR"),
        help="Select and record the best compatible baseline Hinglish mode",
    )
    parser.add_argument(
        "--candidate", help="Fine-tuned/candidate evaluation directory"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--candidate-label", default="fine-tuned")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.select_hinglish_from:
            evaluations = [load_evaluation(path) for path in args.select_hinglish_from]
            selection = select_hinglish_decoding(evaluations)
            selection_path = write_selected_decoding(selection, args.output_dir)
            output = {
                "selected_decoding": str(selection_path),
                "selected_mode": selection["selected_mode"],
                "decoding_fingerprint": selection["decoding_fingerprint"],
            }
        else:
            if not args.candidate:
                parser.error("--candidate is required with --base")
            base = load_evaluation(args.base)
            candidate = load_evaluation(args.candidate)
            comparison = compare_evaluations(
                base,
                candidate,
                base_label=args.base_label,
                candidate_label=args.candidate_label,
            )
            json_path, markdown_path = write_comparison(comparison, args.output_dir)
            output = {
                "comparison_json": str(json_path),
                "comparison_markdown": str(markdown_path),
                "comparison_fingerprint": comparison["comparison_fingerprint"],
            }
    except (ComparisonError, ArtifactError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            output,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
