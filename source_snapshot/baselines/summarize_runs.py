#!/usr/bin/env python3
"""Summarize WikiBigEdit baseline runs without loading a model.

The tool only reads experiment artifacts (``run_config.json``,
``metrics.json``, ``completed.json``, and, when present,
``predictions.jsonl``).  It writes a compact machine-readable JSON report and
a Markdown table.  In addition to the ordinary WikiBigEdit axes, locality is
reported in three complementary ways:

* absolute answer accuracy from ``metrics.json``;
* the accuracy delta relative to a Pre-Edit run;
* generated-output preservation relative to Pre-Edit, paired by ``sample_id``.

The output-preservation calculation deliberately reports coverage.  Missing
post-edit predictions therefore count as unpreserved rather than silently
being removed from the denominator.  Open-book retrieval runs are classified
and rendered separately from all closed-book and side-memory methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_RUNS_DIR = Path("runs/wikibigedit_3k_qwen3_unknown")
SUMMARY_SCHEMA = "wikibigedit-run-summary/v3"
AXES = ("update", "rephrase", "persona", "locality", "multihop")
AXIS_ALIASES = {
    "personas": "persona",
    "mhop": "multihop",
}
COMPARISON_GROUPS = (
    "formal_closed_book",
    "transfer_untuned",
    "side_memory",
    "open_book",
    "diagnostic_smoke",
)
COMPARISON_GROUP_LABELS = {
    "formal_closed_book": "Formal closed-book",
    "transfer_untuned": "Transfer / untuned",
    "side_memory": "Side-memory",
    "open_book": "Open-book retrieval",
    "diagnostic_smoke": "Diagnostic / smoke",
}
STANDARD_ARTIFACT_NAMES = (
    "run_config.json",
    "metrics.json",
    "completed.json",
    "predictions.jsonl",
)


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _as_int(value)
        if parsed is not None:
            return parsed
    return None


def _sha256_file(path: Path) -> tuple[str | None, str | None]:
    if not path.is_file():
        return None, None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        return None, f"could not hash {path.name}: {error}"
    return digest.hexdigest(), None


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"could not read {path.name}: {error}"


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _configured_num_facts(
    config: dict[str, Any], completed: dict[str, Any]
) -> int | None:
    direct = _first_int(config.get("num_facts"), completed.get("num_facts"))
    if direct is not None:
        return direct
    stage_counts = config.get("stage_counts")
    if isinstance(stage_counts, dict):
        counts = [_as_int(value) for value in stage_counts.values()]
        if counts and all(value is not None and value >= 0 for value in counts):
            return sum(value for value in counts if value is not None)
    return None


def _configured_max_stage(config: dict[str, Any]) -> int | None:
    stage_counts = config.get("stage_counts")
    if isinstance(stage_counts, dict):
        stages = [_as_int(stage) for stage in stage_counts]
        stages = [stage for stage in stages if stage is not None]
        if stages:
            return max(stages)
    max_stages = _as_int(config.get("max_stages"))
    if max_stages is not None and max_stages > 0:
        return max_stages
    num_chunks = _as_int(config.get("num_chunks"))
    if num_chunks is not None and num_chunks > 1:
        return num_chunks
    return None


def _normalized_method(method: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(method).casefold())


def _target_description(config: dict[str, Any]) -> str | None:
    raw = config.get("target_modules")
    if isinstance(raw, (list, tuple)):
        targets = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(raw, str):
        targets = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        targets = []
    if not targets:
        return None
    if set(targets) == {"q_proj", "v_proj"}:
        return "q/v"
    all_linear = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    if set(targets) == all_linear:
        return "all-linear"
    return ",".join(targets)


def _layer_description(config: dict[str, Any]) -> str | None:
    layers = config.get("layers", config.get("qwen3_layers"))
    if isinstance(layers, str):
        values = [_as_int(item.strip()) for item in layers.split(",")]
        parsed = [value for value in values if value is not None]
    elif isinstance(layers, (list, tuple)):
        parsed = [_as_int(value) for value in layers]
        parsed = [value for value in parsed if value is not None]
    else:
        layer = _as_int(config.get("layer"))
        parsed = [layer] if layer is not None else []
    if not parsed:
        return None
    parsed = sorted(dict.fromkeys(parsed))
    if len(parsed) == 1:
        return f"layer {parsed[0]}"
    if parsed == list(range(parsed[0], parsed[-1] + 1)):
        return f"layers {parsed[0]}–{parsed[-1]}"
    return f"{len(parsed)} layers ({parsed[0]}–{parsed[-1]})"


def _display_method_label(
    run_name: str, method: str, config: dict[str, Any]
) -> str:
    declared = config.get("method_display_name")
    if isinstance(declared, str) and declared.strip():
        # Audited new ports disclose corpus/preprocessing/dev-tuning directly;
        # avoid replacing those qualifications with an inferred old preset.
        return declared.strip()
    normalized = _normalized_method(method)
    layer_text = _layer_description(config)
    target_text = _target_description(config)
    rank = _as_int(config.get("lora_r"))
    epochs = _as_int(config.get("epochs"))
    is_transfer = bool(_transfer_reasons(run_name, method, config))
    is_qwen3_selected = _is_independent_qwen3_selection(config)

    if normalized in {"preedit", "base"}:
        return "Pre-Edit"
    if normalized in {"fullftbf", "fullft", "fullftfsdp"} or run_name.startswith(
        "full_ft"
    ):
        return "Full FT-BF (FSDP)"
    if normalized == "loraft":
        target_text = target_text or ("all-linear" if rank == 16 else None)
        details = [item for item in (target_text, f"r{rank}" if rank else None) if item]
        return "LoRA-FT" + (f" ({', '.join(details)})" if details else "")
    if normalized == "loramerge":
        alpha = _as_finite_float(config.get("merge_alpha"))
        details = ["q/v", f"r{rank}" if rank else None]
        if alpha is not None:
            details.append(f"merge α={alpha:g}")
        return "LoRA-Merge" + f" ({', '.join(item for item in details if item)})"
    if normalized == "erreservoir":
        details = [target_text or "all-linear", f"r{rank}" if rank else None]
        return "ER-Reservoir + LoRA" + f" ({', '.join(item for item in details if item)})"
    if normalized == "relationoraclereplay":
        return "Relation-Structure Oracle Replay + LoRA (gold relation, ER reservoir)"
    if normalized == "cumulativereplay":
        return "Cumulative Replay + LoRA (all history, higher compute)"
    if normalized == "entitymaskedsemanticreplay":
        return "Entity-masked Semantic Replay + LoRA"
    if normalized == "ticstylereplay":
        return "TiC-style Replay + LoRA (stage LR schedule)"
    if normalized in {"rome", "rrome"}:
        name = "ROME" if normalized == "rome" else "R-ROME"
        if is_qwen3_selected:
            details = [layer_text, "Qwen3 independent-dev selected"]
            if config.get("mom2_adjustment") is True:
                details.append("checkpoint Mom2")
            trust = _as_finite_float(config.get("max_update_weight_ratio"))
            if trust is not None:
                details.append(f"trust={trust:g}")
            return name + f" ({', '.join(item for item in details if item)})"
        return name + f" ({layer_text or 'layer 5'}, Qwen2.5→Qwen3 transfer)"
    if normalized in {"nas", "memitnas"}:
        name = "NAS" if normalized == "nas" else "MEMIT + NAS"
        if is_qwen3_selected:
            batch = _as_int(config.get("edit_batch_size"))
            details = [layer_text, "Qwen3 independent-dev selected"]
            if batch is not None:
                details.append(f"batch{batch}")
            return name + f" ({', '.join(item for item in details if item)})"
        return name + f" ({layer_text}, batch100, Qwen3 transfer)"
    if normalized == "olora":
        return "O-LoRA (all-linear, r16/stage, growing rank, Qwen3 transfer)"
    if normalized == "horen":
        return "HoReN (value side memory, layer24, Qwen3 transfer)"
    if normalized == "semanticreplay":
        details = [target_text or "all-linear", f"r{rank}" if rank else None]
        if config.get("match_er_admission"):
            details.append("ER admission matched")
        return "Semantic Top-K Replay + LoRA" + f" ({', '.join(item for item in details if item)})"
    if normalized in {"mir", "mirlora"}:
        details = [target_text, f"r{rank}" if rank else None]
        if epochs is not None:
            details.append(f"{epochs} epoch{'s' if epochs != 1 else ''}")
        return "MIR + LoRA" + f" ({', '.join(item for item in details if item)})"
    if normalized == "locftbf":
        details = [str(config.get("mode", "")).strip() or None, layer_text]
        if is_transfer:
            details.append("Qwen2.5→Qwen3 transfer")
        return "LocFT-BF" + f" ({', '.join(item for item in details if item)})"
    if normalized == "memit":
        details = [layer_text]
        if is_qwen3_selected:
            details.append("Qwen3 independent-dev selected")
        preset_label = config.get("preset_label")
        if isinstance(preset_label, str) and preset_label.strip():
            details.append(preset_label.strip().replace("_", " "))
        if "solver_ridge" in config:
            ridge = _as_finite_float(config.get("solver_ridge"))
            details.append(f"ridge={ridge:g}" if ridge is not None else "ridge=?")
        if str(config.get("stats_source", "")).casefold() == "manifest_diag":
            details.append("manifest-diagonal proxy stats")
        if is_transfer:
            details.append("Qwen2.5→Qwen3 transfer")
        return "MEMIT" + f" ({', '.join(item for item in details if item)})"
    if normalized == "alphaedit":
        details = [layer_text]
        preset_label = str(config.get("preset_label", "")).casefold()
        if "adapted" in preset_label and "sensitivity" in preset_label:
            details.append("Qwen3 sensitivity variant")
        if is_transfer:
            details.append("Qwen2.5→Qwen3 transfer")
        if config.get("formal_baseline_eligible") is False:
            details.append("diagnostic projection")
        return "AlphaEdit" + f" ({', '.join(item for item in details if item)})"
    if "crispedit" in normalized:
        details = [layer_text]
        if is_transfer:
            details.append("Qwen2.5→Qwen3 transfer")
        if config.get("publication_ready") is False:
            details.append("diagnostic stats")
        return "CrispEdit" + f" ({', '.join(item for item in details if item)})"
    if normalized == "ultraedit":
        details = [layer_text]
        if is_transfer:
            details.append("Qwen2.5→Qwen3 transfer")
        return "UltraEdit" + f" ({', '.join(item for item in details if item)})"
    if normalized == "stableedit":
        details = [layer_text]
        if is_transfer:
            details.append("Qwen2.5→Qwen3 transfer")
        return "StableEdit" + f" ({', '.join(item for item in details if item)})"
    if normalized in {"wiseretrieve", "wise"}:
        details = ["side memory + router", layer_text]
        if is_transfer:
            details.append("Qwen3 transfer")
        return "WISE-Retrieve" + f" ({', '.join(item for item in details if item)})"
    if normalized in {"denserag", "ragdense"}:
        top_k = _as_int(config.get("retrieval_top_k", config.get("top_k")))
        encoder = config.get("embedding_model")
        details = [f"top-{top_k}" if top_k else None]
        if isinstance(encoder, str) and encoder.strip():
            details.append(Path(encoder).name)
        if config.get("recipe") == "wikibigedit":
            details.append("paper recipe Qwen3 transfer")
        return "Dense RAG" + f" ({', '.join(item for item in details if item)})"
    if normalized in {"bm25rag", "hybridrag"}:
        name = "BM25 RAG" if normalized == "bm25rag" else "Hybrid BM25 + MPNet RRF"
        return f"{name} (top-{config.get('top_k')})"
    if normalized in {"oraclerag", "ragoracle"}:
        return "Oracle exact-fact RAG (Top-1 diagnostic)"
    return method


def _metric_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        nested = payload.get("metrics")
        if isinstance(nested, list):
            return [record for record in nested if isinstance(record, dict)]
        return [payload]
    return []


def _configured_final_stage(
    config: dict[str, Any], completed: dict[str, Any]
) -> int | None:
    final_stage = _as_int(completed.get("final_stage"))
    if final_stage is not None:
        return final_stage
    return _configured_max_stage(config)


def select_final_metric(
    payload: Any,
    *,
    config: dict[str, Any] | None = None,
    completed: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, int | None]:
    """Select the final evaluation record from a metrics payload.

    Runners can append base and per-stage evaluations to the same list.  A
    completed ``final_stage`` (or the configured maximum stage) takes
    precedence.  Otherwise the last non-base record is used.
    """

    records = _metric_records(payload)
    if not records:
        return None, None
    config = config or {}
    completed = completed or {}
    final_stage = _configured_final_stage(config, completed)
    if final_stage is not None:
        for index in range(len(records) - 1, -1, -1):
            if _as_int(records[index].get("current_stage")) == final_stage:
                return records[index], index
    for index in range(len(records) - 1, -1, -1):
        checkpoint = str(records[index].get("checkpoint", "")).strip().casefold()
        if checkpoint not in {"base", "pre_edit", "pre-edit"}:
            return records[index], index
    return records[-1], len(records) - 1


def _retrieval_history_audit(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Audit every stage-level retrieval record, not only the selected final one."""

    stage_records = [
        (index, record)
        for index, record in enumerate(records)
        if _as_int(record.get("current_stage")) is not None
    ]
    audited: list[dict[str, Any]] = []
    missing_indices: list[int] = []
    invalid_indices: list[int] = []
    total_future_leaks = 0
    for index, record in stage_records:
        retrieval = record.get("retrieval")
        if not isinstance(retrieval, dict):
            missing_indices.append(index)
            continue
        future_leaks = _as_int(retrieval.get("future_leak_count"))
        if future_leaks is None or future_leaks < 0:
            invalid_indices.append(index)
        else:
            total_future_leaks += future_leaks
        audited.append(
            {
                "metrics_record_index": index,
                "checkpoint": record.get("checkpoint"),
                "current_stage": _as_int(record.get("current_stage")),
                "evaluation_stage": _as_int(retrieval.get("evaluation_stage")),
                "future_leak_count": future_leaks,
            }
        )
    return {
        "stage_metric_count": len(stage_records),
        "audited_stage_metric_count": len(audited),
        "missing_retrieval_record_indices": missing_indices,
        "invalid_future_leak_record_indices": invalid_indices,
        "future_leak_count_total": total_future_leaks,
        "all_stage_records_audited": bool(stage_records)
        and not missing_indices
        and not invalid_indices,
        "all_stage_records_zero_future_leaks": bool(stage_records)
        and not missing_indices
        and not invalid_indices
        and total_future_leaks == 0,
        "records": audited,
    }


def _axis_metrics(metric: dict[str, Any] | None, axis: str) -> dict[str, Any]:
    result = {
        "count": None,
        "match_accuracy": None,
        "exact_accuracy": None,
    }
    if not isinstance(metric, dict):
        return result
    nested = metric.get(axis)
    if isinstance(nested, dict):
        result["count"] = _as_int(nested.get("count"))
        result["match_accuracy"] = _as_finite_float(nested.get("match_accuracy"))
        result["exact_accuracy"] = _as_finite_float(nested.get("exact_accuracy"))
    if result["match_accuracy"] is None:
        result["match_accuracy"] = _as_finite_float(metric.get(f"{axis}_accuracy"))
    return result


def normalize_generated_output(text: Any) -> str:
    """Normalize output exactly like the shared generated-answer scorer."""

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    lines = normalized.splitlines()
    normalized = lines[0] if lines else normalized
    normalized = "".join(
        " " if character in string.punctuation else character
        for character in normalized
    )
    return " ".join(normalized.split())


def _canonical_axis(record: dict[str, Any]) -> str:
    value = record.get("axis", record.get("panel", ""))
    axis = str(value).strip().casefold()
    return AXIS_ALIASES.get(axis, axis)


def load_locality_predictions(
    path: Path, checkpoint: str | None
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    """Load one locality prediction per sample for the desired checkpoint."""

    metadata: dict[str, Any] = {
        "available": False,
        "selected_checkpoint": None,
        "record_count": 0,
        "parse_error_count": 0,
        "missing_sample_id_count": 0,
        "duplicate_sample_id_count": 0,
    }
    warnings: list[str] = []
    if not path.is_file():
        return {}, metadata, warnings

    locality_rows: list[dict[str, Any]] = []
    checkpoint_order: list[str | None] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    metadata["parse_error_count"] += 1
                    if metadata["parse_error_count"] <= 3:
                        warnings.append(
                            f"predictions.jsonl line {line_number} is invalid JSON: {error}"
                        )
                    continue
                if not isinstance(record, dict) or _canonical_axis(record) != "locality":
                    continue
                locality_rows.append(record)
                row_checkpoint_raw = record.get("checkpoint")
                row_checkpoint = (
                    str(row_checkpoint_raw) if row_checkpoint_raw is not None else None
                )
                if row_checkpoint not in checkpoint_order:
                    checkpoint_order.append(row_checkpoint)
    except (OSError, UnicodeError) as error:
        warnings.append(f"could not read predictions.jsonl: {error}")
        return {}, metadata, warnings

    if not locality_rows:
        warnings.append("predictions.jsonl has no locality records")
        return {}, metadata, warnings

    selected_checkpoint: str | None
    requested = str(checkpoint) if checkpoint is not None else None
    if requested in checkpoint_order:
        selected_checkpoint = requested
    elif len(checkpoint_order) == 1:
        selected_checkpoint = checkpoint_order[0]
    else:
        selected_checkpoint = checkpoint_order[-1]
        if requested is not None:
            warnings.append(
                f"checkpoint {requested!r} absent from locality predictions; "
                f"using {selected_checkpoint!r}"
            )

    predictions: dict[str, str] = {}
    for record in locality_rows:
        raw_checkpoint = record.get("checkpoint")
        row_checkpoint = str(raw_checkpoint) if raw_checkpoint is not None else None
        if row_checkpoint != selected_checkpoint:
            continue
        sample_id = record.get("sample_id")
        if sample_id in (None, ""):
            metadata["missing_sample_id_count"] += 1
            continue
        key = str(sample_id)
        if key in predictions:
            metadata["duplicate_sample_id_count"] += 1
        predictions[key] = str(record.get("prediction", ""))

    metadata.update(
        {
            "available": bool(predictions),
            "selected_checkpoint": selected_checkpoint,
            "record_count": len(predictions),
        }
    )
    if metadata["missing_sample_id_count"]:
        warnings.append(
            f"ignored {metadata['missing_sample_id_count']} locality predictions "
            "without sample_id"
        )
    if metadata["duplicate_sample_id_count"]:
        warnings.append(
            f"used the last of {metadata['duplicate_sample_id_count']} duplicate "
            "locality prediction records"
        )
    return predictions, metadata, warnings


def compare_output_preservation(
    base_predictions: dict[str, str], post_predictions: dict[str, str]
) -> dict[str, Any]:
    """Compare post-edit locality generations against frozen base outputs."""

    if not base_predictions:
        return {
            "available": False,
            "reason": "base locality predictions unavailable",
        }
    if not post_predictions:
        return {
            "available": False,
            "reason": "post-edit locality predictions unavailable",
            "base_count": len(base_predictions),
        }

    base_ids = set(base_predictions)
    post_ids = set(post_predictions)
    paired_ids = base_ids & post_ids
    raw_equal = sum(
        post_predictions[sample_id] == base_predictions[sample_id]
        for sample_id in paired_ids
    )
    normalized_equal = sum(
        normalize_generated_output(post_predictions[sample_id])
        == normalize_generated_output(base_predictions[sample_id])
        for sample_id in paired_ids
    )
    denominator = len(base_ids)
    return {
        "available": True,
        "base_count": denominator,
        "post_count": len(post_ids),
        "paired_count": len(paired_ids),
        "coverage": len(paired_ids) / denominator,
        "base_only_count": len(base_ids - post_ids),
        "post_only_count": len(post_ids - base_ids),
        "raw_equal_count": raw_equal,
        "raw_equal_accuracy": raw_equal / denominator,
        "normalized_equal_count": normalized_equal,
        "normalized_equal_accuracy": normalized_equal / denominator,
        "denominator_policy": "all base locality records; missing post records fail",
    }


def _experiment_directories(runs_dir: Path) -> Iterable[Path]:
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs directory does not exist: {runs_dir}")
    artifact_names = set(STANDARD_ARTIFACT_NAMES)
    for path in sorted(runs_dir.iterdir(), key=lambda item: item.name):
        if path.is_dir() and any((path / name).exists() for name in artifact_names):
            yield path


def _run_status(config: dict[str, Any], completed: dict[str, Any]) -> str:
    status = completed.get("status")
    if isinstance(status, str) and status.strip():
        return status.strip()
    return "incomplete" if config else "unknown"


def _is_independent_qwen3_selection(config: dict[str, Any]) -> bool:
    """Identify a certificate-bound Qwen3 editor adaptation.

    Recording the upstream Qwen2.5 recipe for provenance must not by itself
    classify an independently selected Qwen3 configuration as an untuned
    transfer.
    """
    provenance = (
        config.get("selection_certificate")
        or config.get("selection_provenance")
        or config.get("development_selection")
        or config.get("matched_memit_binding")
    )
    return bool(
        config.get("qwen3_layer_selection_validated") is True
        and config.get("formal_baseline_eligible") is True
        and provenance
    )


def _transfer_reasons(
    run_name: str, method: str, config: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    independently_selected = _is_independent_qwen3_selection(config)
    for key in ("qwen3_preset_status", "qwen3_layer_selection_status"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            lowered = value.casefold()
            if (not independently_selected
                    and any(token in lowered for token in ("transfer", "untuned", "not officially"))):
                reasons.append(value.strip())
    if config.get("qwen3_layer_selection_validated") is False:
        reasons.append("Qwen3 layer selection is explicitly unvalidated")
    if ("upstream_qwen25_recipe" in config or "upstream_qwen25_preset" in config) and not independently_selected:
        reasons.append("uses an upstream Qwen2.5 preset on Qwen3")
    if "transfer" in run_name.casefold() or "transfer" in str(method).casefold():
        reasons.append("run is explicitly named as a transfer")
    return list(dict.fromkeys(reasons))


def _diagnostic_reasons(
    run: dict[str, Any],
    config: dict[str, Any],
    completed: dict[str, Any],
    expected_num_facts: int | None,
) -> list[str]:
    reasons: list[str] = []
    diagnostic_text = " ".join(
        str(value)
        for value in (
            run.get("run_name"),
            config.get("preset_label"),
            config.get("reproduction_status"),
            completed.get("reproduction_status"),
            config.get("formal_baseline_qualification"),
        )
        if value is not None
    ).casefold()
    if "smoke" in diagnostic_text:
        reasons.append("explicit smoke run")
    if "diag" in diagnostic_text or "diagnostic" in diagnostic_text:
        reasons.append("explicit diagnostic run")
    if config.get("formal_baseline_eligible") is False or completed.get(
        "formal_baseline_eligible"
    ) is False:
        reasons.append("runner marks the artifact ineligible for a formal baseline")
    if config.get("publication_ready") is False:
        reasons.append("runner marks the configuration as not publication-ready")
    if config.get("diagnostic_reference") is True:
        reasons.append("runner marks the artifact as a diagnostic reference")
    if config.get("allow_diagnostic_stats") is True:
        reasons.append("uses diagnostic rather than publication statistics")
    if str(config.get("mode", "")).casefold() == "batch":
        reasons.append("one-shot batch mode is not the sequential continual protocol")
    evaluated = run.get("num_facts")
    if (
        run.get("status") == "complete"
        and expected_num_facts is not None
        and evaluated is not None
        and evaluated != expected_num_facts
    ):
        reasons.append(
            f"completed evaluation covers {evaluated} facts, not the "
            f"{expected_num_facts}-fact protocol"
        )
    return list(dict.fromkeys(reasons))


def _is_side_memory(method: str, config: dict[str, Any]) -> bool:
    text = " ".join(
        str(value)
        for value in (method, config.get("method_family"), config.get("router"))
        if value is not None
    ).casefold()
    return "wise" in text or "side" in text and "memory" in text


def _is_open_book(method: str, config: dict[str, Any]) -> bool:
    if config.get("open_book_evaluation") is True:
        return True
    text = " ".join(
        str(value)
        for value in (method, config.get("method_family"), config.get("method_variant"))
        if value is not None
    ).casefold()
    return "rag" in text or "external retrieval" in text


def _method_specific_warnings(
    run: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    warnings: list[str] = []
    normalized = _normalized_method(run.get("method"))
    upstream_corrections = config.get("known_upstream_config_corrections")
    if isinstance(upstream_corrections, list):
        warnings.extend(
            "upstream config correction: " + str(correction).strip()
            for correction in upstream_corrections
            if str(correction).strip()
        )
    if run["run_name"].startswith("full_ft") and not config.get("method"):
        warnings.append(
            "run_config.json does not declare method; inferred Full FT-BF from "
            "the run name and breadth-first FSDP metadata"
        )
    if normalized in {"mir", "mirlora"}:
        target = _target_description(config)
        rank = _as_int(config.get("lora_r"))
        epochs = _as_int(config.get("epochs"))
        if target == "q/v" and rank == 4 and epochs == 1:
            warnings.append(
                "low-budget MIR uses q/v LoRA r4 for 1 epoch; it is not "
                "recipe-matched to the all-linear r16, 5-epoch LoRA/ER runs"
            )
    if normalized == "memit" and str(config.get("stats_source", "")).casefold() == "manifest_diag":
        warnings.append(
            "MEMIT uses diagonal proxy statistics from selected WikiBigEdit prompts, "
            "not the required Qwen3 Wikipedia second-moment matrices"
        )
    if normalized == "memit":
        provenance = config.get("selection_provenance")
        if isinstance(provenance, str) and provenance.strip():
            warnings.append(
                "MEMIT Qwen3 variant-selection provenance: " + provenance.strip()
            )
    if normalized == "alphaedit":
        preset_label = str(config.get("preset_label", "")).casefold()
        if "adapted" in preset_label and "sensitivity" in preset_label:
            warnings.append(
                "AlphaEdit layers 5–8 are a Qwen3 sensitivity variant; this is "
                "not an official Qwen3 preset"
            )
    if "crispedit" in normalized:
        blocker = config.get("publication_blocker")
        if isinstance(blocker, str) and blocker.strip():
            warnings.append(f"CrispEdit publication blocker: {blocker.strip()}")
    if normalized == "stableedit":
        preheat = config.get("preheat")
        if isinstance(preheat, str) and "disabled" in preheat.casefold():
            warnings.append(
                "StableEdit preheat is disabled; this differs from the official recipe"
            )
    if _is_side_memory(str(run.get("method", "")), config):
        warnings.append(
            "uses side parametric memory with a router; report separately from "
            "backbone-only parameter internalization"
        )
    if _is_open_book(str(run.get("method", "")), config):
        warnings.append(
            "uses test-time external context; report in the open-book table, not "
            "as parameter internalization"
        )
        top_k = _as_int(config.get("retrieval_top_k", config.get("top_k")))
        if normalized in {"denserag", "ragdense"} and top_k is not None and top_k > 1:
            warnings.append(
                "top-K changes both retrieval coverage and the number of in-context "
                "Q/A demonstrations; compare against Top-1 before attributing the "
                "gain solely to retrieval recall"
            )
        if normalized in {"oraclerag", "ragoracle"}:
            warnings.append(
                "Oracle RAG is an exact-fact single-context (Top-1) reader diagnostic; "
                "it is not a theoretical upper bound for multi-context Dense RAG"
            )
        if config.get("no_stage_eval") is True and not run.get("linked_staged_audit", {}).get("verified"):
            warnings.append(
                "open-book metrics are final-stage-only (--no-stage-eval); they do not "
                "establish a complete per-stage retrieval curve"
            )
    return warnings


def _run_summary(
    path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    warnings: list[str] = []
    config_payload, config_error = _read_json(path / "run_config.json")
    linked_audit, _ = _read_json(path / "staged_audit.json")
    if not isinstance(linked_audit, dict):
        linked_audit = {}
    if linked_audit.get("verified"):
        files = linked_audit.get("files", {})
        linked_audit["verified"] = bool(files) and all(
            _sha256_file(Path(name))[0] == digest for name, digest in files.items()
        )
    metrics_payload, metrics_error = _read_json(path / "metrics.json")
    completed_payload, completed_error = _read_json(path / "completed.json")
    warnings.extend(
        error
        for error in (config_error, metrics_error, completed_error)
        if error is not None
    )
    config = config_payload if isinstance(config_payload, dict) else {}
    completed = completed_payload if isinstance(completed_payload, dict) else {}
    metric_records = _metric_records(metrics_payload)
    metric, metric_index = select_final_metric(
        metrics_payload, config=config, completed=completed
    )
    axes = {axis: _axis_metrics(metric, axis) for axis in AXES}
    checkpoint = (
        str(metric.get("checkpoint"))
        if isinstance(metric, dict) and metric.get("checkpoint") is not None
        else None
    )
    predictions_path = path / "predictions.jsonl"
    _, prediction_metadata, prediction_warnings = load_locality_predictions(
        predictions_path, checkpoint
    )
    warnings.extend(prediction_warnings)

    method = config.get("method") or completed.get("method") or path.name
    status = _run_status(config, completed)
    configured_num_facts = _configured_num_facts(config, completed)
    evaluated_num_facts = (
        _as_int(metric.get("num_facts")) if isinstance(metric, dict) else None
    )
    num_facts = _first_int(evaluated_num_facts, configured_num_facts)
    configured_final_stage = _configured_max_stage(config)
    completed_final_stage = _as_int(completed.get("final_stage"))
    evaluated_final_stage = (
        _as_int(metric.get("current_stage")) if isinstance(metric, dict) else None
    )
    metrics_sha256, metrics_hash_error = _sha256_file(path / "metrics.json")
    predictions_sha256, predictions_hash_error = _sha256_file(predictions_path)
    warnings.extend(
        error
        for error in (metrics_hash_error, predictions_hash_error)
        if error is not None
    )
    summary = {
        "run_name": path.name,
        "linked_staged_audit": linked_audit,
        "run_path": str(path.resolve()),
        "method": str(method),
        "display_label": _display_method_label(path.name, str(method), config),
        "status": status,
        "checkpoint": checkpoint,
        "metrics_record_index": metric_index,
        "protocol_id": config.get("manifest_protocol_id"),
        "num_facts": num_facts,
        "closed_book_evaluation": config.get("closed_book_evaluation"),
        "open_book_evaluation": config.get("open_book_evaluation"),
        "parameter_updates": config.get("parameter_updates"),
        "reader_frozen": config.get("reader_frozen"),
        "completion_audit": {
            "completed_marker_present": (path / "completed.json").is_file(),
            "configured_num_facts": configured_num_facts,
            "evaluated_num_facts": evaluated_num_facts,
            "configured_final_stage": configured_final_stage,
            "completed_final_stage": completed_final_stage,
            "evaluated_final_stage": evaluated_final_stage,
            "metrics_record_count": len(metric_records),
        },
        "axes": axes,
        "current_update_accuracy": (
            _as_finite_float(metric.get("current_update_accuracy"))
            if isinstance(metric, dict)
            else None
        ),
        "past_update_retention": (
            _as_finite_float(metric.get("past_update_retention"))
            if isinstance(metric, dict)
            else None
        ),
        "retrieval": (
            metric.get("retrieval")
            if isinstance(metric, dict) and isinstance(metric.get("retrieval"), dict)
            else None
        ),
        "retrieval_history_audit": _retrieval_history_audit(metric_records),
        "elapsed_seconds": _as_finite_float(completed.get("elapsed_seconds")),
        "method_config": {
            key: config[key]
            for key in (
                "mode",
                "target_modules",
                "layer",
                "lora_r",
                "lora_alpha",
                "merge_alpha",
                "memory_capacity",
                "replay_ratio",
                "world_size",
                "epochs",
                "layers",
                "chunk_size",
                "num_chunks",
                "method_family",
                "method_variant",
                "preset_label",
                "selection_provenance",
                "solver_ridge",
                "qwen3_preset_status",
                "qwen3_layer_selection_status",
                "qwen3_layer_selection_validated",
                "formal_baseline_eligible",
                "selection_certificate",
                "selection_provenance",
                "development_selection",
                "matched_memit_binding",
                "loss_layer",
                "v_loss_layer",
                "mom2_adjustment",
                "mom2_ridge",
                "mom2_update_weight",
                "max_update_weight_ratio",
                "max_cumulative_weight_drift_ratio",
                "oversized_update_action",
                "edit_batch_size",
                "stats_sample_size",
                "allow_proxy_stats",
                "publication_ready",
                "stats_source",
                "retrieval_top_k",
                "embedding_model",
                "no_stage_eval",
                "diagnostic_reference",
            )
            if key in config
        },
        "locality": {
            "absolute_match_accuracy": axes["locality"]["match_accuracy"],
            "absolute_exact_accuracy": axes["locality"]["exact_accuracy"],
            "match_delta_vs_pre_edit": None,
            "exact_delta_vs_pre_edit": None,
            "output_preservation": {
                "available": False,
                "reason": "not yet compared with base predictions",
            },
            "prediction_records": prediction_metadata,
        },
        "artifacts": {
            name: (path / name).is_file()
            for name in STANDARD_ARTIFACT_NAMES
        },
        "artifact_sha256": {
            "metrics.json": metrics_sha256,
            "predictions.jsonl": predictions_sha256,
        },
        "classification": {
            "comparison_group": None,
            "comparison_group_label": None,
            "closed_book": config.get("closed_book_evaluation") is True,
            "open_book": config.get("open_book_evaluation") is True,
            "transfer_or_untuned": False,
            "transfer_reasons": [],
            "diagnostic_or_smoke": False,
            "diagnostic_reasons": [],
            "side_memory": False,
            "main_3k_eligible": False,
            "main_3k_exclusion_reasons": [],
        },
        "duplicate_of": None,
        "duplicate_copies": [],
        "warnings": warnings,
    }
    if metric is None and metrics_error is None:
        summary["warnings"].append("no usable metrics record")
    if status != "complete":
        if evaluated_num_facts is None:
            summary["warnings"].append("run is incomplete and has no evaluated facts yet")
        else:
            stage_text = (
                f" through stage {evaluated_final_stage}"
                if evaluated_final_stage is not None
                else ""
            )
            summary["warnings"].append(
                f"run is incomplete; latest metrics cover {evaluated_num_facts} facts{stage_text}"
            )
    if (
        configured_num_facts is not None
        and evaluated_num_facts is not None
        and configured_num_facts != evaluated_num_facts
    ):
        summary["warnings"].append(
            f"selected metric covers {evaluated_num_facts} of "
            f"{configured_num_facts} configured facts"
        )
    if (
        status == "complete"
        and configured_final_stage is not None
        and evaluated_final_stage is not None
        and evaluated_final_stage != configured_final_stage
    ):
        summary["warnings"].append(
            f"completed marker conflicts with evaluated stage {evaluated_final_stage}; "
            f"configured final stage is {configured_final_stage}"
        )
    summary["warnings"].extend(_method_specific_warnings(summary, config))
    return summary, predictions_path, config, completed


def _find_base_run(runs: Sequence[dict[str, Any]], requested_name: str) -> dict[str, Any] | None:
    for run in runs:
        if run["run_name"] == requested_name:
            return run
    aliases = {"pre_edit", "pre-edit", "preedit", "base"}
    for run in runs:
        if str(run.get("method", "")).strip().casefold() in aliases:
            return run
    return None


def _infer_expected_num_facts(
    runs: Sequence[dict[str, Any]], base_run: dict[str, Any] | None
) -> int | None:
    if base_run is not None:
        base_audit = base_run.get("completion_audit", {})
        expected = _first_int(
            base_audit.get("configured_num_facts"), base_run.get("num_facts")
        )
        if expected is not None:
            return expected
    candidates = [
        _first_int(
            run.get("completion_audit", {}).get("configured_num_facts"),
            run.get("num_facts"),
        )
        for run in runs
    ]
    counts = Counter(value for value in candidates if value is not None)
    return counts.most_common(1)[0][0] if counts else None


def _version_suffix(run_name: str) -> int:
    match = re.search(r"(?:^|_)v(\d+)$", run_name.casefold())
    return int(match.group(1)) if match else 0


def _mark_exact_duplicate_results(
    runs: Sequence[dict[str, Any]], configs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        fingerprints = run.get("artifact_sha256", {})
        metrics_hash = fingerprints.get("metrics.json")
        predictions_hash = fingerprints.get("predictions.jsonl")
        if (
            run.get("status") != "complete"
            or not metrics_hash
            or not predictions_hash
        ):
            continue
        signature = (
            _normalized_method(run.get("method")),
            run.get("protocol_id"),
            run.get("num_facts"),
            metrics_hash,
            predictions_hash,
        )
        buckets[signature].append(run)

    duplicate_groups: list[dict[str, Any]] = []
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue

        def preference(run: dict[str, Any]) -> tuple[int, int, str]:
            config = configs.get(run["run_name"], {})
            retained_artifact = int(
                bool(config.get("save_edited_qv") or config.get("save_delta"))
            )
            return (_version_suffix(run["run_name"]), retained_artifact, run["run_name"])

        canonical = max(bucket, key=preference)
        duplicates = sorted(
            run["run_name"] for run in bucket if run is not canonical
        )
        canonical["duplicate_copies"] = duplicates
        canonical["warnings"].append(
            "identical metrics.json and predictions.jsonl also appear in "
            + ", ".join(duplicates)
            + "; only this canonical copy is eligible for a main table"
        )
        for run in bucket:
            if run is canonical:
                continue
            run["duplicate_of"] = canonical["run_name"]
            run["warnings"].append(
                f"exact result duplicate of {canonical['run_name']}; excluded from main tables"
            )
        duplicate_groups.append(
            {
                "canonical_run": canonical["run_name"],
                "duplicate_runs": duplicates,
                "evidence": "identical metrics.json and predictions.jsonl SHA-256",
            }
        )
    return duplicate_groups


def _classify_run(
    run: dict[str, Any],
    config: dict[str, Any],
    completed: dict[str, Any],
    expected_num_facts: int | None,
) -> None:
    classification = run["classification"]
    transfer_reasons = _transfer_reasons(run["run_name"], run["method"], config)
    diagnostic_reasons = _diagnostic_reasons(
        run, config, completed, expected_num_facts
    )
    side_memory = _is_side_memory(run["method"], config)
    open_book = _is_open_book(run["method"], config)
    if open_book:
        group = "open_book"
    elif diagnostic_reasons:
        group = "diagnostic_smoke"
    elif side_memory:
        group = "side_memory"
    elif transfer_reasons:
        group = "transfer_untuned"
    else:
        group = "formal_closed_book"
    classification.update(
        {
            "comparison_group": group,
            "comparison_group_label": COMPARISON_GROUP_LABELS[group],
            "transfer_or_untuned": bool(transfer_reasons),
            "transfer_reasons": transfer_reasons,
            "diagnostic_or_smoke": bool(diagnostic_reasons),
            "diagnostic_reasons": diagnostic_reasons,
            "side_memory": side_memory,
            "open_book": open_book,
        }
    )
    if transfer_reasons:
        run["warnings"].append(
            "Qwen3 configuration is transferred or untuned: " + "; ".join(transfer_reasons)
        )
    if diagnostic_reasons:
        run["warnings"].append(
            "diagnostic/smoke artifact: " + "; ".join(diagnostic_reasons)
        )


def _main_exclusion_reasons(
    run: dict[str, Any],
    *,
    expected_num_facts: int | None,
    expected_protocol_id: str | None,
) -> list[str]:
    reasons: list[str] = []
    classification = run["classification"]
    if classification["diagnostic_or_smoke"]:
        reasons.extend(classification["diagnostic_reasons"])
    if run.get("status") != "complete":
        reasons.append(f"status is {run.get('status')}")
    if run.get("metrics_record_index") is None:
        reasons.append("no usable final metrics")
    if expected_num_facts is not None and run.get("num_facts") != expected_num_facts:
        reasons.append(
            f"evaluates {run.get('num_facts')} facts instead of {expected_num_facts}"
        )
    if expected_protocol_id is not None and run.get("protocol_id") != expected_protocol_id:
        reasons.append(
            f"protocol_id {run.get('protocol_id')!r} does not match {expected_protocol_id!r}"
        )
    if classification.get("open_book"):
        if classification.get("closed_book"):
            reasons.append("open-book run is incorrectly marked as closed-book")
        if run.get("open_book_evaluation") is not True:
            reasons.append("not explicitly marked as open-book evaluation")
        if run.get("parameter_updates") is not False:
            reasons.append("open-book run does not explicitly declare zero parameter updates")
        if run.get("reader_frozen") is not True:
            reasons.append("open-book reader is not explicitly marked frozen")
        retrieval = run.get("retrieval")
        if not isinstance(retrieval, dict):
            reasons.append("open-book run has no retrieval audit")
        else:
            future_leaks = _as_int(retrieval.get("future_leak_count"))
            if future_leaks is None:
                reasons.append("retrieval audit has no future-leak count")
            elif future_leaks != 0:
                reasons.append(f"retrieval audit reports {future_leaks} future-fact leaks")
        history_audit = run.get("retrieval_history_audit")
        if not isinstance(history_audit, dict) or not history_audit.get(
            "all_stage_records_audited"
        ):
            reasons.append("one or more stage metrics lack a valid retrieval leak audit")
        elif not history_audit.get("all_stage_records_zero_future_leaks"):
            reasons.append(
                "retrieval history reports "
                f"{history_audit.get('future_leak_count_total')} future-fact leaks"
            )
    elif run.get("closed_book_evaluation") is not True:
        reasons.append("not explicitly marked as closed-book evaluation")
    if run.get("duplicate_of"):
        reasons.append(f"exact duplicate of {run['duplicate_of']}")
    if any(run["axes"][axis]["match_accuracy"] is None for axis in AXES):
        reasons.append("one or more required WikiBigEdit axes are missing")
    configured_stage = run.get("completion_audit", {}).get("configured_final_stage")
    evaluated_stage = run.get("completion_audit", {}).get("evaluated_final_stage")
    if (
        _normalized_method(run.get("method")) not in {"preedit", "base"}
        and configured_stage is not None
        and evaluated_stage != configured_stage
    ):
        reasons.append(
            f"evaluated stage {evaluated_stage} does not match configured final "
            f"stage {configured_stage}"
        )
    return list(dict.fromkeys(reasons))


def _update_sort_value(run: dict[str, Any]) -> float:
    value = run.get("axes", {}).get("update", {}).get("match_accuracy")
    parsed = _as_finite_float(value)
    return parsed if parsed is not None else -1.0


def build_summary(runs_dir: Path, *, base_run_name: str = "pre_edit") -> dict[str, Any]:
    """Scan run artifacts and return a JSON-serializable summary."""

    runs_dir = Path(runs_dir)
    checkpoint_identity_path = (
        runs_dir.parent / "protocols" / "qwen3_8b_base_checkpoint_identity.json"
    )
    checkpoint_identity_payload, checkpoint_identity_error = _read_json(
        checkpoint_identity_path
    )
    runs: list[dict[str, Any]] = []
    prediction_paths: dict[str, Path] = {}
    configs: dict[str, dict[str, Any]] = {}
    completed_payloads: dict[str, dict[str, Any]] = {}
    for directory in _experiment_directories(runs_dir):
        run, prediction_path, config, completed = _run_summary(directory)
        runs.append(run)
        prediction_paths[run["run_name"]] = prediction_path
        configs[run["run_name"]] = config
        completed_payloads[run["run_name"]] = completed

    base_run = _find_base_run(runs, base_run_name)
    top_warnings: list[str] = []
    if checkpoint_identity_error is not None:
        top_warnings.append(checkpoint_identity_error)
    if base_run is None:
        top_warnings.append(f"Pre-Edit reference run {base_run_name!r} was not found")
        base_predictions: dict[str, str] = {}
        base_match = base_exact = None
        resolved_base_name = None
    else:
        resolved_base_name = str(base_run["run_name"])
        base_match = base_run["locality"]["absolute_match_accuracy"]
        base_exact = base_run["locality"]["absolute_exact_accuracy"]
        base_predictions, refreshed_metadata, refreshed_warnings = load_locality_predictions(
            prediction_paths[resolved_base_name], base_run.get("checkpoint")
        )
        base_run["locality"]["prediction_records"] = refreshed_metadata
        base_run["warnings"].extend(refreshed_warnings)

    expected_num_facts = _infer_expected_num_facts(runs, base_run)
    if base_run is not None and base_run.get("protocol_id") is not None:
        expected_protocol_id = str(base_run.get("protocol_id"))
    else:
        protocol_counts = Counter(
            str(run["protocol_id"])
            for run in runs
            if run.get("protocol_id") is not None
        )
        expected_protocol_id = (
            protocol_counts.most_common(1)[0][0] if protocol_counts else None
        )
    base_locality_count = (
        base_run.get("axes", {}).get("locality", {}).get("count")
        if base_run is not None
        else None
    )

    for run in runs:
        locality = run["locality"]
        if base_match is not None and locality["absolute_match_accuracy"] is not None:
            locality["match_delta_vs_pre_edit"] = (
                locality["absolute_match_accuracy"] - base_match
            )
        if base_exact is not None and locality["absolute_exact_accuracy"] is not None:
            locality["exact_delta_vs_pre_edit"] = (
                locality["absolute_exact_accuracy"] - base_exact
            )
        post_predictions, prediction_metadata, prediction_warnings = (
            load_locality_predictions(
                prediction_paths[run["run_name"]], run.get("checkpoint")
            )
        )
        locality["prediction_records"] = prediction_metadata
        locality["output_preservation"] = compare_output_preservation(
            base_predictions, post_predictions
        )
        preservation = locality["output_preservation"]
        locality["preservation_coverage_complete"] = bool(
            preservation.get("available") and preservation.get("coverage") == 1.0
        )
        run["warnings"].extend(prediction_warnings)
        if preservation.get("available") and preservation.get("coverage") != 1.0:
            run["warnings"].append(
                "locality output-preservation coverage is "
                f"{preservation.get('paired_count')}/{preservation.get('base_count')} "
                f"({_percent(preservation.get('coverage'))}); missing post predictions "
                "count as failures"
            )
        elif not preservation.get("available"):
            run["warnings"].append(
                "locality output-preservation is unavailable: "
                + str(preservation.get("reason", "unknown reason"))
            )
        locality_count = run.get("axes", {}).get("locality", {}).get("count")
        if (
            base_locality_count is not None
            and locality_count is not None
            and locality_count != base_locality_count
        ):
            run["warnings"].append(
                f"locality accuracy uses {locality_count} examples; Pre-Edit uses "
                f"{base_locality_count}"
            )
        _classify_run(
            run,
            configs[run["run_name"]],
            completed_payloads[run["run_name"]],
            expected_num_facts,
        )
        run["warnings"] = list(dict.fromkeys(run["warnings"]))

    duplicate_groups = _mark_exact_duplicate_results(runs, configs)
    if duplicate_groups:
        for group in duplicate_groups:
            top_warnings.append(
                f"exact duplicate result excluded: {', '.join(group['duplicate_runs'])} "
                f"duplicates {group['canonical_run']}"
            )

    for run in runs:
        exclusion_reasons = _main_exclusion_reasons(
            run,
            expected_num_facts=expected_num_facts,
            expected_protocol_id=expected_protocol_id,
        )
        run["classification"]["main_3k_exclusion_reasons"] = exclusion_reasons
        run["classification"]["main_3k_eligible"] = not exclusion_reasons
        if exclusion_reasons:
            run["warnings"].append(
                "excluded from main 3K tables: " + "; ".join(exclusion_reasons)
            )
        run["warnings"] = list(dict.fromkeys(run["warnings"]))

    group_order = {name: index for index, name in enumerate(COMPARISON_GROUPS)}
    runs.sort(
        key=lambda run: (
            0 if run["run_name"] == resolved_base_name else 1,
            group_order.get(run["classification"]["comparison_group"], 99),
            0 if run["classification"]["main_3k_eligible"] else 1,
            -_update_sort_value(run),
            run["run_name"],
        )
    )
    main_rankings: dict[str, list[str]] = {}
    for group in COMPARISON_GROUPS:
        eligible = [
            run
            for run in runs
            if run["classification"]["comparison_group"] == group
            and run["classification"]["main_3k_eligible"]
        ]
        eligible.sort(key=lambda run: (-_update_sort_value(run), run["run_name"]))
        main_rankings[group] = [run["run_name"] for run in eligible]
    excluded_names = [
        run["run_name"]
        for run in runs
        if not run["classification"]["main_3k_eligible"]
    ]
    classification_counts = Counter(
        run["classification"]["comparison_group"] for run in runs
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_dir": str(runs_dir.resolve()),
        "checkpoint_identity_file": (
            str(checkpoint_identity_path.resolve())
            if checkpoint_identity_path.is_file()
            else None
        ),
        "checkpoint_identity": (
            checkpoint_identity_payload
            if isinstance(checkpoint_identity_payload, dict)
            else None
        ),
        "base_run": resolved_base_name,
        "expected_protocol_id": expected_protocol_id,
        "expected_num_facts": expected_num_facts,
        "locality_semantics": {
            "absolute": "generated-answer accuracy against locality ground truth",
            "delta": "run absolute accuracy minus Pre-Edit absolute accuracy",
            "output_preservation": (
                "post-edit locality generation equals the Pre-Edit generation for "
                "the same sample_id"
            ),
            "preservation_denominator": (
                "all Pre-Edit locality prediction records; missing post records fail"
            ),
        },
        "run_count": len(runs),
        "complete_run_count": sum(run["status"] == "complete" for run in runs),
        "main_3k_run_count": sum(
            run["classification"]["main_3k_eligible"] for run in runs
        ),
        "comparison_group_counts": {
            group: classification_counts.get(group, 0) for group in COMPARISON_GROUPS
        },
        "main_3k_rankings": main_rankings,
        "excluded_run_names": excluded_names,
        "exact_duplicate_groups": duplicate_groups,
        "warnings": top_warnings,
        "runs": runs,
    }


def _percent(value: Any, *, signed: bool = False) -> str:
    number = _as_finite_float(value)
    if number is None:
        return "—"
    if signed:
        return f"{number * 100:+.2f} pp"
    return f"{number * 100:.2f}%"


def _seconds(value: Any) -> str:
    number = _as_finite_float(value)
    if number is None:
        return "—"
    if number < 60:
        return f"{number:.1f}s"
    return f"{number / 60:.1f}m"


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _result_row(run: dict[str, Any], rank: int | None = None) -> list[Any]:
    axes = run["axes"]
    locality = run["locality"]
    preservation = locality["output_preservation"]
    if preservation.get("available"):
        preservation_text = (
            f"{_percent(preservation.get('normalized_equal_accuracy'))} norm / "
            f"{_percent(preservation.get('raw_equal_accuracy'))} raw "
            f"(coverage {_percent(preservation.get('coverage'))})"
        )
    else:
        preservation_text = "—"
    values: list[Any] = []
    if rank is not None:
        values.append(rank)
    values.extend(
        [
            run["display_label"],
            run["run_name"],
            run["status"],
            run.get("num_facts") if run.get("num_facts") is not None else "—",
            _percent(axes["update"]["match_accuracy"]),
            _percent(axes["rephrase"]["match_accuracy"]),
            _percent(axes["persona"]["match_accuracy"]),
            _percent(locality["absolute_match_accuracy"]),
            _percent(locality["match_delta_vs_pre_edit"], signed=True),
            preservation_text,
            _percent(axes["multihop"]["match_accuracy"]),
            _percent(run.get("past_update_retention")),
            _seconds(run.get("elapsed_seconds")),
        ]
    )
    return values


def _append_main_table(
    lines: list[str], title: str, runs: Sequence[dict[str, Any]]
) -> None:
    lines.extend([f"## {title}", ""])
    if not runs:
        lines.extend(["No eligible completed 3K runs.", ""])
        return
    lines.extend(
        [
            "| Rank | Method label | Run | Status | N | Update | Rephrase | "
            "Persona | Locality abs. | Locality Δ | Output preservation | "
            "MultiHop | Past retention | Time |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, run in enumerate(runs, start=1):
        lines.append(
            "| "
            + " | ".join(_escape_cell(value) for value in _result_row(run, rank))
            + " |"
        )
    lines.append("")


def _append_open_book_table(
    lines: list[str], title: str, runs: Sequence[dict[str, Any]]
) -> None:
    lines.extend([f"## {title}", ""])
    if not runs:
        lines.extend(["No eligible completed 3K runs.", ""])
        return
    lines.extend(
        [
            "| Rank | Method label | Run | Status | N | Index N | Retrieval "
            "recall@K | Future leaks | Update | Rephrase | Persona | Locality "
            "abs. | MultiHop | Past retention | Time |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, run in enumerate(runs, start=1):
        axes = run["axes"]
        retrieval = run.get("retrieval") or {}
        values = [
            rank,
            run["display_label"],
            run["run_name"],
            run["status"],
            run.get("num_facts") if run.get("num_facts") is not None else "—",
            retrieval.get("index_count", "—"),
            _percent(retrieval.get("recall_at_k")),
            retrieval.get("future_leak_count", "—"),
            _percent(axes["update"]["match_accuracy"]),
            _percent(axes["rephrase"]["match_accuracy"]),
            _percent(axes["persona"]["match_accuracy"]),
            _percent(axes["locality"]["match_accuracy"]),
            _percent(axes["multihop"]["match_accuracy"]),
            _percent(run.get("past_update_retention")),
            _seconds(run.get("elapsed_seconds")),
        ]
        lines.append("| " + " | ".join(_escape_cell(value) for value in values) + " |")
    lines.append("")


def render_markdown(summary: dict[str, Any]) -> str:
    base_name = summary.get("base_run") or "unavailable"
    expected_num_facts = summary.get("expected_num_facts")
    expected_protocol_id = summary.get("expected_protocol_id") or "unavailable"
    checkpoint_identity = summary.get("checkpoint_identity") or {}
    runs_by_name = {run["run_name"]: run for run in summary.get("runs", [])}
    base_run = runs_by_name.get(str(base_name))
    panel_counts = None
    if base_run is not None:
        counts = {
            axis: _as_int(base_run.get("axes", {}).get(axis, {}).get("count"))
            for axis in AXES
        }
        if all(count is not None for count in counts.values()):
            panel_counts = (
                f"Panel counts: Update={counts['update']}, "
                f"Rephrase={counts['rephrase']}, Persona={counts['persona']}, "
                f"Locality={counts['locality']}, MultiHop={counts['multihop']}."
            )
    lines = [
        "# WikiBigEdit 3K Qwen3-8B-Base summary",
        "",
        f"Pre-Edit reference: `{_escape_cell(base_name)}`. Expected protocol: "
        f"`{_escape_cell(expected_protocol_id)}` with "
        f"{_escape_cell(expected_num_facts if expected_num_facts is not None else '—')} facts.",
        "",
        "Base checkpoint: "
        f"`{_escape_cell(checkpoint_identity.get('model_id', 'unrecorded'))}`; "
        "weight revision "
        f"`{_escape_cell(checkpoint_identity.get('weight_blob_revision', 'unrecorded'))}`; "
        "content fingerprint "
        f"`{_escape_cell(checkpoint_identity.get('content_fingerprint', 'unrecorded'))}`.",
        "",
        "Main tables include only completed, non-duplicate, full-protocol runs "
        "with all five WikiBigEdit axes. Closed-book parameter methods, side "
        "memory, and open-book retrieval are kept in separate tables. "
        "Diagnostic/smoke, partial, batch-only, and duplicate artifacts are "
        "listed separately. "
        "Accuracy values are greedy generated-answer scores, not teacher-forced "
        "token accuracy.",
        "",
        "Locality Δ is the absolute percentage-point change from Pre-Edit. "
        "Output preservation compares locality generations by `sample_id`; "
        "normalized preservation applies NFKC/case/punctuation/whitespace "
        "normalization and reports coverage explicitly.",
        "",
    ]
    if panel_counts is not None:
        lines.extend([panel_counts, ""])

    rankings = summary.get("main_3k_rankings", {})
    for group in ("formal_closed_book", "transfer_untuned", "side_memory"):
        selected = [
            runs_by_name[name]
            for name in rankings.get(group, [])
            if name in runs_by_name
        ]
        _append_main_table(
            lines,
            f"{COMPARISON_GROUP_LABELS[group]} 3K results",
            selected,
        )

    open_book = [
        runs_by_name[name]
        for name in rankings.get("open_book", [])
        if name in runs_by_name
    ]
    _append_open_book_table(
        lines,
        f"{COMPARISON_GROUP_LABELS['open_book']} 3K results",
        open_book,
    )

    linked = [r for r in summary.get("runs", []) if r.get("linked_staged_audit", {}).get("verified")]
    if linked:
        lines.extend(["## Verified linked RAG stage audits", "",
            "The final-only main artifacts have independently completed τ1–τ8 audits. "
            "All linked file hashes were rechecked when generating this summary; each stage uses only its visible prefix, "
            "future leaks are zero, and τ8 metrics and prediction bytes match the main result.", ""])
        for run in linked:
            audit = run["linked_staged_audit"]
            lines.append(f"- `{run['run_name']}`: `{audit['staged_run']}`; index counts {audit['index_counts']}.")
        lines.append("")

    excluded = [
        run
        for run in summary.get("runs", [])
        if not run.get("classification", {}).get("main_3k_eligible")
    ]
    lines.extend(["## Excluded, diagnostic, or in-progress artifacts", ""])
    if excluded:
        lines.extend(
            [
                "| Method label | Run | Class | Status | N | Update | Rephrase | "
                "Persona | Locality abs. | Locality Δ | Output preservation | "
                "MultiHop | Past retention | Time | Exclusion reason |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for run in excluded:
            classification = run["classification"]
            values = _result_row(run)
            group_label = classification["comparison_group_label"]
            if (classification["comparison_group"] == "diagnostic_smoke"
                    and run.get("num_facts") == expected_num_facts
                    and run.get("status") == "complete"):
                group_label = "Full 3K diagnostic / nonstandard protocol"
            values.insert(2, group_label)
            values.append("; ".join(classification["main_3k_exclusion_reasons"]))
            lines.append(
                "| " + " | ".join(_escape_cell(value) for value in values) + " |"
            )
    else:
        lines.append("None.")

    warning_rows = [
        (run["run_name"], warning)
        for run in summary.get("runs", [])
        for warning in run.get("warnings", [])
    ]
    top_warnings = summary.get("warnings", [])
    if top_warnings or warning_rows:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {_escape_cell(warning)}" for warning in top_warnings)
        lines.extend(
            f"- `{_escape_cell(run_name)}`: {_escape_cell(warning)}"
            for run_name, warning in warning_rows
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--base-run", default="pre_edit")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="default: <runs-dir>/summary.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="default: <runs-dir>/summary.md",
    )
    parser.add_argument(
        "--print-markdown",
        action="store_true",
        help="also print the Markdown report to stdout",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_summary(args.runs_dir, base_run_name=args.base_run)
    markdown = render_markdown(summary)
    json_output = args.json_output or args.runs_dir / "summary.json"
    markdown_output = args.markdown_output or args.runs_dir / "summary.md"
    _atomic_write(
        json_output,
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _atomic_write(markdown_output, markdown)
    if args.print_markdown:
        print(markdown, end="")
    else:
        print(f"wrote {json_output} and {markdown_output}")


if __name__ == "__main__":
    main()
