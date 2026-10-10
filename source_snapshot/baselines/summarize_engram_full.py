#!/usr/bin/env python3
"""Summarize continual-writer artifacts without importing a model or torch.

Accept one run or a suite via --root. Missing measurements remain null; legacy
LoRA sequence_tokens are non-padding training tokens, not physical GPU work.
Training methods and selected streams are kept in separate comparison groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AXES = ("update", "rephrase", "persona", "locality", "multihop")
ACTIONS = ("add", "merge", "ignore")
ARTIFACTS = ("run_config.json", "metrics.json", "completed.json", "actions.jsonl",
             "training_log.json", "policy_updates.jsonl")
SCHEMA = "engram-continual-summary/v1"


def number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def first_number(record: dict, *keys: str) -> int | float | None:
    return next((number(record[key]) for key in keys if number(record.get(key)) is not None), None)


def read_json(path: Path, warnings: list[str], expected: type) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, expected):
            raise ValueError(f"expected {expected.__name__}")
        return value
    except (OSError, ValueError, UnicodeError) as error:
        warnings.append(f"{path.name}: {error}")
        return None


def read_jsonl(path: Path, warnings: list[str], *, action_records: bool = False) -> tuple[list[dict] | None, bool]:
    try:
        stream = path.open(encoding="utf-8")
    except OSError as error:
        warnings.append(f"{path.name}: {error}")
        return None, False
    rows, valid = [], True
    with stream:
        try:
            for index, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected an object")
                    if action_records and row.get("action") not in ACTIONS:
                        raise ValueError("expected add/merge/ignore action")
                    rows.append(row)
                except ValueError as error:
                    warnings.append(f"{path.name} line {index}: {error}")
                    valid = False
        except (UnicodeError, OSError) as error:
            warnings.append(f"{path.name}: {error}")
            valid = False
    return rows, valid


def training_kind(config: dict, warnings: list[str]) -> dict:
    """Require parameter evidence: FSDP alone does not establish full FT."""
    method = str(config.get("training_method", "")).lower()
    backend = config.get("backend")
    backend_name = backend.get("backend", "") if isinstance(backend, dict) else backend
    backend_name = backend_name.lower() if isinstance(backend_name, str) else ""
    backend_lora = isinstance(backend, dict) and backend.get("lora") is True
    frozen = config.get("base_weights_frozen")
    fraction = number(config.get("trainable_fraction"))
    full = config.get("full_parameter_update") is True or "full_parameter" in method or "full_ft" in method
    lora = "lora" in method or backend_lora or "lora" in backend_name or (frozen is True and (number(config.get("lora_r")) or 0) > 0)
    contradictory = (full and (frozen is True or lora or config.get("full_parameter_update") is False
                               or (fraction is not None and fraction < 1)))
    contradictory |= lora and (frozen is False or fraction == 1)
    contradictory |= fraction is not None and not 0 <= fraction <= 1
    if contradictory:
        kind = "inconsistent"
        warnings.append("Training metadata conflicts; full-parameter/LoRA classification is not reliable.")
    elif full or (frozen is False and fraction == 1):
        kind = "full_parameter"
    elif lora:
        kind = "lora"
    elif frozen is False and fraction is not None and 0 < fraction < 1:
        kind = "partial_parameter"
    else:
        kind = "unknown"
        warnings.append("Training method is not established by run_config.json.")
    return {"kind": kind, "declared_method": config.get("training_method"),
            "backend": config.get("backend"), "base_weights_frozen": frozen,
            "trainable_fraction": fraction, "full_parameter_update": config.get("full_parameter_update")}


def sum_records(rows: list[dict] | None, *keys: str) -> int | float | None:
    if not rows:
        return None
    values = [first_number(row, *keys) if isinstance(row, dict) else None for row in rows]
    return sum(values) if all(value is not None for value in values) else None


def total(completed: dict, stages: list[dict] | None, events: list[dict] | None, *keys: str) -> dict:
    observations = {"completed.json": first_number(completed, *keys),
                    "training_log.json": sum_records(stages, *keys),
                    "actions.jsonl": sum_records(events, *keys)}
    source = next((name for name, value in observations.items() if value is not None), None)
    return {"value": observations.get(source), "source": source, "observations": observations}


def final_metric(metrics: list[dict]) -> dict:
    def stage(row: dict) -> int | float:
        value = number(row.get("current_stage"))
        match = re.fullmatch(r"tau(\d+)", str(row.get("checkpoint", "")))
        return value if value is not None else int(match[1]) if match else -1
    post = [row for row in metrics if row.get("checkpoint") not in ("base", "pre_edit")]
    return max(enumerate(post), key=lambda item: (stage(item[1]), item[0]))[1] if post else {}


def decision_summary(actions: list[dict] | None, mode: Any, update_log: list[dict] | None = None) -> dict:
    if actions is None:
        result = {key: None for key in ("policy_updates", "policy_update_observed_records", "mask_forced_count",
                                       "forced_observed_events", "deterministic_distribution_count",
                                       "credited_transitions", "policy_updates_source")}
        if update_log is not None:
            observed = decision_summary([], mode, update_log)
            for key in ("policy_updates", "policy_update_observed_records", "credited_transitions", "policy_updates_source"):
                result[key] = observed[key]
        return result
    records = update_log if update_log is not None else [row["policy_update"] for row in actions if isinstance(row.get("policy_update"), dict)]
    updates = [row.get("updated") for row in records]
    update_known = sum(isinstance(value, bool) for value in updates)
    nonpolicy = mode in ("always_add", "rule", "random_replay", "hash_replay")
    forced, deterministic = [], []
    for row in actions:
        explicit = row.get("forced", row.get("mask_forced"))
        if isinstance(explicit, bool):
            forced.append(explicit)
        elif isinstance(row.get("allow_ignore"), bool) and isinstance(row.get("allow_merge"), bool):
            forced.append(not row["allow_ignore"] and not row["allow_merge"])
        probabilities = row.get("probabilities")
        if isinstance(probabilities, dict) and probabilities and all(number(p) is not None for p in probabilities.values()):
            deterministic.append(sum(p > 0 for p in probabilities.values()) == 1)
    return {"policy_updates": sum(value is True for value in updates) if update_known else 0 if (nonpolicy and actions) or update_log == [] else None,
            "policy_update_observed_records": update_known,
            "policy_updates_source": "policy_updates.jsonl" if update_log is not None else "actions.jsonl",
            "credited_transitions": sum_records([row for row in records if row.get("updated") is True], "transition_count"),
            "mask_forced_count": sum(forced) if forced else None,
            "forced_observed_events": len(forced),
            "deterministic_distribution_count": sum(deterministic) if deterministic else None}


def fact_coverage(config: dict, completed: dict, actions: list[dict] | None,
                  actions_valid: bool, configured_facts: int | float | None,
                  fallback_facts: int | float | None, final: dict, warnings: list[str]) -> dict:
    """Check recorded ID/count consistency, not retirement authorization or alias closure."""
    valid_id = lambda value: isinstance(value, str) and bool(value.strip())
    valid_count = lambda value: isinstance(value, int) and not isinstance(value, bool) and value >= 0
    arrival_ids = [row.get("sample_id") for row in actions or []]
    observed = len(set(arrival_ids)) if actions is not None and all(map(valid_id, arrival_ids)) else None
    result = {"configured_facts": configured_facts, "observed_facts": observed,
              "active_facts": None, "retired_facts": None,
              "evaluated_facts": number(final.get("num_facts")),
              "expected_facts": fallback_facts, "basis": "configured_facts",
              "retirement_evidence": "absent", "valid": True, "matches_expected": None}
    retirement_logged = any("retired_now" in row for row in actions or [])
    if retirement_logged:
        errors = []
        expected_events = first_number(config, "num_events", "events")
        marker_events = first_number(completed, "events", "num_events")
        if (not actions_valid or completed.get("status") != "complete"
                or not valid_count(expected_events) or expected_events != len(actions)
                or not valid_count(marker_events) or marker_events != len(actions)):
            errors.append("complete actions and matching configured/completed event counts are required")
        if not valid_count(configured_facts):
            errors.append("a nonnegative integer configured fact count is required")
        for record, name, keys, expected in (
                (config, "run_config.json", ("num_facts", "facts"), configured_facts),
                (config, "run_config.json", ("num_events", "events"), len(actions)),
                (completed, "completed.json", ("num_facts", "facts"), observed),
                (completed, "completed.json", ("events", "num_events"), len(actions))):
            if any(key in record and (not valid_count(record[key]) or record[key] != expected) for key in keys):
                errors.append(f"{name} has an invalid or inconsistent {keys[0]} count")
        configured_ids = config.get("sample_ids")
        if (not isinstance(configured_ids, list) or not all(map(valid_id, configured_ids))
                or len(set(configured_ids)) != len(configured_ids)):
            errors.append("valid unique configured sample_ids are required")
            configured_ids = None
        elif len(configured_ids) != configured_facts:
            errors.append("configured sample_ids and fact count disagree")
        seen, retired = set(), set()
        previous_stage = None
        for index, row in enumerate(actions):
            sample_id = row.get("sample_id")
            if type(row.get("event")) is not int or row["event"] != index:
                errors.append(f"event {index}: contiguous zero-based event indices are required")
            stage = row.get("arrival_stage")
            if not valid_count(stage) or (previous_stage is not None and stage < previous_stage):
                errors.append(f"event {index}: nondecreasing integer arrival_stage evidence is required")
            else:
                previous_stage = stage
            if not valid_id(sample_id):
                errors.append(f"event {index}: a nonempty string arrival sample_id is required")
                continue
            if sample_id in retired:
                errors.append(f"event {index}: a retired sample_id arrived again")
            retiring = row.get("retired_now")
            if not isinstance(retiring, list) or not all(map(valid_id, retiring)):
                errors.append(f"event {index}: an explicit list of retired_now IDs is required")
            elif len(set(retiring)) != len(retiring):
                errors.append(f"event {index}: retired_now contains duplicate IDs")
            elif sample_id in retiring or not set(retiring).issubset(seen):
                errors.append(f"event {index}: retired_now references a self, unknown or future sample_id")
            else:
                # Repeated retirements across events are legal and idempotent.
                retired.update(retiring)
            seen.add(sample_id)
        if configured_ids is not None and seen != set(configured_ids):
            errors.append("arrival sample_ids do not cover exactly the configured sample_ids")
        terminal_stage = actions[-1].get("arrival_stage")
        if not valid_count(final.get("current_stage")) or final["current_stage"] != terminal_stage:
            errors.append("final metrics do not match the terminal arrival_stage")
        if ("final_stage" in completed and (not valid_count(completed["final_stage"])
                                           or completed["final_stage"] != terminal_stage)):
            errors.append("completed.json final_stage does not match the terminal arrival_stage")
        if errors:
            warnings.extend(f"Unverified retirement coverage: {error}." for error in errors)
            result.update(retirement_evidence="invalid", valid=False)
        else:
            active = len(seen - retired)
            result.update(active_facts=active, retired_facts=len(retired), expected_facts=active,
                          basis="verified_active_facts", retirement_evidence="verified")
        if not valid_count(final.get("num_facts")):
            warnings.append("Final evaluated fact count is missing or invalid for retirement-aware coverage.")
            result["valid"] = False
    if result["expected_facts"] is not None and result["evaluated_facts"] is not None:
        result["matches_expected"] = result["expected_facts"] == result["evaluated_facts"]
        if not result["matches_expected"]:
            basis = "verified active" if result["basis"] == "verified_active_facts" else "configured"
            warnings.append(f"Final evaluated fact count differs from the {basis} fact count.")
            result["valid"] = False
    return result


def summarize_run(path: Path) -> dict:
    warnings: list[str] = []
    config = read_json(path / "run_config.json", warnings, dict) or {}
    completed = read_json(path / "completed.json", warnings, dict) or {}
    raw_metrics = read_json(path / "metrics.json", warnings, list)
    metrics = [row for row in raw_metrics or [] if isinstance(row, dict)]
    stages = read_json(path / "training_log.json", warnings, list)
    actions, actions_valid = read_jsonl(path / "actions.jsonl", warnings, action_records=True)
    # Older random-replay logs mislabeled every nonempty random sample as
    # MERGE. Canonicalize the analysis view while preserving raw JSON files.
    if config.get("mode") == "random_replay" and actions is not None:
        invalid = [row for row in actions if row.get("action") != "add"
                   and row.get("replay_source") != "random_history"]
        if invalid:
            warnings.append("random_replay contains non-random action/source records; action counts are unverified.")
            actions_valid = False
        mislabeled = sum(row.get("action") == "merge" for row in actions)
        if mislabeled and not invalid:
            warnings.append(f"Normalized {mislabeled} legacy random-replay MERGE labels to semantic ADD.")
        for row in actions:
            if row.get("replay_source") == "random_history":
                row["action"] = "add"
    update_log = None
    if (path / "policy_updates.jsonl").is_file():
        update_log, _ = read_jsonl(path / "policy_updates.jsonl", warnings)
    final = final_metric(metrics)
    training = training_kind(config, warnings)
    counts = dict(Counter(row["action"] for row in actions)) if actions is not None else None
    reported_counts = completed.get("action_counts", completed.get("actions"))
    if not isinstance(reported_counts, dict):
        reported_counts = None
    if counts is not None:
        counts = {action: counts.get(action, 0) for action in ACTIONS}
    elif reported_counts is not None:
        counts = {action: number(reported_counts.get(action, 0)) for action in ACTIONS}
    expected_events = first_number(config, "num_events", "events")
    recorded_events = len(actions) if actions is not None else first_number(completed, "events", "num_events")
    configured_facts = first_number(config, "num_facts", "facts")
    if configured_facts is None and isinstance(config.get("sample_ids"), list):
        try:
            configured_facts = len(set(config["sample_ids"]))
        except TypeError:
            warnings.append("Configured sample_ids are not scalar IDs; fact count cannot be inferred.")
    facts = configured_facts if configured_facts is not None else first_number(completed, "num_facts", "facts")
    coverage = fact_coverage(config, completed, actions, actions_valid, configured_facts, facts, final, warnings)
    diagnostic = (config.get("stream_is_diagnostic") is True or (number(config.get("repeat_every")) or 0) > 0
                  or any(row.get("diagnostic_repeat") is True for row in actions or []))
    if not diagnostic and path.name == "duplicate_diagnostic":
        diagnostic = True
        warnings.append("Duplicate diagnostic identified from directory name; explicit stream metadata is missing.")
    if diagnostic:
        warnings.append("Artificial duplicate diagnostic, not a natural arrival stream; exclude from natural-stream comparisons.")
    status = "complete" if completed.get("status") == "complete" else "incomplete"
    if not final:
        warnings.append("No post-training metrics record; final quality is unknown.")
        status = "incomplete"
    if not config or not actions_valid:
        status = "incomplete"
    if not coverage["valid"]:
        status = "incomplete"
    if expected_events is not None and recorded_events is not None and expected_events != recorded_events:
        warnings.append(f"Recorded events {recorded_events} differ from configured events {expected_events}.")
        status = "incomplete"
    marker_events = first_number(completed, "events", "num_events")
    if marker_events is not None and recorded_events is not None and marker_events != recorded_events:
        warnings.append("completed.json event count differs from actions.jsonl.")
        status = "incomplete"
    if actions is not None and reported_counts is not None and any(counts[a] != reported_counts.get(a, 0) for a in ACTIONS):
        warnings.append("completed.json action counts differ from actions.jsonl.")
        status = "incomplete"
    final_stage = number(final.get("current_stage"))
    expected_stage = number(completed.get("final_stage"))
    if expected_stage is not None and final_stage != expected_stage:
        warnings.append("Final metrics stage does not match completed.json final_stage.")
        status = "incomplete"
    event_train = None
    if actions is not None:
        event_train = [row.get("train") if isinstance(row.get("train"), dict) else
                       {"optimizer_steps": 0, "sequence_tokens": 0, "answer_tokens": 0}
                       if row.get("action") == "ignore" and "train" in row else {} for row in actions]
    costs = {}
    for name, keys in {
        "optimizer_steps": ("optimizer_steps",),
        "logical_sequence_tokens": ("logical_sequence_tokens", "effective_sequence_tokens", "trained_sequence_tokens", "sequence_tokens"),
        "physical_sequence_tokens": ("physical_sequence_tokens", "physical_train_tokens"),
        "answer_tokens": ("answer_tokens",),
    }.items():
        costs[name] = total(completed, stages, event_train, *keys)
    for name, keys in {
        "reward_forward_tokens": ("reward_forward_tokens", "probe_forward_tokens"),
        "physical_reward_forward_tokens": ("physical_reward_forward_tokens", "physical_probe_tokens"),
        "mastery_generation_tokens": ("mastery_generation_tokens",),
        "replay_count": ("replay_count",),
        "router_reward_seconds": ("router_reward_seconds",),
        "encoder_seconds": ("encoder_seconds",),
    }.items():
        costs[name] = total(completed, stages, actions, *keys)
    costs["train_seconds"] = total(completed, stages, event_train, "train_seconds", "wall_seconds")
    costs["elapsed_seconds"] = {"value": number(completed.get("elapsed_seconds")), "source": "completed.json"}
    for name, measurement in costs.items():
        values = [v for v in measurement.get("observations", {}).values() if v is not None]
        if values and any(not math.isclose(values[0], value, rel_tol=1e-8, abs_tol=1e-6) for value in values[1:]):
            warnings.append(f"Conflicting {name} totals across artifacts; inspect observations in summary.json.")
    if costs["physical_sequence_tokens"]["value"] is None:
        warnings.append("Physical training-token cost was not recorded; logical tokens do not establish equal physical compute.")
    if costs["reward_forward_tokens"]["value"] is None:
        warnings.append("Reward/mastery forward-token cost was not recorded.")
    if costs["mastery_generation_tokens"]["value"] is None:
        warnings.append("Mastery-generation tokens were not recorded; reward_forward_tokens alone is not total probe compute.")
    axes = {}
    for axis in AXES:
        raw = final.get(axis, {})
        raw = raw if isinstance(raw, dict) else {}
        axes[axis] = {"match_accuracy": first_number(raw, "match_accuracy"),
                      "exact_accuracy": first_number(raw, "exact_accuracy"), "count": number(raw.get("count"))}
        if axes[axis]["match_accuracy"] is None:
            axes[axis]["match_accuracy"] = number(final.get(axis + "_accuracy"))
    ids = config.get("sample_ids")
    identity = {"training_kind": training["kind"], "training_method": training["declared_method"],
                "backend": training["backend"], "model_dir": config.get("model_dir"),
                "schema_version": config.get("schema_version"), "action_definition": config.get("action_definition"),
                "manifest_sha256": config.get("manifest_sha256"), "protocol_id": config.get("protocol_id"),
                "source_sha256": config.get("source_sha256"),
                "encoder_provenance_sha256": (config["encoder"].get("provenance_sha256")
                                                if isinstance(config.get("encoder"), dict) else None),
                "sample_ids_sha256": hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest() if isinstance(ids, list) else None,
                "num_facts": facts, "expected_events": expected_events, "diagnostic": diagnostic,
                "fact_mode": config.get("fact_mode"), "seed": config.get("seed"),
                "training_hyperparameters": {key: config.get(key) for key in (
                    "epochs", "learning_rate", "batch_size", "max_length", "lora_r", "lora_alpha", "target_modules",
                    "steps_per_event", "weight_decay", "max_grad_norm", "activation_checkpointing", "flash_attention",
                    "probe_size", "replay_k", "add_replay_k", "ignore_replay_k", "mastery_nll", "similarity_threshold",
                    "training_loss", "current_loss_weight", "replay_selection", "merge_similar_k",
                    "recent_replay_k", "recent_window", "training_views", "training_views_sha256",
                    "parameter_mode", "lora_rank", "lora_dropout", "training_schedule_sha256", "policy_scope",
                    "max_new_tokens", "eval_batch_size", "locality_panel_size", "multihop_panel_size")}}
    if training["kind"] in ("unknown", "inconsistent") or identity["sample_ids_sha256"] is None:
        identity["unverified_run"] = str(path.resolve())
    return {"run_name": path.name, "run_dir": str(path.resolve()), "status": status,
            "completion_marker_status": completed.get("status"), "mode": config.get("mode"),
            "training": training, "comparison_identity": identity,
            "stream_kind": "duplicate_diagnostic" if diagnostic else "natural" if config.get("stream_is_diagnostic") is False else "unspecified",
            "num_facts": facts, "expected_events": expected_events, "recorded_events": recorded_events,
            "configured_facts": coverage["configured_facts"], "observed_facts": coverage["observed_facts"],
            "active_facts": coverage["active_facts"], "retired_facts": coverage["retired_facts"],
            "fact_coverage": coverage,
            "action_counts": counts, "decision": decision_summary(actions, config.get("mode"), update_log),
            "checkpoint": final.get("checkpoint"), "final_stage": final_stage,
            "saved_model_checkpoint": completed.get("final_checkpoint"),
            "evaluated_facts": coverage["evaluated_facts"], "axes": axes,
            "old_retention": {axis: number(final.get(f"past_{axis}_retention")) for axis in AXES[:3]},
            "current_accuracy": {axis: number(final.get(f"current_{axis}_accuracy")) for axis in AXES[:3]},
            "costs": costs, "replay_schedule": config.get("replay_schedule"),
            "closed_book_evaluation": config.get("closed_book_evaluation"),
            "split_role": config["split_audit"].get("role") if isinstance(config.get("split_audit"), dict) else None,
            "warnings": warnings}


def build_summary(root: Path | str) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Run root is not a directory: {root}")
    paths = [root] if any((root / name).is_file() for name in ARTIFACTS) else [
        child for child in sorted(root.iterdir()) if child.is_dir() and any((child / name).is_file() for name in ARTIFACTS)]
    runs = [summarize_run(path) for path in paths]
    groups: dict[str, list[str]] = {}
    for run in runs:
        key = json.dumps(run["comparison_identity"], sort_keys=True)
        group = hashlib.sha256(key.encode()).hexdigest()[:12]
        run["comparison_group"] = group
        groups.setdefault(group, []).append(run["run_name"])
    by_path = {str(Path(run["run_dir"]) / "actions.jsonl"): run for run in runs}
    comparisons = []
    for run in runs:
        if not run["replay_schedule"]:
            continue
        reference = by_path.get(str(Path(run["replay_schedule"]).resolve()))
        if reference is None:
            run["warnings"].append("Replay-schedule source is not present in this report; budget matching cannot be verified.")
            continue
        fields = ("optimizer_steps", "replay_count", "logical_sequence_tokens", "physical_sequence_tokens", "reward_forward_tokens", "physical_reward_forward_tokens", "mastery_generation_tokens")
        same = {}
        for field in fields:
            left, right = run["costs"][field]["value"], reference["costs"][field]["value"]
            same[field] = left == right if left is not None and right is not None else None
        compatible = run["comparison_group"] == reference["comparison_group"]
        comparisons.append({"run": run["run_name"], "reference": reference["run_name"],
                            "compatible_training_and_stream": compatible, "equal_recorded_costs": same,
                            "fully_matched": compatible and all(value is True for value in same.values())})
        if not compatible:
            run["warnings"].append("Replay reference has different or unverified training/stream metadata; do not compare outcomes directly.")
        if not all(value is True for value in same.values()):
            run["warnings"].append("Replay counts/optimizer steps do not establish fully matched budgets: tokens/probe costs differ or are unrecorded.")
    return {"schema": SCHEMA, "root": str(root), "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_count": len(runs), "complete_run_count": sum(run["status"] == "complete" for run in runs),
            "comparison_groups": groups, "budget_comparisons": comparisons, "runs": runs,
            "warnings": (["Multiple source snapshots are present; outcomes are not a same-code comparison."]
                         if len({json.dumps(run["comparison_identity"].get("source_sha256"), sort_keys=True)
                                 for run in runs}) > 1 else [])
                        if runs else ["No run artifacts found; no completed experiment is established."]}


def display(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, int)):
        return f"{100 * value:.2f}%" if percent else f"{value:,}" if isinstance(value, int) else f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(summary: dict) -> str:
    lines = ["# Continual writer: recorded results", "",
             f"Completed artifacts: {summary['complete_run_count']}/{summary['run_count']}. Generated: {summary['generated_at']}.", "",
             "This report summarizes existing artifacts only. Missing measurements are `unknown`; no new experiment is implied.",
             "Runs are separated by training method, model, protocol and selected stream. Groups are not pooled or ranked.",
             "Quality columns use final-checkpoint match accuracy; exact accuracy and evaluation counts are in summary.json.",
             "Active fact counts require complete retirement-aware action evidence; legacy logs retain configured-count coverage.",
             "Old retention is accuracy on earlier fact stages, not a normalized retention ratio. A one-stage run may have no old-fact measurement.", ""]
    for group in summary["comparison_groups"]:
        runs = [run for run in summary["runs"] if run["comparison_group"] == group]
        lines += [f"## Group {group}: {runs[0]['training']['kind']} / {runs[0]['stream_kind']}", "",
                  "| Run | Status | Facts: configured / observed / active / evaluated | Events | ADD / MERGE / IGNORE | Policy updates / credited transitions | Mask-forced / observed |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for run in runs:
            actions = " / ".join(display((run["action_counts"] or {}).get(a)) for a in ACTIONS)
            facts = " / ".join(display(run[key]) for key in ("configured_facts", "observed_facts", "active_facts", "evaluated_facts"))
            d = run["decision"]
            lines.append(f"| {display(run['run_name'])} | {run['status']} | {facts} | {display(run['recorded_events'])} | {actions} | {display(d['policy_updates'])} / {display(d['credited_transitions'])} | {display(d['mask_forced_count'])} / {display(d['forced_observed_events'])} |")
        lines += ["", "| Run / checkpoint | Update | Rephrase | Persona | Locality | Multihop | Old update | Old rephrase | Old persona |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for run in runs:
            values = [run["axes"][a]["match_accuracy"] for a in AXES] + list(run["old_retention"].values())
            lines.append(f"| {display(run['run_name'])} / {display(run['checkpoint'])} | " + " | ".join(display(v, percent=True) for v in values) + " |")
        lines += ["", "| Run | Optimizer steps | Logical train tokens | Physical train tokens | Reward forward tokens | Physical reward tokens | Mastery generation tokens | Replay rows |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for run in runs:
            keys = ("optimizer_steps", "logical_sequence_tokens", "physical_sequence_tokens", "reward_forward_tokens", "physical_reward_forward_tokens", "mastery_generation_tokens", "replay_count")
            lines.append(f"| {display(run['run_name'])} | " + " | ".join(display(run["costs"][key]["value"]) for key in keys) + " |")
        lines.append("")
        for run in runs:
            d = run["decision"]
            lines.append(f"- {display(run['run_name'])}: training={display(run['training']['declared_method'] or run['training']['kind'])}; backend={display(run['training']['backend'])}; split={display(run['split_role'])}; policy source={display(d['policy_updates_source'])}; deterministic distributions={display(d['deterministic_distribution_count'])}.")
            lines.extend(f"- {display(run['run_name'])}: {warning}" for warning in run["warnings"])
        lines.append("")
    lines += ["Mask-forced counts use explicit flags or the legacy add-only action mask; this does not imply that any separate budget choice was forced. A deterministic rule/random distribution is not evidence of policy learning.",
              "Policy update counts prefer policy_updates.jsonl, including terminal flushes. Credited transitions are reported only when transition_count was recorded.",
              "Logical training tokens exclude padding in the legacy runner. Physical work, reward forward passes and mastery generation are separate costs; none are inferred from GPU count.", ""]
    for comparison in summary["budget_comparisons"]:
        equal = ", ".join(f"{key}={display(value)}" for key, value in comparison["equal_recorded_costs"].items())
        lines += [f"Budget check {comparison['run']} vs {comparison['reference']}: fully matched={comparison['fully_matched']}; equal measurements: {equal}.", ""]
    lines.extend(summary["warnings"])
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="One run directory or a suite containing run directories")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing generated reports")
    args = parser.parse_args(argv)
    targets = [args.root / "RESULTS.md", args.root / "summary.json"]
    if not args.overwrite and any(path.exists() for path in targets):
        parser.error("Report already exists; inspect it before using --overwrite.")
    summary = build_summary(args.root)
    targets[0].write_text(render_markdown(summary), encoding="utf-8")
    targets[1].write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Summarized {summary['run_count']} runs ({summary['complete_run_count']} complete): {targets[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
