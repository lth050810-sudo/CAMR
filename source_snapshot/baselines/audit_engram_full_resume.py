#!/usr/bin/env python3
"""Compare explicitly selected completed full-writer runs entirely on CPU.

Only completed.json, actions.jsonl and policy.pt inside the supplied run
directories are read. The audit does not load Qwen checkpoints or execute CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import torch

from baselines.engram_online_policy import OnlineWriterPolicy


EVENT_FIELDS = ("event", "sample_id", "arrival_stage", "diagnostic_repeat", "action",
                "replay_count", "replay_ids", "probe_ids", "gate_candidate_ids")
NUMERIC_FIELDS = ("before_nll", "after_nll", "reward", "probabilities")
TRAIN_COSTS = ("optimizer_steps", "sequence_tokens", "answer_tokens", "physical_sequence_tokens")
PROBE_COSTS = ("reward_forward_tokens", "physical_reward_forward_tokens", "mastery_generation_tokens", "replay_count")


def artifact(run: Path, name: str) -> Path:
    path = run / name
    if path.resolve().parent != run.resolve():
        raise ValueError(f"Refusing artifact outside specified run: {path}")
    return path


def load_run(run: Path) -> dict:
    errors = []
    completed, actions, policy = None, None, None
    try:
        completed = json.loads(artifact(run, "completed.json").read_text(encoding="utf-8"))
        if not isinstance(completed, dict):
            raise ValueError("completed.json must contain an object")
    except (OSError, ValueError, UnicodeError) as error:
        errors.append(str(error))
        completed = None
    try:
        actions = []
        with artifact(run, "actions.jsonl").open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"actions.jsonl line {line_number} must contain an object")
                actions.append(row)
    except (OSError, ValueError, UnicodeError) as error:
        errors.append(str(error))
        actions = None
    try:
        # Never retry with unrestricted pickle. These are user-selected local
        # tensor/basic-type state dictionaries, not remote executable objects.
        policy = torch.load(artifact(run, "policy.pt"), map_location="cpu", weights_only=True)
        if not isinstance(policy, dict) or not isinstance(policy.get("model"), dict):
            raise ValueError("policy.pt must contain the full policy training state, including model")
        # Validate the complete versioned schema, not merely that both files
        # happen to omit the same parameter, optimizer or sampler field.
        verifier = OnlineWriterPolicy(**policy["dimensions"])
        verifier.load_training_state_dict(policy)
    except Exception as error:
        errors.append(f"policy.pt safe CPU load: {type(error).__name__}: {error}")
        policy = None
    return {"path": str(run.resolve()), "completed": completed, "actions": actions, "policy": policy, "errors": errors}


def exact_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        a = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        b = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return bool(torch.equal(a, b))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return struct.pack("!d", left) == struct.pack("!d", right)
    return bool(left == right)


def numeric_comparison(left: Any, right: Any, *, atol: float, rtol: float) -> dict:
    """Tolerance is |resumed-reference| <= atol + rtol*|reference|."""
    count, max_abs, failures = 0, 0.0, []

    def fail(path: str, reason: str) -> None:
        if len(failures) < 20:
            failures.append({"path": path, "reason": reason})

    def visit(a: Any, b: Any, path: str) -> None:
        nonlocal count, max_abs
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                fail(path, "tensor type, shape or dtype differs")
                return
            if not a.is_floating_point() and not a.is_complex():
                count += a.numel()
                if not exact_equal(a, b):
                    fail(path, "integer/bool tensor differs")
                return
            dtype = torch.complex128 if a.is_complex() else torch.float64
            a, b = a.detach().cpu().to(dtype), b.detach().cpu().to(dtype)
            count += a.numel()
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                fail(path, "nonfinite tensor value")
                return
            difference = (b - a).abs()
            if difference.numel():
                max_abs = max(max_abs, float(difference.max()))
            if not bool(torch.all(difference <= atol + rtol * a.abs())):
                fail(path, "tensor exceeds tolerance")
        elif isinstance(a, dict) or isinstance(b, dict):
            if not isinstance(a, dict) or not isinstance(b, dict) or a.keys() != b.keys():
                fail(path, "dictionary keys differ")
                return
            for key in a:
                visit(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                fail(path, "sequence type or length differs")
                return
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{index}]")
        elif isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
            count += 1
            if not math.isfinite(a) or not math.isfinite(b):
                fail(path, "nonfinite number")
                return
            max_abs = max(max_abs, abs(b - a))
            if abs(b - a) > atol + rtol * abs(a):
                fail(path, "number exceeds tolerance")
        elif not exact_equal(a, b):
            fail(path, "nonnumeric value differs")

    visit(left, right, "root")
    return {"within_tolerance": not failures, "numeric_values_compared": count,
            "max_abs_difference": max_abs if count else None,
            "exact": exact_equal(left, right), "mismatches": failures}


def integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def valid_measurement(field: str, value: Any) -> bool:
    def finite_number(item: Any) -> bool:
        return isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
    if field in ("before_nll", "after_nll"):
        return isinstance(value, list) and bool(value) and all(finite_number(item) for item in value)
    if field == "probabilities":
        return (isinstance(value, dict) and set(value) == {"add", "merge", "ignore"}
                and all(finite_number(item) and 0 <= item <= 1 for item in value.values())
                and math.isclose(sum(value.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6))
    return finite_number(value)


def cumulative_costs(run: dict) -> dict:
    actions, done = run["actions"], run["completed"]
    result = {}
    if actions is None or done is None:
        return {"passed": False, "fields": {}, "reason": "missing readable artifacts"}
    totals = done.get("totals", {})
    totals = totals if isinstance(totals, dict) else {}
    for key in TRAIN_COSTS + PROBE_COSTS:
        values = []
        for event in actions:
            if key in TRAIN_COSTS:
                train = event.get("train")
                value = train.get(key) if isinstance(train, dict) else 0 if event.get("action") == "ignore" and "train" in event else None
            else:
                value = event.get(key)
            values.append(value)
        observed = sum(values) if all(integer(value) for value in values) else None
        top_key = "trained_sequence_tokens" if key == "sequence_tokens" else key
        top, nested = done.get(top_key), totals.get(key)
        declared = top if integer(top) else nested if integer(nested) else None
        consistent = observed is not None and declared is not None and observed == declared
        if top is not None:
            consistent &= integer(top) and top == observed
        if key in totals:
            consistent &= integer(nested) and nested == observed
        result[key] = {"event_total": observed, "completed_total": declared,
                       "completed_top_level": top, "completed_nested": nested, "consistent": bool(consistent)}
    return {"passed": all(item["consistent"] for item in result.values()), "fields": result}


def audit_resume(reference: Path | str, resumed: Path | str, *, atol: float = 1e-5, rtol: float = 1e-5) -> dict:
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("Tolerances must be finite and nonnegative")
    left, right = load_run(Path(reference)), load_run(Path(resumed))
    checks: dict[str, Any] = {}
    checks["artifacts_readable"] = {"passed": not left["errors"] and not right["errors"],
                                     "reference_errors": left["errors"], "resumed_errors": right["errors"]}
    completed = [run["completed"] or {} for run in (left, right)]
    checks["both_complete"] = {"passed": all(row.get("status") == "complete" for row in completed),
                                "statuses": [row.get("status") for row in completed]}
    actions = [run["actions"] for run in (left, right)]
    coverage = []
    for rows, done in zip(actions, completed):
        coverage.append(rows is not None and integer(done.get("events")) and done["events"] == len(rows)
                        and integer(done.get("expected_events")) and done["expected_events"] == len(rows)
                        and [row.get("event") for row in rows] == list(range(len(rows))))
    checks["complete_event_coverage"] = {"passed": all(coverage), "reference": coverage[0], "resumed": coverage[1]}
    same_length = actions[0] is not None and actions[1] is not None and len(actions[0]) == len(actions[1])
    discrete = {}
    for field in EVENT_FIELDS:
        present = same_length and all(field in event for rows in actions for event in rows)
        discrete[field] = bool(present and all(exact_equal(a[field], b[field]) for a, b in zip(*actions)))
    checks["event_sequence_and_decisions"] = {"passed": all(discrete.values()), "fields": discrete}
    left_costs, right_costs = cumulative_costs(left), cumulative_costs(right)
    cost_equal = {key: left_costs["fields"].get(key, {}).get("event_total") == right_costs["fields"].get(key, {}).get("event_total")
                  and left_costs["fields"].get(key, {}).get("event_total") is not None for key in TRAIN_COSTS + PROBE_COSTS}
    checks["cumulative_costs"] = {"passed": left_costs["passed"] and right_costs["passed"] and all(cost_equal.values()),
                                  "reference": left_costs, "resumed": right_costs, "equal_fields": cost_equal}
    numeric = {}
    for field in NUMERIC_FIELDS:
        present = same_length and all(field in event and valid_measurement(field, event[field]) for rows in actions for event in rows)
        numeric[field] = numeric_comparison([row[field] for row in actions[0]], [row[field] for row in actions[1]], atol=atol, rtol=rtol) if present else {
            "within_tolerance": False, "max_abs_difference": None, "exact": False, "mismatches": [{"reason": "missing field or event count differs"}]}
        if not numeric[field].get("numeric_values_compared"):
            numeric[field]["within_tolerance"] = False
            numeric[field]["mismatches"].append({"reason": "no numeric measurements available"})
    checks["numeric_traces"] = {"passed": all(item["within_tolerance"] for item in numeric.values()), "fields": numeric}
    policies = [run["policy"] for run in (left, right)]
    parameter_result = {"within_tolerance": False, "max_abs_difference": None, "exact": False, "reason": "missing policy state"}
    optimizer_result = dict(parameter_result)
    if all(policy is not None for policy in policies):
        models = [{key: value for key, value in policy["model"].items() if isinstance(value, torch.Tensor)} for policy in policies]
        if all(models):
            parameter_result = numeric_comparison(*models, atol=atol, rtol=rtol)
        if all(isinstance(policy.get("optimizer"), dict) for policy in policies):
            optimizer_result = numeric_comparison(policies[0]["optimizer"], policies[1]["optimizer"], atol=atol, rtol=rtol)
    checks["policy_parameters"] = {"passed": parameter_result["within_tolerance"], **parameter_result}
    checks["policy_optimizer"] = {"passed": optimizer_result["within_tolerance"], **optimizer_result}
    optimizer_metadata = []
    for policy in policies:
        optimizer = policy.get("optimizer") if policy is not None else None
        if isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict) and isinstance(optimizer.get("param_groups"), list):
            optimizer_metadata.append({"steps": {key: state.get("step") for key, state in optimizer["state"].items()},
                                       "param_groups": optimizer["param_groups"]})
        else:
            optimizer_metadata.append(None)
    checks["policy_optimizer_counters_and_groups"] = {
        "passed": all(value is not None for value in optimizer_metadata) and exact_equal(*optimizer_metadata)}
    extra = [policy["model"].get("_extra_state") if policy is not None else None for policy in policies]
    metadata_equal = all(policy is not None and "dimensions" in policy and "format_version" in policy for policy in policies)
    metadata_equal = bool(metadata_equal and exact_equal(policies[0]["dimensions"], policies[1]["dimensions"]) and exact_equal(policies[0]["format_version"], policies[1]["format_version"]))
    checks["policy_rng_and_counters"] = {"passed": extra[0] is not None and extra[1] is not None and exact_equal(*extra),
                                         "policy_format_and_dimensions_equal": metadata_equal}
    pending = [policy.get("pending") if policy is not None else None for policy in policies]
    counts = [len(value) if isinstance(value, list) else None for value in pending]
    pending_markers = [done.get("policy_pending_transitions") for done in completed]
    checks["terminal_pending_empty"] = {"passed": counts == [0, 0] and all(integer(value) and value == 0 for value in pending_markers),
                                          "state_pending_counts": counts, "completed_pending_counts": pending_markers}
    checks["policy_format_and_dimensions"] = {"passed": metadata_equal}
    fingerprints = []
    for rows in actions:
        last_train = next((row["train"] for row in reversed(rows or []) if isinstance(row.get("train"), dict)), {})
        fingerprints.append(last_train.get("parameter_fingerprint_after"))
    fingerprint_exact = exact_equal(*fingerprints) if all(value is not None for value in fingerprints) else None
    within = all(check["passed"] for check in checks.values())
    bitwise = {"last_training_parameter_fingerprint_exact": fingerprint_exact,
               "last_training_parameter_fingerprints": fingerprints,
               "policy_model_tensors_exact": parameter_result["exact"],
               "policy_training_state_exact": exact_equal(*policies) if all(policy is not None for policy in policies) else None,
               "numeric_traces_exact": all(item["exact"] for item in numeric.values()),
               "full_qwen_weights_compared": False, "full_qwen_optimizer_compared": False}
    differences = [item.get("max_abs_difference") for item in numeric.values()] + [parameter_result.get("max_abs_difference"), optimizer_result.get("max_abs_difference")]
    differences = [value for value in differences if value is not None]
    observed_exact = within and fingerprint_exact is True and bitwise["policy_training_state_exact"] is True and bitwise["numeric_traces_exact"]
    return {"schema": "engram-full-resume-audit/v1", "reference": left["path"], "resumed": right["path"],
            "atol": atol, "rtol": rtol, "checks": checks, "within_tolerance": within,
            "max_abs_difference": max(differences) if differences else None, "bitwise_evidence": bitwise,
            "conclusion": "observed_artifacts_exact" if observed_exact else "within_tolerance_not_bitwise" if within else "differences_or_incomplete_evidence",
            "limitations": [
                "Identical actions alone do not establish exact checkpoint restoration.",
                "The last training fingerprint samples Qwen parameter values; it is not a full-weight comparison or checkpoint hash.",
                "Without enforced deterministic kernels, including FlashAttention, fingerprint/numeric differences alone do not identify a broken resume mechanism.",
                "Passing the tolerance checks establishes agreement of the observed artifacts, not bitwise identity of the complete Qwen model, optimizer or distributed RNG state.",
                "Wall-clock timing is intentionally excluded; only recorded cumulative training/probe counts are compared.",
            ]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing paths are refused")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; use a new report path.")
    result = audit_resume(args.reference, args.resumed, atol=args.atol, rtol=args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "within_tolerance": result["within_tolerance"],
                      "conclusion": result["conclusion"], "max_abs_difference": result["max_abs_difference"]}))
    return 0 if result["within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
