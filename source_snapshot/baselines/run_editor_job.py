"""Restartable queue boundary for staged editor transfers.

Committed stages resume in place. Legacy partial attempts without a stage
checkpoint are preserved before a fresh run. Completed runs are audited and
recovered without retraining. The queue marker follows source-bound checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from baselines.common import atomic_json, load_manifest, manifest_panel, sha256_file
from baselines.wikibigedit_protocol import answer_scores
from baselines.editor_checkpoint import FILENAME as RESUME_FILENAME


AXES = {"update": ("update", "ans"), "rephrase": ("rephrase", "ans"),
        "persona": ("personas", "ans"), "locality": ("loc", "loc_ans"),
        "multihop": ("mhop", "mhop_ans")}


def audit_run(directory: Path):
    config = json.loads((directory / "run_config.json").read_text())
    done = json.loads((directory / "completed.json").read_text())
    manifest_path = Path(config["manifest"])
    rows, manifest = load_manifest(manifest_path)
    by_id = {row["sample_id"]: row for row in rows}
    if (done.get("status") != "complete" or config["num_facts"] != len(rows)
            or done["num_facts"] != len(rows) or done["final_stage"] != 8
            or config["manifest_sha256"] != sha256_file(manifest_path)):
        raise ValueError("Editor completion/manifest identity mismatch")
    metrics = json.loads((directory / "metrics.json").read_text())
    metrics = [row for row in metrics if row.get("current_stage") is not None]
    if [row["current_stage"] for row in metrics] != list(range(1, 9)):
        raise ValueError("Missing staged editor evaluation")
    stages = {row["checkpoint"]: row["current_stage"] for row in metrics}
    panels = {"locality": {row["sample_id"] for row in manifest_panel(manifest, "locality") if row["sample_id"] in by_id},
              "multihop": {row["sample_id"] for row in manifest_panel(manifest, "mhop") if row["sample_id"] in by_id}}
    expected = {(metric["checkpoint"], axis): (panels[axis] if axis in panels else
                {row["sample_id"] for row in rows if row["stage"] <= metric["current_stage"]})
                for metric in metrics for axis in AXES}
    observed = {key: set() for key in expected}
    matches, exacts = Counter(), Counter()
    count = 0
    with (directory / "predictions.jsonl").open() as stream:
        for line in stream:
            pred = json.loads(line)
            if pred["checkpoint"] == "base":
                continue
            key = (pred["checkpoint"], pred["axis"])
            sid = pred["sample_id"]
            if key not in expected or sid not in expected[key] or sid in observed[key]:
                raise ValueError("Missing, duplicate, or future evaluation sample")
            source = by_id[sid]
            question, answer = AXES[pred["axis"]]
            if pred["question"] != source[question] or pred["gold"] != source[answer] or pred["fact_stage"] != source["stage"]:
                raise ValueError("Predictions are not bound to source questions/gold")
            exact, match = answer_scores(pred["prediction"], pred["gold"])
            if pred["exact"] != exact or pred["match"] != match:
                raise ValueError("Stored score differs from recomputation")
            observed[key].add(sid)
            matches[key] += match
            exacts[key] += exact
            count += 1
    if observed != expected:
        raise ValueError("Incomplete generated prediction coverage")
    for metric in metrics:
        if metric["num_facts"] != sum(row["stage"] <= metric["current_stage"] for row in rows):
            raise ValueError("Edited prefix count mismatch")
        for axis in AXES:
            key = (metric["checkpoint"], axis)
            total = len(expected[key])
            if not total or metric[axis]["count"] != total:
                raise ValueError("Axis denominator mismatch")
            for label, counts in [("match", matches), ("exact", exacts)]:
                if (metric[axis][label + "_count"] != counts[key] or
                        not math.isclose(metric[axis][label + "_accuracy"], counts[key] / total, abs_tol=1e-12)):
                    raise ValueError("Metrics disagree with generated predictions")
    logs = json.loads((directory / "training_log.json").read_text())
    if len(logs) != 8:
        raise ValueError("Missing editor training stages")
    seen = 0
    for stage, log in enumerate(logs, 1):
        current = [row for row in rows if row["stage"] == stage]
        seen += len(current)
        if log["num_current_facts"] != len(current) or log["num_seen_facts"] != seen:
            raise ValueError("Training did not consume the entire source interval")
        if "sample_ids" in log and log["sample_ids"] != [row["sample_id"] for row in current]:
            raise ValueError("Wrong training facts/order")
        if "first_sample_id" in log and (log["first_sample_id"] != current[0]["sample_id"] or log["last_sample_id"] != current[-1]["sample_id"]):
            raise ValueError("Wrong training chunk boundaries")
        if not log["epochs"] or any(not math.isfinite(record["loss"]) or record["examples"] != len(current)
                                     or record["optimizer_steps"] < 1 for record in log["epochs"]):
            raise ValueError("Nonfinite or incomplete optimizer trace")
    eligible = config.get("formal_baseline_eligible", False)
    if eligible:
        selection = config.get("dev_selection_audit")
        if not selection or sha256_file(Path(selection["path"])) != selection["sha256"]:
            raise ValueError("Missing or changed independent development certificate")
    result = {"status": "verified_complete", "num_facts": len(rows), "final_stage": 8,
              "prediction_records": count, "scores_recomputed_from_predictions": True,
              "source_questions_and_gold_verified": True, "future_leaks": 0,
              "manifest_sha256": config["manifest_sha256"], "method": config["method"],
              "method_display_name": config.get("method_display_name"),
              "formal_baseline_eligible": eligible, "reproduction_exact": config.get("reproduction_exact", False),
              "final_counts": {axis: metrics[-1][axis]["count"] for axis in AXES},
              "file_sha256": {name: sha256_file(directory / name) for name in
                  ("run_config.json", "metrics.json", "predictions.jsonl", "training_log.json", "completed.json")}}
    atomic_json(directory / "queue_complete.json", result)
    return result


def preserve_partial(output_dir: Path, checkpoint_dir: Path | None):
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return None
    if (output_dir / "completed.json").exists():
        raise ValueError("Refusing to archive a completed result; audit it instead")
    stamp = f"{time.time_ns()}.{os.getpid()}"
    destination = output_dir.parent / "_partial_attempts" / f"{output_dir.name}.{stamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_dir.rename(destination)
    record = {"at": time.time(), "original_output": str(output_dir), "preserved_output": str(destination),
              "reason": "previous attempt incomplete and has no committed stage checkpoint"}
    if checkpoint_dir and checkpoint_dir.exists():
        target = checkpoint_dir.parent / "_partial_attempts" / f"{checkpoint_dir.name}.{stamp}"
        target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.rename(target)
        record["preserved_checkpoints"] = str(target)
    history = output_dir.parent / "_partial_attempts" / f"{output_dir.name}.jsonl"
    with history.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def run_spec(spec, gpu):
    directory = Path(spec["output_dir"]).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = directory.parent / f".{directory.name}.editor-job.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "completed.json").exists():
            print(json.dumps({"recovered": audit_run(directory)}), flush=True)
            return 0
        checkpoint = Path(spec["checkpoint_dir"]) if spec.get("checkpoint_dir") else None
        resumable = (directory / RESUME_FILENAME).is_file()
        if not resumable:
            if record := preserve_partial(directory, checkpoint):
                print(json.dumps({"preserved_incomplete_attempt": record}), flush=True)
        command = [str(arg).replace("{gpu}", str(gpu)) for arg in spec["command"]]
        if resumable and "--resume" not in command:
            command.append("--resume")
        print(json.dumps({"starting_editor": command}), flush=True)
        code = subprocess.run(command, cwd=spec.get("cwd", "/root/engram")).returncode
        if code:
            return code
        print(json.dumps({"verified": audit_run(directory)}), flush=True)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(run_spec(json.loads(args.spec.read_text()), args.gpu))


if __name__ == "__main__":
    main()
