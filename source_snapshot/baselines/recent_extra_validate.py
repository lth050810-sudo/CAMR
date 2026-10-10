"""Reject partial/invalid TSR results before scheduler marks a job complete."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(output: Path, steps=750):
    read = lambda name: json.loads((output / name).read_text())
    done = read("completed.json")
    assert done["status"] == "complete" and done["final_stage"] == 8 and done["smoke_only"] is False
    config = read("run_config.json")["configuration"]
    assert config["num_facts"] == 3000 and config["steps_per_stage"] == steps
    assert config["stage_counts"] == {str(i): 375 for i in range(1, 9)}
    assert config["manifest_sha256"] == "93b84820bfcb4c7367e6ecc41c3fda40fdf45bd03d2770a6efdd7d8e4a0b63a1"
    assert hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest() == config["manifest_sha256"]
    logs, metrics = read("training_log.json"), read("metrics.json")
    assert [r["stage"] for r in logs] == list(range(1, 9))
    assert len(metrics) == 8 and sum(r["steps"] for r in logs) == 8 * steps
    for stage, (log, metric) in enumerate(zip(logs, metrics), 1):
        assert log["steps"] == steps and log["current_count"] == 375
        assert metric["checkpoint"] == f"tau{stage}" and metric["num_facts"] == 375 * stage
        assert set(("update", "rephrase", "persona", "locality", "multihop")).issubset(metric)
        assert metric["train"] == log
        route = read(f"routing_tau{stage}.json")
        assert route["future_leak_count"] == 0
        assert all(int(k) < stage for k in route["replay_buffer_sample_ids"])
        predictions = [json.loads(line) for line in (output / f"predictions_tau{stage}.jsonl").read_text().splitlines()]
        assert predictions and {p["axis"] for p in predictions} == {"update", "rephrase", "persona", "locality", "multihop"}
        for axis in ("update", "rephrase", "persona"):
            values = [p for p in predictions if p["axis"] == axis]
            assert len(values) == 375 * stage and all(int(p["fact_stage"]) <= stage for p in values)
        for axis in ("locality", "multihop"):
            panel_manifest=json.loads(Path(config["manifest"]).read_text())
            panel_key={"locality":"locality","multihop":"mhop"}[axis]
            frozen_ids=panel_manifest["panels"][panel_key]["sample_ids"]
            limit=config[f"{axis}_panel_size"]
            expected_ids=frozen_ids[:limit] if limit>0 else frozen_ids
            actual_ids=[p["sample_id"] for p in predictions if p["axis"]==axis]
            assert len(actual_ids)==len(expected_ids) and set(actual_ids)==set(expected_ids)
        assert log["kd_teacher_forwards"] == (steps if stage > 1 else 0)
    assert (output / "checkpoint.pt").stat().st_size > 1_000_000
    return {"status": "valid", "stages": 8, "optimizer_steps": 8 * steps,
            "num_facts": 3000, "seed": config["seed"], "method": config["method"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps-per-stage", type=int, default=750)
    args = p.parse_args()
    print(json.dumps(validate(args.output_dir, args.steps_per_stage)))
