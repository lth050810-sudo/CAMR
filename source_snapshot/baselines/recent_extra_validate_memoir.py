"""Validate all 3K MEMOIR edits and eight evaluation stages without a GPU."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(output):
    read=lambda name:json.loads((output/name).read_text())
    done=read("completed.json")
    assert done["status"]=="complete" and done["processed"]==3000
    assert done["smoke_only"] is False and done["evaluation_stages"]==8
    cfg=read("run_config.json")
    assert cfg["num_facts"]==3000
    manifest_path=Path(cfg["manifest"])
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==cfg["manifest_sha256"]=="93b84820bfcb4c7367e6ecc41c3fda40fdf45bd03d2770a6efdd7d8e4a0b63a1"
    manifest=json.loads(manifest_path.read_text())
    selected=set(manifest["selection"]["sample_ids"])
    rows=sorted([r for r in manifest["records"] if r["sample_id"] in selected],key=lambda r:(int(r["stage"]),str(r["sample_id"])))
    logs=read("training_log.json");metrics=read("metrics.json")
    assert len(logs)==3000 and len(metrics)==8
    assert [log["sample_id"] for log in logs]==[r["sample_id"] for r in rows]
    assert all(log["index"]==i and 1<=log["optimizer_steps"]<=cfg["iterations"] for i,log in enumerate(logs))
    assert sum(log["optimizer_steps"] for log in logs)==done["optimizer_steps"]
    for stage,metric in enumerate(metrics,1):
        assert metric["checkpoint"]==f"tau{stage}" and metric["num_facts"]==stage*375
        predictions=[json.loads(line) for line in (output/f"predictions_tau{stage}.jsonl").read_text().splitlines()]
        assert {p["axis"] for p in predictions}=={"update","rephrase","persona","locality","multihop"}
        for axis in ("update","rephrase","persona"):
            values=[p for p in predictions if p["axis"]==axis]
            assert len(values)==375*stage and all(int(p["fact_stage"])<=stage for p in values)
        for axis in ("locality","multihop"):
            panel_manifest=json.loads(Path(cfg["manifest"]).read_text())
            panel_key={"locality":"locality","multihop":"mhop"}[axis]
            frozen_ids=panel_manifest["panels"][panel_key]["sample_ids"]
            limit=cfg[f"{axis}_panel_size"]
            expected_ids=frozen_ids[:limit] if limit>0 else frozen_ids
            actual_ids=[p["sample_id"] for p in predictions if p["axis"]==axis]
            assert len(actual_ids)==len(expected_ids) and set(actual_ids)==set(expected_ids)
    background=read("background_provenance.json")
    assert len(background["documents"])==100 and background["feature_shape"][0]==100
    assert background["corpus_sha256"]==cfg["background_corpus_sha256"]
    assert len(read("context_templates.json"))==11
    assert (output/"checkpoint.pt").stat().st_size>1_000_000
    return {"status":"valid","num_facts":3000,"stages":8,"optimizer_steps":done["optimizer_steps"],"method":cfg["method"]}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True)
    print(json.dumps(validate(p.parse_args().output_dir)))
