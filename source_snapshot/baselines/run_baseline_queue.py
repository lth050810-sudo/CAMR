"""Run durable experiment jobs, yielding GPUs to other distributed sessions.

Commands are argument lists in a JSON plan. Completion requires both a zero
exit code and validated output artifacts. Only processes started here may be
stopped. Interrupted GPU jobs restart using their own checkpoint/shard logic.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def distributed_launcher(argv):
    if not argv:
        return False
    if Path(argv[0]).name == "torchrun":
        return True
    if not Path(argv[0]).name.startswith("python"):
        return False
    index = 1
    while index < len(argv):
        value = argv[index]
        if value == "-c":
            return False
        if value == "-m":
            return index + 1 < len(argv) and argv[index + 1] == "torch.distributed.run"
        if value in {"-W", "-X"}:
            index += 2
            continue
        if value.startswith("-"):
            index += 1
            continue
        # Only the interpreter's script is relevant; a script's own arguments
        # mentioning torchrun are not an external distributed launcher.
        return Path(value).name == "torchrun"
    return False


def command_option(job, flag, default=None):
    """Read an argparse-style option, including --flag=value and last-wins."""
    found = default
    arguments = [str(value) for value in job.get("command", [])]
    for index, value in enumerate(arguments):
        if value == flag:
            if index + 1 == len(arguments):
                raise ValueError(f"job command is missing a value after {flag}")
            found = arguments[index + 1]
        elif value.startswith(flag + "="):
            found = value.split("=", 1)[1]
    return found


def job_path(job, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(job.get("cwd", Path.cwd())) / path).resolve()


def launch_command(job):
    command = [str(x).replace("{gpu}", "0") for x in job["command"]]
    marker = job.get("resume_if_exists", False)
    if marker:
        if marker is True:
            output = command_option(job, "--output-dir")
            if output is None:
                raise ValueError("resume_if_exists=true requires --output-dir")
            marker = job_path(job, output) / "run_config.json"
        elif isinstance(marker, str):
            marker = job_path(job, marker)
        else:
            raise ValueError("resume_if_exists must be a marker path or boolean")
        if Path(marker).exists() and "--resume" not in command:
            command.append("--resume")
    return command


def validate_screen_recipe(job, metadata, source):
    if not job.get("command"):
        raise ValueError("screen completion requires its original job command")
    expected = {
        "model_dir": str(job_path(job, command_option(job, "--model-dir", "Qwen3-8B-Base"))),
        "batch_size": int(command_option(job, "--batch-size", 64)),
        "max_length": int(command_option(job, "--max-length", 128)),
        "max_new_tokens": int(command_option(job, "--max-new-tokens", 16)),
        "screen_protocol_id": "qwen3-base-greedy-update-v1",
    }
    command_manifest = command_option(job, "--manifest")
    if command_manifest is None or job_path(job, command_manifest) != source.resolve():
        raise ValueError("screen job manifest and command differ")
    for key, value in expected.items():
        actual = str(Path(metadata[key]).resolve()) if key == "model_dir" and key in metadata else metadata.get(key)
        if actual != value:
            raise ValueError(f"screen recipe mismatch: {key}")
    identity = metadata.get("resume_identity")
    if identity is not None:
        for key, value in {**expected, "manifest_sha256": sha256(source)}.items():
            if identity.get(key) != value:
                raise ValueError(f"screen resume identity mismatch: {key}")
        checks = {"common_source_sha256": job_path(job, "baselines/common.py"),
                  "source_sha256": job_path(job, "baselines/screen_qwen3_resumable.py"),
                  "model_config_sha256": Path(expected["model_dir"]) / "config.json",
                  "weight_index_sha256": Path(expected["model_dir"]) / "model.safetensors.index.json"}
        for key, path in checks.items():
            if not path.is_file() or identity.get(key) != sha256(path):
                raise ValueError(f"screen resume source/model identity mismatch: {key}")


def validate_full_replay(job, output):
    """Check complete full-stream traversal and sampled five-axis evaluation."""
    from baselines.run_full_replay import AXES, load_stream, make_blocks, object_sha256, parse_args
    directory = output.parent
    manifest = job_path(job, job.get("manifest") or command_option(job, "--manifest"))
    rows, metadata = load_stream(manifest)
    if (metadata.get("screening", {}).get("status") != "complete"
            or metadata.get("screening", {}).get("accepted_statuses") != ["unknown"]
            or any(row.get("base_screen", {}).get("status") != "unknown" for row in rows)):
        raise ValueError("full replay manifest is not a completed base-unknown selection")
    if sorted({row["stage"] for row in rows}) != list(range(1, 9)):
        raise ValueError("full replay must cover all eight original stages")
    config = json.loads((directory / "run_config.json").read_text())
    done = json.loads(output.read_text())
    if config.get("manifest_sha256") != sha256(manifest) or config.get("num_facts") != len(rows):
        raise ValueError("full replay manifest hash/count mismatch")
    if config.get("config_fingerprint") != object_sha256({k: v for k, v in config.items() if k != "config_fingerprint"}):
        raise ValueError("full replay config fingerprint mismatch")
    # Bind the full recipe to the actual queue argv rather than accepting any
    # complete artifact that happens to occupy the configured output directory.
    command = [str(value).replace("{gpu}", "0") for value in job["command"]]
    try:
        start = next(i for i, value in enumerate(command)
                     if value == "baselines.run_full_replay" or Path(value).name == "run_full_replay.py") + 1
    except StopIteration as error:
        raise ValueError("full replay job must invoke its runner directly") from error
    arguments = parse_args(command[start:])
    runtime = {"resume", "gpu", "max_blocks", "plan_only", "checkpoint_every_blocks"}
    for key, value in vars(arguments).items():
        if key in runtime:
            continue
        expected = str(job_path(job, value)) if isinstance(value, Path) else value
        if config.get(key) != expected:
            raise ValueError(f"full replay recipe mismatch: {key}")
    blocks = make_blocks(rows, config["block_size"])
    if (done.get("status") != "complete" or done.get("num_facts") != len(rows)
            or done.get("num_blocks") != len(blocks) or done.get("final_stage") != 8
            or done.get("method") != config["method"]):
        raise ValueError("full replay completion counts/status mismatch")
    indices = {row["sample_id"]: index for index, row in enumerate(rows)}
    seen_blocks = 0
    with (directory / "training_log.jsonl").open() as stream:
        for seen_blocks, line in enumerate(stream, 1):
            if seen_blocks > len(blocks):
                raise ValueError("full replay training log has duplicate/extra blocks")
            log = json.loads(line)
            block = blocks[seen_blocks - 1]
            if (log.get("block"), log.get("stage"), log.get("start"), log.get("end")) != (
                    seen_blocks, block.stage, block.start, block.end):
                raise ValueError("full replay log is not a continuous source prefix")
            current_ids = [row["sample_id"] for row in rows[block.start:block.end]]
            if log.get("current_sample_ids") != current_ids or log.get("current_count") != len(current_ids):
                raise ValueError("full replay current-block identities mismatch")
            replay = log.get("replay_sample_ids", [])
            expected_count = (0 if config["method"] == "lora_ft" else
                min(round(len(current_ids) * config["replay_ratio"]), block.start, config["memory_capacity"]))
            if len(replay) != expected_count or log.get("replay_count") != len(replay) or len(set(replay)) != len(replay):
                raise ValueError("full replay budget/duplicate mismatch")
            if any(sid not in indices or indices[sid] >= block.start for sid in replay):
                raise ValueError("full replay contains future/untrained facts")
            if log.get("future_leak_count") != 0 or log.get("reservoir_count") != min(config["memory_capacity"], block.end):
                raise ValueError("full replay reservoir/temporal audit mismatch")
            if (not isinstance(log.get("loss"), (int, float)) or not math.isfinite(log["loss"])
                    or any(log.get(key, 0) <= 0 for key in ["optimizer_steps", "sequence_tokens", "answer_tokens"])):
                raise ValueError("full replay training statistics incomplete/nonfinite")
    if seen_blocks != len(blocks):
        raise ValueError("full replay log ends before the full selected stream")
    panels = json.loads((directory / "evaluation_panels.json").read_text())
    if object_sha256(panels) != config["panel_fingerprint"]:
        raise ValueError("full replay panel fingerprint mismatch")
    for axis, identifiers in panels["sample_ids"].items():
        if axis not in AXES or len(identifiers) != len(set(identifiers)) or any(sid not in indices for sid in identifiers):
            raise ValueError("full replay panel identities invalid")
        source_axis = {"persona": "personas", "multihop": "mhop"}.get(axis, axis)
        allowed = set(metadata["panels"][source_axis]["sample_ids"])
        if any(sid not in allowed or any(audit is not None and audit.get("available") is not True
                   for audit in [rows[indices[sid]].get("panel_audit", {}).get(source_axis),
                                 rows[indices[sid]].get("token_length_audit", {}).get("panels", {}).get(source_axis)])
               for sid in identifiers):
            raise ValueError("full replay panel resurrects masked source probes")
        if any(sum(rows[indices[sid]]["stage"] == stage for sid in identifiers) > config["panel_size_per_stage"]
               for stage in range(1, 9)):
            raise ValueError("full replay panel exceeds its frozen per-stage sample budget")
    metrics = json.loads((directory / "metrics.json").read_text())
    if [row.get("current_stage") for row in metrics] != list(range(1, 9)) or done.get("evaluated_stages") != list(range(1, 9)):
        raise ValueError("full replay requires eight evaluated stages")
    for stage, metric in enumerate(metrics, 1):
        trained = sum(row["stage"] <= stage for row in rows)
        if metric.get("trained_facts") != trained or metric.get("num_facts") != trained or metric.get("checkpoint") != f"tau{stage}":
            raise ValueError("full replay evaluation trained-prefix mismatch")
        if metric != json.loads((directory / "evaluation" / f"tau{stage}" / "metrics.json").read_text()):
            raise ValueError("full replay stage metric copies disagree")
        expected_ids = {axis: {sid for sid in panels["sample_ids"][axis]
                              if axis in {"locality", "multihop"} or rows[indices[sid]]["stage"] <= stage}
                        for axis in AXES}
        for axis in AXES:
            value, count = metric[axis], len(expected_ids[axis])
            if count < 1 or value.get("count") != count:
                raise ValueError("full replay sampled-panel denominator mismatch")
            for metric_name in ["match", "exact"]:
                accuracy, correct = value[metric_name + "_accuracy"], value[metric_name + "_count"]
                if not math.isfinite(accuracy) or not 0 <= correct <= count or not math.isclose(accuracy, correct / count, abs_tol=1e-12):
                    raise ValueError("full replay metric invalid/nonfinite")
        observed = {axis: set() for axis in AXES}
        with (directory / "evaluation" / f"tau{stage}" / "predictions.jsonl").open() as predictions:
            for line in predictions:
                value = json.loads(line)
                axis, sid = value["axis"], value["sample_id"]
                if axis not in observed or sid in observed[axis] or sid not in expected_ids[axis] or value["checkpoint"] != f"tau{stage}":
                    raise ValueError("full replay prediction coverage mismatch")
                observed[axis].add(sid)
        if observed != expected_ids:
            raise ValueError("full replay missing sampled predictions")
    if done.get("panel_counts") != {axis: metrics[-1][axis]["count"] for axis in AXES}:
        raise ValueError("full replay final denominators mismatch")
    pointer = json.loads((directory / "latest_checkpoint.json").read_text())
    checkpoint = (directory / pointer["file"]).resolve()
    if (not checkpoint.is_relative_to((directory / "checkpoints").resolve())
            or pointer.get("next_block") != len(blocks) or checkpoint.stat().st_size != pointer["bytes"]
            or pointer.get("config_fingerprint") != config["config_fingerprint"]):
        raise ValueError("full replay final checkpoint missing or inconsistent")
    if not all((directory / "adapters" / "final" / name).is_file()
               for name in ["adapter_model.safetensors", "adapter_config.json"]):
        raise ValueError("full replay final adapter is missing")
    return {"sha256": sha256(output), "manifest_sha256": config["manifest_sha256"],
            "training_facts": len(rows), "training_blocks": len(blocks), "evaluated_stages": 8,
            "panel_counts": done["panel_counts"], "future_leaks": 0,
            "prediction_scores_recomputed": False}


def validate_full_rag(job, output):
    """Keep complete retrieval-index coverage separate from sampled queries."""
    from baselines import run_rag as rag
    from baselines.common import load_manifest, answer_scores, axis_fields
    directory = output.parent
    command = [str(value).replace("{gpu}", "0") for value in job["command"]]
    try:
        start = next(i for i, value in enumerate(command)
                     if value == "baselines.run_rag" or Path(value).name == "run_rag.py") + 1
    except StopIteration as error:
        raise ValueError("full RAG job must invoke its runner directly") from error
    args = rag.parse_args(command[start:])
    if args.evaluation_panels is None or args.no_stage_eval or args.method == "oracle_rag":
        raise ValueError("full RAG requires frozen query panels and all eight stages")
    manifest = job_path(job, args.manifest)
    if job.get("manifest") and job_path(job, job["manifest"]) != manifest:
        raise ValueError("full RAG job manifest differs from its command")
    rows, metadata = load_manifest(manifest)
    if (sorted({row["stage"] for row in rows}) != list(range(1, 9))
            or metadata.get("screening", {}).get("status") != "complete"
            or metadata.get("screening", {}).get("accepted_statuses") != ["unknown"]
            or any(row.get("base_screen", {}).get("status") != "unknown" for row in rows)):
        raise ValueError("full RAG requires the eight-stage screened-unknown selection")
    manifest_hash = sha256(manifest)
    panels, provenance = rag.load_evaluation_panels(job_path(job, args.evaluation_panels),
        rows, metadata, manifest_sha256=manifest_hash)
    config = json.loads((directory / "run_config.json").read_text())
    done = json.loads(output.read_text())
    for key in ["method", "top_k", "recipe", "rrf_depth", "rrf_constant", "reader_max_length",
                "max_new_tokens", "embedding_model", "embedding_max_length", "query_prefix", "document_prefix",
                "no_stage_eval", "max_stages", "examples_per_stage", "locality_panel_size", "multihop_panel_size"]:
        if config.get(key) != getattr(args, key):
            raise ValueError(f"full RAG recipe mismatch: {key}")
    for key in ["manifest", "model_dir", "output_dir", "evaluation_panels"]:
        if config.get(key) is None or job_path(job, config[key]) != job_path(job, getattr(args, key)):
            raise ValueError(f"full RAG path binding mismatch: {key}")
    if (config.get("manifest_sha256") != manifest_hash or config.get("num_facts") != len(rows)
            or config.get("index_count") != len(rows) or not config.get("reader_frozen")
            or config.get("parameter_updates") is not False):
        raise ValueError("full RAG index/config coverage mismatch")
    if rag.bind_evaluation_panels({}, config.get("evaluation_panels_provenance")) != rag.bind_evaluation_panels({}, provenance):
        raise ValueError("full RAG frozen panel provenance mismatch")
    index = config["index_provenance"]
    if (index.get("document_count") != len(rows) or index.get("manifest_sha256") != manifest_hash
            or index.get("selected_sample_ids_sha256") != rag._canonical_digest([row["sample_id"] for row in rows])
            or index.get("stage_counts") != {str(stage): sum(row["stage"] == stage for row in rows) for stage in range(1, 9)}):
        raise ValueError("full RAG index provenance is not the full selected stream")
    identity = {"method": args.method, "reader": rag._reader_provenance(job_path(job, args.model_dir)),
        "manifest_sha256": manifest_hash, "selected_sample_ids_sha256": index["selected_sample_ids_sha256"],
        "top_k": args.top_k, "encoder": config["embedding_model_provenance"],
        "query_prefix": args.query_prefix if args.method == "dense_rag" else None,
        "document_prefix": args.document_prefix if args.method == "dense_rag" else None,
        "reader_max_length": args.reader_max_length, "max_new_tokens": args.max_new_tokens,
        "rag_prompt_template": "<Q: question A: answer contexts> Q: <question> A:" if args.recipe == "wikibigedit" else rag.RAG_PROMPT_TEMPLATE,
        "recipe": args.recipe, "rrf_depth": args.rrf_depth if args.method == "hybrid_rag" else None,
        "rrf_constant": args.rrf_constant if args.method == "hybrid_rag" else None, "index": index}
    fingerprint = rag._canonical_digest(rag.bind_evaluation_panels(identity, provenance))
    if config.get("run_fingerprint") != fingerprint or done.get("run_fingerprint") != fingerprint:
        raise ValueError("full RAG run fingerprint does not bind its reader/index/panel")
    expected_final = rag.sampled_evaluation_counts(panels)
    for value in [config, done]:
        if any(value.get(key) != expected for key, expected in expected_final.items()):
            raise ValueError("full RAG final evaluated denominator differs from its sampled panels")
    if (done.get("status") != "complete" or done.get("method") != args.method or done.get("final_stage") != 8
            or done.get("num_facts") != len(rows) or done.get("index_count") != len(rows)
            or done.get("evaluation_panels_file_sha256") != provenance["file_sha256"]
            or done.get("parameter_updates") != 0 or done.get("future_leak_count") != 0):
        raise ValueError("full RAG completion index/count/status mismatch")
    by_id = {row["sample_id"]: row for row in rows}
    expected_ids = {stage: {axis: {row["sample_id"] for row in panels[axis] if row["stage"] <= stage}
                           for axis in rag.AXES} for stage in range(1, 9)}
    observed = {stage: {axis: {} for axis in rag.AXES} for stage in range(1, 9)}
    total = 0
    with (directory / "predictions.jsonl").open() as predictions:
        for line in predictions:
            value = json.loads(line)
            stage, axis, sid = value["current_stage"], value["axis"], value["sample_id"]
            if (stage not in observed or axis not in observed[stage] or sid not in expected_ids[stage][axis]
                    or sid in observed[stage][axis] or value.get("checkpoint") != f"tau{stage}"):
                raise ValueError("full RAG sampled prediction identity/coverage mismatch")
            row = by_id[sid]
            question, answer = axis_fields(axis)
            if value.get("fact_stage") != row["stage"] or value.get("question") != row[question] or value.get("gold") != row[answer]:
                raise ValueError("full RAG prediction question/gold/fact-stage mismatch")
            hits, stages, scores = value["retrieved_sample_ids"], value["retrieved_fact_stages"], value["retrieval_scores"]
            if (len(hits) != len(set(hits)) or len(hits) > args.top_k or len(hits) != len(stages)
                    or len(hits) != len(scores) or any(not math.isfinite(score) for score in scores)
                    or any(hit not in by_id or by_id[hit]["stage"] > stage for hit in hits)
                    or stages != [by_id[hit]["stage"] for hit in hits]):
                raise ValueError("full RAG retrieval contains future/invalid facts")
            score = answer_scores(value["prediction"], value["gold"])
            if any(value.get(key) != expected for key, expected in score.items()):
                raise ValueError("full RAG prediction score mismatch")
            observed[stage][axis][sid] = value
            total += 1
    metrics = json.loads((directory / "metrics.json").read_text())
    if [metric.get("current_stage") for metric in metrics] != list(range(1, 9)):
        raise ValueError("full RAG does not have eight evaluated stages")
    for stage, metric in enumerate(metrics, 1):
        prefix = [row for row in rows if row["stage"] <= stage]
        axis_rows = {axis: [row for row in panels[axis] if row["stage"] <= stage] for axis in rag.AXES}
        if ({axis: set(values) for axis, values in observed[stage].items()} != expected_ids[stage]
                or metric.get("num_facts") != len(prefix) or metric.get("index_count") != len(prefix)
                or metric.get("run_fingerprint") != fingerprint
                or metric.get("evaluation_panels_file_sha256") != provenance["file_sha256"]
                or any(metric.get(key) != expected for key, expected in rag.sampled_evaluation_counts(axis_rows).items())):
            raise ValueError("full RAG stage index/sample denominator/coverage mismatch")
        retrieval = metric["retrieval"]
        if (retrieval.get("index_count") != len(prefix) or retrieval.get("future_leak_count") != 0
                or retrieval.get("index_sample_ids_sha256") != rag._canonical_digest([row["sample_id"] for row in prefix])):
            raise ValueError("full RAG stage retrieval index mismatch")
        for axis, values in observed[stage].items():
            count = len(values)
            value = metric[axis]
            if count < 1 or value.get("count") != count:
                raise ValueError("full RAG five-axis denominator mismatch")
            for score in ["exact", "match"]:
                correct = sum(record[score] for record in values.values())
                if value.get(score + "_count") != correct or not math.isclose(value[score + "_accuracy"], correct / count, abs_tol=1e-12):
                    raise ValueError("full RAG aggregate score mismatch")
        if metric != json.loads((directory / "cache/evaluation" / f"tau{stage}/metrics.json").read_text()):
            raise ValueError("full RAG canonical/cached stage metrics disagree")
    if done.get("evaluated_total_across_stages") != total:
        raise ValueError("full RAG total sampled-query count mismatch")
    return {"sha256": sha256(output), "manifest_sha256": manifest_hash, "index_count": len(rows),
            "evaluated_count": expected_final["evaluated_count"], "evaluated_count_by_axis": expected_final["evaluated_count_by_axis"],
            "prediction_records": total, "evaluated_stages": 8, "future_leaks": 0,
            "run_fingerprint": fingerprint, "panel_file_sha256": provenance["file_sha256"],
            "prediction_scores_recomputed": True}


def external_distributed(excluded):
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) in excluded:
            continue
        try:
            argv = [x.decode(errors="replace") for x in (p / "cmdline").read_bytes().split(b"\0") if x]
            if distributed_launcher(argv):
                # Read only CUDA visibility; never copy an environment to logs.
                visible = next((v.split(b"=", 1)[1].decode() for v in
                                (p / "environ").read_bytes().split(b"\0")
                                if v.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
                gpus = ([int(v) for v in visible.split(",")]
                        if visible and all(v.strip().isdigit() for v in visible.split(",")) else None)
                found.append({"pid": int(p.name), "gpus": gpus})
        except (OSError, ProcessLookupError):
            pass
    return found


def free_gpus():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       check=True, capture_output=True, text=True, timeout=20)
    return {int(parts[0]) for line in r.stdout.splitlines()
            if len(parts := line.split(",")) == 2 and int(parts[1]) < 512}


def validate_output(job):
    """Return a completion certificate, or fail rather than promote partial work."""
    kind = job.get("validation", "json")
    path = Path(job["output"])
    if kind == "full_replay":
        return validate_full_replay(job, path)
    if kind == "full_rag":
        return validate_full_rag(job, path)
    if kind == "screen":
        source = Path(job["manifest"])
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        if metadata["manifest_sha256"] != sha256(source):
            raise ValueError("screen manifest hash mismatch")
        validate_screen_recipe(job, metadata, source)
        records = json.loads(source.read_text())["records"]
        from baselines.common import answer_scores
        with path.open() as f:
            answers = [json.loads(line) for line in f if line.strip()]
        if len(answers) != len(records) or len(answers) != job["expected_count"]:
            raise ValueError("screen coverage mismatch")
        statuses = {"known": 0, "unknown": 0}
        for expected, actual in zip(records, answers):
            if any(actual[k] != expected[s] for k, s in (
                ("sample_id", "sample_id"), ("stage", "stage"),
                ("question", "update"), ("gold", "ans"))):
                raise ValueError("screen row identity mismatch")
            score = answer_scores(actual["prediction"], actual["gold"])
            status = "known" if score["match"] else "unknown"
            if actual["screen_status"] != status or actual["base_unknown"] != (status == "unknown"):
                raise ValueError("screen classification mismatch")
            if any(actual[k] != v for k, v in score.items()):
                raise ValueError("screen score mismatch")
            statuses[status] += 1
        if metadata["status_counts"] != statuses or metadata["candidate_count"] != len(records):
            raise ValueError("screen metadata counts mismatch")
        return {"sha256": sha256(path), "count": len(answers), "status_counts": statuses}
    obj = json.loads(path.read_text())
    for key, expected in job.get("required_fields", {}).items():
        if obj.get(key) != expected:
            raise ValueError(f"unexpected completion field {key}: {obj.get(key)!r}")
    return {"sha256": sha256(path)}


def preserve_uncommitted_screen(job):
    """Keep a result written before its metadata commit, then allow retry."""
    path = Path(job["output"])
    if job.get("validation") != "screen" or not path.exists() or path.with_suffix(".metadata.json").exists():
        return None
    archive = path.with_name(path.name + f".uncommitted.{time.time_ns()}")
    path.replace(archive)
    return str(archive)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--gpus", default="6,7")
    p.add_argument("--max-parallel", type=int, default=2)
    p.add_argument("--poll-seconds", type=float, default=3)
    p.add_argument("--idle-seconds", type=float, default=15)
    args = p.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lease = args.state.with_suffix(".lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw = args.plan.read_bytes()
    plan_hash = hashlib.sha256(raw).hexdigest()
    plan = json.loads(raw)
    jobs = plan["jobs"]
    by_name = {j["name"]: j for j in jobs}
    if len(by_name) != len(jobs):
        raise ValueError("duplicate job names")
    if any(d not in by_name for j in jobs for d in j.get("dependencies", [])):
        raise ValueError("missing dependency")
    state = json.loads(args.state.read_text()) if args.state.exists() else {
        "plan_sha256": plan_hash, "started_at": time.time(), "jobs": {}}
    if state["plan_sha256"] != plan_hash:
        raise ValueError("plan changed; use a new queue state")
    state.update(pid=os.getpid(), status="running")
    for j in jobs:
        s = state["jobs"].setdefault(j["name"], {"status": "pending", "attempts": 0})
        if archived := preserve_uncommitted_screen(j):
            s.setdefault("uncommitted_results", []).append(archived)
            s["status"] = "pending"
        if s["status"] == "complete":
            s["certificate"] = validate_output(j)
        elif Path(j["output"]).exists():
            s.update(status="complete", certificate=validate_output(j), recovered=True)
        elif s["status"] in ("running", "waiting_external", "interrupted"):
            s["status"] = "pending"
    running = {}
    idle = {}
    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def stop_job(name, why):
        item = running.pop(name)
        proc = item["proc"]
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        item["log"].close()
        if item["lease"]:
            item["lease"].close()
        state["jobs"][name].update(status="pending", last_interruption=why,
                                  interrupted_at=time.time(), pid=None)
        job = by_name[name]
        if archived := preserve_uncommitted_screen(job):
            state["jobs"][name].setdefault("uncommitted_results", []).append(archived)
        elif Path(job["output"]).exists():
            try:
                state["jobs"][name].update(status="complete", certificate=validate_output(job))
            except Exception as e:
                state["jobs"][name].update(status="failed", error=repr(e))
        print(json.dumps({"yielded": name, "reason": why}), flush=True)

    try:
        while True:
            now = time.time()
            for name, item in list(running.items()):
                code = item["proc"].poll()
                if code is None:
                    continue
                item["log"].close()
                if item["lease"]:
                    item["lease"].close()
                del running[name]
                s = state["jobs"][name]
                s.update(exit_code=code, finished_at=now, pid=None)
                try:
                    if code:
                        raise RuntimeError(f"process exit code {code}")
                    s.update(status="complete", certificate=validate_output(by_name[name]))
                except Exception as e:
                    s.update(status="failed", error=repr(e))
                print(json.dumps({"job": name, "status": s["status"], "exit_code": code}), flush=True)
            if stopping:
                for name in list(running):
                    stop_job(name, "queue_stop")
                state["status"] = "interrupted"
                break
            external = external_distributed({os.getpid()} | {v["proc"].pid for v in running.values()})
            state["external_distributed_pids"] = external
            blocked_gpus = ({int(x) for x in args.gpus.split(",")} if any(e["gpus"] is None for e in external)
                            else {g for e in external for g in e["gpus"]})
            if blocked_gpus:
                for name, item in list(running.items()):
                    if item["gpu"] in blocked_gpus:
                        stop_job(name, f"external_distributed:{external}")
                for gpu in blocked_gpus:
                    idle.pop(gpu, None)
            available = free_gpus() - blocked_gpus
            allowed = [int(x) for x in args.gpus.split(",")]
            for gpu in allowed:
                if gpu in available:
                    idle.setdefault(gpu, now)
                else:
                    idle.pop(gpu, None)
            occupied = {v["gpu"] for v in running.values()}
            for j in jobs:
                s = state["jobs"][j["name"]]
                if s["status"] != "pending" or len(running) >= args.max_parallel:
                    continue
                dependencies = [state["jobs"][d]["status"] for d in j.get("dependencies", [])]
                if any(x in {"failed", "blocked_dependency"} for x in dependencies):
                    s["status"] = "blocked_dependency"
                    continue
                if any(x != "complete" for x in dependencies):
                    continue
                gpu, gpu_lease = None, None
                if j.get("gpu", True):
                    for candidate in allowed:
                        if candidate in occupied or now - idle.get(candidate, now) < args.idle_seconds:
                            continue
                        candidate_lease = Path(f"/tmp/engram_baseline_gpu_{candidate}.lock").open("a+")
                        try:
                            fcntl.flock(candidate_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            candidate_lease.close()
                            continue
                        gpu, gpu_lease = candidate, candidate_lease
                        break
                    if gpu is None:
                        continue
                log_path = Path(j.get("log", str(args.state.parent / "logs" / (j["name"] + ".log"))))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("a", buffering=1)
                command = launch_command(j)
                env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                if gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                log.write(f"\nQUEUE START {time.time()} GPU={gpu} argv={json.dumps(command)}\n")
                proc = subprocess.Popen(command, cwd=j.get("cwd", str(Path.cwd())), env=env,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[j["name"]] = {"proc": proc, "log": log, "lease": gpu_lease, "gpu": gpu}
                occupied.add(gpu)
                s.update(status="running", pid=proc.pid, gpu=gpu, launched_at=now,
                         attempts=s["attempts"] + 1, log=str(log_path))
                print(json.dumps({"started": j["name"], "pid": proc.pid, "gpu": gpu}), flush=True)
            statuses = [s["status"] for s in state["jobs"].values()]
            if all(s == "complete" for s in statuses):
                state["status"] = "complete"
            elif not running and all(s in {"complete", "failed", "blocked_dependency"} for s in statuses):
                state["status"] = "failed"
            else:
                state["status"] = "waiting_external" if external and not running else "running"
            state["updated_at"] = now
            atomic_json(args.state, state)
            if state["status"] in {"complete", "failed"}:
                break
            time.sleep(args.poll_seconds)
    finally:
        for name in list(running):
            stop_job(name, "queue_exit")
        state["updated_at"] = time.time()
        atomic_json(args.state, state)
        lease.close()
    if state["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
