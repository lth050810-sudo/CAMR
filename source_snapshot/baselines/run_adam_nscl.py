"""Adam-NSCL with fixed Wikipedia input covariance on the WikiBigEdit stages.

The update operator is verified against the CVPR 2021 source. Applying it to
Qwen3 down projections and a fixed pretraining corpus is an explicit transfer
choice, following the fixed-corpus WikiBigEdit comparison setting. It is not
the defective gradient-before-Adam implementation in the CrispEdit fork.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import hashlib
from pathlib import Path

import torch

from baselines.adam_nscl_core import AdamNSCL, source_transform, UPSTREAM_COMMIT, UPSTREAM_REPOSITORY
from baselines.common import atomic_json, environment_metadata, group_by_stage, load_manifest, load_model, load_tokenizer, manifest_panel, sha256_file
from baselines.crispedit_core import model_identity, parse_layers, resolve_parameters, stats_filename, validate_stats_payload
from baselines.run_crispedit import train_epoch
from baselines.run_lora_family import evaluate_and_write
from baselines.editor_checkpoint import resume_identity, read_checkpoint, restore_stage, save_stage


def cached_transform(covariance, stats_path, threshold, parameter, stats_dir):
    """Keep expensive source SVD results across preempted initializations."""
    signature = {"stats_sha256": sha256_file(stats_path), "threshold": threshold,
                 "parameter_dtype": str(parameter.dtype), "upstream_commit": UPSTREAM_COMMIT,
                 "core_sha256": sha256_file(Path(__file__).with_name("adam_nscl_core.py"))}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    directory = stats_dir.parent / "adam_nscl_transform_cache"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["signature"] != signature or saved["transform"].shape != covariance.shape:
            raise ValueError("Invalid cached Adam-NSCL transform")
        return saved["transform"].to(parameter), saved["info"]
    transform, info = source_transform(covariance.to(device=parameter.device, dtype=torch.float32), threshold)
    result = transform.to(parameter)
    tmp = path.with_name(path.name + f".{__import__('os').getpid()}.tmp")
    torch.save({"signature": signature, "transform": result.detach().cpu(), "info": info}, tmp)
    tmp.replace(path)
    return result, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--layers", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--nullspace-threshold", type=float, default=10.)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--loss-threshold", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--allow-diagnostic-stats", action="store_true")
    parser.add_argument("--dev-selection", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(args.output_dir)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    layers = parse_layers(args.layers)
    rows, manifest = load_manifest(args.manifest)
    stages = group_by_stage(rows)
    ids = {row["sample_id"] for row in rows}
    loc = [row for row in manifest_panel(manifest, "locality") if row["sample_id"] in ids]
    mh = [row for row in manifest_panel(manifest, "mhop") if row["sample_id"] in ids]
    identity = model_identity(args.model_dir)
    method = "Adam-NSCL-Qwen3-transfer"
    selection = None
    if args.dev_selection:
        from baselines.editor_dev_selection import verify_selection
        selection = verify_selection(args.dev_selection, args, method, layers)
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)
    model.requires_grad_(False)
    parameters = resolve_parameters(model, layers)
    previous_config = json.loads((args.output_dir / "run_config.json").read_text()) if args.resume else None
    resume_payload = read_checkpoint(args.output_dir) if args.resume else None
    transforms, provenance = {}, []
    for layer, parameter in parameters.items():
        path = args.stats_dir / stats_filename(layer)
        if args.resume:
            entry = next(item for item in previous_config["statistics_provenance"] if item["layer"] == layer)
            if entry["path"] != str(path.resolve()) or entry["sha256"] != sha256_file(path):
                raise ValueError("Adam-NSCL statistics changed across resume")
            transforms[parameter] = resume_payload["transforms"][str(layer)].to(parameter)
            provenance.append(entry)
            parameter.requires_grad_(True)
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_stats_payload(payload, identity=identity, layer=layer)
        corpus = payload.get("corpus", {})
        qualified = corpus.get("publication_ready") is True or corpus.get("verified_dated_corpus_transfer") is True
        if not qualified and not args.allow_diagnostic_stats:
            raise ValueError("Unverified corpus; --allow-diagnostic-stats is only for development")
        covariance = payload["A"]
        if covariance.shape != (parameter.shape[1], parameter.shape[1]):
            raise ValueError("KFAC A must describe the Linear input dimension")
        transform, info = cached_transform(covariance, path, args.nullspace_threshold, parameter, args.stats_dir)
        transforms[parameter] = transform.to(parameter)
        provenance.append({"layer": layer, "path": str(path.resolve()), "sha256": sha256_file(path),
                           "source_corpus": corpus, "source_num_documents": payload["num_documents"], **info})
        parameter.requires_grad_(True)
        del covariance, payload, transform
    optimizer = AdamNSCL(parameters.values(), transforms, lr=args.learning_rate,
                         weight_decay=args.weight_decay)
    originals = {layer: parameter.detach().cpu().clone() for layer, parameter in parameters.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qualified = all(p["source_corpus"].get("publication_ready") is True or
                    p["source_corpus"].get("verified_dated_corpus_transfer") is True for p in provenance)
    corpus_label = ("20220301 sampled-clone, whitespace-normalized" if all(
        p["source_corpus"].get("verified_dated_corpus_transfer") is True for p in provenance
    ) else "Wikipedia corpus" if qualified else "diagnostic corpus")
    config = {**{key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "layers": layers, "qwen3_layers": layers, "method": method,
              "method_variant": "fixed-Wikipedia-input-covariance",
              "num_facts": len(rows), "manifest_protocol_id": manifest["protocol_id"],
              "manifest_sha256": sha256_file(args.manifest), "model_identity": identity,
              "upstream_repository": UPSTREAM_REPOSITORY, "upstream_commit": UPSTREAM_COMMIT,
              "statistics_provenance": provenance, "qwen3_layer_selection_validated": selection is not None,
              "dev_selection_audit": selection, "closed_book_evaluation": True,
              "formal_baseline_eligible": bool(qualified and selection),
              "publication_ready": bool(qualified and selection),
              "reproduction_exact": False,
              "method_display_name": f"Adam-NSCL ({corpus_label}, Qwen3 " + ("dev-tuned transfer)" if selection else "dev candidate)"),
              "qualification": f"{corpus_label}, Qwen3 transfer; fixed old-task covariance; not an official Qwen3 preset or canonical source-corpus reproduction",
              "transfer_differences": ["Qwen3 Linear down projections and answer-only autoregressive QA",
                  "fixed Wikipedia covariance rather than accumulating supervised image-task features",
                  "layer selection on independent dev before 3K evaluation; other hyperparameters fixed",
                  "shared bfloat16 parameter/optimizer states and generated five-axis evaluation"],
              **environment_metadata()}
    identity_for_resume = resume_identity(args, config["manifest_sha256"], identity, provenance,
        ["baselines/run_adam_nscl.py", "baselines/adam_nscl_core.py", "baselines/run_crispedit.py",
         "baselines/crispedit_core.py", "baselines/editor_dev_selection.py"])
    config["resume_identity"] = identity_for_resume
    if args.resume:
        if previous_config.get("resume_identity") != identity_for_resume:
            raise ValueError("Adam-NSCL run configuration changed across resume")
        config = previous_config
    else:
        atomic_json(args.output_dir / "run_config.json", config)
    metrics, training, seen = [], [], []
    start_cursor = 0
    if args.resume:
        checkpoint = restore_stage(args.output_dir, parameters, optimizer, identity_for_resume, payload=resume_payload)
        start_cursor = checkpoint["cursor"]
        if sum(len(stage_rows) for stage, stage_rows in stages.items() if stage <= start_cursor) != checkpoint["seen_count"]:
            raise ValueError("Adam-NSCL checkpoint stage cursor mismatch")
        seen = list(rows[:checkpoint["seen_count"]])
        metrics, training = checkpoint["metrics"], checkpoint["training_log"]
        print({"resumed_from_stage": start_cursor, "seen_count": len(seen)}, flush=True)
        del checkpoint, resume_payload
    started = time.time()
    checkpoint_dir = Path("/mnt/yy2607004_hs/engram_checkpoints") / args.output_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for stage, current in stages.items():
        if stage <= start_cursor:
            continue
        epochs = []
        for epoch in range(1, args.epochs + 1):
            record = train_epoch(model, tokenizer, current, optimizer, device,
                                 seed=args.seed + stage * 1000 + epoch, batch_size=args.batch_size,
                                 max_length=args.max_length, max_grad_norm=0.)
            if not all(torch.isfinite(parameter).all() for parameter in parameters.values()):
                raise FloatingPointError("Non-finite Adam-NSCL parameter")
            epochs.append({"epoch": epoch, **record})
            if record["loss"] < args.loss_threshold:
                break
        seen.extend(current)
        training.append({"stage": stage, "num_current_facts": len(current), "num_seen_facts": len(seen),
                         "sample_ids": [row["sample_id"] for row in current], "finite": True,
                         "future_leak_count": 0, "epochs": epochs})
        atomic_json(args.output_dir / "training_log.json", training)
        metrics.append(evaluate_and_write(model, tokenizer, seen, loc, mh, device, args, f"tau{stage}", stage))
        atomic_json(args.output_dir / "metrics.json", metrics)
        torch.save({"stage": stage, "deltas": {
            f"model.layers.{layer}.mlp.down_proj.weight": parameter.detach().cpu() - originals[layer]
            for layer, parameter in parameters.items()}}, checkpoint_dir / f"tau{stage}_delta.pt")
        save_stage(args.output_dir, parameters, optimizer, identity_for_resume,
                   cursor=stage, stage=stage, seen_count=len(seen), metrics=metrics, training_log=training)
    atomic_json(args.output_dir / "completed.json", {"status": "complete", "method": method,
        "num_facts": len(rows), "edits": len(rows), "final_stage": max(stages),
        "elapsed_seconds": time.time() - started, "formal_baseline_eligible": bool(qualified and selection),
        "resumed_from_stage": start_cursor})


if __name__ == "__main__":
    main()
