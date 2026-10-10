#!/usr/bin/env python3
"""Official TSR-v2 LoRA mechanics transferred to the frozen WikiBigEdit stream.

Not a full-parameter reproduction. Uses upstream LoRA/signature/routing code,
with stable FP32 KL and teacher forward before student to avoid swapping live
autograd tensors. No evaluation question participates in training or routing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/recent_extra/tsr"
sys.path.insert(0, str(SOURCE / "src"))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=ROOT / "Qwen3-8B-Base")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--memory-fraction", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps-per-stage", type=int, default=500)
    p.add_argument("--single-pass", action="store_true",
                   help="Consume each current-stage example once; historical replay uses separate steps")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--buffer-per-task", type=int, default=100)
    p.add_argument("--replay-frac", type=float, default=0.5)
    p.add_argument("--probe-batches", type=int, default=10)
    p.add_argument("--sig-tau", type=float, default=0.1)
    p.add_argument("--kd-lambda", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=2.)
    p.add_argument("--max-stages", type=int, default=8)
    p.add_argument("--locality-panel-size", type=int, default=0)
    p.add_argument("--multihop-panel-size", type=int, default=0)
    p.add_argument("--skip-eval", action="store_true", help="smoke only")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def source_hashes():
    return {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SOURCE.rglob("*.py"))}


def cpu_snapshot(layers):
    return [(l.A.detach().cpu().clone(), l.B.detach().cpu().clone()) for l in layers]


def stable_kl(student, teacher, mask, temperature):
    # Algebraically identical to upstream all-attended-token KL; FP32 prevents
    # BF16 softmax tails and log(0) from producing non-finite distillation loss.
    import torch
    import torch.nn.functional as F
    log_s = F.log_softmax(student.float() / temperature, dim=-1)
    log_t = F.log_softmax(teacher.float() / temperature, dim=-1)
    kl = (log_t.exp() * (log_t - log_s)).sum(-1)
    weights = mask.to(kl.dtype)
    return (kl * weights).sum() / weights.sum().clamp_min(1) * temperature ** 2


def main():
    args = arguments()
    import torch
    from data import infinite_loader, subsample, make_collate_fn
    from baselines.single_pass_stream import training_batches
    from lora import attach_lora, lora_layers, reset_all, restore, snapshot, trainable_params
    from routing import probe_signature, cosine_scores, routing_weights
    from baselines.common import (load_manifest, group_by_stage, qa_prompt,
        load_model, load_tokenizer, atomic_json, sha256_file, manifest_panel,
        fixed_panel, evaluate_checkpoint, append_jsonl, environment_metadata)
    from baselines.run_lora_family import summarize_checkpoint_records

    if not (args.steps_per_stage > 0 and args.batch_size > 0 and args.probe_batches > 0
            and args.buffer_per_task > 0 and 0 <= args.replay_frac <= 1
            and args.sig_tau > 0 and args.kd_temperature > 0
            and 0 < args.memory_fraction <= 1):
        raise ValueError("invalid training/routing arguments")
    if args.single_pass and args.replay_frac >= 1:
        raise ValueError('Single-pass training requires replay_frac < 1')
    rows, metadata = load_manifest(args.manifest)
    stages = group_by_stage(rows)
    if not 1 <= args.max_stages <= len(stages):
        raise ValueError("invalid max-stages")
    configuration = {k: str(v.resolve()) if isinstance(v, Path) else v
                     for k, v in vars(args).items() if k not in {"resume", "dry_run", "gpu"}}
    configuration.update(method="TSR-v2-LoRA-WikiBigEdit-transfer",
        replication_scope="Official TSR-v2 algorithm; Qwen3 Base and WikiBigEdit adaptation, not paper TRACE reproduction or full-parameter baseline",
        upstream_repository="https://github.com/ymeng3/transfer-selective-replay",
        upstream_archive_sha256=sha256_file(SOURCE.parent / "tsr.zip"),
        upstream_file_sha256=source_hashes(),
        manifest_sha256=sha256_file(args.manifest),
        manifest_protocol_id=metadata.get("protocol_id"),
        num_facts=len(rows), stage_counts={str(k): len(v) for k, v in stages.items()},
        training_fields=["update", "ans"], evaluation_fields_never_used_for_training=True,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        differences_from_upstream=["Qwen3-8B-Base replaces Qwen2.5-7B-Instruct",
          "all 375 training facts per stage replace TRACE's 50-example target subsample",
          "shared closed-book QA template/evaluation; 128-token training limit",
          "batch size/steps and KD defaults retained; SDPA and gradient checkpointing bound VRAM",
          "FP32 KL computation; teacher forward precedes student; CPU checkpoint storage",
          "stage checkpoint resume and cost/coverage logging"],
        budget_note=f"{args.steps_per_stage} updates/stage ({args.steps_per_stage * args.max_stages} total) plus {args.probe_batches} signature probe batches/stage; {args.replay_frac:.0%} expected replay after stage 1; KD teacher forwards counted separately. Original uses 500 updates/stage. Matching update counts does not match compute with full-parameter Policy.")
    if args.single_pass:
        configuration.update(current_data_passes=1, steps_per_stage=None,
            budget_note='One shuffled pass over current examples per stage; separate historical replay steps. Signature probes and KD forwards are additional compute, not additional current-data optimizer passes.')
        configuration['differences_from_upstream'].append('User-specified single-pass current-data budget replaces the fixed update-count schedule')
    if args.dry_run:
        print(json.dumps(configuration, indent=2), flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if not args.resume:
            raise FileExistsError("run exists; use --resume")
        if old.get("configuration") != configuration:
            raise ValueError("resume configuration/source/manifest mismatch")
    elif any(args.output_dir.iterdir()):
        raise FileExistsError("nonempty output without run_config")
    else:
        atomic_json(config_path, {"configuration": configuration, **environment_metadata()})
    if (args.output_dir / "completed.json").exists():
        print("Already complete", flush=True)
        return
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device, use_flash_attention=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    n_layers = attach_lora(model, r=args.rank, alpha=args.alpha)
    if not n_layers:
        raise RuntimeError("no LoRA modules attached")
    layers = list(lora_layers(model))
    reset_all(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    init_snap = snapshot(layers)
    buffer, signatures, snapshots = {}, {}, {}
    logs, metrics, completed_stage = [], [], 0
    checkpoint_path = args.output_dir / "checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt["configuration"] != configuration:
            raise ValueError("checkpoint identity mismatch")
        restore(layers, ckpt["adapter"])
        init_snap = [(a.to(device), b.to(device)) for a, b in ckpt["initial_adapter"]]
        snapshots = {k: [(a.to(device), b.to(device)) for a, b in v]
                     for k, v in ckpt["snapshots"].items()}
        signatures, buffer = ckpt["signatures"], ckpt["buffer"]
        logs, metrics, completed_stage = ckpt["logs"], ckpt["metrics"], ckpt["completed_stage"]
        torch.set_rng_state(ckpt["torch_rng"])
        torch.cuda.set_rng_state(ckpt["cuda_rng"], device)
        random.setstate(ckpt["python_rng"])
    locality = manifest_panel(metadata, "locality", limit=args.locality_panel_size)
    if locality is None:
        locality = fixed_panel(rows, "loc", "loc_ans", args.locality_panel_size, args.seed + 1001)
    multihop = manifest_panel(metadata, "mhop", limit=args.multihop_panel_size)
    if multihop is None:
        multihop = fixed_panel(rows, "mhop", "mhop_ans", args.multihop_panel_size, args.seed + 2001)
    def dataset(stage_rows):
        return [{"prompt": qa_prompt(row["update"]), "answer": " " + row["ans"].strip(),
                 "sample_id": row["sample_id"]} for row in stage_rows]
    def loader(ds, seed):
        return infinite_loader(ds, tokenizer, args.batch_size, args.max_length, seed)
    start_time = time.time()
    for stage, current in list(stages.items())[:args.max_stages]:
        if stage <= completed_stage:
            continue
        begin = time.time()
        train_ds = dataset(current)
        # Only update/ans are passed to official probe and training loaders.
        signature = probe_signature(model, layers, init_snap,
            loader(train_ds, args.seed + 977), args.probe_batches, device)
        scores = cosine_scores(signatures, signature) if signatures else {}
        top_source = max(scores, key=scores.get) if scores else None
        keys, probs = routing_weights(scores, args.sig_tau) if scores else ([], [])
        signatures[stage] = signature
        atomic_json(args.output_dir / f"routing_tau{stage}.json", {
            "stage": stage, "scores": scores, "task_keys": keys, "probabilities": probs,
            "current_sample_ids": [r["sample_id"] for r in current],
            "replay_buffer_sample_ids": {str(k): [x["sample_id"] for x in v] for k, v in buffer.items()},
            "future_leak_count": 0})
        opt = torch.optim.AdamW(list(trainable_params(model)), lr=args.learning_rate)
        target_loader = loader(train_ds, args.seed + stage - 1)
        current_batch_count = None
        current_ids = []
        if args.single_pass:
            generator = torch.Generator().manual_seed(args.seed + stage - 1)
            collate = make_collate_fn(tokenizer, max_length=args.max_length)
            def record_current(batch):
                current_ids.extend(row['sample_id'] for row in batch)
                return collate(batch)
            finite = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                shuffle=True, generator=generator, collate_fn=record_current)
            current_batch_count = len(finite)
            target_loader = iter(finite)
        replay_loaders = [loader(buffer[k], args.seed + 31 + k - 1) for k in keys]
        rng = random.Random(args.seed * 7919 + stage - 1)
        replay_steps = kd_forwards = answer_tokens = sequence_tokens = 0
        replay_by_stage = {k: 0 for k in keys}
        sum_loss = 0.
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        for step, ri, raw_batch in training_batches(target_loader, replay_loaders, rng,
                args.replay_frac, probs, args.steps_per_stage, current_batch_count):
            replay_task = None
            if ri is not None:
                replay_task = keys[ri]
                if replay_task >= stage:
                    raise RuntimeError("future replay leak")
                replay_steps += 1
                replay_by_stage[replay_task] += 1
            batch = {k: v.to(device) for k, v in raw_batch.items()}
            teacher = None
            teacher_snap = snapshots.get(replay_task if replay_task is not None else top_source)
            if teacher_snap is not None:
                current_snap = snapshot(layers)
                with torch.no_grad():
                    restore(layers, teacher_snap)
                    teacher = model(**batch).logits.detach()
                    restore(layers, current_snap)
                del current_snap
                kd_forwards += 1
            out = model(**batch)
            loss = out.loss
            if teacher is not None:
                loss = loss + args.kd_lambda * stable_kl(out.logits, teacher,
                    batch["attention_mask"], args.kd_temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at stage {stage} step {step}")
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sum_loss += float(loss.detach())
            answer_tokens += int((batch["labels"] != -100).sum())
            sequence_tokens += int(batch["attention_mask"].sum())
            del out, loss, teacher, batch
            if (step + 1) % 10 == 0 or step == 0 or (args.single_pass and len(current_ids) == len(current)) or (not args.single_pass and step + 1 == args.steps_per_stage):
                state = {"status": "training", "stage": stage, "step": step + 1,
                         "steps_per_stage": None if args.single_pass else args.steps_per_stage,
                         "current_examples_consumed": len(current_ids) if args.single_pass else None,
                         "current_examples_total": len(current), "loss": sum_loss / (step + 1),
                         "replay_steps": replay_steps, "kd_forwards": kd_forwards,
                         "elapsed_stage_seconds": time.time() - begin}
                atomic_json(args.output_dir / "progress.json", state)
                print(json.dumps(state), flush=True)
        buffer[stage] = list(subsample(train_ds, args.buffer_per_task, args.seed + 53))
        snapshots[stage] = snapshot(layers)
        actual_steps = step + 1
        if args.single_pass:
            assert len(current_ids) == len(set(current_ids)) == len(current)
            assert set(current_ids) == {row['sample_id'] for row in current}
        log = {"stage": stage, "current_count": len(current), "steps": actual_steps,
               "current_data_passes": 1 if args.single_pass else None,
               "current_training_ids": current_ids if args.single_pass else None,
               "replay_steps": replay_steps, "replay_by_stage": replay_by_stage,
               "kd_teacher_forwards": kd_forwards, "signature_probe_forwards_backwards": args.probe_batches,
               "mean_loss": sum_loss / actual_steps, "answer_tokens": answer_tokens,
               "sequence_tokens": sequence_tokens, "wall_seconds": time.time() - begin,
               "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
               "trainable_parameters": sum(p.numel() for p in trainable_params(model))}
        logs.append(log)
        del opt
        if not args.skip_eval:
            raw, records = evaluate_checkpoint(model, tokenizer,
                [r for r in rows if r["stage"] <= stage], device,
                checkpoint=f"tau{stage}", axes=("update", "rephrase", "persona", "locality", "multihop"),
                batch_size=args.eval_batch_size,
                max_length=args.max_length, max_new_tokens=args.max_new_tokens,
                locality_panel=locality, multihop_panel=multihop)
            prediction_path = args.output_dir / f"predictions_tau{stage}.jsonl"
            temporary = prediction_path.with_suffix(".tmp")
            temporary.write_text("", encoding="utf-8")
            append_jsonl(temporary, records)
            temporary.replace(prediction_path)
            metrics.append({**raw, **summarize_checkpoint_records(records, stage), "train": log})
        atomic_json(args.output_dir / "metrics.json", metrics)
        atomic_json(args.output_dir / "training_log.json", logs)
        saved = {"configuration": configuration, "adapter": cpu_snapshot(layers),
            "initial_adapter": [(a.cpu(), b.cpu()) for a, b in init_snap],
            "snapshots": {k: [(a.cpu(), b.cpu()) for a, b in v] for k, v in snapshots.items()},
            "buffer": buffer, "signatures": signatures, "logs": logs, "metrics": metrics,
            "completed_stage": stage, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device), "python_rng": random.getstate()}
        temporary = checkpoint_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(saved, temporary)
        temporary.replace(checkpoint_path)
        print(json.dumps({"committed_stage": stage, **log}), flush=True)
    atomic_json(args.output_dir / "completed.json", {
        "status": "complete", "method": configuration["method"],
        "final_stage": args.max_stages, "smoke_only": args.skip_eval or args.max_stages < len(stages),
        "elapsed_seconds_this_invocation": time.time() - start_time})


if __name__ == "__main__":
    main()
