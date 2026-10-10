#!/usr/bin/env python3
"""Sequential breadth-first full fine-tuning with native PyTorch FSDP.

The runner is intentionally independent of Trainer, Accelerate launch, and
DeepSpeed.  Launch it with ``torchrun`` so every process owns one CUDA device::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    .venv-qwen3/bin/torchrun --standalone --nproc-per-node=8 \
      -m baselines.run_full_ft_fsdp \
      --model-dir Qwen3-8B-Base \
      --manifest runs/protocols/wikibigedit_3k_seed42.json \
      --output-dir runs/full_ft_bf_3k_seed42

Each chronological stage is optimized breadth-first: its complete mini-batch
loader is shuffled independently on every epoch.  Prompt tokens are masked,
so the causal-LM objective is computed on answer tokens only.  Checkpoints use
``torch.distributed.checkpoint`` and are sharded across ranks; they are not
ordinary Hugging Face ``from_pretrained`` directories.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_sharded_optimizer_state_dict,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from baselines.common import (
    AnswerOnlyDataset,
    append_jsonl,
    atomic_json,
    environment_metadata,
    evaluate_checkpoint,
    fixed_panel,
    group_by_stage,
    load_manifest,
    load_model,
    load_tokenizer,
    make_collator,
    manifest_panel,
    sha256_file,
)


DEFAULT_CHECKPOINT_DIR = Path(
    "/mnt/yy2607004_hs/engram_checkpoints/full_ft_qwen3_8b_wikibigedit_3k_seed42"
)
EVAL_AXES = ("update", "rephrase", "persona", "locality", "multihop")


@dataclass
class DistributedTrainStats:
    loss: float
    optimizer_steps: int
    sequence_tokens: int
    answer_tokens: int
    wall_seconds: float
    peak_vram_gb: float
    examples_per_epoch_with_sampler_padding: int
    sampler_padding_examples_per_epoch: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--joint-training", action="store_true",
        help="Shuffle all selected facts together; no chronological training or replay",
    )
    parser.add_argument(
        "--fp32-master-weights", action="store_true",
        help="Keep trainable shards and Adam state in FP32 with BF16 computation",
    )
    parser.add_argument(
        "--probe-steps", type=int, default=0,
        help="Memory probe only: stop after N updates, without evaluation/checkpoint",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-format", choices=["fsdp", "hf"], default="fsdp")
    parser.add_argument("--hf-staging-root", default="/dev/shm/engram_joint_hf_staging")
    parser.add_argument("--probe-save-checkpoint", action="store_true")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--sync-every-microbatch", action="store_true",
        help=(
            "Reduce and shard gradients after every microbatch during accumulation; "
            "uses less GPU memory at the cost of additional communication"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument(
        "--eval-axes",
        nargs="+",
        choices=EVAL_AXES,
        default=list(EVAL_AXES),
    )
    parser.add_argument(
        "--locality-panel-size",
        type=int,
        default=0,
        help="0 uses the complete locality panel frozen in the manifest",
    )
    parser.add_argument(
        "--multihop-panel-size",
        type=int,
        default=0,
        help="0 uses the complete multihop panel frozen in the manifest",
    )
    parser.add_argument(
        "--eval-every-stage",
        action="store_true",
        help="Evaluate after every stage; the final selected stage is always evaluated",
    )
    parser.add_argument(
        "--eval-before-training",
        action="store_true",
        help="Also materialize and evaluate the initial base checkpoint",
    )
    parser.add_argument(
        "--save-every-stage",
        action="store_true",
        help="Save every stage; the final selected stage is always saved",
    )
    parser.add_argument(
        "--keep-latest-checkpoint",
        action="store_true",
        help=(
            "Save a resumable checkpoint after every stage and remove the preceding "
            "tauN directory only after the new checkpoint is committed"
        ),
    )
    parser.add_argument(
        "--save-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include sharded AdamW state so a stage-boundary checkpoint is resumable",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--flash-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-stages",
        type=int,
        default=0,
        help="Debug/smoke only: 0 keeps all stages, otherwise keep the first N stages",
    )
    parser.add_argument(
        "--examples-per-stage",
        type=int,
        default=0,
        help="Debug/smoke only: 0 keeps the frozen stream, otherwise take each stage prefix",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--distributed-timeout-hours", type=float, default=12.0)
    parser.add_argument("--allow-existing-output", action="store_true")
    # Accepted for compatibility with torch.distributed.launch and older torchrun.
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.per_device_batch_size <= 0:
        parser.error("--per-device-batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.max_stages < 0 or args.examples_per_stage < 0:
        parser.error("debug limits must be non-negative")
    if args.probe_steps < 0 or args.log_every < 1:
        parser.error("probe steps must be non-negative and log-every positive")
    if args.checkpoint_format == "hf" and (args.save_optimizer or args.resume_checkpoint):
        parser.error("HF final weights require --no-save-optimizer and no FSDP resume checkpoint")
    return args


def setup_distributed(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("launch this runner with torchrun (RANK/WORLD_SIZE are missing)")
    local_rank = (
        args.local_rank
        if args.local_rank is not None
        else int(os.environ.get("LOCAL_RANK", "0"))
    )
    torch.cuda.set_device(local_rank)
    timeout = timedelta(hours=args.distributed_timeout_hours)
    dist.init_process_group(backend="nccl", timeout=timeout)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def limited_stages(
    rows: Sequence[dict[str, Any]],
    *,
    max_stages: int,
    examples_per_stage: int,
) -> dict[int, list[dict[str, Any]]]:
    stages = group_by_stage(rows)
    if max_stages > 0:
        stages = dict(list(stages.items())[:max_stages])
    if examples_per_stage > 0:
        stages = {
            stage: stage_rows[:examples_per_stage]
            for stage, stage_rows in stages.items()
        }
    return {stage: stage_rows for stage, stage_rows in stages.items() if stage_rows}


def axis_accuracy(records: Sequence[dict[str, Any]], axis: str) -> float | None:
    selected = [record for record in records if record["axis"] == axis]
    if not selected:
        return None
    return sum(bool(record["match"]) for record in selected) / len(selected)


def summarize_checkpoint_records(
    records: Sequence[dict[str, Any]], current_stage: int | None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        f"{axis}_accuracy": axis_accuracy(records, axis) for axis in EVAL_AXES
    }
    if current_stage is None:
        return result
    factual = [
        record
        for record in records
        if record["axis"] in {"update", "rephrase", "persona"}
    ]
    current = [
        record for record in factual if int(record["fact_stage"]) == current_stage
    ]
    past = [record for record in factual if int(record["fact_stage"]) < current_stage]
    for axis in ("update", "rephrase", "persona"):
        result[f"current_{axis}_accuracy"] = axis_accuracy(current, axis)
        result[f"past_{axis}_retention"] = axis_accuracy(past, axis)
    result["per_fact_stage"] = {
        str(stage): {
            axis: axis_accuracy(
                [record for record in factual if int(record["fact_stage"]) == stage],
                axis,
            )
            for axis in ("update", "rephrase", "persona")
        }
        for stage in sorted({int(record["fact_stage"]) for record in factual})
    }
    return result


def make_fsdp_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[FSDP, Any]:
    tokenizer = load_tokenizer(args.model_dir)
    base = load_model(
        args.model_dir,
        device,
        use_flash_attention=args.flash_attention,
    )
    if args.fp32_master_weights:
        base.float()
    base.config.use_cache = False
    if args.activation_checkpointing:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3DecoderLayer},
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    wrapped = FSDP(
        base,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        sync_module_states=False,
        limit_all_gathers=True,
        use_orig_params=False,
    )
    return wrapped, tokenizer


def make_optimizer(model: FSDP, args: argparse.Namespace) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    try:
        return torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
        )
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )


def _all_reduce_sum(value: float | int, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _all_reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def joint_answer_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Identical causal answer-only CE without FP32 copies of prompt logits."""
    targets = labels[:, 1:]
    active = targets.ne(-100)
    if not bool(active.any()):
        raise ValueError("Joint batch contains no answer tokens")
    return torch.nn.functional.cross_entropy(
        logits[:, :-1][active].float(), targets[active], reduction="mean"
    )


def train_stage(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    stage: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> DistributedTrainStats:
    dataset = AnswerOnlyDataset(rows, tokenizer, args.max_length)
    if args.joint_training and len(dataset) != len(rows):
        raise ValueError("Joint training must cover every selected fact after tokenization")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed + stage * 1000,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        collate_fn=make_collator(tokenizer),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if not loader:
        raise ValueError(f"stage {stage} produced an empty loader")

    model.train()
    model.module.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    local_loss_sum = 0.0
    local_loss_tokens = 0
    local_sequence_tokens = 0
    local_answer_tokens = 0
    optimizer_steps = 0
    started = time.perf_counter()

    for epoch in range(args.epochs):
        sampler.set_epoch(args.seed + stage * 1000 + epoch)
        num_batches = len(loader)
        for batch_index, batch in enumerate(loader):
            batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            group_start = (
                batch_index // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            group_size = min(
                args.gradient_accumulation_steps, num_batches - group_start
            )
            is_boundary = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == num_batches
            )
            # FSDP no_sync retains full, unsharded gradients between microbatches.
            # Synchronizing each microbatch instead accumulates reduced shards;
            # loss scaling and optimizer update boundaries stay the same.
            sync_context = (
                nullcontext()
                if is_boundary or args.sync_every_microbatch
                else model.no_sync()
            )
            answer_count = int((batch["labels"] != -100).sum().item())
            with sync_context:
                if args.joint_training:
                    outputs = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"], use_cache=False)
                    loss = joint_answer_loss(outputs.logits, batch["labels"])
                else:
                    outputs = model(**batch, use_cache=False)
                    loss = outputs.loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training loss")
                # Match the global answer-token objective even when ranks have
                # unequal answer lengths. FSDP averages gradients across ranks.
                scale = 1.0
                if args.joint_training:
                    global_answer_count = _all_reduce_sum(answer_count, device)
                    scale = world_size * answer_count / max(global_answer_count, 1)
                (loss * scale / group_size).backward()
            local_loss_sum += float(loss.detach()) * answer_count
            local_loss_tokens += answer_count
            local_sequence_tokens += int(batch["attention_mask"].sum().item())
            local_answer_tokens += answer_count
            if is_boundary:
                grad_norm = model.clip_grad_norm_(args.max_grad_norm)
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError("Nonfinite gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if rank == 0 and (optimizer_steps == 1 or optimizer_steps % args.log_every == 0):
                    progress = {
                        "status": "probing" if args.probe_steps else "training",
                        "stage": stage, "epoch": epoch + 1,
                        "optimizer_steps": optimizer_steps,
                        "loss": float(loss.detach()),
                        "grad_norm": float(grad_norm),
                        "peak_vram_gb_rank0": torch.cuda.max_memory_allocated(device) / 1024**3,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    atomic_json(args.output_dir / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                del outputs, loss
                if args.probe_steps and optimizer_steps >= args.probe_steps:
                    break
        if args.probe_steps and optimizer_steps >= args.probe_steps:
            break

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    global_loss_sum = _all_reduce_sum(local_loss_sum, device)
    global_loss_tokens = int(_all_reduce_sum(local_loss_tokens, device))
    global_sequence_tokens = int(_all_reduce_sum(local_sequence_tokens, device))
    global_answer_tokens = int(_all_reduce_sum(local_answer_tokens, device))
    max_elapsed = _all_reduce_max(elapsed, device)
    max_peak = _all_reduce_max(peak, device)
    # Every rank has the same loader length and therefore the same step count.
    total_per_epoch = sampler.num_samples * world_size
    return DistributedTrainStats(
        loss=global_loss_sum / max(global_loss_tokens, 1),
        optimizer_steps=optimizer_steps,
        sequence_tokens=global_sequence_tokens,
        answer_tokens=global_answer_tokens,
        wall_seconds=max_elapsed,
        peak_vram_gb=max_peak,
        examples_per_epoch_with_sampler_padding=total_per_epoch,
        sampler_padding_examples_per_epoch=total_per_epoch - len(dataset),
    )


def save_hf_checkpoint(model, destination, *, stage, args, manifest_sha256, rank):
    """Save CPU full weights without NCCL checkpoint-planner object gathers."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        full_state = model.state_dict()
    if rank == 0:
        if destination.exists():
            raise FileExistsError(f"checkpoint already exists: {destination}")
        token = uuid.uuid4().hex
        staging = Path(args.hf_staging_root) / (args.output_dir.name + "-" + token)
        staging.mkdir(parents=True, exist_ok=False)
        model.module.save_pretrained(staging, state_dict=full_state,
                                     safe_serialization=True, max_shard_size="4GB")
        load_tokenizer(args.model_dir).save_pretrained(staging)
        roundtrip_verified = False
        if args.probe_steps:
            restored = AutoModelForCausalLM.from_pretrained(
                staging, torch_dtype=torch.float32, local_files_only=True,
                low_cpu_mem_usage=True, attn_implementation="sdpa")
            loaded = restored.state_dict()
            if set(loaded) != set(full_state):
                raise ValueError("HF checkpoint parameter names do not match")
            for name, value in full_state.items():
                if not torch.equal(value.cpu(), loaded[name].cpu()):
                    raise ValueError(f"HF checkpoint changed parameter {name}")
            roundtrip_verified = True
            del loaded, restored
        metadata = {
            "schema_version": "engram-hf-joint-checkpoint/v1", "format": "huggingface-safetensors",
            "stage": stage, "world_size": dist.get_world_size(),
            "model_dir": str(args.model_dir.resolve()), "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_sha256, "has_optimizer": False,
            "master_dtype": "float32", "compute_dtype": "bfloat16",
            "joint_training": args.joint_training, "hf_roundtrip_verified": roundtrip_verified,
            "created_unix_time": time.time(),
        }
        atomic_json(staging / "checkpoint_meta.json", metadata)
        # Keep a complete local weights backup before durable storage copy.
        atomic_json(args.output_dir / "hf_staging_checkpoint.json", {
            "path": str(staging), "complete": True, "parameter_tensor_count": len(full_state),
            "durable_destination": str(destination), "hf_roundtrip_verified": roundtrip_verified,
        })
        temporary = destination.with_name(destination.name + ".tmp-" + token)
        shutil.copytree(staging, temporary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
        print(json.dumps({"saved_hf_checkpoint": str(destination),
                          "hf_roundtrip_verified": roundtrip_verified}), flush=True)
    del full_state
    torch.cuda.empty_cache()
    dist.barrier()


def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    destination: Path,
    *,
    stage: int,
    args: argparse.Namespace,
    manifest_sha256: str,
    rank: int,
) -> None:
    if getattr(args, "checkpoint_format", "fsdp") == "hf":
        return save_hf_checkpoint(model, destination, stage=stage, args=args,
                                  manifest_sha256=manifest_sha256, rank=rank)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    token = str(uuid.uuid4()) if rank == 0 else ""
    objects: list[str] = [token]
    dist.broadcast_object_list(objects, src=0)
    temporary = destination.with_name(destination.name + f".tmp-{objects[0]}")
    if rank == 0:
        if destination.exists():
            raise FileExistsError(f"checkpoint already exists: {destination}")
        temporary.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    options = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, options,
                             ShardedOptimStateDictConfig(offload_to_cpu=True)):
        model_state = model.state_dict()
        state: dict[str, Any] = {"model": model_state}
        if args.save_optimizer:
            state["optimizer"] = FSDP.optim_state_dict(model, optimizer)
        save_state_dict(
            state,
            storage_writer=FileSystemWriter(str(temporary)),
            process_group=dist.group.WORLD,
        )
    dist.barrier()
    if rank == 0:
        atomic_json(
            temporary / "checkpoint_meta.json",
            {
                "schema_version": "engram-fsdp-checkpoint/v1",
                "format": "torch.distributed.checkpoint",
                "stage": stage,
                "world_size": dist.get_world_size(),
                "model_dir": str(args.model_dir.resolve()),
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": manifest_sha256,
                "has_optimizer": args.save_optimizer,
                "bf16": True,
                "answer_loss_only": True,
                "joint_training": args.joint_training,
                "fp32_master_weights": args.fp32_master_weights,
                "created_unix_time": time.time(),
            },
        )
        os.replace(temporary, destination)
    dist.barrier()


def load_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    source: Path,
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    metadata = json.loads((source / "checkpoint_meta.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "engram-fsdp-checkpoint/v1":
        raise ValueError(f"unsupported checkpoint metadata in {source}")
    if metadata.get("manifest_sha256") != manifest_sha256:
        raise ValueError("resume checkpoint was created from a different manifest")
    reader = FileSystemReader(str(source))
    options = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, options):
        state: dict[str, Any] = {"model": model.state_dict()}
        load_state_dict(
            state,
            storage_reader=reader,
            process_group=dist.group.WORLD,
        )
        model.load_state_dict(state["model"])
        if metadata.get("has_optimizer"):
            optimizer_state = load_sharded_optimizer_state_dict(
                model_state_dict=state["model"],
                optimizer_key="optimizer",
                storage_reader=reader,
            )["optimizer"]
            flattened = FSDP.optim_state_dict_to_load(
                model, optimizer, optimizer_state
            )
            optimizer.load_state_dict(flattened)
    dist.barrier()
    return metadata


def remove_previous_checkpoint(previous: Path, checkpoint_root: Path) -> None:
    """Remove one superseded runner-owned tau checkpoint with strict guards."""

    resolved = previous.resolve()
    root = checkpoint_root.resolve()
    if resolved.parent != root or not resolved.name.startswith("tau"):
        raise ValueError(f"refusing to prune checkpoint outside {root}: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def evaluate_fsdp_checkpoint(
    model: FSDP,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    locality_panel: Sequence[dict[str, Any]],
    multihop_panel: Sequence[dict[str, Any]],
    *,
    checkpoint: str,
    current_stage: int | None,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
) -> dict[str, Any] | None:
    """Gather on rank 0, evaluate with an ordinary HF model, then release it.

    Calling ``model.module.generate`` directly would bypass the root FSDP
    hooks.  A rank-0 materialization is slower to construct but gives exactly
    the same generation/evaluator path used by the other baselines.
    """

    model.eval()
    full_options = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_options):
        full_state = model.state_dict()
    result: dict[str, Any] | None = None
    if rank == 0:
        config = copy.deepcopy(model.module.config)
        config.use_cache = True
        # ``from_config`` otherwise allocates fp32 parameters first, which adds
        # ~32 GB of host traffic for Qwen3-8B before the later bf16 conversion.
        eval_model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
        incompatible = eval_model.load_state_dict(full_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"full-state materialization mismatch: {incompatible}")
        del full_state
        eval_model.to(device=device, dtype=torch.bfloat16)
        eval_model.eval()
        raw, records = evaluate_checkpoint(
            eval_model,
            tokenizer,
            rows,
            device,
            checkpoint=checkpoint,
            axes=args.eval_axes,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            locality_panel=locality_panel,
            multihop_panel=multihop_panel,
        )
        append_jsonl(args.output_dir / "predictions.jsonl", records)
        result = {
            **raw,
            **summarize_checkpoint_records(records, current_stage),
            "current_stage": current_stage,
        }
        del eval_model
        torch.cuda.empty_cache()
    # Nonzero ranks may wait for a long rank-0 evaluation; process-group timeout
    # is intentionally configurable and defaults to twelve hours.
    dist.barrier()
    return result


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list in {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rank = world_size = local_rank = -1
    distributed_started = False
    try:
        rank, world_size, local_rank, device = setup_distributed(args)
        distributed_started = True
        seed_everything(args.seed, rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        rows, manifest_metadata = load_manifest(args.manifest)
        stages = limited_stages(
            rows,
            max_stages=args.max_stages,
            examples_per_stage=args.examples_per_stage,
        )
        if not stages:
            raise ValueError("empty selection after applying debug limits")
        selected_rows = [row for stage_rows in stages.values() for row in stage_rows]
        final_stage = max(stages)
        original_stage_counts = {str(k): len(v) for k, v in stages.items()}
        if args.joint_training:
            stages = {final_stage: selected_rows}
        manifest_hash = sha256_file(args.manifest)
        locality_panel = manifest_panel(
            manifest_metadata, "locality", limit=args.locality_panel_size
        ) or fixed_panel(
            selected_rows,
            "loc",
            "loc_ans",
            args.locality_panel_size,
            args.seed + 1001,
        )
        multihop_panel = manifest_panel(
            manifest_metadata, "mhop", limit=args.multihop_panel_size
        ) or fixed_panel(
            selected_rows,
            "mhop",
            "mhop_ans",
            args.multihop_panel_size,
            args.seed + 2001,
        )

        if rank == 0:
            nonempty = args.output_dir.exists() and any(args.output_dir.iterdir())
            if nonempty and not (
                args.allow_existing_output or args.resume_checkpoint is not None
            ):
                raise FileExistsError(
                    f"non-empty output directory: {args.output_dir}; "
                    "pass --allow-existing-output to reuse"
                )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        model, tokenizer = make_fsdp_model(args, device)
        optimizer = make_optimizer(model, args)
        if rank == 0:
            tokenizer.save_pretrained(args.output_dir / "tokenizer")
            run_config = {
                **vars(args),
                "model_dir": str(args.model_dir.resolve()),
                "manifest": str(args.manifest.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "checkpoint_dir": str(args.checkpoint_dir.resolve()),
                "resume_checkpoint": (
                    str(args.resume_checkpoint.resolve())
                    if args.resume_checkpoint is not None
                    else None
                ),
                "manifest_sha256": manifest_hash,
                "manifest_protocol_id": manifest_metadata.get("protocol_id"),
                "num_facts": len(selected_rows),
                "stage_counts": {
                    str(stage): len(stage_rows) for stage, stage_rows in stages.items()
                },
                "original_stage_counts": original_stage_counts,
                "world_size": world_size,
                "global_micro_batch_size": args.per_device_batch_size * world_size,
                "effective_global_batch_size": (
                    args.per_device_batch_size
                    * world_size
                    * args.gradient_accumulation_steps
                ),
                "locality_panel_size_actual": len(locality_panel),
                "multihop_panel_size_actual": len(multihop_panel),
                "closed_book_evaluation": True,
                "prompt_template": "Question: <question>\\nAnswer:",
                "answer_loss_only": True,
                "stage_training_order": (
                    "all facts jointly shuffled each epoch; no replay or continual learning"
                    if args.joint_training else "breadth-first mini-batch epochs"
                ),
                "training_fields": ["update", "ans"],
                "master_parameter_dtype": str(next(model.parameters()).dtype),
                "source_sha256": sha256_file(Path(__file__)),
                "optimizer_continues_across_stages": True,
                "fsdp_sharding_strategy": "FULL_SHARD",
                "fsdp_auto_wrap_class": "Qwen3DecoderLayer",
                "checkpoint_format": "torch.distributed.checkpoint sharded state dict",
                **environment_metadata(),
            }
            atomic_json(args.output_dir / "run_config.json", run_config)

        resume_stage = 0
        if args.resume_checkpoint is not None:
            resume_metadata = load_checkpoint(
                model,
                optimizer,
                args.resume_checkpoint,
                manifest_sha256=manifest_hash,
            )
            resume_stage = int(resume_metadata["stage"])
            if resume_stage not in stages:
                raise ValueError(
                    f"resume stage {resume_stage} is outside the selected stage stream"
                )

        metrics = (
            load_json_list(args.output_dir / "metrics.json") if rank == 0 else []
        )
        train_log = (
            load_json_list(args.output_dir / "training_log.json") if rank == 0 else []
        )
        if args.eval_before_training and resume_stage == 0:
            base_result = evaluate_fsdp_checkpoint(
                model,
                tokenizer,
                selected_rows,
                locality_panel,
                multihop_panel,
                checkpoint="base",
                current_stage=None,
                args=args,
                rank=rank,
                device=device,
            )
            if rank == 0:
                assert base_result is not None
                metrics.append(base_result)
                atomic_json(args.output_dir / "metrics.json", metrics)

        run_started = time.time()
        previous_checkpoint: Path | None = None
        if args.resume_checkpoint is not None:
            candidate = args.resume_checkpoint.resolve()
            if candidate.parent == args.checkpoint_dir.resolve():
                previous_checkpoint = candidate
        for stage, current_rows in stages.items():
            if stage <= resume_stage:
                continue
            stats = train_stage(
                model,
                optimizer,
                tokenizer,
                current_rows,
                stage,
                args,
                rank,
                world_size,
                device,
            )
            stage_log = {
                "stage": stage,
                "current_count": len(current_rows),
                "epochs": args.epochs,
                "per_device_batch_size": args.per_device_batch_size,
                "effective_global_batch_size": (
                    args.per_device_batch_size
                    * world_size
                    * args.gradient_accumulation_steps
                ),
                **asdict(stats),
            }
            if rank == 0:
                train_log.append(stage_log)
                atomic_json(args.output_dir / "training_log.json", train_log)

            if args.probe_steps:
                if args.probe_save_checkpoint:
                    save_checkpoint(model, optimizer, args.checkpoint_dir / "probe",
                        stage=stage, args=args, manifest_sha256=manifest_hash, rank=rank)
                if rank == 0:
                    atomic_json(args.output_dir / "probe_complete.json", {
                        "status": "probe_complete", "world_size": world_size,
                        "num_facts": len(selected_rows),
                        "checkpoint_saved": args.probe_save_checkpoint, **stage_log,
                    })
                dist.barrier()
                return

            if (
                args.save_every_stage
                or args.keep_latest_checkpoint
                or stage == final_stage
            ):
                new_checkpoint = args.checkpoint_dir / f"tau{stage}"
                save_checkpoint(
                    model,
                    optimizer,
                    new_checkpoint,
                    stage=stage,
                    args=args,
                    manifest_sha256=manifest_hash,
                    rank=rank,
                )
                if args.keep_latest_checkpoint and previous_checkpoint is not None:
                    if rank == 0 and previous_checkpoint != new_checkpoint.resolve():
                        remove_previous_checkpoint(
                            previous_checkpoint, args.checkpoint_dir
                        )
                    dist.barrier()
                previous_checkpoint = new_checkpoint.resolve()

            if args.eval_every_stage or stage == final_stage:
                seen_rows = [
                    row
                    for seen_stage, stage_rows in stages.items()
                    if seen_stage <= stage
                    for row in stage_rows
                ]
                stage_result = evaluate_fsdp_checkpoint(
                    model,
                    tokenizer,
                    seen_rows,
                    locality_panel,
                    multihop_panel,
                    checkpoint=f"tau{stage}",
                    current_stage=stage,
                    args=args,
                    rank=rank,
                    device=device,
                )
                if rank == 0:
                    assert stage_result is not None
                    stage_result["train"] = stage_log
                    metrics.append(stage_result)
                    atomic_json(args.output_dir / "metrics.json", metrics)

        # A resume of an already-final checkpoint still gets a final evaluation.
        if resume_stage == final_stage:
            final_result = evaluate_fsdp_checkpoint(
                model,
                tokenizer,
                selected_rows,
                locality_panel,
                multihop_panel,
                checkpoint=f"tau{final_stage}-resumed",
                current_stage=final_stage,
                args=args,
                rank=rank,
                device=device,
            )
            if rank == 0:
                assert final_result is not None
                metrics.append(final_result)
                atomic_json(args.output_dir / "metrics.json", metrics)

        if rank == 0:
            final_checkpoint = (
                args.resume_checkpoint
                if resume_stage == final_stage and args.resume_checkpoint is not None
                else args.checkpoint_dir / f"tau{final_stage}"
            )
            atomic_json(
                args.output_dir / "completed.json",
                {
                    "status": "complete",
                    "method": "joint_full_ft_fsdp" if args.joint_training else "full_ft_bf_fsdp",
                    "num_facts": len(selected_rows),
                    "joint_training": args.joint_training,
                    "elapsed_seconds": time.time() - run_started,
                    "final_stage": final_stage,
                    "world_size": world_size,
                    "final_checkpoint": str(final_checkpoint.resolve()),
                },
            )
        dist.barrier()
    finally:
        if distributed_started and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
