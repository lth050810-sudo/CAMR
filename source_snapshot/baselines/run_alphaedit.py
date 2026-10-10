#!/usr/bin/env python3
"""Run an auditable Qwen3-8B AlphaEdit transfer on WikiBigEdit.

There is no tuned AlphaEdit preset for Qwen3-8B-Base.  Defaults transfer the
EasyEdit Qwen2.5 layers and optimization constants, while moving its final
loss layer from Qwen2.5 block 27 to Qwen3 block 35.  The run is therefore
always labelled *untuned transfer*.  It requires projection artifacts built
from this exact Qwen3 checkpoint by ``build_alphaedit_projection``; a bare or
Qwen2.5 ``null_space_project.pt`` file is rejected.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from baselines.alphaedit_core import (
    EASYEDIT_COMMIT,
    EASYEDIT_REPOSITORY,
    LICENSE_PROVENANCE,
    PROJECTION_SCHEMA,
    ProjectedHistory,
    UPSTREAM_COMMIT,
    UPSTREAM_QWEN25_PRESET,
    UPSTREAM_REPOSITORY,
    load_projection_payload,
    module_name_for_layer,
    projection_filename,
)
from baselines.collect_memit_stats import atomic_torch_save, file_sha256, model_fingerprint
from baselines.common import (
    append_jsonl,
    atomic_json,
    environment_metadata,
    fixed_panel,
    group_by_stage,
    load_manifest,
    load_model,
    load_tokenizer,
    manifest_panel,
    sha256_file,
)
from baselines.run_lora_family import evaluate_and_write
from baselines.run_memit import (
    ZOptimizationStats,
    build_context_templates,
    chunks,
    compute_current_zs,
    compute_keys,
    compute_z_star,
    flatten_context_templates,
    get_module,
    make_request,
    model_stats_key,
    parse_int_range,
    should_evaluate_stage,
)
from baselines.run_ultraedit import restrict_stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layers",
        type=parse_int_range,
        default=parse_int_range("4-8"),
        help="untuned exact-index transfer of EasyEdit's Qwen2.5 layers",
    )
    parser.add_argument(
        "--v-loss-layer",
        type=int,
        default=35,
        help="Qwen3 final block; Qwen2.5's source preset used final block 27",
    )
    parser.add_argument("--v-num-grad-steps", type=int, default=25)
    parser.add_argument("--v-lr", type=float, default=5e-1)
    parser.add_argument("--v-weight-decay", type=float, default=1e-3)
    parser.add_argument("--clamp-norm-factor", type=float, default=4.0)
    parser.add_argument("--kl-factor", type=float, default=0.0625)
    parser.add_argument("--nullspace-threshold", type=float, default=2e-2)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--edit-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--representation-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--locality-panel-size", type=int, default=0)
    parser.add_argument("--multihop-panel-size", type=int, default=0)
    parser.add_argument("--max-stages", type=int, default=0)
    parser.add_argument("--examples-per-stage", type=int, default=0)
    parser.add_argument("--max-edits", type=int, default=0)
    parser.add_argument("--no-stage-eval", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument(
        "--context-mode", choices=("upstream_generated", "static"), default="upstream_generated"
    )
    parser.add_argument(
        "--projection-dir",
        type=Path,
        default=Path("/mnt/yy2607004_hs/engram_alphaedit_stats"),
    )
    parser.add_argument(
        "--projection-file-template",
        type=str,
        default=None,
        help="optional Python format string containing {layer} and {module}",
    )
    parser.add_argument("--projection-sample-size", type=int, default=100000)
    parser.add_argument(
        "--model-fingerprint-mode", choices=("content", "metadata"), default="content"
    )
    parser.add_argument("--allow-diagnostic-projection", action="store_true")
    parser.add_argument(
        "--preset-label",
        default="qwen25-exact-layer-indices-final-block-transfer-untuned",
    )
    parser.add_argument("--save-delta", type=Path, default=None)
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_total_edits(
    stages: dict[int, list[dict[str, Any]]], max_edits: int
) -> dict[int, list[dict[str, Any]]]:
    if max_edits <= 0:
        return stages
    result: dict[int, list[dict[str, Any]]] = {}
    remaining = max_edits
    for stage, rows in stages.items():
        if remaining <= 0:
            break
        selected = rows[:remaining]
        if selected:
            result[stage] = selected
            remaining -= len(selected)
    return result


class ProjectionProvider:
    def __init__(
        self,
        model: torch.nn.Module,
        model_dir: Path,
        device: torch.device,
        args: argparse.Namespace,
    ) -> None:
        self.model = model
        self.model_dir = model_dir.resolve()
        self.device = device
        self.args = args
        self.cache: dict[int, torch.Tensor] = {}
        self.audit: dict[str, dict[str, Any]] = {}
        self.model_fingerprint = model_fingerprint(
            self.model_dir, args.model_fingerprint_mode
        )
        self.config_sha256 = file_sha256(self.model_dir / "config.json")

    def _resolve(self, layer: int) -> Path:
        module_name = module_name_for_layer(layer)
        if self.args.projection_file_template:
            path = Path(
                self.args.projection_file_template.format(
                    layer=layer, module=module_name
                )
            )
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        model_key = model_stats_key(self.model_dir)
        candidates = [
            self.args.projection_dir / model_key / projection_filename(layer),
            self.args.projection_dir / projection_filename(layer),
        ]
        for path in candidates:
            if path.exists():
                return path
        rendered = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            "Qwen3 AlphaEdit projection is missing. Expected one of:\n" + rendered
        )

    def get(self, layer: int) -> torch.Tensor:
        if layer in self.cache:
            return self.cache[layer]
        module_name = module_name_for_layer(layer)
        module = get_module(self.model, module_name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"{module_name} is not nn.Linear")
        path = self._resolve(layer)
        basis, metadata = load_projection_payload(
            path,
            layer=layer,
            module_name=module_name,
            input_dimension=module.in_features,
            nullspace_threshold=self.args.nullspace_threshold,
            model_fingerprint=self.model_fingerprint,
            model_config_sha256=self.config_sha256,
            sample_size=self.args.projection_sample_size,
            allow_diagnostic=self.args.allow_diagnostic_projection,
        )
        if metadata.get("model_fingerprint_mode") != self.args.model_fingerprint_mode:
            raise ValueError(
                "projection model fingerprint mode differs from the requested verification mode"
            )
        self.audit[str(layer)] = {
            **metadata,
            "path": str(path.resolve()),
            "artifact_sha256": file_sha256(path),
        }
        self.cache[layer] = basis.to(device=self.device, dtype=torch.float32)
        return self.cache[layer]


@dataclass
class AlphaEditBatchStats:
    stage: int
    batch_index: int
    sample_ids: list[str]
    wall_seconds: float
    layer_update_norms: dict[str, float]
    layer_weight_norms: dict[str, float]
    layer_nullities: dict[str, int]
    history_key_counts: dict[str, int]
    history_key_norms: dict[str, float]
    history_cholesky_retry_counts: dict[str, int]
    history_max_cholesky_jitters: dict[str, float]
    z_optimizations: list[dict[str, Any]]


def validate_edit_context_lengths(
    tokenizer: Any,
    requests: Sequence[dict[str, Any]],
    context_groups: Sequence[Sequence[str]],
    max_length: int,
) -> None:
    """Reject silent right-truncation of AlphaEdit's target-bearing prompts."""

    contexts = flatten_context_templates(context_groups)
    for request in requests:
        target_ids = tokenizer.encode(
            request["target_new"], add_special_tokens=False
        )
        special_ids = {tokenizer.bos_token_id, tokenizer.unk_token_id}
        if target_ids and target_ids[0] in {
            value for value in special_ids if value is not None
        }:
            target_ids = target_ids[1:]
        target_prefix = tokenizer.decode(target_ids[:-1], skip_special_tokens=False)
        for context in contexts:
            template = context.format(request["prompt"])
            text = template.format(request["subject"]) + target_prefix
            token_count = len(tokenizer.encode(text, add_special_tokens=True))
            if token_count > max_length:
                raise ValueError(
                    f"AlphaEdit context for {request['sample_id']} needs {token_count} "
                    f"tokens but --max-length is {max_length}; increase the limit rather "
                    "than silently truncating target supervision"
                )


def apply_alphaedit_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[dict[str, Any]],
    context_groups: Sequence[Sequence[str]],
    projections: ProjectionProvider,
    histories: dict[int, ProjectedHistory],
    device: torch.device,
    args: argparse.Namespace,
    *,
    stage: int,
    batch_index: int,
    z_function=None,
) -> AlphaEditBatchStats:
    """Apply one upstream-equivalent AlphaEdit batch in null-basis form."""

    started = time.perf_counter()
    z_layer = args.layers[-1]
    validate_edit_context_lengths(tokenizer, requests, context_groups, args.max_length)
    targets: list[torch.Tensor] = []
    optimization_stats: list[ZOptimizationStats] = []
    for request in requests:
        target, stats = (z_function or compute_z_star)(
            model,
            tokenizer,
            request,
            context_groups,
            z_layer,
            args.v_loss_layer,
            device,
            args,
        )
        targets.append(target)
        optimization_stats.append(stats)
    z_targets = torch.stack(targets, dim=1)

    update_norms: dict[str, float] = {}
    weight_norms: dict[str, float] = {}
    nullities: dict[str, int] = {}
    for layer_offset, layer in enumerate(args.layers):
        module = get_module(model, module_name_for_layer(layer))
        basis = projections.get(layer)
        keys = compute_keys(
            model,
            tokenizer,
            requests,
            layer,
            context_groups,
            device,
            batch_size=args.representation_batch_size,
            max_length=args.max_length,
        ).T
        current_zs = compute_current_zs(
            model,
            tokenizer,
            requests,
            z_layer,
            device,
            batch_size=args.representation_batch_size,
            max_length=args.max_length,
        ).T
        residual = (z_targets.to(current_zs) - current_zs) / (
            len(args.layers) - layer_offset
        )
        history = histories.setdefault(
            layer,
            ProjectedHistory.empty(basis.shape[1], args.l2, device),
        )
        projected_keys = basis.T @ keys.float()
        adjusted_keys = history.adjusted_keys(projected_keys)
        # Associate the products through the usually small edit-batch axis.
        # This is identical to (R @ adjusted.T) @ U.T but avoids constructing
        # and multiplying an output_dim x nullity intermediate.
        adjusted_original_space = basis @ adjusted_keys.to(basis)
        update = residual.float() @ adjusted_original_space.T
        if tuple(update.shape) != tuple(module.weight.shape):
            if tuple(update.T.shape) == tuple(module.weight.shape):
                update = update.T
            else:
                raise ValueError(
                    f"AlphaEdit update {tuple(update.shape)} does not match "
                    f"{module_name_for_layer(layer)} weight {tuple(module.weight.shape)}"
                )
        update_norms[str(layer)] = float(torch.linalg.vector_norm(update))
        weight_norms[str(layer)] = float(torch.linalg.vector_norm(module.weight.float()))
        nullities[str(layer)] = int(basis.shape[1])
        with torch.no_grad():
            module.weight.add_(update.to(dtype=module.weight.dtype))

    # Match upstream AlphaEdit: cache keys are recomputed after all selected
    # layer updates in the batch, then used to protect prior edits.
    history_counts: dict[str, int] = {}
    history_norms: dict[str, float] = {}
    history_retry_counts: dict[str, int] = {}
    history_max_jitters: dict[str, float] = {}
    with torch.no_grad():
        for layer in args.layers:
            basis = projections.get(layer)
            keys = compute_keys(
                model,
                tokenizer,
                requests,
                layer,
                context_groups,
                device,
                batch_size=args.representation_batch_size,
                max_length=args.max_length,
            ).T.float()
            projected = basis.T @ keys
            histories[layer].append(projected)
            history_counts[str(layer)] = histories[layer].count
            history_norms[str(layer)] = float(
                torch.linalg.vector_norm(histories[layer].keys)
            )
            history_retry_counts[str(layer)] = histories[layer].cholesky_retry_count
            history_max_jitters[str(layer)] = histories[layer].max_cholesky_jitter

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return AlphaEditBatchStats(
        stage=stage,
        batch_index=batch_index,
        sample_ids=[request["sample_id"] for request in requests],
        wall_seconds=time.perf_counter() - started,
        layer_update_norms=update_norms,
        layer_weight_norms=weight_norms,
        layer_nullities=nullities,
        history_key_counts=history_counts,
        history_key_norms=history_norms,
        history_cholesky_retry_counts=history_retry_counts,
        history_max_cholesky_jitters=history_max_jitters,
        z_optimizations=[asdict(stats) for stats in optimization_stats],
    )


def validate_qwen3_model(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError(f"runner is scoped to Qwen3, got {model.config.model_type!r}")
    layer_count = int(model.config.num_hidden_layers)
    if not args.layers or min(args.layers) < 0 or max(args.layers) >= layer_count:
        raise ValueError(f"rewrite layers must lie in [0, {layer_count - 1}]")
    if args.v_loss_layer < max(args.layers) or args.v_loss_layer >= layer_count:
        raise ValueError("v-loss-layer must be at or after every rewrite layer")
    for layer in args.layers:
        module = get_module(model, module_name_for_layer(layer))
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"layer {layer} down_proj is not nn.Linear")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(args.output_dir)
    if args.edit_batch_size <= 0 or args.projection_sample_size <= 0:
        raise ValueError("edit batch size and projection sample size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    all_rows, manifest_metadata = load_manifest(args.manifest)
    stages = restrict_stages(
        group_by_stage(all_rows),
        max_stages=args.max_stages,
        examples_per_stage=args.examples_per_stage,
    )
    stages = limit_total_edits(stages, args.max_edits)
    selected_rows = [row for stage_rows in stages.values() for row in stage_rows]
    if not selected_rows:
        raise ValueError("no selected WikiBigEdit rows")
    request_alignment = Counter(
        make_request(row, index)["subject_match"]
        for index, row in enumerate(selected_rows)
    )
    locality_panel = manifest_panel(
        manifest_metadata, "locality", limit=args.locality_panel_size
    ) or fixed_panel(
        all_rows, "loc", "loc_ans", args.locality_panel_size, args.seed + 1001
    )
    multihop_panel = manifest_panel(
        manifest_metadata, "mhop", limit=args.multihop_panel_size
    ) or fixed_panel(
        all_rows, "mhop", "mhop_ans", args.multihop_panel_size, args.seed + 2001
    )

    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)
    validate_qwen3_model(model, args)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    context_groups = build_context_templates(model, tokenizer, device, args.context_mode)
    projections = ProjectionProvider(model, args.model_dir, device, args)
    # Materialize and validate every Qwen3 cache before applying the first edit.
    for layer in args.layers:
        projections.get(layer)
    histories: dict[int, ProjectedHistory] = {}

    original_weights: dict[int, torch.Tensor] | None = None
    if args.save_delta is not None:
        original_weights = {
            layer: get_module(model, module_name_for_layer(layer))
            .weight.detach().cpu().clone()
            for layer in args.layers
        }

    formal_projection = all(
        entry.get("formal_baseline_eligible") is True
        for entry in projections.audit.values()
    )
    reproduction_status = (
        "qwen3_transfer_preset_untuned"
        if formal_projection
        else "diagnostic_qwen3_transfer_projection"
    )
    config = {
        **vars(args),
        "layers": list(args.layers),
        "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "projection_dir": str(args.projection_dir.resolve()),
        "save_delta": str(args.save_delta.resolve()) if args.save_delta else None,
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_id": manifest_metadata.get("protocol_id"),
        "method": "AlphaEdit",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "easyedit_repository": EASYEDIT_REPOSITORY,
        "easyedit_commit": EASYEDIT_COMMIT,
        "projection_schema": PROJECTION_SCHEMA,
        "upstream_qwen25_preset": UPSTREAM_QWEN25_PRESET,
        "license_provenance": LICENSE_PROVENANCE,
        "reproduction_status": reproduction_status,
        "formal_baseline_eligible": formal_projection,
        "formal_baseline_qualification": (
            "untuned Qwen3 transfer with checkpoint-specific 100K Wikipedia null spaces"
            if formal_projection
            else "diagnostic/custom Qwen3 projection"
        ),
        "qwen3_layer_selection_status": (
            "untuned exact layer-index transfer from Qwen2.5-7B; "
            "loss target moved from source final block 27 to Qwen3 final block 35"
        ),
        "port_differences_from_upstream": {
            "model": "Qwen3-8B-Base instead of the Qwen2.5-7B-Instruct source preset",
            "context_generation": (
                "Hugging Face generate(max_new_tokens=10) regenerates checkpoint-specific "
                "contexts; upstream generate_fast(max_out_len=10) is not token-identical"
            ),
            "sequence_limit": (
                f"port explicitly truncates edit/evaluation inputs at {args.max_length} "
                "tokens; the frozen protocol examples fit this configured bound"
            ),
            "linear_solver": (
                "orthonormal null-basis plus float64 Woodbury/history Cholesky instead "
                "of materializing the dense projector; bounded roundoff-only diagonal "
                "jitter retries are recorded per batch"
            ),
        },
        "preset_label": args.preset_label,
        "projection_audit": projections.audit,
        "subject_alignment_counts": dict(request_alignment),
        "edit_prompt_template": "Question: <question-with-{subject}>\\nAnswer:",
        "closed_book_evaluation": True,
        "evaluation_prompt_template": "Question: <question>\\nAnswer:",
        "context_templates": context_groups,
        "num_facts": len(selected_rows),
        "stage_counts": {str(stage): len(stage_rows) for stage, stage_rows in stages.items()},
        "locality_panel_size_actual": len(locality_panel),
        "multihop_panel_size_actual": len(multihop_panel),
        **environment_metadata(),
    }
    atomic_json(args.output_dir / "run_config.json", config)
    atomic_json(args.output_dir / "projection_audit.json", projections.audit)

    metrics: list[dict[str, Any]] = []
    if not args.skip_base_eval:
        metrics.append(
            evaluate_and_write(
                model,
                tokenizer,
                selected_rows,
                locality_panel,
                multihop_panel,
                device,
                args,
                checkpoint="base",
                current_stage=None,
            )
        )
        atomic_json(args.output_dir / "metrics.json", metrics)

    case_id = 0
    for stage, stage_rows in stages.items():
        for batch_index, batch_rows in enumerate(chunks(stage_rows, args.edit_batch_size)):
            requests = [
                make_request(row, case_id + offset)
                for offset, row in enumerate(batch_rows)
            ]
            case_id += len(requests)
            batch_stats = apply_alphaedit_batch(
                model,
                tokenizer,
                requests,
                context_groups,
                projections,
                histories,
                device,
                args,
                stage=stage,
                batch_index=batch_index,
            )
            append_jsonl(args.output_dir / "edit_trace.jsonl", [asdict(batch_stats)])
        if should_evaluate_stage(
            stage, list(stages), no_stage_eval=args.no_stage_eval
        ):
            seen_rows = [
                row
                for seen_stage, seen_stage_rows in stages.items()
                if seen_stage <= stage
                for row in seen_stage_rows
            ]
            metrics.append(
                evaluate_and_write(
                    model,
                    tokenizer,
                    seen_rows,
                    locality_panel,
                    multihop_panel,
                    device,
                    args,
                    checkpoint=f"stage_{stage}",
                    current_stage=stage,
                )
            )
            atomic_json(args.output_dir / "metrics.json", metrics)

    if args.save_delta is not None:
        assert original_weights is not None
        delta = {
            f"{module_name_for_layer(layer)}.weight": (
                get_module(model, module_name_for_layer(layer)).weight.detach().cpu()
                - original_weights[layer]
            )
            for layer in args.layers
        }
        atomic_torch_save(
            args.save_delta,
            {
                "method": "AlphaEdit",
                "reproduction_status": reproduction_status,
                "layers": list(args.layers),
                "delta": delta,
            },
        )

    final_stage = max(stages)
    atomic_json(
        args.output_dir / "completed.json",
        {
            "status": "complete",
            "method": "AlphaEdit",
            "reproduction_status": reproduction_status,
            "formal_baseline_eligible": formal_projection,
            "final_stage": final_stage,
            "num_edits": len(selected_rows),
            "projection_audit": projections.audit,
        },
    )


if __name__ == "__main__":
    main()
