"""NAS and matched MEMIT+NAS Qwen3 transfers under the frozen five-axis protocol.

NAS uses its source Qwen2.5 layers 7-11 and AlphaEdit-style nullspace update.
MEMIT+NAS must be explicitly bound to a completed MEMIT ``run_config.json``;
it never silently falls back to an old layer-4 reference. Both use
original-model CounterFact anchors.
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from baselines.build_nas_anchor import COMMIT, load_upstream
from baselines.common import (
    atomic_json, append_jsonl, environment_metadata, group_by_stage, load_manifest,
    load_model, load_tokenizer, manifest_panel, sha256_file,
)
from baselines.run_alphaedit import ProjectionProvider, apply_alphaedit_batch
from baselines.run_memit import (
    CovarianceProvider, CumulativeWeightDriftAudit, apply_memit_batch, build_context_templates, capture_representations,
    chunks, compute_z_star, get_module, make_request,
)
from baselines.run_lora_family import evaluate_and_write


def load_matched_memit_config(
    path: Path | None,
    model_dir: Path,
    *,
    development_only: bool,
) -> tuple[dict, dict]:
    """Load and audit the explicit MEMIT configuration used by MEMIT+NAS.

    The configuration and its sibling ``completed.json`` are treated as an
    immutable binding.  Formal MEMIT+NAS runs additionally require an explicit
    Qwen3 layer-selection validation/provenance marker; development runs may
    bind an unvalidated candidate for controlled experimentation.
    """
    if path is None:
        raise ValueError(
            "--matched-memit-config is required when --method=memit_nas; "
            "implicit historical MEMIT references are disabled"
        )
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"matched MEMIT config does not exist: {config_path}")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid matched MEMIT JSON: {config_path}") from error
    if config.get("method") != "MEMIT":
        raise ValueError(
            f"matched config must describe method MEMIT, got {config.get('method')!r}"
        )
    completed_path = config_path.parent / "completed.json"
    if not completed_path.is_file():
        raise ValueError(
            f"matched MEMIT run is not complete: missing {completed_path}"
        )
    try:
        completed = json.loads(completed_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid MEMIT completion JSON: {completed_path}") from error
    if completed.get("status") != "complete":
        raise ValueError(
            f"matched MEMIT run must be complete, got status={completed.get('status')!r}"
        )
    if completed.get("method") not in (None, "MEMIT"):
        raise ValueError(
            f"matched completion method must be MEMIT, got {completed.get('method')!r}"
        )
    recorded_model = config.get("model_dir")
    if recorded_model is None:
        raise ValueError("matched MEMIT config is missing model_dir")
    if Path(recorded_model).expanduser().resolve() != Path(model_dir).expanduser().resolve():
        raise ValueError(
            "matched MEMIT model mismatch: "
            f"{Path(recorded_model).resolve()} != {Path(model_dir).resolve()}"
        )
    layers = config.get("layers")
    if (
        not isinstance(layers, list)
        or not layers
        or any(not isinstance(layer, int) or layer < 0 for layer in layers)
        or len(layers) != len(set(layers))
        or layers != sorted(layers)
    ):
        raise ValueError(
            "matched MEMIT layers must be a non-empty sorted list of unique non-negative ints"
        )
    required = (
        "mom2_update_weight",
        "solver_ridge",
        "v_num_grad_steps",
        "v_lr",
        "v_weight_decay",
        "clamp_norm_factor",
        "kl_factor",
        "v_loss_layer",
        "context_templates",
        "edit_batch_size",
        "eval_batch_size",
        "representation_batch_size",
        "max_length",
        "max_new_tokens",
        "stats_source",
        "stats_dir",
        "stats_sample_size",
        "allow_proxy_stats",
        "allow_unverified_stats",
        "proxy_stat_samples",
        "proxy_variance_floor",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            "matched MEMIT config is missing required compatibility fields: "
            + ", ".join(missing)
        )
    contexts = config["context_templates"]
    if not isinstance(contexts, list) or not contexts or any(
        not isinstance(group, list) or not group for group in contexts
    ):
        raise ValueError("matched MEMIT context_templates must contain non-empty groups")
    if not development_only and not (
        config.get("qwen3_layer_selection_validated") is True
        or bool(config.get("selection_provenance"))
    ):
        raise ValueError(
            "formal MEMIT+NAS requires a validated Qwen3 layer selection or "
            "non-empty selection_provenance in the matched MEMIT config"
        )
    if not development_only:
        if config.get("development_only") is True:
            raise ValueError(
                "formal MEMIT+NAS cannot bind a development-only MEMIT config"
            )
        if config.get("formal_baseline_eligible") is not True:
            raise ValueError(
                "formal MEMIT+NAS requires matched MEMIT formal_baseline_eligible=true"
            )
    metadata = {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
        "completed_path": str(completed_path.resolve()),
        "completed_sha256": sha256_file(completed_path),
        "layers": list(layers),
        "development_only_source": bool(config.get("development_only", False)),
        "qwen3_layer_selection_validated": bool(
            config.get("qwen3_layer_selection_validated", False)
        ),
        "selection_provenance": config.get("selection_provenance"),
    }
    return config, metadata


def validate_anchor_compatibility(
    anchor: dict,
    *,
    layers,
    expected_hyperparameters: dict,
    matched_memit_binding: dict | None,
) -> None:
    """Reject an anchor calibrated for a different NAS/MEMIT configuration."""
    if anchor.get('target_layer') != layers[-1]:
        raise ValueError('NAS anchor layer mismatch')
    anchor_hp = anchor.get('hyperparameters')
    if not isinstance(anchor_hp, dict):
        raise ValueError('NAS anchor is missing hyperparameters')
    common_keys = (
        'v_lr', 'v_weight_decay', 'v_loss_layer', 'clamp_norm_factor', 'kl_factor'
    )
    keys = common_keys
    if matched_memit_binding is not None:
        if anchor.get('preset') != 'selected_memit':
            raise ValueError(
                'MEMIT+NAS requires an anchor built with --preset selected_memit'
            )
        if anchor.get('matched_memit_config_sha256') != matched_memit_binding['sha256']:
            raise ValueError('NAS anchor matched MEMIT config hash mismatch')
        if (
            anchor.get('matched_memit_completed_sha256')
            != matched_memit_binding['completed_sha256']
        ):
            raise ValueError('NAS anchor matched MEMIT completion hash mismatch')
        if anchor_hp.get('layers') != list(layers):
            raise ValueError('NAS anchor matched MEMIT layer list mismatch')
        keys += ('v_num_grad_steps', 'mom2_update_weight', 'solver_ridge')
    for key in keys:
        if anchor_hp.get(key) != expected_hyperparameters[key]:
            raise ValueError(f'Anchor optimizer mismatch: {key}')


def nas_transform(
    target_init,
    delta,
    v0,
    anchor,
    scale_function,
    *,
    outlier_mode='clip',
):
    """Preserve the pre-MLP residual and safely normalize the proposed MLP value.

    The upstream outlier branch replaced an oversized ``v0 + delta`` with
    ``v0``, silently discarding the edit signal.  ``clip`` keeps the proposal
    direction while bounding it before normalization; ``error`` is available
    for strict diagnostics.
    """
    if outlier_mode not in {'clip', 'error'}:
        raise ValueError("NAS outlier_mode must be 'clip' or 'error'")
    if not math.isfinite(float(anchor)) or anchor <= 0:
        raise ValueError('NAS anchor must be a finite positive number')
    for name, value in (('target_init', target_init), ('delta', delta), ('v0', v0)):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Non-finite NAS {name}')
    residual = target_init.float() - v0.float()
    proposed = delta.float() + v0.float()
    proposed_norm = float(proposed.norm())
    outlier_limit = 2.0 * float(anchor)
    outlier = proposed_norm > outlier_limit
    if outlier and outlier_mode == 'error':
        raise FloatingPointError(
            f'NAS proposal norm {proposed_norm:.6g} exceeds outlier limit '
            f'{outlier_limit:.6g}'
        )
    clip_scale = outlier_limit / proposed_norm if outlier else 1.0
    bounded_proposal = proposed * clip_scale
    value = scale_function(bounded_proposal, anchor)
    target = residual + value
    if not torch.isfinite(value).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Non-finite NAS target')
    return target, {'v0_norm': float(v0.float().norm()), 'v_star_norm_orig': proposed_norm,
                    'v_star_norm_real': float(value.norm()), 'anchor': anchor,
                    'outlier_limit': outlier_limit, 'outlier_detected': outlier,
                    'outlier_action': 'clip' if outlier else 'none',
                    'proposal_clip_scale': clip_scale,
                    # Kept for old trace readers; fixed behavior never drops delta.
                    'outlier_skipped_delta': False, 'target_norm': float(target.norm())}


def validate_batch_update_ratios(stats, max_ratio):
    """Fail closed on non-finite or disproportionately large weight updates."""
    if not math.isfinite(max_ratio) or max_ratio <= 0:
        raise ValueError('max update/weight ratio must be finite and positive')
    ratios = {}
    for layer, update_norm in stats.layer_update_norms.items():
        weight_norm = stats.layer_weight_norms.get(layer)
        if weight_norm is None:
            raise ValueError(f'missing pre-update weight norm for NAS layer {layer}')
        ratio = update_norm / weight_norm if weight_norm > 0 else math.inf
        ratios[layer] = ratio
        if not all(math.isfinite(value) for value in (update_norm, weight_norm, ratio)):
            raise FloatingPointError(f'Non-finite NAS update statistics at layer {layer}')
        if ratio > max_ratio:
            raise FloatingPointError(
                f'NAS layer {layer} update rejected: ||delta W||/||W||={ratio:.6g} '
                f'exceeds {max_ratio:.6g}'
            )
    return ratios


def apply_alphaedit_transaction(model, layers, histories, apply_call, max_ratio):
    """Apply AlphaEdit atomically, including its mutable projected-key history."""
    originals = {
        layer: get_module(model, f'model.layers.{layer}.mlp.down_proj')
        .weight.detach().clone()
        for layer in layers
    }
    history_snapshot = {
        layer: (
            history,
            history.keys.detach().clone(),
            history.cholesky.detach().clone(),
            history.cholesky_retry_count,
            history.max_cholesky_jitter,
        )
        for layer, history in histories.items()
    }
    try:
        stats = apply_call()
        ratios = validate_batch_update_ratios(stats, max_ratio)
        for layer in layers:
            weight = get_module(model, f'model.layers.{layer}.mlp.down_proj').weight
            if not torch.isfinite(weight).all():
                raise FloatingPointError(f'Non-finite NAS weights at layer {layer}')
        return stats, ratios
    except Exception:
        with torch.no_grad():
            for layer, original in originals.items():
                get_module(
                    model, f'model.layers.{layer}.mlp.down_proj'
                ).weight.copy_(original)
        histories.clear()
        for layer, snapshot in history_snapshot.items():
            history, keys, cholesky, retry_count, max_jitter = snapshot
            history.keys = keys
            history.cholesky = cholesky
            history.cholesky_retry_count = retry_count
            history.max_cholesky_jitter = max_jitter
            histories[layer] = history
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['nas', 'memit_nas'], required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('Qwen3-8B-Base'))
    p.add_argument('--upstream-dir', type=Path, default=Path('external/NAS'))
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--edit-batch-size', type=int, default=100)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--max-edits', type=int, default=0, help='Integration only; zero consumes entire manifest')
    p.add_argument(
        '--matched-memit-config',
        type=Path,
        default=None,
        help=(
            'completed MEMIT run_config.json to bind for --method=memit_nas; '
            'required for that method'
        ),
    )
    p.add_argument('--development-only', action='store_true',
                   help='mark this run as a development candidate, never a benchmark result')
    p.add_argument('--max-update-weight-ratio', type=float, default=0.5,
                   help='fail when a layer update norm exceeds this fraction of its weight norm')
    p.add_argument('--nas-outlier-mode', choices=['clip', 'error'], default='clip',
                   help='bound oversized NAS proposals while preserving direction, or fail fast')
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    matched = None
    matched_metadata = None
    if a.method == 'memit_nas':
        matched, matched_metadata = load_matched_memit_config(
            a.matched_memit_config,
            a.model_dir,
            development_only=a.development_only,
        )
    elif a.matched_memit_config is not None:
        raise ValueError('--matched-memit-config is only valid with --method=memit_nas')
    upstream = load_upstream(a.upstream_dir)
    anchor = json.loads(a.anchor.read_text())
    if anchor['status'] != 'complete' or anchor['samples'] != 1000 or not anchor['base_model_only']:
        raise ValueError('A complete original-model NAS anchor is required')
    if anchor['model_dir'] != str(a.model_dir.resolve()):
        raise ValueError('NAS anchor checkpoint mismatch')
    if anchor['model_checkpoint_identity_sha256'] != sha256_file(Path('runs/protocols/qwen3_8b_base_checkpoint_identity.json')):
        raise ValueError('NAS anchor checkpoint provenance mismatch')
    source = json.loads((a.upstream_dir / 'hparams/NAS/Qwen2p5-7B.json').read_text())
    hp = matched if matched else source
    for key in ('layers', 'v_num_grad_steps', 'v_lr', 'v_weight_decay', 'clamp_norm_factor', 'kl_factor', 'mom2_update_weight'):
        setattr(a, key, hp[key])
    a.v_loss_layer = matched['v_loss_layer'] if matched else 35
    if matched:
        a.edit_batch_size = int(matched['edit_batch_size'])
        a.eval_batch_size = int(matched['eval_batch_size'])
    a.max_length = int(hp.get('max_length', 128))
    a.max_new_tokens = int(hp.get('max_new_tokens', 16))
    a.representation_batch_size = int(hp.get('representation_batch_size', 64))
    a.solver_ridge = float(hp.get('solver_ridge', 0.0))
    a.stats_source = hp.get('stats_source', 'precomputed')
    a.stats_dir = Path(hp.get('stats_dir', '/mnt/yy2607004_hs/engram_memit_stats'))
    a.stats_file_template = hp.get('stats_file_template')
    a.stats_sample_size = int(hp.get('stats_sample_size', 100000))
    a.allow_proxy_stats = bool(hp.get('allow_proxy_stats', False))
    a.allow_unverified_stats = bool(hp.get('allow_unverified_stats', False))
    a.proxy_stat_samples = int(hp.get('proxy_stat_samples', 64))
    a.proxy_variance_floor = float(hp.get('proxy_variance_floor', 1e-5))
    a.projection_dir = Path('/mnt/yy2607004_hs/engram_nas_stats')
    a.projection_file_template = None
    a.projection_sample_size = 100000
    a.model_fingerprint_mode = 'content'
    a.allow_diagnostic_projection = False
    a.nullspace_threshold = source['nullspace_threshold']
    a.l2 = source['L2']
    if (not a.layers or len(a.layers) != len(set(a.layers))
            or list(a.layers) != sorted(a.layers) or min(a.layers) < 0):
        raise ValueError('NAS layers must be unique, non-negative, and strictly increasing')
    if not math.isfinite(a.max_update_weight_ratio) or a.max_update_weight_ratio <= 0:
        raise ValueError('max-update-weight-ratio must be finite and positive')
    validate_anchor_compatibility(
        anchor,
        layers=a.layers,
        expected_hyperparameters={
            key: getattr(a, key)
            for key in (
                'v_num_grad_steps', 'v_lr', 'v_weight_decay', 'v_loss_layer',
                'clamp_norm_factor', 'kl_factor', 'mom2_update_weight', 'solver_ridge',
            )
        },
        matched_memit_binding=matched_metadata,
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.cuda.set_device(a.gpu)
    device = torch.device(f'cuda:{a.gpu}')
    rows, manifest = load_manifest(a.manifest)
    if a.max_edits:
        rows = rows[:a.max_edits]
    ids = {r['sample_id'] for r in rows}
    stages = group_by_stage(rows)
    loc = [r for r in manifest_panel(manifest, 'locality') if r['sample_id'] in ids]
    mh = [r for r in manifest_panel(manifest, 'mhop') if r['sample_id'] in ids]
    tok = load_tokenizer(a.model_dir)
    model = load_model(a.model_dir, device)
    if max(a.layers) >= int(model.config.num_hidden_layers):
        raise ValueError('NAS rewrite layer is outside the Qwen3 model')
    if a.v_loss_layer < max(a.layers) or a.v_loss_layer >= int(model.config.num_hidden_layers):
        raise ValueError('NAS loss layer must be at or after all rewrite layers')
    model.eval().requires_grad_(False)
    contexts = matched['context_templates'] if matched else build_context_templates(model, tok, device, 'upstream_generated')
    provider = (CovarianceProvider(model, tok, a.model_dir, rows, device, a)
                if matched else ProjectionProvider(model, a.model_dir, device, a))
    for layer in a.layers:
        provider.get(layer)
    state = {}
    originals = {l: get_module(model, f'model.layers.{l}.mlp.down_proj').weight.detach().cpu().clone() for l in a.layers}
    cumulative_drift_audit = (
        CumulativeWeightDriftAudit.from_model(model, a.layers) if matched else None
    )
    formal_artifacts = all(
        entry.get('formal_baseline_eligible') is True
        for entry in provider.audit.values()
    )
    matched_qwen3_selected = bool(
        matched
        and matched.get('qwen3_layer_selection_validated') is True
        and matched.get('formal_baseline_eligible') is True
        and not matched.get('development_only', False)
    )
    formal_baseline_eligible = (
        not a.max_edits and not a.development_only and formal_artifacts
    )
    if matched and not matched_qwen3_selected and not a.development_only:
        raise ValueError(
            'formal MEMIT+NAS requires a certificate-bound, independently '
            'selected Qwen3 MEMIT configuration'
        )
    config = {**{k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'method': 'MEMIT+NAS' if matched else 'NAS',
              'upstream_repository': 'SasyaTitech/NAS', 'upstream_commit': COMMIT,
              'upstream_qwen25_preset': source, 'anchor_audit': anchor, 'anchor_sha256': sha256_file(a.anchor),
              'manifest_protocol_id': manifest['protocol_id'], 'manifest_sha256': sha256_file(a.manifest),
              'num_facts': len(rows), 'stage_counts': {str(s): len(rs) for s, rs in stages.items()},
              'closed_book_evaluation': True, 'development_only': bool(a.development_only),
              'formal_baseline_eligible': formal_baseline_eligible,
              'qwen3_preset_status': (
                  'Qwen3 independent-dev selected MEMIT+NAS adaptation'
                  if matched_qwen3_selected else
                  ('Qwen2.5 -> Qwen3 development candidate; no official Qwen3-8B-Base preset'
                   if a.development_only else
                   'Qwen2.5 -> Qwen3 untuned transfer; no official Qwen3-8B-Base preset')
              ),
              'reproduction_status': (
                  'qwen3_development_candidate' if a.development_only
                  else ('qwen3_independent_dev_selected' if matched_qwen3_selected
                        else 'qwen3_transfer_preset_untuned')
              ),
              'qwen3_layer_selection_validated': bool(
                  matched_qwen3_selected
                  or (matched and matched.get('qwen3_layer_selection_validated') is True)
              ),
              'selection_certificate': (
                  matched.get('selection_certificate') if matched else None
              ),
              'selection_provenance': (
                  matched.get('development_selection')
                  or matched.get('selection_provenance') if matched else None
              ),
              'statistics_audit': provider.audit, 'context_templates': contexts,
              'matched_memit_binding': matched_metadata,
              'matched_memit_config_sha256': (
                  matched_metadata['sha256'] if matched_metadata else None
              ),
              'matched_memit_completed_sha256': (
                  matched_metadata['completed_sha256'] if matched_metadata else None
              ),
              'port_differences': ['Qwen3 final loss layer 35', 'shared QA prompts and greedy five-axis evaluation',
                  'shared bf16 model with float32 latent losses and stable existing linear solves',
                  f'edit batches of {a.edit_batch_size} within each time interval; not single-edit paper scores',
                  'MEMIT+NAS settings and contexts are bound to an explicit completed MEMIT config',
                  'Qwen3 100K Wikipedia20231101 moments; original-model 1000 MultiCounterFact norm calibration'],
              **environment_metadata()}
    atomic_json(a.output_dir / 'run_config.json', config)
    mean_norm = anchor['mean_vstar_norm']

    def optimize(model, tokenizer, request, context_groups, z_layer, loss_layer, device, args):
        def transform(initial, delta, template):
            v0 = capture_representations(model, tokenizer, [template], [request['subject']],
                                        f'model.layers.{z_layer}.mlp', 'out', device,
                                        batch_size=1, max_length=args.max_length)[0]
            target, debug = nas_transform(
                initial, delta, v0, mean_norm, upstream.scale_vector_to_norm,
                outlier_mode=a.nas_outlier_mode,
            )
            append_jsonl(a.output_dir / 'nas_targets.jsonl', [{'sample_id': request['sample_id'], **debug}])
            return target
        return compute_z_star(model, tokenizer, request, context_groups, z_layer, loss_layer,
                              device, args, target_transform=transform)

    metrics, seen = [], []
    count = 0
    started = time.time()
    checkpoint_dir = Path('/mnt/yy2607004_hs/engram_checkpoints') / a.output_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for stage, stage_rows in stages.items():
        for bi, batch in enumerate(chunks(stage_rows, a.edit_batch_size)):
            requests = [make_request(r, count + i) for i, r in enumerate(batch)]
            apply = apply_memit_batch if matched else apply_alphaedit_batch
            if matched:
                apply_call = lambda: apply(
                    model, tok, requests, contexts, provider, state, device, a,
                    stage=stage, batch_index=bi, z_function=optimize,
                    cumulative_drift_audit=cumulative_drift_audit,
                )
                # apply_memit_batch already performs exact cross-layer rollback.
                stats = apply_call()
                ratios = validate_batch_update_ratios(stats, a.max_update_weight_ratio)
            else:
                apply_call = lambda: apply(
                    model, tok, requests, contexts, provider, state, device, a,
                    stage=stage, batch_index=bi, z_function=optimize,
                )
                stats, ratios = apply_alphaedit_transaction(
                    model, a.layers, state, apply_call, a.max_update_weight_ratio
                )
            values = asdict(stats)
            values['layer_update_weight_ratios'] = ratios
            for layer in a.layers:
                if not torch.isfinite(get_module(model, f'model.layers.{layer}.mlp.down_proj').weight).all():
                    raise FloatingPointError('Non-finite NAS weights')
            append_jsonl(a.output_dir / 'edit_trace.jsonl', [{**values, 'finite': True}])
            count += len(batch)
            atomic_json(a.output_dir / 'progress.json', {'edits': count, 'total': len(rows), 'stage': stage, 'elapsed_seconds': time.time() - started})
        seen.extend(stage_rows)
        metrics.append(evaluate_and_write(model, tok, seen, loc, mh, device, a, f'tau{stage}', stage))
        atomic_json(a.output_dir / 'metrics.json', metrics)
        torch.save({'stage': stage, 'edits': count, 'deltas': {
            f'model.layers.{l}.mlp.down_proj.weight': get_module(model, f'model.layers.{l}.mlp.down_proj').weight.detach().cpu() - originals[l]
            for l in a.layers}}, checkpoint_dir / f'tau{stage}_delta.pt')
    cumulative_drift_summary = (
        cumulative_drift_audit.completed_summary(model, a.layers)
        if cumulative_drift_audit is not None else {}
    )
    atomic_json(a.output_dir / 'completed.json', {'status': 'complete', 'method': config['method'],
                'num_facts': len(rows), 'edits': count, 'final_stage': max(stages),
                'development_only': bool(a.development_only),
                'formal_baseline_eligible': formal_baseline_eligible,
                'matched_memit_config_sha256': (
                    matched_metadata['sha256'] if matched_metadata else None
                ),
                **cumulative_drift_summary,
                'elapsed_seconds': time.time() - started})


if __name__ == '__main__':
    main()
