"""Run the pinned EasyEdit ROME family with an auditable Qwen3 port.

The upstream recipe was written for Qwen2.5 and is not safe to apply verbatim
to Qwen3.  This runner therefore makes the rewrite/loss layers explicit,
optionally uses checkpoint-specific Qwen3 second moments, and applies a small
trust region to each sequential rank-one update.  The trust region is not
hidden: every clipped update is recorded in ``edit_trace.jsonl`` and the run
is labelled as an adapted candidate rather than an upstream reproduction.
"""
import argparse
import importlib
import json
import math
import random
import subprocess
import sys
import time
import types
from pathlib import Path
import torch
import yaml
from baselines.common import atomic_json, append_jsonl, load_manifest, group_by_stage, manifest_panel, load_model, load_tokenizer, sha256_file, environment_metadata
from baselines.run_lora_family import evaluate_and_write
from baselines.run_memit import load_moment_file, make_request

COMMIT='14cea8245f06715684592ab55184939b99d70784'
DEFAULT_QWEN3_LAYERS = [5]
DEFAULT_MOM2_STATS_DIR = Path('/mnt/yy2607004_hs/engram_memit_stats')

def upstream_modules(root):
    source=root/'easyeditor'
    for name,path in [('easyeditor',source),('easyeditor.models',source/'models'),('easyeditor.util',source/'util'),('easyeditor.models.rome',source/'models/rome'),('easyeditor.models.r_rome',source/'models/r_rome')]:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    return {key:importlib.import_module(f'easyeditor.models.{key}.{key}_main') for key in ('rome','r_rome')}


def _resolve_moment_path(stats_dir: Path, model_dir: Path, layer: int) -> Path:
    """Resolve a checkpoint-specific full moment used by EasyEdit's ROME u."""
    model_key = model_dir.resolve().name.replace('/', '_')
    candidates = [
        stats_dir / model_key / f'layer_{layer}_mom2.pt',
        stats_dir / f'layer_{layer}_mom2.pt',
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        'Qwen3 ROME covariance statistics are missing; expected:\n' +
        '\n'.join(f'  - {path}' for path in candidates)
    )


def _install_qwen3_covariance(
    model: torch.nn.Module,
    model_dir: Path,
    stats_dir: Path,
    layers: list[int],
    ridge: float,
) -> dict[str, dict[str, object]]:
    """Install inverse moments into both pinned EasyEdit compute_u modules.

    EasyEdit's ROME implementation looks up ``inv_mom2_cache`` by the model
    name and module name.  Loading our audited Qwen3 moments here avoids the
    upstream ``layer_stats`` path (which would otherwise collect unrelated
    Qwen2.5 statistics on demand).
    """
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError('--mom2-ridge must be finite and non-negative')
    model_key = str(getattr(model.config, '_name_or_path', '')).replace('/', '_')
    if not model_key:
        raise ValueError('Qwen3 model has no _name_or_path for ROME covariance cache')
    audit: dict[str, dict[str, object]] = {}
    for layer in layers:
        module_name = f'model.layers.{layer}.mlp.down_proj'
        path = _resolve_moment_path(stats_dir, model_dir, layer)
        moment, metadata = load_moment_file(path)
        actual_sha256 = sha256_file(path)
        module = model
        for part in module_name.split('.'):
            module = getattr(module, part)
        if moment.ndim != 2 or tuple(moment.shape) != (module.in_features, module.in_features):
            raise ValueError(
                f'{path}: moment shape {tuple(moment.shape)} does not match '
                f'{module_name} input dimension {module.in_features}'
            )
        recorded_model = metadata.get('model_dir') or metadata.get('model_name')
        if recorded_model is None or Path(recorded_model).resolve() != model_dir.resolve():
            raise ValueError(f'{path}: moment was not collected from {model_dir.resolve()}')
        if metadata.get('module_name') != module_name:
            raise ValueError(f'{path}: moment module does not match {module_name}')
        if metadata.get('kind') != 'full_second_moment':
            raise ValueError(f'{path}: ROME requires a full second moment')
        if metadata.get('formal_memit_eligible') is not True:
            raise ValueError(f'{path}: moment provenance is not formal/audited')
        recorded_sha256 = metadata.get('artifact_sha256')
        if recorded_sha256 is not None and recorded_sha256 != actual_sha256:
            raise ValueError(f'{path}: moment artifact hash changed')
        device = next(model.parameters()).device
        base = moment.to(device=device, dtype=torch.float64)
        if ridge:
            base = base.clone()
            base.diagonal().add_(ridge)
        try:
            inverse = torch.cholesky_inverse(torch.linalg.cholesky(base)).float()
        except RuntimeError as error:
            raise ValueError(
                f'Qwen3 ROME moment is not positive definite at layer {layer}; '
                f'increase --mom2-ridge or regenerate {path}'
            ) from error
        inverse = inverse.to(device)
        for method in ('rome', 'r_rome'):
            compute_u_module = importlib.import_module(
                f'easyeditor.models.{method}.compute_u'
            )
            cache = getattr(compute_u_module, 'inv_mom2_cache')
            cache[(model_key, module_name)] = inverse
        audit[str(layer)] = {
            'path': str(path),
            'file_sha256': actual_sha256,
            'shape': list(moment.shape),
            'ridge': ridge,
            'metadata': metadata,
        }
    return audit


def _bounded_scale(
    current_delta: torch.Tensor,
    update: torch.Tensor,
    original_norm: float,
    *,
    max_update_ratio: float,
    max_cumulative_ratio: float,
) -> tuple[float, dict[str, float]]:
    """Return a scale in [0,1] keeping both update and cumulative norms bounded."""
    update_norm = float(update.float().norm())
    scale = 1.0
    if update_norm == 0:
        return scale, {'update_norm': 0.0, 'update_ratio': 0.0, 'cumulative_ratio_before':
                       float(current_delta.float().norm()) / original_norm if original_norm else math.inf}
    if not math.isfinite(update_norm) or original_norm <= 0:
        raise FloatingPointError('non-finite or degenerate ROME update norm')
    # Per-update bound.  max_update_ratio <= 0 means no usable trust region.
    if math.isfinite(max_update_ratio):
        scale = min(scale, max_update_ratio * original_norm / update_norm)
    # Solve ||d + s*u|| <= radius for the largest s in [0,1].
    radius = max_cumulative_ratio * original_norm
    if math.isfinite(radius):
        d = current_delta.float().reshape(-1)
        u = update.float().reshape(-1)
        c = float(torch.dot(d, d)) - radius * radius
        tolerance = max(1e-4, radius * radius * 1e-6)
        if c > tolerance:
            raise FloatingPointError('ROME cumulative update already exceeds its trust region')
        a = float(torch.dot(u, u))
        b = 2.0 * float(torch.dot(d, u))
        if a > 0:
            discriminant = max(0.0, b * b - 4.0 * a * min(c, 0.0))
            root = (-b + math.sqrt(discriminant)) / (2.0 * a)
            scale = min(scale, max(0.0, root))
    scale = max(0.0, min(1.0, scale))
    return scale, {
        'update_norm': update_norm,
        'update_ratio': update_norm / original_norm,
        'cumulative_ratio_before': float(current_delta.float().norm()) / original_norm,
    }

def _parse_layers(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('layers must be comma-separated integers') from error


def _validate_qwen3(model: torch.nn.Module, layers: list[int], loss_layer: int) -> None:
    if getattr(model.config, 'model_type', None) != 'qwen3':
        raise ValueError(f'ROME runner is scoped to Qwen3, got {model.config.model_type!r}')
    layer_count = int(model.config.num_hidden_layers)
    if (not layers or len(layers) != len(set(layers)) or layers != sorted(layers)
            or min(layers) < 0 or max(layers) >= layer_count):
        raise ValueError(f'--layers must be unique, increasing, and lie in [0, {layer_count - 1}]')
    # The final-block rewrite candidate produced no usable optimization signal
    # in Qwen3; require at least one downstream block for the loss target.
    if loss_layer <= max(layers) or loss_layer >= layer_count:
        raise ValueError('--loss-layer must be after every rewrite layer and inside the model')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['rome', 'r_rome'], required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('Qwen3-8B-Base'))
    p.add_argument('--upstream-dir', type=Path, default=Path('external/EasyEdit'))
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--layers', type=_parse_layers, default=None,
                   help='comma-separated Qwen3 rewrite layers; default is development layer 5')
    p.add_argument('--loss-layer', type=int, default=35)
    p.add_argument('--mom2-adjustment', action=argparse.BooleanOptionalAction, default=False,
                   help='use the audited Qwen3 full second moment for the ROME key direction')
    p.add_argument('--mom2-stats-dir', type=Path, default=DEFAULT_MOM2_STATS_DIR)
    p.add_argument('--mom2-ridge', type=float, default=1e-5)
    p.add_argument('--max-update-weight-ratio', type=float, default=0.075,
                   help='per-edit ROME trust-region radius relative to the base weight norm')
    p.add_argument('--max-cumulative-weight-drift-ratio', type=float, default=2.0,
                   help='cumulative ROME trust-region radius relative to the base weight norm')
    p.add_argument('--oversized-update-action', choices=['clip', 'skip', 'error'], default='clip')
    p.add_argument('--selection-certificate', type=Path, default=None)
    p.add_argument('--development-only', action='store_true')
    p.add_argument('--max-edits', type=int, default=0)
    p.add_argument('--eval-batch-size', type=int, default=48)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--max-new-tokens', type=int, default=16)
    a = p.parse_args()

    if not math.isfinite(a.max_update_weight_ratio) or a.max_update_weight_ratio <= 0:
        raise ValueError('--max-update-weight-ratio must be finite and positive')
    if (not math.isfinite(a.max_cumulative_weight_drift_ratio)
            or a.max_cumulative_weight_drift_ratio <= 0):
        raise ValueError('--max-cumulative-weight-drift-ratio must be finite and positive')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)

    preset_name = 'ROME' if a.method == 'rome' else 'R-ROME'
    raw = subprocess.check_output(
        ['git', '-C', str(a.upstream_dir), 'show',
         f'{COMMIT}:hparams/{preset_name}/qwen2.5-7b.yaml'],
        text=True,
    )
    preset = yaml.safe_load(raw)
    source_layers = list(preset['layers'])
    layers = list(a.layers if a.layers is not None else DEFAULT_QWEN3_LAYERS)
    if not layers or len(layers) != len(set(layers)) or layers != sorted(layers) or min(layers) < 0:
        raise ValueError('--layers must be a non-empty increasing list of unique non-negative indices')
    a.layers = layers
    preset['layers'] = layers
    for key in ('v_lr', 'v_weight_decay', 'kl_factor', 'clamp_norm_factor'):
        preset[key] = float(preset[key])
    preset.update(device=a.gpu, model_name=str(a.model_dir.resolve()),
                  v_loss_layer=a.loss_layer, mom2_adjustment=bool(a.mom2_adjustment))
    if a.method == 'r_rome':
        preset.update(enable_prompt_keys=False, enable_random_prefix_keys=True,
                      original_implementation=False)

    selection = None
    formal_request = not a.development_only and not a.max_edits
    if formal_request and a.selection_certificate is None:
        raise ValueError(
            'A full ROME/R-ROME benchmark requires --selection-certificate from '
            'the independent development sweep'
        )
    if a.selection_certificate is not None:
        from baselines.editor_dev_selection import verify_selection
        selection = verify_selection(a.selection_certificate, a, preset_name, layers)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / 'source_preset.yaml').write_text(raw)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.cuda.set_device(a.gpu)

    rows, manifest = load_manifest(a.manifest)
    if a.max_edits:
        rows = rows[:a.max_edits]
    if not rows:
        raise ValueError('manifest selected no edits')
    stages = group_by_stage(rows)
    ids = {row['sample_id'] for row in rows}
    locality = manifest_panel(manifest, 'locality')
    multihop = manifest_panel(manifest, 'mhop')
    if a.max_edits:
        locality = [row for row in locality if row['sample_id'] in ids]
        multihop = [row for row in multihop if row['sample_id'] in ids]

    modules = upstream_modules(a.upstream_dir)
    module = modules[a.method]
    execute = module.execute_rome if a.method == 'rome' else module.execute_r_rome
    tokenizer = load_tokenizer(a.model_dir)
    tokenizer.padding_side = 'right'
    model = load_model(a.model_dir, torch.device(f'cuda:{a.gpu}'))
    _validate_qwen3(model, layers, a.loss_layer)
    model.eval().requires_grad_(False)

    mom2_audit: dict[str, dict[str, object]] = {}
    if a.mom2_adjustment:
        mom2_audit = _install_qwen3_covariance(
            model, a.model_dir, a.mom2_stats_dir, layers, a.mom2_ridge
        )

    hp = types.SimpleNamespace(**preset)
    formal_eligible = selection is not None and formal_request
    serialized_args = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(a).items()
    }
    config = {
        **serialized_args,
        'method': preset_name,
        'num_facts': len(rows),
        'stage_counts': {str(stage): len(stage_rows) for stage, stage_rows in stages.items()},
        'manifest': str(a.manifest.resolve()),
        'manifest_protocol_id': manifest['protocol_id'],
        'manifest_sha256': sha256_file(a.manifest),
        'closed_book_evaluation': True,
        'development_only': bool(a.development_only),
        'formal_baseline_eligible': formal_eligible,
        'source_qwen25_layers': source_layers,
        'source_mom2_adjustment': bool(yaml.safe_load(raw)['mom2_adjustment']),
        'mom2_audit': mom2_audit,
        'qwen3_preset_status': (
            'independent-dev-selected Qwen3 adaptation'
            if selection else 'Qwen3 development candidate; not a benchmark result'
        ),
        'qwen3_layer_selection_validated': selection is not None,
        'selection_provenance': selection,
        'upstream_repository': 'zjunlp/EasyEdit',
        'upstream_commit': COMMIT,
        'upstream_hparams': preset,
        'trust_region': {
            'per_update_weight_ratio': a.max_update_weight_ratio,
            'cumulative_weight_drift_ratio': a.max_cumulative_weight_drift_ratio,
            'oversized_action': a.oversized_update_action,
        },
        'compatibility_changes': [
            f'source layers {source_layers} -> Qwen3 layers {layers}; loss layer 27 -> {a.loss_layer}',
            f'Qwen3 checkpoint moment adjustment={a.mom2_adjustment}',
            'transactional finite and relative-norm checks for every sequential update',
            'shared QA completion prompts and greedy five-axis evaluation',
            'robust subject surface matching shared with MEMIT port',
        ],
        **environment_metadata(),
    }
    atomic_json(a.output_dir / 'run_config.json', config)
    contexts = module.get_context_templates(model, tokenizer, hp.context_template_length_params)
    atomic_json(a.output_dir / 'contexts.json', contexts)

    named_parameters = dict(model.named_parameters())
    parameter_names = [f'{hp.rewrite_module_tmp.format(layer)}.weight' for layer in layers]
    originals = {name: named_parameters[name].detach().cpu().clone() for name in parameter_names}
    original_norms = {
        name: float(named_parameters[name].detach().float().norm()) for name in parameter_names
    }
    cumulative = {
        name: torch.zeros_like(named_parameters[name], dtype=torch.float32) for name in parameter_names
    }

    seen: list[dict] = []
    metrics: list[dict] = []
    started = time.time()
    edit_count = 0
    clipped_edits = 0
    skipped_edits = 0
    checkpoint_dir = Path('/mnt/yy2607004_hs/engram_checkpoints') / a.output_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for stage, current_rows in stages.items():
        for row in current_rows:
            request = make_request(row, edit_count)
            request['target_new'] = str(row['ans']).strip()
            before = time.time()
            deltas = execute(model, tokenizer, request, hp)
            candidates: dict[str, torch.Tensor] = {}
            next_cumulative: dict[str, torch.Tensor] = {}
            update_norms: dict[str, float] = {}
            applied_norms: dict[str, float] = {}
            update_ratios: dict[str, float] = {}
            cumulative_ratios: dict[str, float] = {}
            scales: dict[str, float] = {}
            clipped = False
            skipped = False

            for name, (left, right) in deltas.items():
                if name not in original_norms:
                    raise ValueError(f'ROME returned an unexpected parameter {name}')
                weight = named_parameters[name]
                update = module.upd_matrix_match_shape(
                    left[:, None] @ right[None, :], weight.shape
                )
                if not torch.isfinite(update).all():
                    raise FloatingPointError(f'non-finite ROME update for {row["sample_id"]}')
                scale, audit = _bounded_scale(
                    cumulative[name], update, original_norms[name],
                    max_update_ratio=a.max_update_weight_ratio,
                    max_cumulative_ratio=a.max_cumulative_weight_drift_ratio,
                )
                if scale < 1.0 - 1e-8:
                    clipped = True
                    if a.oversized_update_action == 'error':
                        raise FloatingPointError(
                            f'ROME update for {row["sample_id"]} exceeds the configured trust region'
                        )
                    if a.oversized_update_action == 'skip':
                        scale = 0.0
                applied = (update * scale).to(dtype=weight.dtype)
                if not torch.isfinite(applied).all():
                    raise FloatingPointError('ROME update overflows model weight dtype')
                candidate_cumulative = cumulative[name] + applied.float()
                cumulative_ratio = (
                    float(candidate_cumulative.norm()) / original_norms[name]
                )
                if cumulative_ratio > a.max_cumulative_weight_drift_ratio * (1.0 + 1e-5):
                    raise FloatingPointError('ROME cumulative trust-region calculation failed')
                candidates[name] = applied
                next_cumulative[name] = candidate_cumulative
                update_norms[name] = audit['update_norm']
                applied_norms[name] = float(applied.float().norm())
                update_ratios[name] = audit['update_ratio']
                cumulative_ratios[name] = cumulative_ratio
                scales[name] = scale
                skipped = skipped or scale == 0.0

            snapshots = {name: named_parameters[name].detach().clone() for name in candidates}
            try:
                with torch.no_grad():
                    for name, applied in candidates.items():
                        named_parameters[name].add_(applied)
                        if not torch.isfinite(named_parameters[name]).all():
                            raise FloatingPointError(f'non-finite ROME weight after updating {name}')
            except Exception:
                with torch.no_grad():
                    for name, snapshot in snapshots.items():
                        named_parameters[name].copy_(snapshot)
                raise
            cumulative.update(next_cumulative)
            edit_count += 1
            clipped_edits += int(clipped)
            skipped_edits += int(skipped)
            append_jsonl(a.output_dir / 'edit_trace.jsonl', [{
                'edit': edit_count,
                'stage': stage,
                'sample_id': row['sample_id'],
                'update_norms': update_norms,
                'applied_update_norms': applied_norms,
                'update_weight_ratios': update_ratios,
                'cumulative_weight_drift_ratios': cumulative_ratios,
                'update_scales': scales,
                'trust_region_clipped': clipped,
                'trust_region_skipped': skipped,
                'finite': True,
                'seconds': time.time() - before,
                'subject_match': request['subject_match'],
            }])
            if edit_count % 25 == 0:
                atomic_json(a.output_dir / 'progress.json', {
                    'edits': edit_count,
                    'total': len(rows),
                    'stage': stage,
                    'clipped_edits': clipped_edits,
                    'skipped_edits': skipped_edits,
                    'elapsed_seconds': time.time() - started,
                })

        seen.extend(current_rows)
        metrics.append(evaluate_and_write(
            model, tokenizer, seen, locality, multihop, model.device, a,
            f'tau{stage}', stage,
        ))
        atomic_json(a.output_dir / 'metrics.json', metrics)
        torch.save({
            'stage': stage,
            'edits': edit_count,
            'deltas': {
                name: named_parameters[name].detach().cpu() - original
                for name, original in originals.items()
            },
        }, checkpoint_dir / f'tau{stage}_delta.pt')

    atomic_json(a.output_dir / 'completed.json', {
        'status': 'complete',
        'method': preset_name,
        'num_facts': len(rows),
        'final_stage': max(stages),
        'edits': edit_count,
        'clipped_edits': clipped_edits,
        'skipped_edits': skipped_edits,
        'elapsed_seconds': time.time() - started,
        'development_only': bool(a.development_only),
        'formal_baseline_eligible': formal_eligible,
    })


if __name__ == '__main__':
    main()
