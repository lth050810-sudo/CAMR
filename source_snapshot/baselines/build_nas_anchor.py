"""Compute NAS's original-model CounterFact norm anchor using pinned upstream code."""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from baselines.common import atomic_json, load_model, load_tokenizer, sha256_file
from baselines.run_memit import UPSTREAM_QWEN25_PRESET

COMMIT = '5acab48c88effb95cc9492559872093dc085c79e'
SELECTED_MEMIT_HPARAM_KEYS = (
    'layers',
    'v_num_grad_steps',
    'v_lr',
    'v_weight_decay',
    'clamp_norm_factor',
    'kl_factor',
    'mom2_update_weight',
    'v_loss_layer',
    'solver_ridge',
)


def load_selected_memit_config(path: Path, model_dir: Path) -> tuple[dict, dict]:
    """Load a completed, formal MEMIT run for a provenance-bound NAS anchor."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'MEMIT config does not exist: {config_path}')
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'invalid MEMIT config JSON: {config_path}') from error
    if config.get('method') != 'MEMIT':
        raise ValueError(
            f'selected MEMIT config must have method=MEMIT, got {config.get("method")!r}'
        )
    if config.get('development_only') is True:
        raise ValueError('selected MEMIT anchor cannot be built from development_only config')
    if config.get('formal_baseline_eligible') is not True:
        raise ValueError('selected MEMIT anchor requires formal_baseline_eligible=true')
    if not (
        config.get('qwen3_layer_selection_validated') is True
        or config.get('selection_provenance')
    ):
        raise ValueError(
            'selected MEMIT anchor requires qwen3_layer_selection_validated=true '
            'or non-empty selection_provenance'
        )
    recorded_model = config.get('model_dir')
    if recorded_model is None:
        raise ValueError('selected MEMIT config is missing model_dir')
    expected_model = Path(model_dir).expanduser().resolve()
    if Path(recorded_model).expanduser().resolve() != expected_model:
        raise ValueError(
            f'MEMIT model mismatch: {Path(recorded_model).resolve()} != {expected_model}'
        )
    layers = config.get('layers')
    if (
        not isinstance(layers, list)
        or not layers
        or any(not isinstance(layer, int) or layer < 0 for layer in layers)
        or len(layers) != len(set(layers))
        or layers != sorted(layers)
    ):
        raise ValueError('selected MEMIT layers must be sorted, unique, non-negative ints')
    completed_path = config_path.parent / 'completed.json'
    if not completed_path.is_file():
        raise ValueError(f'selected MEMIT run is missing {completed_path}')
    try:
        completed = json.loads(completed_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'invalid MEMIT completed JSON: {completed_path}') from error
    if completed.get('status') != 'complete' or completed.get('method') not in (None, 'MEMIT'):
        raise ValueError('selected MEMIT run must have completed.json status=complete and method=MEMIT')
    if completed.get('formal_baseline_eligible') is not True:
        raise ValueError(
            'selected MEMIT completed.json requires formal_baseline_eligible=true'
        )
    missing = [key for key in SELECTED_MEMIT_HPARAM_KEYS if key not in config]
    if missing:
        raise ValueError(
            'selected MEMIT config is missing anchor hyperparameters: '
            + ', '.join(missing)
        )
    binding = {
        'path': str(config_path),
        'sha256': sha256_file(config_path),
        'completed_path': str(completed_path.resolve()),
        'completed_sha256': sha256_file(completed_path),
        'layers': list(layers),
    }
    return config, binding


def build_anchor_hparams(
    source: dict,
    preset: str,
    model_dir: Path,
    *,
    selected_memit: dict | None = None,
) -> dict:
    """Resolve legacy or selected-MEMIT anchor hyperparameters."""
    hp = dict(source)
    if preset == 'selected_memit':
        if selected_memit is None:
            raise ValueError('selected_memit preset requires a loaded MEMIT config')
        for key in SELECTED_MEMIT_HPARAM_KEYS:
            hp[key] = selected_memit[key]
        for key in ('max_length', 'max_new_tokens'):
            if key in selected_memit:
                hp[key] = selected_memit[key]
    elif preset == 'easyedit_memit':
        hp.update({k: v for k, v in UPSTREAM_QWEN25_PRESET.items() if k in hp})
        hp['v_loss_layer'] = 35
    elif preset == 'nas_source':
        hp['v_loss_layer'] = 35
    else:
        raise ValueError(f'unknown NAS anchor preset: {preset}')
    hp.update(
        model_name=str(Path(model_dir).resolve()),
        use_nas=False,
        nas_restart=1000,
    )
    return hp


def load_upstream(root):
    import baselines
    root = root.resolve()
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('NAS upstream commit differs from audited source')
    baselines.__path__.append(str(root / 'baselines'))
    sys.path.insert(0, str(root))
    from util import vstar_stats
    return vstar_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-dir', type=Path, default=Path('external/NAS'))
    p.add_argument('--model-dir', type=Path, default=Path('Qwen3-8B-Base'))
    p.add_argument('--counterfact', type=Path, default=Path('external/CrispEdit/data/multi_counterfact.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument(
        '--preset',
        choices=['nas_source', 'easyedit_memit', 'selected_memit'],
        required=True,
    )
    p.add_argument(
        '--memit-config',
        type=Path,
        default=None,
        help='formal completed MEMIT run_config.json for --preset selected_memit',
    )
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.preset == 'selected_memit' and a.memit_config is None:
        raise ValueError('--memit-config is required with --preset selected_memit')
    if a.preset != 'selected_memit' and a.memit_config is not None:
        raise ValueError('--memit-config is only valid with --preset selected_memit')
    upstream = load_upstream(a.upstream_dir)
    # The pinned loader resolves exactly this filename within DATA_DIR.
    if a.counterfact.name != 'multi_counterfact.json':
        raise ValueError('Expected the official MultiCounterFact filename')
    upstream.DATA_DIR = a.counterfact.resolve().parent
    source_hp = json.loads((a.upstream_dir / 'hparams/MEMIT/Qwen2p5-7B.json').read_text())
    matched_memit = None
    matched_binding = None
    if a.preset == 'selected_memit':
        matched_memit, matched_binding = load_selected_memit_config(
            a.memit_config, a.model_dir
        )
    # Carry the selected run's exact edit/solver settings into the CounterFact
    # norm calibration. This prevents a new 5--8/50k MEMIT run from reusing
    # the historical 4--8/15k anchor by accident.
    hp = build_anchor_hparams(
        source_hp,
        a.preset,
        a.model_dir,
        selected_memit=matched_memit,
    )
    hparams = SimpleNamespace(**hp)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(a.gpu)
    tok = load_tokenizer(a.model_dir)
    tok.padding_side = 'right'
    model = load_model(a.model_dir, torch.device(f'cuda:{a.gpu}'))
    model.eval().requires_grad_(False)
    versions = {name: param._version for name, param in model.named_parameters()}
    mean, meta = upstream._compute_mean_vstar_norm_from_multicounterfact(
        model=model, tok=tok, target_layer=hp['layers'][-1], memit_hparams=hparams,
        nas_restart=1000, seed=0, batch_size=16, outlier_mode='drop', outlier_upper_quantile=0.95,
    )
    assert versions == {name: param._version for name, param in model.named_parameters()}
    assert mean > 0 and torch.isfinite(torch.tensor(mean))
    atomic_json(a.output, {
        'schema_version': 'engram-nas-anchor/v1', 'status': 'complete',
        'upstream_repository': 'SasyaTitech/NAS', 'upstream_commit': COMMIT,
        'upstream_file_sha256': sha256_file(a.upstream_dir / 'util/vstar_stats.py'),
        'model_dir': str(a.model_dir.resolve()), 'model_checkpoint_identity_sha256': sha256_file(Path('runs/protocols/qwen3_8b_base_checkpoint_identity.json')),
        'counterfact_path': str(a.counterfact.resolve()), 'counterfact_sha256': sha256_file(a.counterfact),
        'counterfact_rows': len(json.loads(a.counterfact.read_text())),
        'target_layer': hp['layers'][-1], 'preset': a.preset, 'hyperparameters': hp,
        'matched_memit_config_sha256': (
            matched_binding['sha256'] if matched_binding else None
        ),
        'matched_memit_completed_sha256': (
            matched_binding['completed_sha256'] if matched_binding else None
        ),
        'matched_memit_binding': matched_binding,
        'mean_vstar_norm': mean, 'samples': 1000, 'seed': 0,
        'outlier_policy': {'mode': 'drop', 'upper_quantile': 0.95},
        'parameter_versions_unchanged': True, 'base_model_only': True, 'metadata': meta,
        'qualification': (
            'NAS upstream CounterFact optimization and upper-trimmed norm mean, '
            'evaluated on the Qwen3 original checkpoint and bound to the selected '
            'formal MEMIT configuration.'
            if matched_binding else
            'NAS upstream CounterFact optimization and upper-trimmed norm mean, '
            'evaluated on Qwen3 original checkpoint; transferred final loss layer '
            '27 -> 35.'
        ),
    })
    print(a.output, mean, flush=True)


if __name__ == '__main__':
    main()
