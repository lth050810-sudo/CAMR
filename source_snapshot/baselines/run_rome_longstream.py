"""Sequential ROME/R-ROME edits on the frozen WikiBigEdit long stream."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
import types
from pathlib import Path

import torch
import yaml

from baselines.common import append_jsonl, generate_answers, load_model, load_tokenizer, sha256_file
from baselines.longstream_protocol import PREFIXES, atomic, evaluate
from baselines.run_memit import make_request
from baselines.run_rome_family import (
    COMMIT, _bounded_scale, _install_qwen3_covariance, _validate_qwen3, upstream_modules,
)


class Generator:
    def __init__(self, model, tokenizer, device, batch_size, max_length, max_new_tokens):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.batch_size, self.max_length, self.max_new_tokens = batch_size, max_length, max_new_tokens

    def generate(self, questions):
        return generate_answers(self.model, self.tokenizer, questions, self.device,
                                batch_size=self.batch_size, max_length=self.max_length,
                                max_new_tokens=self.max_new_tokens)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('r_rome', 'rome'), required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('/root/engram/Qwen3-8B-Base'))
    p.add_argument('--upstream-dir', type=Path, default=Path('/root/engram/external/EasyEdit'))
    p.add_argument('--mom2-stats-dir', type=Path, default=Path('/mnt/yy2607004_hs/engram_memit_stats'))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target', type=int, default=100000)
    p.add_argument('--eval-batch-size', type=int, default=48)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--max-new-tokens', type=int, default=16)
    return p.parse_args(argv)


def run(a):
    manifest = json.loads(a.manifest.read_text())
    if manifest.get('schema_version') != 'wikibigedit-longstream/v1':
        raise ValueError('Expected frozen longstream manifest')
    rows = manifest['records']
    if [r['sample_id'] for r in rows] != manifest['selection']['sample_ids']:
        raise ValueError('Frozen stream order differs')
    target = min(a.target, int(json.loads(a.control.read_text())['target_events']))
    if target not in (50000, 100000):
        raise ValueError('Shared target must be 50K or 100K')
    manifest_hash = sha256_file(a.manifest)
    config_path = a.output_dir / 'run_config.json'
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        config = json.loads(config_path.read_text())
        if config['manifest_sha256'] != manifest_hash or config['method'] != a.method or config['target'] != target:
            raise ValueError('Resume configuration differs')
    else:
        a.output_dir.mkdir(parents=True)
        config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
        config.update(manifest_sha256=manifest_hash, target=target,
                      backend={'world_size': 1, 'parameter_mode': 'editor'},
                      layers=[5], loss_layer=35, mom2_adjustment=True,
                      max_update_weight_ratio=0.075, max_cumulative_weight_drift_ratio=4.0,
                      upstream_commit=COMMIT, source_sha256=sha256_file(Path(__file__)),
                      selection_provenance='Transferred 3K Qwen3-selected recipe; longstream overlap not audited')
        atomic(config_path, config)
    a.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    preset_name = 'R-ROME' if a.method == 'r_rome' else 'ROME'
    raw = subprocess.check_output(['git', '-C', str(a.upstream_dir), 'show',
                                   f'{COMMIT}:hparams/{preset_name}/qwen2.5-7b.yaml'], text=True)
    preset = yaml.safe_load(raw)
    preset.update(layers=[5], device=a.gpu, model_name=str(a.model_dir.resolve()),
                  v_loss_layer=35, mom2_adjustment=True)
    if a.method == 'r_rome':
        preset.update(enable_prompt_keys=False, enable_random_prefix_keys=True,
                      original_implementation=False)
    for key in ('v_lr', 'v_weight_decay', 'kl_factor', 'clamp_norm_factor'):
        preset[key] = float(preset[key])
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.cuda.set_device(a.gpu)
    device = torch.device(f'cuda:{a.gpu}')
    modules = upstream_modules(a.upstream_dir)
    module = modules[a.method]
    execute = module.execute_r_rome if a.method == 'r_rome' else module.execute_rome
    tokenizer = load_tokenizer(a.model_dir)
    tokenizer.padding_side = 'right'
    model = load_model(a.model_dir, device)
    _validate_qwen3(model, [5], 35)
    model.eval().requires_grad_(False)
    _install_qwen3_covariance(model, a.model_dir, a.mom2_stats_dir, [5], 1e-5)
    hp = types.SimpleNamespace(**preset)
    module.get_context_templates(model, tokenizer, hp.context_template_length_params)
    parameter_name = hp.rewrite_module_tmp.format(5) + '.weight'
    weight = dict(model.named_parameters())[parameter_name]
    original = weight.detach().cpu().clone()
    original_norm = float(weight.detach().float().norm())
    cumulative = torch.zeros_like(weight, dtype=torch.float32)
    generator = Generator(model, tokenizer, device, a.eval_batch_size, a.max_length, a.max_new_tokens)
    pointer = a.output_dir / 'latest_checkpoint.json'
    metrics_path = a.output_dir / 'metrics.json'
    predictions_path = a.output_dir / 'predictions.jsonl'
    trace_path = a.output_dir / 'edit_trace.jsonl'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    cursor = 0
    if pointer.exists():
        latest = json.loads(pointer.read_text())
        cursor = latest['event_cursor']
        saved = torch.load(Path(latest['path']) / 'weight.pt', map_location='cpu', weights_only=True)
        if saved['event_cursor'] != cursor or saved['manifest_sha256'] != manifest_hash:
            raise ValueError('Checkpoint identity differs')
        with torch.no_grad():
            weight.copy_(saved['weight'].to(device=device, dtype=weight.dtype))
        cumulative.copy_((weight.detach().float() - original.to(device=device).float()))
        committed = [m for m in metrics if m['events'] <= cursor]
        if len(committed) != len(metrics):
            metrics = committed
            atomic(metrics_path, metrics)
    elif not metrics:
        base, records = evaluate(generator, rows, manifest, 0, 'base', base=True)
        metrics.append(base)
        atomic(metrics_path, metrics)
        append_jsonl(predictions_path, records)
    if trace_path.exists():
        lines = trace_path.read_text().splitlines()
        if len(lines) < cursor:
            raise ValueError('Edit trace is shorter than checkpoint')
        if len(lines) > cursor:
            trace_path.write_text('\n'.join(lines[:cursor]) + '\n' if cursor else '')
    if predictions_path.exists():
        committed_checkpoints = {m['checkpoint'] for m in metrics}
        temporary = predictions_path.with_suffix('.jsonl.tmp')
        with predictions_path.open() as source, temporary.open('w') as target_file:
            for line in source:
                if json.loads(line)['checkpoint'] in committed_checkpoints:
                    target_file.write(line)
        temporary.replace(predictions_path)
    started = time.monotonic()
    boundaries = set(PREFIXES) | set(range(12500, target + 1, 12500))
    for index in range(cursor, target):
        row = rows[index]
        request = make_request(row, index)
        request['target_new'] = str(row['ans']).strip()
        before = time.monotonic()
        factors = execute(model, tokenizer, request, hp)
        if set(factors) != {parameter_name}:
            raise ValueError('Unexpected ROME parameter set')
        left, right = factors[parameter_name]
        update = module.upd_matrix_match_shape(left[:, None] @ right[None, :], weight.shape)
        if not torch.isfinite(update).all():
            raise FloatingPointError('Nonfinite ROME update')
        scale, audit = _bounded_scale(cumulative, update, original_norm,
                                      max_update_ratio=0.075, max_cumulative_ratio=4.0)
        applied = (update * scale).to(weight.dtype)
        with torch.no_grad():
            weight.add_(applied)
        if not torch.isfinite(weight).all():
            raise FloatingPointError('Nonfinite ROME weight')
        cumulative.add_(applied.float())
        drift = float(cumulative.norm()) / original_norm
        if drift > 4.0 * (1.0 + 1e-5):
            raise FloatingPointError('ROME cumulative trust region exceeded')
        count = index + 1
        append_jsonl(trace_path, [{'edit': count, 'sample_id': row['sample_id'],
                                   'update_weight_ratios': {parameter_name: audit['update_ratio']},
                                   'cumulative_weight_drift_ratios': {parameter_name: drift},
                                   'update_scales': {parameter_name: scale}, 'finite': True,
                                   'seconds': time.monotonic() - before}])
        if count % 25 == 0 or count in boundaries:
            atomic(a.output_dir / 'progress.json', {'status': 'running', 'events': count,
                   'target_events': target, 'elapsed_seconds': time.monotonic() - started})
        if count not in boundaries:
            continue
        checkpoint = f'prefix{count:06d}' if count in PREFIXES else f'stage{row["stage"]:02d}'
        result, records = evaluate(generator, rows, manifest, count, checkpoint,
                                   full_prefix=count in PREFIXES)
        result['cumulative_cost'] = {'current_fact_presentations': count,
                                     'replay_fact_presentations': 0,
                                     'elapsed_seconds': time.monotonic() - started,
                                     'gpu_hours': (time.monotonic() - started) / 3600}
        folder = a.checkpoint_dir / checkpoint
        folder.mkdir(parents=True, exist_ok=True)
        saved = {'event_cursor': count, 'manifest_sha256': manifest_hash,
                 'weight': weight.detach().cpu().clone()}
        temporary = folder / 'weight.pt.tmp'
        torch.save(saved, temporary)
        temporary.replace(folder / 'weight.pt')
        atomic(folder / 'checkpoint_meta.json', {'event_cursor': count, 'manifest_sha256': manifest_hash,
                                                 'method': a.method, 'weight_file': 'weight.pt'})
        metrics.append(result)
        atomic(metrics_path, metrics)
        append_jsonl(predictions_path, records)
        atomic(pointer, {'event_cursor': count, 'path': str(folder)})
    atomic(a.output_dir / 'completed.json', {'status': 'complete', 'events': target,
           'edits': target, 'method': a.method, 'totals': {'current_fact_presentations': target,
           'replay_fact_presentations': 0}, 'elapsed_seconds': time.monotonic() - started})
    atomic(a.output_dir / 'progress.json', {'status': 'complete', 'events': target,
           'target_events': target, 'elapsed_seconds': time.monotonic() - started})


if __name__ == '__main__':
    run(parse_args())
