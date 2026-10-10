"""O-LoRA's frozen growing subspaces and orthogonality penalty on the shared Qwen3 recipe."""

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from baselines.common import (atomic_json, environment_metadata, group_by_stage, load_manifest,
                              load_model, load_tokenizer, manifest_panel, sha256_file, train_answer_only)
from baselines.run_lora_family import evaluate_and_write

COMMIT = '07117e1fc4a5f5ad9308a815a42cee8f46502dc8'
TARGETS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}


class OrthogonalLinear(torch.nn.Module):
    def __init__(self, base, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.scaling = rank, alpha / rank
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('past_a', torch.empty(0, base.in_features, device=base.weight.device))
        self.register_buffer('past_b', torch.empty(base.out_features, 0, device=base.weight.device))
        self.register_parameter('current_a', None)
        self.register_parameter('current_b', None)

    @property
    def weight(self):
        return self.base.weight

    def begin_stage(self):
        if self.current_a is not None:
            raise RuntimeError('Consolidate the current stage before starting another')
        self.current_a = torch.nn.Parameter(torch.empty(self.rank, self.base.in_features, device=self.weight.device))
        self.current_b = torch.nn.Parameter(torch.zeros(self.base.out_features, self.rank, device=self.weight.device))
        torch.nn.init.kaiming_uniform_(self.current_a, a=math.sqrt(5))

    def consolidate(self):
        self.past_a = torch.cat((self.past_a, self.current_a.detach()), dim=0)
        self.past_b = torch.cat((self.past_b, self.current_b.detach()), dim=1)
        self.current_a = None
        self.current_b = None

    def orthogonal_loss(self):
        return (self.past_a @ self.current_a.T).abs().sum()

    def forward(self, x):
        result = self.base(x)
        dropped = self.dropout(x.float())
        if self.past_a.shape[0]:
            result = result + (F.linear(F.linear(dropped, self.past_a), self.past_b) * self.scaling).to(result.dtype)
        if self.current_a is not None:
            result = result + (F.linear(F.linear(dropped, self.current_a), self.current_b) * self.scaling).to(result.dtype)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('Qwen3-8B-Base'))
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--max-edits', type=int, default=0)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=5)
    a = p.parse_args()
    if a.epochs < 1:
        p.error('--epochs must be positive')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    source = Path('external/O-LoRA')
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('Unexpected O-LoRA source revision')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    a.max_length, a.max_new_tokens = 128, 16
    random.seed(a.seed)
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
    model.requires_grad_(False)
    backbone_versions = {n: p._version for n, p in model.named_parameters()}
    backbone_params = dict(model.named_parameters())
    modules = {}
    for name, module in list(model.named_modules()):
        if name.split('.')[-1] in TARGETS and isinstance(module, torch.nn.Linear):
            parent_name, child = name.rsplit('.', 1)
            wrapped = OrthogonalLinear(module)
            setattr(model.get_submodule(parent_name), child, wrapped)
            modules[name] = wrapped
    if not modules:
        raise ValueError('No Qwen3 target modules found')
    config = {**{k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'method': 'O-LoRA', 'upstream_repository': 'cmnfriend/O-LoRA', 'upstream_commit': COMMIT,
              'manifest_protocol_id': manifest['protocol_id'], 'manifest_sha256': sha256_file(a.manifest),
              'num_facts': len(rows), 'stage_counts': {str(s): len(v) for s, v in stages.items()},
              'closed_book_evaluation': True, 'formal_baseline_eligible': not a.max_edits,
              'qwen3_preset_status': 'Qwen3 transfer with shared LoRA optimization budget; no official Qwen3 preset',
              'rank_per_stage': 16, 'alpha': 32, 'dropout': 0.05, 'target_modules': sorted(TARGETS),
              'epochs': a.epochs, 'batch_size': 8, 'learning_rate': 2e-4, 'optimizer_reset_each_stage': True,
              'lambda_orthogonal': 0.5, 'lambda_l2': 0.0,
              'orthogonal_loss': 'sum over modules of abs(A_past @ A_current.T).sum(), source trainer line 96',
              'past_adapter_policy': 'freeze and concatenate all previous A/B; all are active at inference; no task routing',
              'budget_note': f'Shared all-linear r16 recipe with {a.epochs} pass(es) per stage. Stored rank grows by 16 per stage; report memory separately.',
              **environment_metadata()}
    atomic_json(a.output_dir / 'run_config.json', config)
    logs, metrics, seen = [], [], []
    started = time.time()
    for stage, current in stages.items():
        for module in modules.values():
            module.begin_stage()
        components = {'answer_loss_sum': 0., 'orthogonal_loss_sum': 0., 'batches': 0}

        def add_regularizer(_model, _inputs, output):
            if output.loss is not None:
                regularizer = sum(module.orthogonal_loss() for module in modules.values())
                components['answer_loss_sum'] += float(output.loss.detach())
                components['orthogonal_loss_sum'] += float(regularizer.detach())
                components['batches'] += 1
                output.loss = output.loss + 0.5 * regularizer
            return output

        hook = model.register_forward_hook(add_regularizer)
        try:
            stats, optimizer = train_answer_only(model, tok, current, device,
                seed=a.seed + stage, epochs=a.epochs, batch_size=8, gradient_accumulation_steps=1,
                learning_rate=2e-4, weight_decay=0., max_grad_norm=1., max_length=128)
        finally:
            hook.remove()
        del optimizer
        assert backbone_versions == {n: p._version for n, p in backbone_params.items()}
        for module in modules.values():
            module.consolidate()
        log = {'stage': stage, 'current_count': len(current), 'replay_count': 0, 'future_leak_count': 0,
               'stored_rank_per_module': stage * 16, 'frozen_backbone_verified': True,
               'adapter_parameters_stored': sum(m.past_a.numel() + m.past_b.numel() for m in modules.values()),
               **asdict(stats), **components}
        logs.append(log)
        atomic_json(a.output_dir / 'training_log.json', logs)
        seen.extend(current)
        metrics.append(evaluate_and_write(model, tok, seen, loc, mh, device, a, f'tau{stage}', stage))
        atomic_json(a.output_dir / 'metrics.json', metrics)
    adapter_dir = a.output_dir / 'adapters'
    adapter_dir.mkdir(exist_ok=True)
    torch.save({'method': 'O-LoRA', 'config': config, 'modules': {
        n: {'past_a': m.past_a.cpu(), 'past_b': m.past_b.cpu(), 'scaling': m.scaling}
        for n, m in modules.items()}}, adapter_dir / 'olora.pt')
    atomic_json(a.output_dir / 'completed.json', {'status': 'complete', 'method': 'O-LoRA',
                'num_facts': len(rows), 'final_stage': max(stages), 'elapsed_seconds': time.time() - started})


if __name__ == '__main__':
    main()
