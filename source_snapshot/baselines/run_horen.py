"""Pinned HoReN value-memory editor with shared Qwen3 generated evaluation."""

import argparse
import importlib
import json
import math
import random
import subprocess
import sys
import time
import types
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import yaml

from baselines.common import (atomic_json, append_jsonl, environment_metadata, group_by_stage,
                              load_manifest, load_model, load_tokenizer, manifest_panel, qa_prompt, sha256_file)
from baselines.run_lora_family import evaluate_and_write

COMMIT = '38bbf34d4f37a2bb1b24dbef6b8bc1fb5554fbae'


@contextmanager
def frozen_prefix(model, stop_layer):
    """Cache the unchanged Qwen3 prefix for repeated forwards of one edit's tokens."""
    if model.training or model.config.use_cache:
        raise ValueError('Prefix caching requires eval mode and use_cache=False')
    layers = model.model.layers[:stop_layer]
    if any(p.requires_grad for layer in layers for p in layer.parameters()):
        raise ValueError('Cannot cache a trainable prefix')
    originals = [(layer, layer.forward) for layer in layers]

    def cached_forward(original):
        cache = []
        def forward(*args, **kwargs):
            if not cache:
                out = original(*args, **kwargs)
                if not isinstance(out, torch.Tensor) or out.requires_grad:
                    raise ValueError('Expected a detached Qwen3 prefix tensor')
                cache.append(out.detach().clone())
                return out
            return cache[0]
        return forward

    try:
        for layer, original in originals:
            layer.forward = cached_forward(original)
        yield
    finally:
        for layer, original in originals:
            layer.forward = original


def upstream_modules(root):
    for name, path in [('horen_upstream', root / 'src'), ('horen_upstream.models', root / 'src/models'),
                       ('horen_upstream.models.horen', root / 'src/models/horen'),
                       ('horen_upstream.util', root / 'src/util')]:
        module = types.ModuleType(name)
        module.__path__ = [str(path.resolve())]
        sys.modules[name] = module
    editor = importlib.import_module('horen_upstream.models.horen.editor')
    utils = importlib.import_module('horen_upstream.models.horen.utils')
    return editor, utils


def unpadded_query(original, layer_input, last_index, mask):
    """Preserve upstream single-query pooling for each left-padded prefill row."""
    if mask is None or layer_input.shape[1] == 1:
        return original(layer_input, last_index)
    if mask.shape != layer_input.shape[:2]:
        raise ValueError('Unexpected HoReN prefill attention-mask shape')
    queries = []
    for row, row_mask in zip(layer_input, mask):
        active = row[row_mask.bool()]
        queries.append(original(active.unsqueeze(0), len(active) - 1))
    return torch.cat(queries, dim=0)


class MemoryReader(torch.nn.Module):
    def __init__(self, editor, adapter, slot_stages):
        super().__init__()
        self.editor = editor
        self.config = editor.model.config
        self.slot_stages = slot_stages
        self.stage = 0
        self.audit = Counter()
        self._mask = None
        original = adapter._select_query
        adapter._select_query = lambda x, i: unpadded_query(original, x, i, self._mask)

        def record_route(module, _inputs, _output):
            if self._mask is not None:
                chosen = module.chosen_key.tolist()
                for slot in chosen:
                    self.audit['routing_decisions'] += 1
                    if self.slot_stages[slot] > self.stage:
                        self.audit['future_leak_count'] += 1
                        raise RuntimeError('HoReN read a future memory slot')
                self.audit['max_slot_read'] = max([self.audit['max_slot_read'], *chosen])
        adapter.register_forward_hook(record_route)

    def generate(self, *args, **kwargs):
        self._mask = kwargs['attention_mask']
        try:
            return self.editor.generate(*args, **kwargs)
        finally:
            self._mask = None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('Qwen3-8B-Base'))
    p.add_argument('--upstream-dir', type=Path, default=Path('external/HoReN_snapshot'))
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--eval-batch-size', type=int, default=48)
    p.add_argument('--max-edits', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cache-frozen-prefix', action='store_true')
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    if subprocess.check_output(['git', '-C', str(a.upstream_dir), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('Unexpected HoReN source revision')
    code, utils = upstream_modules(a.upstream_dir)
    source = a.upstream_dir / 'hparams/HOREN/qwen2.5-7b.yaml'
    hp = yaml.safe_load(source.read_text())
    hp.update(device=a.gpu, model_name=str(a.model_dir.resolve()))
    a.max_length, a.max_new_tokens = 128, 16
    a.output_dir.mkdir(parents=True, exist_ok=True)
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
    tok.padding_side = 'right'
    model = load_model(a.model_dir, device)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    backbone = dict(model.named_parameters())
    versions = {n: p._version for n, p in backbone.items()}
    editor = code.HOREN(types.SimpleNamespace(**hp), model)
    adapter = model.get_submodule(utils.brackets_to_periods(editor.layer))
    slot_stages, slot_ids = [0], [None]
    reader = MemoryReader(editor, adapter, slot_stages)
    config = {**{k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'method': 'HoReN', 'method_family': 'side memory', 'upstream_repository': 'ha11ucin8/HoReN',
              'upstream_commit': COMMIT, 'source_preset_sha256': sha256_file(source), 'hyperparameters': hp,
              'manifest_protocol_id': manifest['protocol_id'], 'manifest_sha256': sha256_file(a.manifest),
              'num_facts': len(rows), 'stage_counts': {str(s): len(v) for s, v in stages.items()},
              'closed_book_evaluation': True, 'open_book_evaluation': False, 'formal_baseline_eligible': not a.max_edits,
              'qwen3_preset_status': 'Qwen2.5 layer24 source preset transferred to Qwen3; no official Qwen3-8B-Base preset',
              'router': 'normalized Hopfield side memory, threshold0.55, one retrieval iteration',
              'backbone_frozen': True, 'source_algorithm_called_directly': True,
              'compatibility_changes': ['shared QA prompts and greedy five-axis evaluation',
                  'batch prefill pools only unpadded prompt tokens, matching source per-example query selection',
                  'preserve source cached-generation behavior: re-query on the single decoding token after prefill'],
              'qualification': 'Untuned source Qwen2.5 value-mode transfer; not the LLaMA paper result',
              **environment_metadata()}
    atomic_json(a.output_dir / 'run_config.json', config)
    metrics, seen, routing = [], [], []
    started, count = time.time(), 0
    saved = Path('/mnt/yy2607004_hs/engram_checkpoints') / a.output_dir.name
    saved.mkdir(parents=True, exist_ok=True)
    for stage, stage_rows in stages.items():
        for row in stage_rows:
            before, old_slots = time.time(), len(adapter.keys)
            tokens = utils.tokenize_request({'prompt': qa_prompt(row['update']), 'target_new': row['ans']}, tok, device)
            if tokens['input_ids'].shape[1] > 128 or not (tokens['labels'] != -100).any():
                raise ValueError('HoReN edit is truncated or has no answer tokens')
            with frozen_prefix(model, 24) if a.cache_frozen_prefix else nullcontext():
                editor.edit(tokens)
            new_slots = len(adapter.keys)
            if new_slots not in (old_slots, old_slots + 1):
                raise ValueError('Unexpected HoReN memory growth')
            if new_slots > old_slots:
                slot_stages.append(stage)
                slot_ids.append(row['sample_id'])
            if not all(math.isfinite(x) for x in editor.losses):
                raise FloatingPointError('Non-finite HoReN loss')
            chosen = editor.edit_log['chosen_key']
            if not torch.isfinite(adapter.values[chosen]).all():
                raise FloatingPointError('Non-finite HoReN value')
            count += 1
            append_jsonl(a.output_dir / 'edit_trace.jsonl', [{'edit': count, 'stage': stage, 'sample_id': row['sample_id'],
                'finite': True, 'losses': editor.losses, 'codebook_slots': new_slots, 'chosen_key': chosen,
                'seconds': time.time() - before}])
            if count % 25 == 0:
                atomic_json(a.output_dir / 'progress.json', {'edits': count, 'total': len(rows), 'stage': stage,
                    'codebook_slots': new_slots, 'elapsed_seconds': time.time() - started})
        assert versions == {n: p._version for n, p in backbone.items()}
        seen.extend(stage_rows)
        reader.stage = stage
        reader.audit.clear()
        reader.audit['future_leak_count'] = 0
        metrics.append(evaluate_and_write(reader, tok, seen, loc, mh, device, a, f'tau{stage}', stage))
        atomic_json(a.output_dir / 'metrics.json', metrics)
        routing.append({'stage': stage, 'codebook_slots': len(slot_ids), 'frozen_backbone_verified': True, **reader.audit})
        atomic_json(a.output_dir / 'routing_audit.json', routing)
        torch.save({'stage': stage, 'edits': count, 'keys': adapter.keys.detach().cpu(),
            'values': adapter.values.detach().cpu(), 'key_labels': [v.cpu() for v in adapter.key_labels],
            'slot_stages': list(slot_stages), 'slot_ids': list(slot_ids), 'hyperparameters': hp}, saved / f'tau{stage}_memory.pt')
    atomic_json(a.output_dir / 'completed.json', {'status': 'complete', 'method': 'HoReN',
        'num_facts': len(rows), 'final_stage': max(stages), 'edits': count, 'elapsed_seconds': time.time() - started})


if __name__ == '__main__':
    main()
