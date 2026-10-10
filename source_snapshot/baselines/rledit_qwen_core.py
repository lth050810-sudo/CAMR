"""RLEdit's released hypernetwork and delayed sequence-gradient update on Qwen.

The network is imported from the pinned upstream source. Ridge solves use an
equivalent dual system when there are fewer answer tokens than input features.
Temporary locality branches restore weights exactly, avoiding BF16 add/subtract
round-trip drift. No optimizer is applied directly to the language model.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from baselines.run_ultraedit import ActivationGradientCapture, collate_examples, get_module


UPSTREAM = Path('/root/engram/external/recent_extra/RLEdit-source')


def upstream_network(source=UPSTREAM):
    identity = json.loads((source / 'SOURCE.json').read_text())
    path = source / 'nets.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != identity['file_sha256']['nets.py']:
        raise ValueError('Pinned RLEdit network changed')
    spec = importlib.util.spec_from_file_location('engram_upstream_rledit_nets', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RLEditNet


def ridge_update(keys, values, ridge):
    """Return (K'K + lambda I)^-1 K'V without differentiating the LLM."""
    if keys.shape[0] < keys.shape[1]:
        matrix = keys @ keys.T + ridge * torch.eye(keys.shape[0], device=keys.device, dtype=keys.dtype)
        return keys.T @ torch.linalg.solve(matrix, values)
    matrix = keys.T @ keys + ridge * torch.eye(keys.shape[1], device=keys.device, dtype=keys.dtype)
    return torch.linalg.solve(matrix, keys.T @ values)


def ridge_adjoint(keys, values, shift, parameter_gradient, ridge):
    """Exact dL/dV and dL/d(log lambda), including the ridge dependence."""
    if keys.shape[0] < keys.shape[1]:
        matrix = keys @ keys.T + ridge * torch.eye(keys.shape[0], device=keys.device, dtype=keys.dtype)
        value_gradient = torch.linalg.solve(matrix, keys @ parameter_gradient)
        solved_values = torch.linalg.solve(matrix, values)
        log_ridge_gradient = -ridge * (value_gradient * solved_values).sum()
    else:
        matrix = keys.T @ keys + ridge * torch.eye(keys.shape[1], device=keys.device, dtype=keys.dtype)
        adjoint = torch.linalg.solve(matrix, parameter_gradient)
        value_gradient = keys @ adjoint
        log_ridge_gradient = -ridge * (adjoint * shift).sum()
    return value_gradient, log_ridge_gradient


def answer_logits(model, batch):
    labels = batch['labels']
    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                   use_cache=False).logits
    return logits[labels != -100].float(), labels[labels != -100]


class RLEditQwen:
    def __init__(self, model, module_names, *, rank=1024, n_blocks=4,
                 edit_lr=1e-6, meta_lr=1e-5, network_batch_size=128,
                 max_grad_norm=1.0, source=UPSTREAM):
        self.model = model
        self.module_names = list(module_names)
        self.device = next(model.parameters()).device
        self.network_batch_size = network_batch_size
        self.max_grad_norm = max_grad_norm
        shapes = Counter()
        self.module_index = {}
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name in self.module_names:
            module = get_module(model, name)
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f'RLEdit requires a linear target: {name}')
            module.weight.requires_grad_(True)
            shape = (module.in_features, module.out_features)
            self.module_index[name] = (shape, shapes[shape])
            shapes[shape] += 1
        network = upstream_network(source)
        self.net = torch.nn.ModuleDict({str(shape): network(*shape, rank, n_blocks, count, edit_lr)
                                       for shape, count in shapes.items()}).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=meta_lr)
        self.original = {name: get_module(model, name).weight.detach().cpu().clone()
                         for name in self.module_names}
        self.meta_optimizer_steps = 0
        self.forward_counts = Counter()

    def batches(self, examples, batch_size, pad_token_id):
        for start in range(0, len(examples), batch_size):
            batch = collate_examples(examples[start:start + batch_size], pad_token_id)
            yield {key: value.to(self.device) for key, value in batch.items()}

    def cache(self, examples, batch_size, pad_token_id):
        captured = {name: [[], []] for name in self.module_names}
        capture = ActivationGradientCapture(self.model, self.module_names)
        self.model.eval()
        try:
            for batch in self.batches(examples, batch_size, pad_token_id):
                self.model.zero_grad(set_to_none=True)
                capture.begin(batch['labels'] != -100)
                logits, labels = answer_logits(self.model, batch)
                F.cross_entropy(logits, labels).backward()
                self.forward_counts['edit_backward_batches'] += 1
                for name in self.module_names:
                    keys = capture.keys[name].float()
                    values = capture.value_grads[name].float()
                    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
                        raise FloatingPointError('Nonfinite cached edit features')
                    shape, _ = self.module_index[name]
                    if keys.shape[0] < 2:
                        raise ValueError('Upstream sample variance needs at least two answer tokens')
                    with torch.no_grad():
                        self.net[str(shape)].normalizer.update(torch.cat((keys, values), -1))
                    captured[name][0].append(keys.detach())
                    captured[name][1].append(values.detach())
        finally:
            capture.close()
            self.model.zero_grad(set_to_none=True)
        return {name: (torch.cat(parts[0]), torch.cat(parts[1])) for name, parts in captured.items()}

    def value_differences(self, name, keys, values):
        shape, index = self.module_index[name]
        net = self.net[str(shape)]
        index = torch.tensor([index], device=self.device, dtype=torch.long)
        pseudo_keys, pseudo_values = net(keys, values, index)
        coefficients = -net.lr(index) * (keys * pseudo_keys).sum(-1, keepdim=True)
        return coefficients * pseudo_values

    @torch.no_grad()
    def predict(self, features):
        shifts, differences = {}, {}
        for name, (keys, values) in features.items():
            chunks = [self.value_differences(name, keys[i:i + self.network_batch_size],
                                             values[i:i + self.network_batch_size])
                      for i in range(0, keys.shape[0], self.network_batch_size)]
            differences[name] = torch.cat(chunks)
            shape, index = self.module_index[name]
            ridge = self.net[str(shape)].lamda.weight[index].exp().squeeze()
            shifts[name] = ridge_update(keys, differences[name], ridge)
            if not torch.isfinite(shifts[name]).all():
                raise FloatingPointError('Nonfinite RLEdit parameter update')
        return shifts, differences

    @torch.no_grad()
    def snapshot(self):
        return {name: get_module(self.model, name).weight.detach().clone() for name in self.module_names}

    @torch.no_grad()
    def restore(self, snapshot):
        for name, weight in snapshot.items():
            get_module(self.model, name).weight.copy_(weight)

    @torch.no_grad()
    def apply(self, shifts):
        for name, shift in shifts.items():
            weight = get_module(self.model, name).weight
            weight.add_(shift.T.to(weight.dtype))

    def update_hypernetwork(self, features, differences, shifts):
        for name, (keys, values) in features.items():
            gradient = get_module(self.model, name).weight.grad
            if gradient is None:
                raise RuntimeError('Missing language-model reward gradient')
            shape, index_number = self.module_index[name]
            net = self.net[str(shape)]
            index = torch.tensor([index_number], device=self.device, dtype=torch.long)
            with torch.no_grad():
                value_gradient, ridge_gradient = ridge_adjoint(
                    keys, differences[name], shifts[name], gradient.float().T,
                    net.lamda(index).exp().squeeze())
            (net.lamda(index).sum() * ridge_gradient).backward()
            for start in range(0, keys.shape[0], self.network_batch_size):
                end = start + self.network_batch_size
                predicted = self.value_differences(name, keys[start:end], values[start:end])
                (predicted * value_gradient[start:end]).sum().backward()
        norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm,
                                             error_if_nonfinite=True)
        return float(norm)

    def meta_edit(self, current, history, *, batch_size, pad_token_id,
                  time_decay=0.95, back_depth=10, locality_coefficient=0.6):
        features = self.cache(current['edit'], batch_size, pad_token_id)
        shifts, differences = self.predict(features)
        before = self.snapshot()
        self.apply(shifts)
        after = self.snapshot()
        self.model.zero_grad(set_to_none=True)
        generalization_loss = locality_loss = 0.0
        try:
            for distance, previous in enumerate(reversed(history[-back_depth:])):
                weight = time_decay ** distance
                for batch in self.batches(previous['equiv'], batch_size, pad_token_id):
                    logits, labels = answer_logits(self.model, batch)
                    loss = F.cross_entropy(logits, labels)
                    (weight * loss).backward()
                    generalization_loss += float(loss.detach()) * weight
                    self.forward_counts['meta_generalization_backward_batches'] += 1
            for distance, previous in enumerate(reversed(history[-back_depth:])):
                weight = locality_coefficient * time_decay ** distance
                for batch in self.batches(previous['locality'], batch_size, pad_token_id):
                    self.restore(before)
                    with torch.no_grad():
                        reference, _ = answer_logits(self.model, batch)
                        reference = F.log_softmax(reference, -1)
                    self.restore(after)
                    actual, _ = answer_logits(self.model, batch)
                    loss = F.kl_div(F.log_softmax(actual, -1), reference,
                                    log_target=True, reduction='batchmean')
                    (weight * loss).backward()
                    locality_loss += float(loss.detach()) * weight
                    self.forward_counts['meta_locality_reference_batches'] += 1
                    self.forward_counts['meta_locality_backward_batches'] += 1
            norm = self.update_hypernetwork(features, differences, shifts)
        except BaseException:
            self.restore(before)
            raise
        finally:
            self.model.zero_grad(set_to_none=True)
        return {'generalization_loss': generalization_loss, 'locality_loss': locality_loss,
                'hypernetwork_grad_norm': norm, 'backtracked_batches': min(len(history), back_depth),
                'shift_squared_norm': sum(float(s.square().sum()) for s in shifts.values())}

    def finish_meta_sequence(self):
        if not any(p.grad is not None and bool(p.grad.abs().sum()) for p in self.net.parameters()):
            raise RuntimeError('No nonzero editor-training gradient')
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.meta_optimizer_steps += 1

    def reset_language_model(self):
        self.restore(self.original)
        self.model.zero_grad(set_to_none=True)

    def edit(self, examples, *, batch_size, pad_token_id):
        features = self.cache(examples, batch_size, pad_token_id)
        shifts, _ = self.predict(features)
        before = self.snapshot()
        self.apply(shifts)
        changed = any(not torch.equal(get_module(self.model, name).weight, weight)
                      for name, weight in before.items())
        return {'parameter_changed': changed,
                'shift_squared_norm': sum(float(s.square().sum()) for s in shifts.values())}
