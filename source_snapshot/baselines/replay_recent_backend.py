"""Extra collective operations without changing the shared full-parameter backend."""
from __future__ import annotations

import math
import time
import torch
import torch.distributed as dist
from baselines.common import qa_prompt
from baselines.engram_full_backend import FullParameterBackend, _answer_losses
from baselines.run_engram_full import BackendCoordinator


class RecentReplayBackend(FullParameterBackend):
    def train_review(self, rows):
        """Capture pre-update per-item NLL from the actual training forward pass."""
        labels = self._batch(rows)['labels']
        captured = []
        def capture(module, inputs, output):
            with torch.no_grad():
                sums, counts = _answer_losses(output.logits.detach(), labels)
                captured.append((sums/counts).float().cpu().tolist())
        handle = self.model.register_forward_hook(capture)
        try:
            stats = self.train(rows, 1)
        finally:
            handle.remove()
        if len(captured) != 1 or len(captured[0]) != len(rows):
            raise RuntimeError('Expected exactly one root training forward')
        return captured[0], stats

    @torch.no_grad()
    def sample(self, questions, temperature):
        """Synchronized FSDP categorical sampling, temperature=.1, top-p=1."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Positive finite sampling temperature required')
        self.model.eval()
        started = time.perf_counter()
        answers, tokens, padded, calls, output_tokens = [], 0, 0, 0, 0
        previous = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos
        try:
            for start in range(0, len(questions), self.eval_batch_size):
                encoded = self.tokenizer([qa_prompt(q) for q in questions[start:start+self.eval_batch_size]],
                    return_tensors='pt', padding=True, truncation=True,
                    max_length=self.max_length, add_special_tokens=True)
                ids = encoded['input_ids'].to(self.device)
                mask = encoded['attention_mask'].to(self.device)
                finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=self.device)
                generated = []
                for _ in range(self.max_new_tokens):
                    positions = mask.long().cumsum(-1)-1
                    positions.masked_fill_(mask == 0, 0)
                    logits = self.model(input_ids=ids, attention_mask=mask,
                                        position_ids=positions, use_cache=False).logits
                    probabilities = torch.softmax(logits[:,-1].float()/temperature, dim=-1)
                    next_tokens = torch.multinomial(probabilities, 1).squeeze(-1)
                    if self.world_size > 1:
                        dist.broadcast(next_tokens, src=0)
                    active = ~finished
                    next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens,pad))
                    generated.append(next_tokens)
                    tokens += int(mask.sum())
                    padded += ids.numel()
                    calls += 1
                    output_tokens += int(active.sum())
                    if eos is not None:
                        finished |= next_tokens == eos
                    del logits, probabilities
                    if bool(finished.all()):
                        break
                    ids = torch.cat((ids,next_tokens[:,None]),dim=1)
                    mask = torch.cat((mask,active.long()[:,None]),dim=1)
                answers.extend(self.tokenizer.batch_decode(torch.stack(generated,dim=1),skip_special_tokens=True))
        finally:
            self.tokenizer.padding_side = previous
        self.last_generation_stats = {
            'forward_calls': calls, 'logical_sequence_tokens': tokens,
            'physical_sequence_tokens': tokens*self.world_size,
            'physical_padded_sequence_tokens': padded*self.world_size,
            'output_tokens': output_tokens,
            'wall_seconds': self._reduce(time.perf_counter()-started, dist.ReduceOp.MAX),
        }
        self.forward_accounting['generation_calls'] += 1
        for key in ('forward_calls','logical_sequence_tokens','physical_sequence_tokens',
                    'physical_padded_sequence_tokens','output_tokens'):
            self.forward_accounting['generation_'+key] += self.last_generation_stats[key]
        return answers

    def mir_scores(self, incoming, candidates, virtual_lr):
        """MIR true loss-increase ranking; exact CPU shard snapshot/restoration.

        No optimizer state or real-step counter is changed by virtual SGD.
        CPU snapshots avoid a second model-size GPU allocation on two GPUs.
        """
        before, _ = self.answer_nll(candidates)
        params = list(self.model.parameters())
        snapshots = [p.detach().cpu().clone() for p in params]
        self.optimizer.zero_grad(set_to_none=True)
        self.model.eval()
        try:
            batch = self._batch(incoming)
            logits = self.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
            sums, counts = _answer_losses(logits,batch['labels'])
            loss = sums.sum()/counts.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite MIR virtual loss')
            loss.backward()
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-virtual_lr)
            self.optimizer.zero_grad(set_to_none=True)
            del loss, logits, sums, counts, batch
            after, _ = self.answer_nll(candidates)
        finally:
            with torch.no_grad():
                for p, saved in zip(params,snapshots):
                    p.copy_(saved)
            self.optimizer.zero_grad(set_to_none=True)
        return [{'before':a,'after':b,'interference':b-a} for a,b in zip(before,after)]


class RecentCoordinator(BackendCoordinator):
    def train_review(self, rows):
        self._send('train_review',rows=rows)
        return self.backend.train_review(rows)

    def sample(self, questions, temperature):
        self._send('sample',questions=questions,temperature=temperature)
        return self.backend.sample(questions,temperature)

    def mir_scores(self, incoming, candidates, virtual_lr):
        self._send('mir_scores',incoming=incoming,candidates=candidates,virtual_lr=virtual_lr)
        return self.backend.mir_scores(incoming,candidates,virtual_lr)


def serve_recent(backend):
    while True:
        command = [None]
        dist.broadcast_object_list(command,src=0)
        c = command[0]
        op = c.pop('op')
        if op == 'shutdown':
            return
        if op == 'nll':
            backend.answer_nll(c['rows'])
        elif op == 'generate':
            backend.generate(c['questions'])
        elif op == 'train':
            backend.train(c['rows'],c['steps'])
        elif op == 'save':
            backend.save_checkpoint(c['path'],None)
        elif op == 'load':
            backend.load_checkpoint(c['path'])
        elif op in ('train_review','sample','mir_scores'):
            getattr(backend,op)(**c)
        else:
            raise ValueError('Unsupported recent replay collective: '+op)
