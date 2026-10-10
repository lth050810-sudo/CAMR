"""TRACE batching adapter: unchanged logical CE, views, and optimizer schedule.

FSDP averages rank gradients. Each view is assigned to exactly one rank and
its weighted summed CE is multiplied by world_size before backward. Equal
numbers of microbatches on every rank keep FSDP collectives aligned. Dummy
rows have zero loss weight. Long prompts never materialize full-vocabulary
logits: only answer-predicting positions enter the LM head.
"""
import math
import time
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F

from baselines.engram_full_backend import FullParameterBackend
from baselines.engram_grad_norm import clip_full_shard_grad_norm_
from baselines.engram_training_views import expand_training_views
from trace_protocol import CONFIG, loss_coefficients, training_encoding, trim_prompt


class TraceBackend(FullParameterBackend):
    def __init__(self, args):
        super().__init__(args)
        self.task = args.trace_task
        self.micro_token_budget = args.trace_micro_tokens
        self.metadata.update(batch_strategy='views partitioned across ranks; exact logical objective; zero-weight dummy padding',
            generation='greedy KV-cached TRACE completion, distributed examples, full fixed test',
            trace_task=self.task, training_prompt='original TRACE prompt with one-space answer separator',
            cost_accounting='physical tokens count rank partitions plus dummy work; backward/recomputation excluded')

    @lru_cache(maxsize=30000)
    def encoded(self, question, answer):
        return training_encoding(self.tokenizer, {'update': question, 'ans': answer}, self.task)

    def _sum_loss(self, encoded):
        return self._sum_losses([encoded])[0]

    def _sum_losses(self, examples):
        length = max(len(x[0]) for x in examples)
        ids = torch.full((len(examples), length), self.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        needed = sorted({i for _, prefix, count in examples for i in range(prefix-1, prefix+count-1)})
        positions = torch.tensor(needed, device=self.device)
        mapping = {position: i for i, position in enumerate(needed)}
        for i, (seq, _, _) in enumerate(examples):
            ids[i, :len(seq)] = torch.tensor(seq, device=self.device)
            mask[i, :len(seq)] = 1
        logits = self.model(input_ids=ids, attention_mask=mask,
                            use_cache=False, logits_to_keep=positions).logits
        sums = []
        for i, (seq, prefix, count) in enumerate(examples):
            select = torch.tensor([mapping[j] for j in range(prefix-1, prefix+count-1)], device=self.device)
            sums.append(F.cross_entropy(logits[i, select].float(), ids[i, prefix:len(seq)], reduction='sum'))
        return torch.stack(sums)

    def _dummy(self):
        eos = self.tokenizer.eos_token_id
        return [eos, eos], 1, 1

    def _gather(self, obj):
        if self.world_size == 1:
            return [obj]
        out = [None] * self.world_size
        dist.all_gather_object(out, obj)
        return out

    def train(self, rows, steps, *, has_current=True):
        if steps == 0:
            return super().train(rows, steps, has_current=has_current)
        expanded, facts = expand_training_views(rows, self.training_views,
            self.training_view_cache, self.training_view_policy)
        encoded = [self.encoded(r['update'], r['ans']) for r in expanded]
        counts = [x[2] for x in encoded]
        coeff = loss_coefficients(counts, facts, self.training_loss, self.current_loss_weight, has_current)
        current = {i for i, r in enumerate(rows) if r.get('_stream_role',
            'current' if has_current and i == 0 else 'replay') == 'current'}
        current_views = [i for i, f in enumerate(facts) if f in current]
        nseq = sum(len(x[0]) for x in encoded)
        cseq = sum(len(encoded[i][0]) for i in current_views)
        cans = sum(counts[i] for i in current_views)
        before = self._sample_parameters()
        fp_before = self._fingerprint_sample(before)
        self.model.train()
        torch.cuda.reset_peak_memory_stats(self.device)
        self._sync()
        start = time.perf_counter()
        local_loss = 0.0
        micro = max(1, min(math.ceil(len(encoded)/self.world_size),
                           self.micro_token_budget//max(len(x[0]) for x in encoded)))
        rounds = math.ceil(len(encoded) / (self.world_size*micro))
        physical_tokens, padded_tokens, dummy_count = 0, 0, 0
        batches = []
        for offset in range(rounds):
            first = offset*self.world_size*micro
            indices = list(range(first+self.rank, min(first+self.world_size*micro, len(encoded)), self.world_size))
            examples = [encoded[i] for i in indices] or [self._dummy()]
            batches.append((indices, examples))
            physical_tokens += sum(len(x[0]) for x in examples)
            padded_tokens += len(examples)*max(len(x[0]) for x in examples)
            dummy_count += int(not indices)
        physical_tokens = int(self._reduce(physical_tokens))
        padded_tokens = int(self._reduce(padded_tokens))
        dummy_count = int(self._reduce(dummy_count))
        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            for indices, examples in batches:
                ce = self._sum_losses(examples)
                weights = [coeff[i]*self.world_size for i in indices] if indices else [0.0]
                loss = (ce*torch.tensor(weights, device=self.device)).sum()
                if self._reduce(float(bool(torch.isfinite(loss))), dist.ReduceOp.MIN) != 1:
                    raise FloatingPointError('Nonfinite TRACE training loss')
                loss.backward()
                local_loss += float(loss.detach()) / self.world_size
            norm = (clip_full_shard_grad_norm_(self.model, self.max_grad_norm)
                    if self.parameter_mode == 'full' else self.model.clip_grad_norm_(self.max_grad_norm))
            if not math.isfinite(float(norm)):
                raise FloatingPointError('Nonfinite TRACE gradient norm')
            self.optimizer.step()
            self.total_optimizer_steps += 1
        self.optimizer.zero_grad(set_to_none=True)
        self._sync()
        elapsed = self._reduce(time.perf_counter()-start, dist.ReduceOp.MAX)
        after = self._sample_parameters()
        delta = self._reduce(float((after-before).double().square().sum()))
        fp_after = self._fingerprint_sample(after)
        self.metadata['optimizer_state_dtypes'] = self._optimizer_dtypes()
        return dict(loss=self._reduce(local_loss)/steps, has_current=has_current,
            training_role='current_and_history' if has_current else 'history_only',
            training_facts=len(rows), training_view_rows=len(facts), training_view_length_fallbacks=0,
            optimizer_steps=steps, total_optimizer_steps=self.total_optimizer_steps,
            sequence_tokens=nseq*steps, answer_tokens=sum(counts)*steps,
            current_sequence_tokens=cseq*steps, replay_sequence_tokens=(nseq-cseq)*steps,
            current_answer_tokens=cans*steps, replay_answer_tokens=(sum(counts)-cans)*steps,
            current_fact_presentations=len(current)*steps, replay_fact_presentations=(len(rows)-len(current))*steps,
            current_view_presentations=len(current_views)*steps,
            replay_view_presentations=(len(facts)-len(current_views))*steps,
            physical_sequence_tokens=physical_tokens*steps, physical_padded_sequence_tokens=padded_tokens*steps,
            physical_answer_tokens=(sum(counts)+dummy_count)*steps,
            microbatch_rounds=rounds, local_microbatch_size=micro,
            wall_seconds=elapsed, peak_vram_gb=self._reduce(torch.cuda.max_memory_allocated(self.device)/2**30, dist.ReduceOp.MAX),
            last_global_grad_norm=float(norm), sampled_parameter_delta_l2=math.sqrt(delta),
            parameter_fingerprint_before=fp_before, parameter_fingerprint_after=fp_after,
            parameter_changed=fp_before != fp_after)

    @torch.no_grad()
    def answer_nll(self, rows):
        self.model.eval()
        start = time.perf_counter()
        results = []
        encoded = [self.encoded(r['update'], r['ans']) for r in rows]
        micro = max(1, min(math.ceil(len(rows)/self.world_size),
                           self.micro_token_budget//max(len(x[0]) for x in encoded))) if rows else 1
        rounds = math.ceil(len(rows)/(self.world_size*micro))
        logical = sum(len(x[0]) for x in encoded)
        local_physical, local_padded = 0, 0
        for offset in range(rounds):
            first = offset*self.world_size*micro
            indices = list(range(first+self.rank, min(first+self.world_size*micro, len(rows)), self.world_size))
            examples = [encoded[i] for i in indices] or [self._dummy()]
            losses = self._sum_losses(examples)
            local_physical += sum(len(x[0]) for x in examples)
            local_padded += len(examples)*max(len(x[0]) for x in examples)
            results.extend((i, float(losses[j])/encoded[i][2]) for j, i in enumerate(indices))
        gathered = sorted(v for part in self._gather(results) for v in part)
        values = [v for _, v in gathered]
        if len(values) != len(rows) or not all(map(math.isfinite, values)):
            raise ValueError('Invalid distributed NLL result')
        physical = int(self._reduce(local_physical))
        self.last_nll_stats = dict(forward_calls=rounds, logical_sequence_tokens=logical,
            physical_sequence_tokens=physical, physical_padded_sequence_tokens=int(self._reduce(local_padded)),
            wall_seconds=self._reduce(time.perf_counter()-start, dist.ReduceOp.MAX))
        self.forward_accounting['nll_calls'] += 1
        for key in ['forward_calls', 'logical_sequence_tokens', 'physical_sequence_tokens', 'physical_padded_sequence_tokens']:
            self.forward_accounting['nll_'+key] += self.last_nll_stats[key]
        return values, logical

    @torch.no_grad()
    def generate(self, questions):
        self.model.eval()
        start = time.perf_counter()
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id or eos
        limit = CONFIG[self.task][0]
        raw = [trim_prompt(self.tokenizer(str(q).rstrip(), add_special_tokens=False).input_ids,
                           limit, self.task) for q in questions]
        # Length sorting changes neither prompts nor the output order.
        order = sorted(range(len(raw)), key=lambda i: len(raw[i]))
        output, physical, logical, padded, calls, out_tokens = [], 0, 0, 0, 0, 0
        batch_limit = max(self.world_size, self.eval_batch_size)
        for start_i in range(0, len(order), batch_limit):
            ids_global = order[start_i:start_i+batch_limit]
            local_ids = ids_global[self.rank::self.world_size]
            seqs = [raw[i] for i in local_ids] or [[eos]]
            length = max(map(len, seqs))
            ids = torch.full((len(seqs), length), pad, device=self.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(seqs):
                ids[i, -len(seq):] = torch.tensor(seq, device=self.device)
                mask[i, -len(seq):] = 1
            finished = torch.zeros(len(seqs), dtype=torch.bool, device=self.device)
            if not local_ids:
                finished[:] = True
            generated = []
            cache = None
            for step in range(self.max_new_tokens):
                pos = mask.cumsum(-1)-1
                pos.masked_fill_(mask == 0, 0)
                if cache is not None:
                    pos = pos[:, -1:]
                result = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = result.past_key_values
                token = result.logits[:, -1].argmax(-1)
                active = ~finished
                token = torch.where(active, token, torch.full_like(token, pad))
                generated.append(token)
                physical += int(mask.sum()) if step == 0 else len(seqs)
                logical += sum(len(raw[i]) for i in local_ids) if step == 0 else int(active.sum())
                padded += ids.numel()
                calls += 1
                out_tokens += int(active.sum())
                finished |= token == eos
                if self._reduce(float(bool(finished.all())), dist.ReduceOp.MIN) == 1:
                    break
                ids = token[:, None]
                mask = torch.cat((mask, active.long()[:, None]), dim=1)
            text = self.tokenizer.batch_decode(torch.stack(generated, dim=1), skip_special_tokens=True)
            output.extend(zip(local_ids, text))
            del cache, result
        values = sorted(v for part in self._gather(output) for v in part)
        assert [i for i, _ in values] == list(range(len(questions)))
        physical = int(self._reduce(physical))
        self.last_generation_stats = dict(forward_calls=int(self._reduce(calls)),
            logical_sequence_tokens=int(self._reduce(logical)), physical_sequence_tokens=physical,
            physical_padded_sequence_tokens=int(self._reduce(padded)), output_tokens=int(self._reduce(out_tokens)),
            wall_seconds=self._reduce(time.perf_counter()-start, dist.ReduceOp.MAX))
        self.forward_accounting['generation_calls'] += 1
        for key in ['forward_calls', 'logical_sequence_tokens', 'physical_sequence_tokens',
                    'physical_padded_sequence_tokens', 'output_tokens']:
            self.forward_accounting['generation_'+key] += self.last_generation_stats[key]
        return [s for _, s in values]
