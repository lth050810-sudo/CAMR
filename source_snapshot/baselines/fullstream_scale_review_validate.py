"""CPU-only semantic equivalence and real-metadata scaling evidence."""
import copy
import ast
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, '/root/engram/runs/engram_v3_hybrid_3k_20260921/source_snapshot')
from baselines.engram_continual_core import FactMemory, training_row
from baselines import engram_full_safety as old
from baselines.engram_replay_v3 import select_replay as old_select_replay
source = Path('/root/engram/runs/engram_v3_hybrid_3k_20260921/source_snapshot/baselines/run_engram_full.py').read_text()
tree = ast.parse(source)
retirement_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'explicit_retirements')
retirement_globals = {'training_row': training_row, 'strict_mastery_match': old.strict_mastery_match}
exec(compile(ast.Module(body=[retirement_function], type_ignores=[]), '<original explicit_retirements>', 'exec'), retirement_globals)
old_retirements = retirement_globals['explicit_retirements']

spec = importlib.util.spec_from_file_location('scale', '/root/engram/baselines/fullstream_scale_review.py')
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if not k.startswith('_scale')}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [plain(v) for v in value]
    return value


checks = 0
hybrid_checks = 0


def compare(memory, row, vector, retired=()):
    global checks, hybrid_checks
    a = old.safe_query(memory, row, vector, retired)
    b = new.safe_query(memory, row, vector, retired)
    assert plain(a) == plain(b), ('query', a, plain(b))
    checks += 1
    for size in (0, 1, 4, 15):
        rng_a, rng_b = random.Random(810 + checks), random.Random(810 + checks)
        pa = old.choose_safe_probe_and_replay_pools(memory, a, row, rng_a, size, retired)
        pb = new.choose_safe_probe_and_replay_pools(memory, b, row, rng_b, size, retired)
        assert plain(pa) == plain(pb), ('pools', plain(pa), plain(pb))
        assert rng_a.getstate() == rng_b.getstate(), 'probe rng'
        stats = {str(oldrow['sample_id']): {'inserted_event': j // 2 - 1,
                 'last_replayed_event': j // 3 if j % 2 else None, 'replay_count': j % 5}
                 for j, oldrow in enumerate(memory.rows)}
        for action in ('add', 'merge', 'ignore'):
            for k, similar_k in ((4, 2), (1, 0), (15, 8)):
                ha, hb = random.Random(200), random.Random(200)
                sa = old_select_replay(action, pa[1], pa[2], k, ha, stats, len(memory.rows) + 1, similar_k)
                sb = new.select_replay(action, pb[1], pb[2], k, hb, stats, len(memory.rows) + 1, similar_k)
                assert plain(sa) == plain(sb), ('hybrid selection', sa, sb)
                assert ha.getstate() == hb.getstate(), 'hybrid RNG'
                hybrid_checks += 1
        for k in (0, 1, 4, 10, len(pa[1])):
            k = min(k, len(pa[1]))
            ra, rb = random.Random(190 + checks), random.Random(190 + checks)
            assert ra.sample(pa[1], k) == rb.sample(pb[1], k), 'replay sample'
            assert ra.getstate() == rb.getstate(), 'replay rng'
        checks += 1


def equivalence():
    generator = torch.Generator().manual_seed(121)
    for mode in ('structured', 'text'):
        memory = FactMemory(16, tables=3, bits=5, max_candidates=13, fact_mode=mode)
        vectors = torch.randn(120, 16, generator=generator)
        rows = []
        retired = set()
        for index in range(80):
            row = {'sample_id': str(index // 2) if index % 17 == 0 else str(index), 'stage': index // 20,
                'update': f'Question {index % 19}', 'ans': ('US', 'us', '①', '1', 'caf\u00e9', 'cafe\u0301')[index % 6],
                'subject_id': f'Q{index % 11}', 'relation_id': f'P{index % 3}', 'object_id': f'A{index % 7}',
                'scope': {'time': index % 2}}
            if index % 9 == 0:
                row.pop('subject_id')
            if index == 40:
                retired.update({'1', '4', '12'})
            if index == 60:
                retired.remove('4')
            compare(memory, row, vectors[index], retired)
            memory.insert(row, vectors[index])
            rows.append(row)
            if index % 8 == 0:
                compare(memory, rows[max(0, index - 7)], vectors[index], retired)
        # Permanent supersession, current provisional replacement, and group
        # conflict removal must all be represented exactly.
        for index in range(80, 90):
            row = copy.deepcopy(rows[index - 80])
            row['sample_id'] = str(index)
            row['supersedes'] = [rows[index - 80]['sample_id']]
            row['ans'] = f'new {index}'
            compare(memory, row, vectors[index], retired)
            memory.insert(row, vectors[index])
        restored = FactMemory.from_state_dict(memory.state_dict())
        compare(restored, rows[-1], vectors[-1], retired)

    # Runtime retirement adapter: regular arrival, aliases, authoritative slot
    # replacement and exact validation exceptions.
    memory = FactMemory(16)
    v = vectors[0]
    for i, answer in enumerate(('old', 'old2', 'old3')):
        memory.insert({'sample_id': str(i), 'update': 'Q', 'ans': answer,
                       'subject_id': 'Q1', 'relation_id': 'P1'}, v)
    for extra in ({}, {'supersedes': ['0']}, {'supersedes': ['alias']},
                  {'single_valued': True, 'authoritative_update': True},
                  {'supersedes': ['missing']}, {'supersedes': ['4']},
                  {'supersedes': '0'}, {'supersedes': ['0'], 'supersedes_sample_ids': ['1']}):
        raw = {'sample_id': '4', 'update': 'Q', 'ans': 'new', 'subject_id': 'Q1', 'relation_id': 'P1', **extra}
        result = []
        for implementation in (old_retirements, new.explicit_retirements):
            retired = set()
            try:
                result.append((implementation({'row': raw}, memory, retired, {'alias': '0'}), retired))
            except Exception as exc:
                result.append((type(exc).__name__, str(exc), retired))
        assert result[0] == result[1], result
    invalid_cases = 0
    for bad_stats in ({'replay_count': -1}, {'replay_count': True}, {'inserted_event': 5},
                      {'inserted_event': -2}, {'last_replayed_event': 6}, {'last_replayed_event': False}):
        for action in ('add', 'merge'):
            available = [{'sample_id': 'a'}, {'sample_id': 'b'}]
            answers = []
            for implementation in (old_select_replay, new.select_replay):
                rng = random.Random(41)
                before = rng.getstate()
                try:
                    answer = implementation(action, available, available[:1], 4, rng,
                                            {'b': bad_stats}, 5, 2)
                except Exception as exc:
                    answer = (type(exc).__name__, str(exc))
                answers.append(answer)
                assert before == rng.getstate(), 'invalid statistics consumed RNG'
            assert answers[0] == answers[1], answers
            invalid_cases += 1
    # Query/pool cache cannot hide a changed retirement set or changed row.
    original_row = {'sample_id': '4', 'update': 'different', 'ans': 'a'}
    query_a = old.safe_query(memory, original_row, v)
    query_b = new.safe_query(memory, original_row, v)
    for row, retired in ((original_row, {'0'}), ({**original_row, 'update': 'Q', 'ans': 'old'}, set())):
        pa = old.choose_safe_probe_and_replay_pools(memory, query_a, row, random.Random(10), 4, retired)
        pb = new.choose_safe_probe_and_replay_pools(memory, query_b, row, random.Random(10), 4, retired)
        assert plain(pa) == plain(pb), 'changed pool inputs'
    print(json.dumps({'equivalence_checks': checks, 'hybrid_checks': hybrid_checks, 'retirement_cases': 8,
                      'invalid_stats_cases': invalid_cases, 'changed_pool_inputs': 2}), flush=True)


def build_memory(rows, dim=32):
    """Bulk-build exactly the same state as insert, without 100k tiny Torch ops."""
    memory = FactMemory(dim, tables=8, bits=8, max_candidates=64, fact_mode='text')
    memory.rows = [training_row(row) for row in rows]
    generator = torch.Generator().manual_seed(2026)
    vectors = torch.nn.functional.normalize(torch.randn(len(rows), dim, generator=generator), dim=1)
    memory._vectors = list(vectors.unbind(0))
    # Use the original hash routine's float accumulation through einsum in
    # batches. Tiny round-off at sign zero is irrelevant to old/new comparison:
    # both query implementations read these same stored bucket memberships.
    signs = torch.einsum('tbd,nd->ntb', memory.hyperplanes, vectors) >= 0
    codes = (signs.numpy().astype(np.uint64) * (np.uint64(1) << np.arange(memory.bits, dtype=np.uint64))).sum(axis=2)
    for index, row in enumerate(memory.rows):
        for table in range(memory.tables):
            memory._buckets[table][int(codes[index, table])].append(index)
        fact, slot = memory._fact_key(row), memory._slot_key(row)
        if fact is not None:
            memory._fact_index[fact].append(index)
        if slot is not None:
            memory._slot_index[slot].append(index)
        memory._sample_index[str(row['sample_id'])].append(index)
    return memory


def benchmark():
    rows = []
    for stage, path in enumerate(sorted(Path('/root/engram/WikiBigEdit/full_hf_viewer').glob('wiki_big_edit_*.json')), 1):
        for index, row in enumerate(json.loads(path.read_text())):
            row['stage'], row['sample_id'] = stage, f'{stage}:{index}'
            rows.append(row)
        if len(rows) > 100001:
            break
    records = []
    for n in (10000, 50000, 100000):
        memory = build_memory(rows[:n])
        row = training_row(rows[n])
        vector = torch.ones(memory.dim)
        initialization = time.perf_counter()
        query = new.safe_query(memory, row, vector)
        initialization = time.perf_counter() - initialization
        old_times, new_times, old_hybrid, new_hybrid = [], [], [], []
        stats = {str(oldrow['sample_id']): {'inserted_event': j, 'last_replayed_event': None, 'replay_count': 0}
                 for j, oldrow in enumerate(memory.rows)}
        for _ in range(3):
            started = time.perf_counter()
            a = old.safe_query(memory, row, vector)
            pa = old.choose_safe_probe_and_replay_pools(memory, a, row, random.Random(88), 4)
            old_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            sa = old_select_replay('merge', pa[1], pa[2], 4, random.Random(60), stats, n + 1, 2)
            old_hybrid.append(time.perf_counter() - started)
            started = time.perf_counter()
            b = new.safe_query(memory, row, vector)
            pb = new.choose_safe_probe_and_replay_pools(memory, b, row, random.Random(88), 4)
            new_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            sb = new.select_replay('merge', pb[1], pb[2], 4, random.Random(60), stats, n + 1, 2)
            new_hybrid.append(time.perf_counter() - started)
        assert plain(a) == plain(b), f'large query {n}'
        assert plain(pa) == plain(pb), f'large pools {n}'
        assert plain(sa) == plain(sb), f'large hybrid selection {n}'
        record = {'history_rows': n, 'old_query_plus_pool_seconds': statistics.median(old_times),
            'new_query_plus_pool_seconds': statistics.median(new_times),
            'cache_build_seconds': initialization, 'conflict_rows': len(a['conflicting_history_ids']),
            'available_rows': len(pa[1]), 'hash_recalled_count': a['hash_recalled_count'],
            'old_v3_select_replay_seconds': statistics.median(old_hybrid),
            'new_v3_select_replay_seconds': statistics.median(new_hybrid),
            'identical_query_pools': True}
        records.append(record)
        print(json.dumps(record), flush=True)
    Path('/root/engram/runs/fullstream_scale_review_benchmark.json').write_text(json.dumps({
        'description': 'CPU only; real WikiBigEdit training metadata first 100k, synthetic normalized Gaussian dim32 embeddings; 8 hash tables, 8 bits; text fact mode; no CUDA work',
        'equivalence_checks': checks, 'hybrid_checks': hybrid_checks, 'records': records}, indent=2))


if __name__ == '__main__':
    equivalence()
    if '--equivalence-only' not in sys.argv:
        benchmark()
