"""Resumable full-row evaluation of the committed 150K checkpoint on eight GPUs."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source_snapshot'))
sys.path.insert(0, str(HERE))
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader, load_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, ShardedStateDictConfig
from baselines.common import answer_scores
from train import WikiBackend, arguments
from suite_utils import AXES, LABELS, atomic, read, sha, checkpoint_inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-output', type=Path, required=True)
    opts = parser.parse_args()
    done = read(opts.training_output / 'completed.json')
    audit = read(opts.training_output / 'audit.json')
    assert done['status'] == 'complete' and done['events'] == done['optimizer_steps'] == 150000
    checkpoint = Path(done['final_checkpoint'])
    inventory = checkpoint_inventory(checkpoint, 150000)
    meta = read(checkpoint / 'checkpoint_meta.json')
    payload = read(HERE / 'manifest.json')
    rows = payload['samples']
    plan = {
        'method': 'CAMR V5 more_learned',
        'manifest_sha256': sha(HERE / 'manifest.json'),
        'selection': payload['selection'],
        'evaluation_denominators': {
            axis: sum(bool(isinstance(row.get(question), str) and row[question].strip()
                      and isinstance(row.get(answer), str) and row[answer].strip())
                      for row in rows)
            for axis, (question, answer) in AXES.items()
        },
        'primary_evaluation_scope': (
            'Historical 100K eligible direct-axis masks and fixed 64-per-stage locality panel; '
            'all paired rows also scored and reported'
        ),
        'locality_definition': 'Gold-answer Match on paired loc/loc_ans; row-weighted, not pre/post consistency',
    }
    output = HERE / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    args = arguments(opts.training_output, HERE / 'manifest.json')
    torch.set_num_threads(4)
    started = time.monotonic()
    backend = WikiBackend(args)
    assert backend.world_size == meta['world_size'] == 8
    assert backend._source_identity == meta['source_identity']
    assert backend.numerical_runtime == meta['numerical_runtime']
    if backend.rank == 0:
        atomic(output / 'progress.json', dict(status='loading_checkpoint', checkpoint=str(checkpoint)))
    with FSDP.state_dict_type(backend.model, StateDictType.SHARDED_STATE_DICT,
                             ShardedStateDictConfig(offload_to_cpu=True)):
        state = {'model': backend.model.state_dict()}
        load_state_dict(state, storage_reader=FileSystemReader(str(checkpoint / 'distributed')),
                        process_group=backend._checkpoint_pg)
        backend.model.load_state_dict(state['model'])
    del state
    fingerprint = backend.parameter_fingerprint()
    assert fingerprint == audit['last_parameter_fingerprint'], 'Final checkpoint does not match the last training update'
    metrics, all_row_metrics = {}, {}
    stage_metrics = defaultdict(lambda: defaultdict(Counter))
    total_expected = sum(plan['evaluation_denominators'].values())
    total_finished = 0
    for axis, (question, answer) in AXES.items():
        valid = [r for r in rows if isinstance(r.get(question), str) and r[question].strip()
                 and isinstance(r.get(answer), str) and r[answer].strip()]
        assert len(valid) == plan['evaluation_denominators'][axis]
        totals, primary_totals = Counter(), Counter()
        primary_ids = set(payload['stage_evaluation_ids'][axis] if axis == 'locality'
                          else payload['primary_evaluation_ids'][axis]) if 'primary_evaluation_ids' in payload else {r['sample_id'] for r in valid}
        for start in range(0, len(valid), backend.eval_batch_size):
            chunk = valid[start:start + backend.eval_batch_size]
            path = output / axis / f'batch{start:06d}.json'
            # Every completed chunk is atomically committed. Incomplete chunks
            # are regenerated; completed chunks have their inputs revalidated.
            cached = backend._rank_zero_io(lambda: read(path) if path.exists() else None)
            if cached is not None:
                assert cached['manifest_sha256'] == plan['manifest_sha256']
                assert cached['parameter_fingerprint'] == fingerprint
                records = cached['records']
                assert [(r['sample_id'], r['question'], r['gold']) for r in records] == [
                    (r['sample_id'], r[question], r[answer]) for r in chunk]
                assert all(answer_scores(r['prediction'], r['gold']) == {'exact': r['exact'], 'match': r['match']} for r in records)
            else:
                predictions = backend.generate([r[question] for r in chunk])
                assert len(predictions) == len(chunk)
                records = [dict(sample_id=r['sample_id'], fact_stage=r['stage'], axis=axis,
                    question=r[question], gold=r[answer], prediction=p, **answer_scores(p, r[answer]))
                    for r, p in zip(chunk, predictions)]
                backend._rank_zero_io(lambda: atomic(path, dict(manifest_sha256=plan['manifest_sha256'],
                    parameter_fingerprint=fingerprint, records=records)))
            for r in records:
                totals.update(count=1, exact_count=int(r['exact']), match_count=int(r['match']))
                if r['sample_id'] in primary_ids:
                    primary_totals.update(count=1, exact_count=int(r['exact']), match_count=int(r['match']))
                stage_metrics[str(r['fact_stage'])][axis].update(count=1, exact_count=int(r['exact']), match_count=int(r['match']))
            total_finished += len(records)
            if backend.rank == 0:
                atomic(output / 'progress.json', dict(status='evaluating', axis=LABELS[axis],
                    axis_completed=totals['count'], axis_total=len(valid), completed=total_finished,
                    total=total_expected, elapsed_seconds=time.monotonic() - started, world_size=8,
                    checkpoint=str(checkpoint)))
                if start % (backend.eval_batch_size * 20) == 0:
                    print(json.dumps({'axis': axis, 'completed': totals['count'], 'total': len(valid)}), flush=True)
        count = totals['count']
        assert count == len(valid)
        all_row_metrics[axis] = dict(totals, match_accuracy=totals['match_count'] / count if count else None,
                                    exact_accuracy=totals['exact_count'] / count if count else None)
        primary_count = primary_totals['count']
        assert primary_count == len(primary_ids)
        metrics[axis] = dict(primary_totals,
            match_accuracy=primary_totals['match_count'] / primary_count if primary_count else None,
            exact_accuracy=primary_totals['exact_count'] / primary_count if primary_count else None)
        if backend.rank == 0:
            atomic(output / 'partial_metrics.json', metrics)
    assert backend.parameter_fingerprint() == fingerprint, 'Evaluation changed the parameters'
    if backend.rank == 0:
        result = dict(status='complete', method=plan['method'], trained_events=150000, k=8, seed=42,
            parameter_mode='full', training_steps=150000, evaluation_training_steps=0, world_size=8,
            checkpoint=inventory, checkpoint_parameter_fingerprint=fingerprint,
            manifest_sha256=plan['manifest_sha256'], selection=plan['selection'], metrics=metrics,
            primary_evaluation_scope=plan['primary_evaluation_scope'], all_paired_row_metrics=all_row_metrics,
            per_native_stage={s: {a: dict(v) for a, v in axes.items()} for s, axes in stage_metrics.items()},
            locality_definition=plan['locality_definition'], max_new_tokens=64, eval_batch_size=512,
            generation='Closed-book greedy Qwen3-8B-Base completion; no retrieval at inference',
            evaluation_elapsed_seconds=time.monotonic() - started,
            training_output=str(opts.training_output), training_elapsed_seconds=done['elapsed_seconds'])
        atomic(HERE / 'results_150k.json', result)
        with (HERE / 'results_150k.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'events', 'K', 'axis', 'count', 'match_count', 'match_percent', 'exact_count', 'exact_percent'])
            for axis, values in metrics.items():
                writer.writerow([plan['method'], 150000, 8, LABELS[axis], values['count'], values['match_count'],
                    100 * values['match_accuracy'], values['exact_count'], 100 * values['exact_accuracy']])
        lines = ['# WikiBigEdit 150K：CAMR K8 全参数结果', '',
            plan['selection'], '',
            '方法：CAMR V5 more_learned，Qwen3-8B-Base 全参数，seed 42，150,000 次更新。', '',
            '主表沿用旧 100K 口径：直接问法使用预先冻结的歧义/冲突排除名单，Locality 使用每阶段 64 题的冻结面板。', '',
            '| 指标 | Match | Exact | Match 正确数 / 有效题数 |', '|---|---:|---:|---:|']
        for axis, values in metrics.items():
            lines.append(f"| {LABELS[axis]} | {100 * values['match_accuracy']:.4f}% | {100 * values['exact_accuracy']:.4f}% | {values['match_count']:,} / {values['count']:,} |")
        lines += ['', '全量配对题目统计（包含主口径排除的歧义题与全部 Locality）：', '',
                  '| 指标 | Match | Exact | 有效题数 |', '|---|---:|---:|---:|']
        for axis, values in all_row_metrics.items():
            lines.append(f"| {LABELS[axis]} | {100 * values['match_accuracy']:.4f}% | {100 * values['exact_accuracy']:.4f}% | {values['count']:,} |")
        lines += ['', 'Locality 使用每条记录配对的 loc/loc_ans，对 gold answer 计分；不是编辑前后输出一致率。',
                  '缺失评测问法不计入该项分母，原始事实仍参与训练。全部有效行按记录等权计分。',
                  '最新版配置曾在前 5K 开发集上选择，本次包含该开发数据；此处报告实际结果，不声称独立验证。',
                  '', f'最终 checkpoint：`{checkpoint}`', '',
                  '逐题结果：`evaluation/<axis>/batch*.json`；完整统计：`results_150k.json`、`results_150k.csv`。']
        (HERE / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
        atomic(output / 'progress.json', dict(status='complete', completed=total_finished, total=total_expected))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
