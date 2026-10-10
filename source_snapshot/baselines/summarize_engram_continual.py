"""Report completed writer experiments and audit replay matching without claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.common import atomic_json


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def traces(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def accuracy(metric, axis):
    value = metric.get(f'{axis}_accuracy')
    if value is None and isinstance(metric.get(axis), dict):
        value = metric[axis].get('match_accuracy')
    return value


def audit_pair(reference, comparison):
    identity = ('event', 'sample_id', 'arrival_stage', 'diagnostic_repeat')
    same_stream = len(reference) == len(comparison) and all(
        all(left.get(key) == right.get(key) for key in identity)
        for left, right in zip(reference, comparison))
    audit = {'same_event_stream': same_stream, 'events': len(reference)}
    for field in ('action', 'replay_count'):
        audit[f'{field}_matched'] = same_stream and all(a.get(field) == b.get(field) for a, b in zip(reference, comparison))
    for field in ('optimizer_steps', 'sequence_tokens', 'effective_sequence_tokens', 'answer_tokens'):
        left = [(record.get('train') or {}).get(field, 0) for record in reference]
        right = [(record.get('train') or {}).get(field, 0) for record in comparison]
        audit[field] = {'reference': sum(left), 'comparison': sum(right),
                        'per_event_matched': same_stream and left == right,
                        'total_matched': same_stream and sum(left) == sum(right)}
    audit['strict_training_token_and_step_match'] = same_stream and all(
        audit[field]['total_matched'] for field in ('optimizer_steps', 'sequence_tokens', 'answer_tokens'))
    audit['total_compute_matched'] = False
    audit['note'] = 'Count/step matching alone is not token or total-compute matching; forward and routing overhead also matter.'
    return audit


def summarize(root):
    rows, by_mode = [], {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / 'run_config.json').exists():
            continue
        config = read_json(directory / 'run_config.json', {})
        done = read_json(directory / 'completed.json', read_json(directory / 'bounded_stop.json', {}))
        metrics = read_json(directory / 'metrics.json', [])
        actions = traces(directory / 'actions.jsonl')
        by_mode[directory.name] = actions
        final = metrics[-1] if metrics else {}
        row = {'run': directory.name, 'mode': config.get('mode'), 'status': done.get('status', 'incomplete'),
               'full_parameter': config.get('base_weights_frozen') is False,
               'events': done.get('events', len(actions)), 'rephrase': accuracy(final, 'rephrase'),
               'update': accuracy(final, 'update'), 'persona': accuracy(final, 'persona'),
               'locality': accuracy(final, 'locality'), 'action_counts': done.get('action_counts', {}),
               'steps': done.get('optimizer_steps'), 'tokens': done.get('trained_sequence_tokens'),
               'effective_tokens': done.get('effective_sequence_tokens'),
               'policy_updates': sum(bool((record.get('policy_update') or {}).get('updated')) for record in actions),
               'parameter_update_max_abs': done.get('parameter_update_max_abs'),
               'resumed_from': config.get('resume_checkpoint')}
        rows.append(row)
    audits = {}
    if 'policy' in by_mode:
        for name in ('random_matched', 'random_replay'):
            if name in by_mode:
                audits[name] = audit_pair(by_mode['policy'], by_mode[name])
    report = {'runs': rows, 'budget_audits': audits,
              'interpretation': 'Development/infrastructure evidence only. No multi-seed or strict total-compute superiority claim.'}
    atomic_json(root / 'summary.json', report)
    lines = ['# Writer 实验记录', '',
             '依据当前目录内实际日志汇总；未完成的运行和未执行的评测不会被当作零分。', '',
             '| 运行 | 状态 | 全参 | 事件 | Update | Rephrase | 步数 | 实际训练 token | Policy 更新 |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    pct = lambda value: '未评测' if value is None else f'{100 * value:.2f}%'
    for row in rows:
        lines.append(f"| {row['run']} | {row['status']} | {'是' if row['full_parameter'] else '否'} | {row['events']} | {pct(row['update'])} | {pct(row['rephrase'])} | {row['steps']} | {row['tokens']} | {row['policy_updates']} |")
    lines += ['', 'ADD 只学当前；MERGE 使用兼容历史；随机匹配运行只在日志证实的维度上算匹配。', '',
              'FSDP 实际 token 包含为各 rank 同步执行而产生的 dummy 前向/反向；effective token 表示真正监督的样本。', '',
              '这些开发或流程实验不能证明优于基线，也不验证长期信用分配或任意自然语言事实等价。', '']
    if audits:
        lines += ['## 预算核对', '', '```json', json.dumps(audits, ensure_ascii=False, indent=2), '```', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
