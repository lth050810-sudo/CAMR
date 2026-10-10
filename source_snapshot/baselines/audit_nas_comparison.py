"""Audit the completed MEMIT/NAS control and summarize its norm trajectory."""
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs/wikibigedit_3k_qwen3_unknown'
BASE = 'memit_full100k_layers4_8_batch100_3k'
NAS = 'memit_nas_layers4_8_batch100_3k'
FIELDS = ['layers', 'v_loss_layer', 'v_lr', 'v_num_grad_steps', 'kl_factor',
          'v_weight_decay', 'clamp_norm_factor', 'mom2_update_weight', 'solver_ridge',
          'context_templates', 'edit_batch_size', 'max_length', 'max_new_tokens',
          'seed', 'stats_sample_size', 'manifest_protocol_id', 'manifest_sha256']
AXES = ['update', 'rephrase', 'persona', 'locality', 'multihop']


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def main():
    configs = {name: read(RUNS / name / 'run_config.json') for name in (BASE, NAS)}
    checks = {k: configs[BASE].get(k) == configs[NAS].get(k) for k in FIELDS}
    assert all(checks.values()), checks
    results, batches = {}, {}
    for name in (BASE, NAS):
        p = RUNS / name
        assert read(p / 'completed.json')['status'] == 'complete', f'{name} not completed'
        traces = jsonl(p / 'edit_trace.jsonl')
        assert len(traces) == 32 and sum(len(t['sample_ids']) for t in traces) == 3000
        assert finite(traces)
        batches[name] = [t['sample_ids'] for t in traces]
        metrics = read(p / 'metrics.json')[-1]
        assert metrics['current_stage'] == 8 and metrics['num_facts'] == 3000
        predictions = Counter(r['prediction'] for r in jsonl(p / 'predictions.jsonl')
                              if r['checkpoint'] == metrics['checkpoint'])
        results[name] = {'axes': {a: metrics[a]['match_accuracy'] for a in AXES},
                         'trace_count': len(traces), 'all_trace_statistics_finite': True,
                         'max_layer_update_norm': max(v for t in traces for v in t['layer_update_norms'].values()),
                         'final_weight_norms': traces[-1]['layer_weight_norms'],
                         'most_common_final_predictions': predictions.most_common(5),
                         'final_prediction_count': sum(predictions.values())}
    assert batches[BASE] == batches[NAS], 'Editing order or batch boundaries differ'
    checks['identical_edit_order_and_batches'] = True
    targets = jsonl(RUNS / NAS / 'nas_targets.jsonl')
    assert len(targets) == 3000 and finite(targets)
    ids = {r['sample_id']: r['stage'] for r in read(ROOT / 'runs/protocols/wikibigedit_3k_seed42_qwen3_unknown.json')['records']}
    stage_rows = []
    for stage in range(1, 9):
        rows = [r for r in targets if ids[r['sample_id']] == stage]
        assert len(rows) == 375
        norms = sorted(r['v_star_norm_orig'] for r in rows)
        stage_rows.append({'stage': stage, 'targets': len(rows),
                           'outlier_skipped_delta': sum(r['outlier_skipped_delta'] for r in rows),
                           'proposed_norm_median': norms[len(norms) // 2],
                           'proposed_norm_max': max(norms)})
    payload = {'control_checks': checks, 'results': results, 'nas_stage_targets': stage_rows,
               'evaluation_batch_sizes': {n: configs[n].get('eval_batch_size') for n in (BASE, NAS)},
               'scope': 'Single-seed Qwen3 transfer control, layers4-8. Does not isolate the standalone NAS7-11 configuration or establish a general NAS claim.'}
    out = ROOT / 'runs/baseline_expansion_20260920'
    (out / 'nas_matched_comparison.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Matched MEMIT + NAS comparison', '', payload['scope'], '',
             'All listed editing settings, contexts, manifest, seed, and moment sample counts match. '
             'Both runs completed 3,000 facts with 32 finite batch traces.', '',
             '| Run | U | R | P | L | MH |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, result in results.items():
        lines.append('| ' + name + ' | ' + ' | '.join(f'{100 * result["axes"][a]:.2f}' for a in AXES) + ' |')
    lines += ['', 'Generation batch sizes (same greedy QA scoring): '
              + ', '.join(f'{n}={configs[n].get("eval_batch_size")}' for n in (BASE, NAS)) + '.', '']
    lines += ['', '| Stage | NAS outliers reverting to original MLP value / 375 | Proposed norm median | Proposed norm max |',
              '| ---: | ---: | ---: | ---: |']
    for r in stage_rows:
        lines.append(f'| {r["stage"]} | {r["outlier_skipped_delta"]} | {r["proposed_norm_median"]:.2f} | {r["proposed_norm_max"]:.2f} |')
    lines += ['', 'The outlier branch preserves the current original MLP value and skips its optimized delta. '
              'Finite arithmetic does not imply preserved generated behavior. '
              'Standalone NAS uses different layers and nullspace constraints and is reported separately.', '']
    (out / 'nas_matched_comparison.md').write_text('\n'.join(lines))
    print('Matched MEMIT + NAS comparison verified and written.')


if __name__ == '__main__':
    main()
