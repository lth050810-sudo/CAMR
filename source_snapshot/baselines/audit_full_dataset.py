"""Inventory the complete local source without selecting data or launching training."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from baselines.freeze_heldout_dev import probes
from baselines.wikibigedit_protocol import (
    PANEL_FIELDS, SOURCE_FIELDS, _fact_fingerprint, qa_prompt, sha256_file,
    stable_sample_id,
)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                    sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('WikiBigEdit/full_hf_viewer'))
    parser.add_argument('--dev-manifest', type=Path, default=Path('runs/protocols/wikibigedit_dev1600_seed20260920_qwen3_unknown.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/full_dataset_preparation'))
    parser.add_argument('--tokenizer', type=Path)
    args = parser.parse_args()
    source = json.loads((args.source_dir / 'manifest.json').read_text())
    dev = json.loads(args.dev_manifest.read_text())
    dev_ids = set(dev['selection']['sample_ids'])
    dev_rows = [r for r in dev['records'] if r['sample_id'] in dev_ids]
    dev_subjects = {r.get('subject_id') or r['subject'] for r in dev_rows}
    dev_facts = {r['fact_fingerprint'] for r in dev_rows}
    dev_probes = set().union(*(probes(r) for r in dev_rows))
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, all_qa, all_facts = set(), set(), set()
    question_answers = defaultdict(set)
    stages = []
    with (args.output_dir / 'rows_requiring_exclusion_review.jsonl').open('w') as review:
        for f in source['files']:
            path = args.source_dir / f['path']
            actual_hash = sha256_file(path)
            if actual_hash != f['sha256_reconstructed']:
                raise ValueError(f'Source hash mismatch: {path}')
            rows = json.loads(path.read_text())
            if len(rows) != f['rows']:
                raise ValueError(f'Source count mismatch: {path}')
            tags, valid_tags, missing, overlap, lengths = (Counter() for _ in range(5))
            qa_missing = valid = valid_dev_disjoint = 0
            duplicate_rows = repeated_qa = repeated_facts = 0
            within_stage_questions = defaultdict(set)
            valid_for_lengths = []
            for i, row in enumerate(rows):
                sid = stable_sample_id(f['interval'], i)
                tag = row.get('tag')
                tag = '<null>' if tag is None else '<empty>' if tag == '' else str(tag)
                tags[tag] += 1
                row_hash = digest({k: row.get(k) for k in SOURCE_FIELDS})
                duplicate_rows += row_hash in all_rows
                all_rows.add(row_hash)
                for axis, (q, a) in PANEL_FIELDS.items():
                    if not (text(row.get(q)) and text(row.get(a))):
                        missing[axis] += 1
                reasons = []
                if not (text(row.get('update')) and text(row.get('ans'))):
                    qa_missing += 1
                    reasons.append('missing_training_question_or_answer')
                else:
                    valid += 1
                    valid_tags[tag] += 1
                    norm_q = ' '.join(row['update'].casefold().split())
                    norm_a = ' '.join(row['ans'].casefold().split())
                    qa = digest([norm_q, norm_a])
                    fact = _fact_fingerprint(row)
                    repeated_qa += qa in all_qa
                    repeated_facts += fact in all_facts
                    all_qa.add(qa)
                    all_facts.add(fact)
                    question_answers[norm_q].add(norm_a)
                    within_stage_questions[norm_q].add(norm_a)
                    checks = {
                        'dev_sample_id': sid in dev_ids,
                        'dev_subject': (row.get('subject_id') or row.get('subject')) in dev_subjects,
                        'dev_fact_fingerprint': fact in dev_facts,
                        'dev_probe_text': bool(probes(row) & dev_probes),
                    }
                    for key, value in checks.items():
                        overlap[key] += value
                        if value:
                            reasons.append(key)
                    if not any(checks.values()):
                        valid_dev_disjoint += 1
                    valid_for_lengths.append(row)
                if reasons:
                    review.write(json.dumps({'sample_id': sid, 'stage': f['stage'],
                                             'reasons': reasons}) + '\n')
            if tok:
                for start in range(0, len(valid_for_lengths), 512):
                    batch = valid_for_lengths[start:start + 512]
                    prompts = tok([qa_prompt(r['update']) for r in batch],
                                  add_special_tokens=True, truncation=False)['input_ids']
                    answers = tok([' ' + r['ans'].strip() + (tok.eos_token or '') for r in batch],
                                  add_special_tokens=False, truncation=False)['input_ids']
                    for p, a in zip(prompts, answers):
                        n = len(p) + len(a)
                        lengths['sequence_tokens'] += n
                        lengths['answer_tokens_including_eos'] += len(a)
                        lengths['max_sequence_tokens'] = max(lengths['max_sequence_tokens'], n)
                        lengths['max_answer_tokens_including_eos'] = max(lengths['max_answer_tokens_including_eos'], len(a))
                        for limit in (128, 256, 512, 1024):
                            lengths[f'sequences_above_{limit}'] += n > limit
                        lengths['answers_above_16_including_eos'] += len(a) > 16
            stage = {'stage': f['stage'], 'interval': f['interval'], 'path': str(path.resolve()),
                     'sha256': actual_hash, 'hash_verified': True, 'raw_rows': len(rows),
                     'tag_counts': dict(tags), 'qa_valid': valid, 'qa_valid_tag_counts': dict(valid_tags),
                     'missing_training_qa': qa_missing, 'missing_panels': dict(missing),
                     'dev_overlap_among_qa_valid_nonexclusive': dict(overlap),
                     'qa_valid_disjoint_from_dev': valid_dev_disjoint,
                     'exact_source_row_repetitions': duplicate_rows,
                     'normalized_qa_repetitions': repeated_qa, 'fact_repetitions': repeated_facts,
                     'questions_with_multiple_answers_within_stage': sum(len(v) > 1 for v in within_stage_questions.values()),
                     'token_lengths_all_qa_valid': dict(lengths)}
            stages.append(stage)
            print(f"tau{f['stage']}: {valid}/{len(rows)} valid QA, {valid_dev_disjoint} dev-disjoint", flush=True)
    total = {k: sum(s[k] for s in stages) for k in (
        'raw_rows', 'qa_valid', 'missing_training_qa', 'qa_valid_disjoint_from_dev',
        'exact_source_row_repetitions', 'normalized_qa_repetitions', 'fact_repetitions')}
    total['tag_counts'] = dict(sum((Counter(s['tag_counts']) for s in stages), Counter()))
    total['qa_valid_tag_counts'] = dict(sum((Counter(s['qa_valid_tag_counts']) for s in stages), Counter()))
    total['questions_with_multiple_answers_across_stream'] = sum(len(v) > 1 for v in question_answers.values())
    assert total['raw_rows'] == source['total_rows']
    assert total['raw_rows'] == total['qa_valid'] + total['missing_training_qa']
    payload = {'schema_version': 'wikibigedit-full-census/v1',
               'generated_at': datetime.now(timezone.utc).isoformat(),
               'source_revision': source['revision'], 'source_manifest_sha256': sha256_file(args.source_dir / 'manifest.json'),
               'dev_manifest_sha256': sha256_file(args.dev_manifest),
               'purpose': 'Preparation only; no training selection, filtering policy or full-data run is launched.',
               'tokenizer': str(args.tokenizer.resolve()) if tok else None,
               'notes': [
                   'Preserve eight original chronological intervals; their sizes are unequal.',
                   'Full data contains new, update and unlabeled records; it is not all base-unknown.',
                   'Train only update -> ans. Other panels are evaluation probes.',
                   'QA validity checks question and answer only; missing other probes do not remove training QA.',
                   'Multiple answers for a question can be legitimate multi-valued relations, not necessarily contradictory labels.',
                   'Repetitions are observations, not automatically removed; future full protocol must define temporal target handling.',
                   'Dev-disjoint count excludes union of sample ID, subject, fact and any probe overlap; no length/tag/dedup filter applied.',
                   'Token lengths follow separate prompt/answer encoding of the current trainer, including EOS in answer.',
               ], 'totals': total, 'stages': stages}
    (args.output_dir / 'census.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    lines = ['# WikiBigEdit 全量数据核验（仅准备，尚未启动全量训练）', '',
             f"来源：`{args.source_dir.resolve()}`，revision `{source['revision']}`。八个文件的本地 SHA-256 均与来源 manifest 一致。", '',
             '| 阶段 | 原始记录 | 有效事实问答 | 按现有 dev 隔离规则保留 |',
             '| --- | ---: | ---: | ---: |']
    lines += [f"| τ{s['stage']} | {s['raw_rows']:,} | {s['qa_valid']:,} | {s['qa_valid_disjoint_from_dev']:,} |" for s in stages]
    lines += [f"| 合计 | {total['raw_rows']:,} | {total['qa_valid']:,} | {total['qa_valid_disjoint_from_dev']:,} |", '',
              '上表未应用标签筛选、去重或长度过滤，不是最终训练 manifest。dev 隔离包含样本 ID、subject、fact fingerprint 和五轴问题文本的并集。', '',
              '标签计数：`' + json.dumps(total['tag_counts'], ensure_ascii=False) + '`。', '',
              f"有 {total['questions_with_multiple_answers_across_stream']:,} 个问题在完整流中对应多个答案。需要区分多值关系与时间更新，不能简单把旧答案和新答案同时当作唯一金标准。", '',
              '本轮先完成并审阅 3K baseline，再由用户决定模型/方法阵容，之后启动全量实验。现有 3K manifest 保持独立。', '']
    (args.output_dir / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
