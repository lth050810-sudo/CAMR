"""Prepare the 150K arrival stream and audit every training/evaluation input."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source_snapshot'))
from baselines.common import atomic_json, qa_prompt, normalize_answer as normalize_text

PAPER_MAX_LENGTH = 128


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', choices=['contiguous', 'balanced'], default='contiguous')
    parser.add_argument('--data-root', type=Path,
                        default=Path(os.environ['CAMR_DATA_ROOT']) if os.getenv('CAMR_DATA_ROOT') else None)
    parser.add_argument('--model-dir', type=Path,
                        default=Path(os.environ['CAMR_MODEL_DIR']) if os.getenv('CAMR_MODEL_DIR') else None)
    parser.add_argument('--encoder-dir', type=Path,
                        default=Path(os.environ['CAMR_ENCODER_DIR']) if os.getenv('CAMR_ENCODER_DIR') else None)
    parser.add_argument('--reference-manifest', type=Path,
                        help='Optional prior manifest used only for overlap auditing.')
    opts = parser.parse_args()
    for name in ('data_root', 'model_dir', 'encoder_dir'):
        if getattr(opts, name) is None:
            parser.error(f'--{name.replace("_", "-")} or CAMR_{name.upper()} is required')
    snapshot = HERE / 'source_snapshot/baselines'
    required = [snapshot / 'run_engram_full.py', snapshot / 'longstream_protocol.py',
                HERE / 'trace_backend.py', HERE / 'trace_protocol.py']
    if any(not path.is_file() for path in required):
        raise FileNotFoundError('The committed CAMR source snapshot is incomplete')
    data_root = opts.data_root
    source = json.loads((data_root / 'manifest.json').read_text())
    rows, sources = [], []
    for desc in sorted(source['files'], key=lambda x: x['stage']):
        if opts.selection == 'contiguous' and len(rows) == 150000:
            break
        path = data_root / desc['path']
        assert sha(path) == desc['sha256_reconstructed']
        raw = json.loads(path.read_text())
        assert isinstance(raw, list) and len(raw) == desc['rows']
        count = min(len(raw), 150000 - len(rows)) if opts.selection == 'contiguous' else 18750
        assert len(raw) >= count
        sources.append(dict(path=str(path), sha256=sha(path), stage=desc['stage'],
                            interval=desc['interval'], source_rows=len(raw), selected_rows=count))
        for index, item in enumerate(raw[:count]):
            row = dict(item, sample_id=f"{desc['interval']}:row{index:06d}",
                       stage=desc['stage'], source_row_idx=index)
            assert all(isinstance(row.get(k), str) and row[k].strip() for k in ['update', 'ans']), row['sample_id']
            rows.append(row)
    assert len(rows) == len({r['sample_id'] for r in rows}) == 150000
    old_rows = (json.loads(opts.reference_manifest.read_text())['samples']
                if opts.reference_manifest else None)
    protocol = None
    if opts.selection == 'balanced':
        # Match the historical 100K tag/relation stratification, eligibility,
        # deduplication and deterministic sampling; change only the stage quota.
        protocol_source = snapshot / 'longstream_protocol.py'
        spec = importlib.util.spec_from_file_location('protocol_150k', protocol_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.build(HERE / 'protocol' / 'manifest.json', data_root, opts.model_dir, total=150000, seed=42)
        protocol = json.loads((HERE / 'protocol/manifest.json').read_text())
        rows = protocol['records']
        assert len(rows) == 150000
    elif old_rows is not None:
        assert rows[:5000] == old_rows[:5000]
    selection = ('First 150,000 consecutive native records in chronological interval order; no shuffle, filtering, deduplication or resampling.'
                 if opts.selection == 'contiguous' else
                 '18,750 records per native interval, eight chronological intervals. Historical 100K tag/relation stratification, token eligibility, deduplication and source ordering; seed 42. Prefer new/update labels; fill interval shortages with QA-complete native rows whose original tag is empty, retaining that tag and auditing their count.')
    payload = dict(schema_version='wikibigedit-camr-150k/v1', selection=selection,
                   selection_mode=opts.selection, sources=sources, samples=rows)
    if protocol:
        payload['primary_evaluation_ids'] = protocol['primary_evaluation_ids']
        payload['stage_evaluation_ids'] = protocol['stage_evaluation_ids']
        payload['historical_protocol_id'] = protocol['protocol_id']
        payload['sampling'] = protocol['sampling']
    atomic_json(HERE / 'manifest.json', payload)
    print(json.dumps({'status': 'auditing_token_lengths', 'rows': len(rows),
                      'stages': dict(Counter(r['stage'] for r in rows))}), flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(opts.model_dir, local_files_only=True)
    encoder = AutoTokenizer.from_pretrained(opts.encoder_dir, local_files_only=True)
    maximum = Counter()
    axes = {'update': ('update', 'ans'), 'rephrase': ('rephrase', 'ans'),
            'persona': ('personas', 'ans'), 'locality': ('loc', 'loc_ans')}
    counts, stress_ids = Counter(), set()
    train_lengths = []
    for start in range(0, len(rows), 2000):
        chunk = rows[start:start + 2000]
        prompts = tokenizer([qa_prompt(r['update']) for r in chunk], add_special_tokens=True)['input_ids']
        answers = tokenizer([' ' + r['ans'] for r in chunk], add_special_tokens=False)['input_ids']
        emb = encoder([f"Question: {r['update'].strip()}\nAnswer: {r['ans'].strip()}" for r in chunk],
                      add_special_tokens=True)['input_ids']
        for i, (p, a, e) in enumerate(zip(prompts, answers, emb)):
            length = len(p) + len(a) + 1
            assert length <= PAPER_MAX_LENGTH and len(e) <= PAPER_MAX_LENGTH, chunk[i]['sample_id']
            train_lengths.append((length, start + i))
            maximum['training_tokens'] = max(maximum['training_tokens'], length)
            maximum['embedding_tokens'] = max(maximum['embedding_tokens'], len(e))
            maximum['answer_tokens_with_eos'] = max(maximum['answer_tokens_with_eos'], len(a) + 1)
        for axis, (q, a) in axes.items():
            valid = [r for r in chunk if isinstance(r.get(q), str) and r[q].strip()
                     and isinstance(r.get(a), str) and r[a].strip()]
            counts[axis] += len(valid)
            if valid:
                lengths = [len(x) for x in tokenizer([qa_prompt(r[q]).rstrip() for r in valid],
                                                   add_special_tokens=False)['input_ids']]
                assert max(lengths) <= PAPER_MAX_LENGTH, (axis, start, max(lengths))
                maximum[axis + '_prompt_tokens'] = max(maximum[axis + '_prompt_tokens'], max(lengths))
        if (start + len(chunk)) % 20000 == 0:
            print(json.dumps({'status': 'auditing', 'rows': start + len(chunk)}), flush=True)
    stress_ids.update(index for _, index in sorted(train_lengths, reverse=True)[:48])
    for stage in sorted({r['stage'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['stage'] == stage]
        stress_ids.update(indices[:2] + indices[-2:])
    for i in range(len(rows)):
        if len(stress_ids) >= 80:
            break
        stress_ids.add(i)
    stress_rows = [rows[i] for i in sorted(stress_ids)]
    atomic_json(HERE / 'stress_manifest.json', dict(payload, samples=stress_rows,
                selection='Independent stress test: longest training rows and native stage boundaries'))
    train_questions = {normalize_text(r['update']) for r in rows}
    locality_answers = defaultdict(set)
    locality_overlap = 0
    for r in rows:
        if isinstance(r.get('loc'), str) and r['loc'].strip() and isinstance(r.get('loc_ans'), str) and r['loc_ans'].strip():
            question = normalize_text(r['loc'])
            locality_answers[question].add(normalize_text(r['loc_ans']))
            locality_overlap += question in train_questions
    audit = dict(rows=len(rows), selection=selection, selection_mode=opts.selection, sources=sources,
        manifest_sha256=sha(HERE / 'manifest.json'), native_stage_counts=dict(Counter(r['stage'] for r in rows)),
        tag_counts=dict(Counter(str(r.get('tag')) for r in rows)), valid_evaluation_rows=dict(counts),
        missing_evaluation_rows={k: len(rows) - v for k, v in counts.items()}, maximum_tokens=dict(maximum),
        truncations=0, unique_training_qa=len({(r['update'], r['ans']) for r in rows}),
        locality_unique_questions=len(locality_answers),
        locality_multitarget_questions=sum(len(a) > 1 for a in locality_answers.values()),
        locality_training_question_overlap_rows=locality_overlap,
        first_5000_identical_to_reference=(rows[:5000] == old_rows[:5000]
                                             if old_rows is not None else None),
        stress_rows=len(stress_rows),
        reference_qa_overlap=(len({(r['update'], r['ans']) for r in rows} &
                                  {(r['update'], r['ans']) for r in old_rows})
                              if old_rows is not None else None))
    primary_counts = ({a: len(protocol['stage_evaluation_ids'][a] if a == 'locality'
                                else protocol['primary_evaluation_ids'][a]) for a in axes}
                      if protocol else dict(counts))
    audit['primary_evaluation_rows'] = primary_counts
    print(json.dumps({'status': 'prepared', **audit}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
