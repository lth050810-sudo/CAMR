"""Freeze closed-book base answers on the shared full-phase sampled panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from baselines.common import answer_scores, atomic_json, axis_fields, evaluate_axis, load_manifest, sha256_file
from baselines.run_rag import (
    AXES, _canonical_digest, _reader_provenance, atomic_jsonl, freeze_reader,
    load_axis_cache, load_evaluation_panels, load_reader, sampled_evaluation_counts, write_axis_cache,
)


def validate_axis_records(records, rows, axis):
    question, answer = axis_fields(axis)
    if len(records) != len(rows):
        raise ValueError(f'Pre-Edit reference panel coverage differs: {axis}')
    for record, row in zip(records, rows):
        if any(record.get(key) != value for key, value in {
            'sample_id': row['sample_id'], 'fact_stage': row['stage'], 'axis': axis,
            'checkpoint': 'base', 'question': row[question], 'gold': row[answer],
        }.items()):
            raise ValueError(f'Pre-Edit reference source identity differs: {axis}')
        if not isinstance(record.get('prediction'), str):
            raise ValueError('Pre-Edit reference prediction is missing')
        scores = answer_scores(record['prediction'], record['gold'])
        if any(not isinstance(record.get(key), bool) or record[key] != value for key, value in scores.items()):
            raise ValueError('Pre-Edit reference scorer fields differ from predictions')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--evaluation-panels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval-batch-size', type=int, default=48)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows, metadata = load_manifest(args.manifest)
    manifest_hash = sha256_file(args.manifest)
    panels, provenance = load_evaluation_panels(args.evaluation_panels, rows, metadata, manifest_sha256=manifest_hash)
    identity = {
        'method': 'pre_edit_reference', 'reader': _reader_provenance(args.model_dir),
        'manifest_sha256': manifest_hash,
        'evaluation_panels': {key: value for key, value in provenance.items() if key != 'source'},
        'prompt': 'Question: <question>\nAnswer:', 'context_fact_count': 0,
        'max_length': args.max_length, 'max_new_tokens': args.max_new_tokens,
        'eval_batch_size': args.eval_batch_size,
        'common_source_sha256': sha256_file(Path(__file__).with_name('common.py')),
        'reference_source_sha256': sha256_file(Path(__file__)),
    }
    fingerprint = _canonical_digest(identity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / 'run_config.json'
    if config_path.exists():
        if not args.resume:
            raise FileExistsError('Pre-Edit reference exists; use --resume')
        if json.loads(config_path.read_text()).get('run_fingerprint') != fingerprint:
            raise ValueError('Pre-Edit reference resume configuration differs')
    config = {
        **identity, 'run_fingerprint': fingerprint,
        'manifest': str(args.manifest.resolve()), 'model_dir': str(args.model_dir.resolve()),
        'source_selected_fact_count': len(rows), 'parameter_updates': 0,
        'closed_book_evaluation': True, 'reader_frozen': True,
        'reference_only': True, 'num_training_facts': 0,
        'evaluation_panels_provenance': provenance,
        'evaluation_scope': 'frozen sampled Pre-Edit reference; no retrieved context; not full-data evaluation',
        **sampled_evaluation_counts(panels),
    }
    atomic_json(config_path, config)
    cached, missing = {}, []
    for axis in AXES:
        records = load_axis_cache(args.output_dir, 'base', axis, run_fingerprint=fingerprint,
                                  expected_count=len(panels[axis])) if args.resume else None
        if records is not None:
            validate_axis_records(records, panels[axis], axis)
            cached[axis] = records
        else:
            missing.append(axis)
    if missing:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required to generate the frozen base answers')
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        tokenizer, model = load_reader(args.model_dir, device)
        freeze_reader(model)
        for axis in missing:
            _, records = evaluate_axis(model, tokenizer, panels[axis], axis, device,
                                       batch_size=args.eval_batch_size, max_length=args.max_length,
                                       max_new_tokens=args.max_new_tokens, checkpoint='base')
            validate_axis_records(records, panels[axis], axis)
            write_axis_cache(args.output_dir, 'base', axis, records, run_fingerprint=fingerprint)
            cached[axis] = records
    all_records = [record for axis in AXES for record in cached[axis]]
    metrics = {'checkpoint': 'base', 'reference_only': True, 'run_fingerprint': fingerprint,
               'evaluation_scope': config['evaluation_scope'], **sampled_evaluation_counts(panels)}
    for axis in AXES:
        records = cached[axis]
        metrics[axis] = {'count': len(records)}
        for name in ('exact', 'match'):
            correct = sum(record[name] for record in records)
            metrics[axis].update({name + '_count': correct,
                                  name + '_accuracy': correct / len(records) if records else None})
    atomic_json(args.output_dir / 'metrics.json', metrics)
    atomic_jsonl(args.output_dir / 'predictions.jsonl', all_records)
    atomic_json(args.output_dir / 'completed.json', {
        'status': 'verified_complete', 'method': 'pre_edit_reference', 'reference_only': True,
        'source_selected_fact_count': len(rows), 'parameter_updates': 0, 'context_fact_count': 0,
        'manifest_sha256': manifest_hash, 'panel_file_sha256': provenance['file_sha256'],
        'panel_payload_sha256': provenance['panel_payload_sha256'], 'run_fingerprint': fingerprint,
        'scores_recomputed_from_predictions': True, 'evaluated_axes': list(AXES),
        'predictions_sha256': sha256_file(args.output_dir / 'predictions.jsonl'),
        'metrics_sha256': sha256_file(args.output_dir / 'metrics.json'),
        'resumed_axes': len(AXES) - len(missing), 'generated_axes': len(missing),
        **sampled_evaluation_counts(panels),
    })


if __name__ == '__main__':
    main()
