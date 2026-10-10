#!/usr/bin/env python3
"""CPU-only, prefix-causal offline audit of the Engram writer's candidate recall.

The full-prefix dense oracle exists only in this diagnostic.  It is never fed
back into FactMemory, the writer policy, or training.  Cosine-threshold relevance
is an embedding-space reference, not ground-truth semantic/factual equivalence.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from baselines.common import atomic_json, load_manifest, sha256_file
from baselines.engram_continual_core import FactMemory
from baselines.run_engram_continual import select_rows, training_view
from baselines.run_rag import HFMeanPoolEncoder, fact_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run-config', type=Path,
        default=Path('runs/engram_continual_pilot/policy/run_config.json'),
        help='Use the pilot selection, encoder, and hash configuration verbatim.',
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('runs/engram_continual_pilot/retrieval_audit.json'),
    )
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    return args


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def make_memory(dim: int, config: dict[str, Any]) -> FactMemory:
    return FactMemory(
        dim, seed=config['seed'], tables=config['hash_tables'], bits=config['hash_bits'],
        max_candidates=config['max_candidates'],
        similarity_threshold=config['similarity_threshold'], fact_mode=config['fact_mode'],
    )


def audit_prefixes(
    rows: list[dict[str, Any]], vectors: torch.Tensor, config: dict[str, Any],
) -> dict[str, Any]:
    """Query only inserted rows; keep dense oracle results external to memory."""
    memory = make_memory(vectors.shape[1], config)
    threshold = config['similarity_threshold']
    details: list[dict[str, Any]] = []
    hash_seconds = 0.0
    oracle_seconds = 0.0
    for index, (row, vector) in enumerate(zip(rows, vectors)):
        started = time.perf_counter()
        query = memory.query(training_view(row), vector)
        hash_seconds += time.perf_counter() - started
        candidates = query['candidates']
        assert len(memory.rows) == index
        assert all(item['memory_index'] < index for item in candidates)
        candidate_ids = {item['memory_index'] for item in candidates}
        lsh_ids = {item['memory_index'] for item in candidates if item['lsh_votes'] > 0}

        # A full dense scan is permitted here only for the independent oracle.
        started = time.perf_counter()
        scores = vectors[:index] @ vector
        oracle_seconds += time.perf_counter() - started
        positives = {j for j, score in enumerate(scores.tolist()) if score >= threshold}
        oracle_top = max(range(index), key=lambda j: (float(scores[j]), j)) if index else None
        candidate_top = candidates[0]['memory_index'] if candidates else None
        lsh_top = next((item['memory_index'] for item in candidates if item['lsh_votes'] > 0), None)
        overlap = positives & candidate_ids
        lsh_overlap = positives & lsh_ids
        details.append({
            'event': index, 'sample_id': row['sample_id'], 'stage': row['stage'],
            'history_count': index, 'candidate_count': len(candidate_ids),
            'hash_and_exact_recalled_count_before_cap': query['hash_recalled_count'],
            'lsh_supported_shortlist_count': len(lsh_ids),
            'dense_top1_memory_index': oracle_top,
            'dense_top1_sample_id': rows[oracle_top]['sample_id'] if oracle_top is not None else None,
            'dense_top1_cosine': float(scores[oracle_top]) if oracle_top is not None else None,
            'candidate_rerank_top1_memory_index': candidate_top,
            'candidate_rerank_top1_cosine': candidates[0]['similarity'] if candidates else None,
            'dense_top1_in_candidates': oracle_top in candidate_ids if index else None,
            'rerank_top1_matches_dense_top1': candidate_top == oracle_top if index else None,
            'lsh_supported_top1_matches_dense_top1': lsh_top == oracle_top if index else None,
            'dense_above_threshold_count': len(positives),
            'retrieved_above_threshold_count': len(overlap),
            'lsh_supported_above_threshold_count': len(lsh_overlap),
            'candidate_recall_above_threshold': ratio(len(overlap), len(positives)),
            'candidate_precision_above_threshold': ratio(len(overlap), len(candidate_ids)),
            'actual_compatible_replay_candidates': len(query['similar_rows']),
            'duplicate_count': len(query['duplicate_rows']),
            'ambiguous_count': len(query['ambiguous_rows']),
            'missed_above_threshold_sample_ids': [rows[j]['sample_id'] for j in sorted(positives - candidate_ids)],
        })
        # Insert after both independent observations; no future/current fact can
        # be retrieved for its own event.  The shared selection is fact-unique.
        memory.insert(training_view(row), vector)

    with_history = [item for item in details if item['history_count']]
    with_positives = [item for item in details if item['dense_above_threshold_count']]
    positive_total = sum(item['dense_above_threshold_count'] for item in details)
    candidate_total = sum(item['candidate_count'] for item in details)
    retrieved_total = sum(item['retrieved_above_threshold_count'] for item in details)
    lsh_total = sum(item['lsh_supported_shortlist_count'] for item in details)
    lsh_retrieved_total = sum(item['lsh_supported_above_threshold_count'] for item in details)
    summary = {
        'queries': len(details), 'queries_with_history': len(with_history),
        'queries_with_dense_above_threshold_history': len(with_positives),
        'queries_with_no_candidates_despite_history': sum(not item['candidate_count'] for item in with_history),
        'dense_top1_covered': sum(bool(item['dense_top1_in_candidates']) for item in with_history),
        'dense_top1_coverage_rate': ratio(sum(bool(item['dense_top1_in_candidates']) for item in with_history), len(with_history)),
        'rerank_top1_agreement_rate': ratio(sum(bool(item['rerank_top1_matches_dense_top1']) for item in with_history), len(with_history)),
        'lsh_supported_top1_agreement_rate': ratio(sum(bool(item['lsh_supported_top1_matches_dense_top1']) for item in with_history), len(with_history)),
        'mean_candidates_all_queries': ratio(candidate_total, len(details)),
        'mean_candidates_given_history': ratio(candidate_total, len(with_history)),
        'mean_history_count': ratio(sum(item['history_count'] for item in details), len(details)),
        'dense_above_threshold_pairs': positive_total,
        'retrieved_above_threshold_pairs': retrieved_total,
        'shortlisted_candidate_pairs': candidate_total,
        'micro_candidate_recall_above_threshold': ratio(retrieved_total, positive_total),
        'micro_candidate_precision_above_threshold': ratio(retrieved_total, candidate_total),
        'macro_candidate_recall_on_positive_queries': ratio(sum(item['candidate_recall_above_threshold'] for item in with_positives), len(with_positives)),
        'lsh_supported_shortlisted_pairs': lsh_total,
        'lsh_supported_above_threshold_pairs': lsh_retrieved_total,
        'lsh_supported_micro_recall_above_threshold': ratio(lsh_retrieved_total, positive_total),
        'lsh_supported_micro_precision_above_threshold': ratio(lsh_retrieved_total, lsh_total),
        'candidate_query_seconds': hash_seconds,
        'offline_dense_oracle_seconds': oracle_seconds,
    }
    return {'summary': summary, 'events': details}


def handwritten_rows() -> list[dict[str, Any]]:
    """Fresh English examples; no official rephrase/persona probes are used."""
    specifications = [
        ('china_initial', 'What is the capital city of China?', 'Beijing',
         'China', 'Q148', 'capital', 'P36', 'Beijing', 'Q956'),
        ('china_paraphrase', "Which city serves as China's capital?", 'Beijing',
         'China', 'Q148', 'capital', 'P36', 'Beijing', 'Q956'),
        ('south_korea_related', 'What is the capital city of South Korea?', 'Seoul',
         'South Korea', 'Q884', 'capital', 'P36', 'Seoul', 'Q8684'),
        ('newton_new_domain', "How does Newton's second law relate force, mass, and acceleration?",
         'Force equals mass times acceleration (F = ma).',
         "Newton's second law", 'handwritten:newtons_second_law', 'mathematical formula',
         'handwritten:formula', 'F = ma', 'handwritten:force_formula'),
    ]
    return [dict(zip(
        ('sample_id', 'update', 'ans', 'subject', 'subject_id', 'relation', 'relation_id', 'object', 'object_id'),
        specification,
    ), stage=1, tag='handwritten_diagnostic') for specification in specifications]


def audit_handwritten(
    encoder: HFMeanPoolEncoder, config: dict[str, Any],
) -> dict[str, Any]:
    rows = handwritten_rows()
    vectors = encoder.encode([fact_document(row) for row in rows])
    # This explicitly diagnostic case uses handwritten IDs to certify identity.
    diagnostic_config = {**config, 'fact_mode': 'structured'}
    memory = make_memory(vectors.shape[1], diagnostic_config)
    expected = ('add', 'ignore', 'merge', 'add')
    events = []
    for index, (row, vector) in enumerate(zip(rows, vectors)):
        query = memory.query(row, vector)
        allow_ignore = bool(query['duplicate_rows'])  # Mastery assumed, not measured.
        allow_merge = bool(query['similar_rows'])
        rule_action = 'ignore' if allow_ignore else 'merge' if allow_merge else 'add'
        events.append({
            'event': index, 'row': row, 'encoded_text': fact_document(row),
            'history_count': len(memory.rows),
            'expected_rule_action': expected[index], 'observed_rule_action': rule_action,
            'matches_expected': rule_action == expected[index],
            'allow_ignore_if_currently_mastered': allow_ignore,
            'allow_merge': allow_merge,
            'duplicate_sample_ids': [old['sample_id'] for old in query['duplicate_rows']],
            'compatible_similar_sample_ids': [old['sample_id'] for old in query['similar_rows']],
            'candidates': [{
                'sample_id': item['row']['sample_id'], 'cosine': item['similarity'],
                'lsh_votes': item['lsh_votes'], 'equivalent': item['equivalent'],
                'same_slot': item['same_slot'],
            } for item in query['candidates']],
        })
        memory.insert(row, vector)
    return {
        'source': 'Four hand-written English QA examples; no dataset rephrase/persona used.',
        'fact_mode': 'structured',
        'mastery': 'Assumed true solely to show ignore eligibility; Qwen is not loaded or tested.',
        'action_rule': 'Ignore if duplicate and assumed mastered; else merge if eligible; else add.',
        'policy_sampling': False,
        'equivalence_limitation': 'Same hand-authored triple IDs and exact target certify duplicate; this is not a general text equivalence test.',
        'pairwise_cosines': vectors.matmul(vectors.T).tolist(),
        'all_expected_rule_actions_observed': all(item['matches_expected'] for item in events),
        'events': events,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite audit: {args.output}')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    config = json.loads(args.run_config.read_text())
    manifest = Path(config['manifest'])
    all_rows, metadata = load_manifest(manifest)
    rows = select_rows(all_rows, config['max_stages'], config['examples_per_stage'])
    if not rows:
        raise ValueError('Empty shared selection')
    if [row['sample_id'] for row in rows] != config['sample_ids']:
        raise ValueError('Audit selection differs from pilot run_config sample IDs')
    if sha256_file(manifest) != config['manifest_sha256']:
        raise ValueError('Manifest changed since the pilot configuration was saved')

    started = time.perf_counter()
    encoder = HFMeanPoolEncoder(
        config['embedding_model'], device=torch.device('cpu'), batch_size=args.batch_size,
        max_length=config['max_length'], query_prefix='', document_prefix='', local_files_only=True,
    )
    load_seconds = time.perf_counter() - started
    assert next(encoder.model.parameters()).device.type == 'cpu'
    started = time.perf_counter()
    vectors = encoder.encode([fact_document(training_view(row)) for row in rows])
    encode_seconds = time.perf_counter() - started
    result = {
        'schema_version': 'engram-prefix-retrieval-audit/v1',
        'scope': 'Offline CPU development diagnostic; no training, policy update, Qwen loading, or routing changes.',
        'device': 'cpu', 'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'torch_version': torch.__version__,
        'dense_oracle_used_for_routing': False,
        'online_order': 'Each query sees strictly earlier inserted rows; insert current only after audit.',
        'embedding_precomputation': 'Fixed encoder may batch all texts; later vectors never enter an earlier retrieval or oracle prefix.',
        'run_config': str(args.run_config), 'run_config_sha256': sha256_file(args.run_config),
        'manifest': str(manifest), 'manifest_sha256': sha256_file(manifest),
        'protocol_id': metadata.get('protocol_id'), 'split_audit': metadata.get('split_audit'),
        'selected_sample_ids': [row['sample_id'] for row in rows],
        'stage_counts': dict(sorted(Counter(str(row['stage']) for row in rows).items())),
        'hash_config': {key: config[key] for key in ('seed', 'hash_tables', 'hash_bits', 'max_candidates', 'similarity_threshold', 'fact_mode')},
        'encoder': encoder.provenance,
        'metric_definitions': {
            'candidate_pool': 'Actual FactMemory shortlist: multi-probe LSH plus exact fact/slot keys, capped, then cosine-reranked.',
            'lsh_supported': 'Members of the actual shortlist with positive LSH votes; not a separate rerun with a different cap.',
            'dense_top1_coverage': 'Fraction of queries with nonempty history whose highest-cosine prior vector is shortlisted; tie breaks toward later prior index.',
            'above_threshold_recall': 'Retrieved prior pairs with cosine >= threshold / all such prior pairs in dense prefix.',
            'candidate_precision': 'Shortlisted prior pairs with cosine >= threshold / all shortlisted prior pairs, before cosine filtering.',
            'null': 'Undefined rate because its denominator is zero.',
            'limitation': 'Dense cosine oracle measures embedding-neighbor recall, not semantic truth, factual equivalence, or downstream learning quality.',
            'cpu_precision_note': 'Audit encoder runs float32 on CPU; pilot GPU uses float16, so very near-threshold scores or hashes may differ.',
        },
        'timing': {'encoder_load_seconds': load_seconds, 'selected_stream_encode_seconds': encode_seconds},
        'prefix_retrieval': audit_prefixes(rows, vectors, config),
        'handwritten_diagnostic': audit_handwritten(encoder, config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({
        'output': str(args.output),
        'prefix_summary': result['prefix_retrieval']['summary'],
        'handwritten_actions': [{
            'sample_id': item['row']['sample_id'], 'action': item['observed_rule_action'],
            'expected': item['expected_rule_action'],
        } for item in result['handwritten_diagnostic']['events']],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
