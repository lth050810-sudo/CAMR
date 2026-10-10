"""Source-bound training views, without evaluation input.

Accepted cache entries may contain separately generated training paraphrases.
By default only accepted cache entries receive a second training view. The
explicit legacy_wrapper policy retains the former instruction-only fallback
for reproducing existing experiments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


TRAINING_VIEW_VERSION = 'original-plus-training-view/v1'
TRAINING_VIEW_CACHE_SCHEMA = 'engram-training-views/v1'
TRAINING_VIEW_POLICIES = ('accepted_only', 'legacy_wrapper')
REFORMULATION_PREFIX = 'Please answer this question: '


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def load_training_view_cache(path):
    if path is None:
        return {}, None
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if (not isinstance(payload, dict) or payload.get('schema_version') != TRAINING_VIEW_CACHE_SCHEMA
            or not isinstance(payload.get('rows'), dict)):
        raise ValueError('Invalid training-view cache schema.')
    for sample_id, item in payload['rows'].items():
        if (not sample_id or not isinstance(item, dict)
                or type(item.get('accepted')) is not bool
                or not all(isinstance(item.get(key), str) and len(item[key]) == 64
                           for key in ('update_sha256', 'answer_sha256'))):
            raise ValueError(f'Invalid training-view cache record: {sample_id}')
        if item['accepted'] and (not isinstance(item.get('question'), str) or not item['question'].strip()):
            raise ValueError(f'Accepted training view is empty: {sample_id}')
    return payload['rows'], hashlib.sha256(raw).hexdigest()


def expand_training_views(rows: Sequence[Mapping[str, Any]], training_views: int = 1,
                          cached_views: Mapping[str, Mapping[str, Any]] | None = None,
                          training_view_policy: str = 'accepted_only'):
    """Return QA-only rows and their original fact indices, in fact order.

    Reads only sample_id/update/ans even if callers supply richer records. Preserving
    the answer verbatim prevents aliases or held-out fields becoming labels.
    ``training_views`` is the maximum per fact, not a fixed batch stride: under
    accepted_only a rejected or absent cached view leaves just the original.
    """
    if isinstance(training_views, bool) or training_views not in (1, 2):
        raise ValueError('training_views must be 1 or 2.')
    if training_view_policy not in TRAINING_VIEW_POLICIES:
        raise ValueError('training_view_policy must be accepted_only or legacy_wrapper.')
    expanded, fact_indices = [], []
    for fact_index, row in enumerate(rows):
        question, answer = row['update'], row['ans']
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Training questions must be nonempty strings.')
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('Training answers must be nonempty strings.')
        expanded.append({'update': question, 'ans': answer})
        fact_indices.append(fact_index)
        if training_views == 2:
            reformulated = (REFORMULATION_PREFIX + question
                            if training_view_policy == 'legacy_wrapper' else None)
            cached = (cached_views or {}).get(str(row.get('sample_id', '')))
            if cached is not None:
                if (cached['update_sha256'] != text_sha256(question)
                        or cached['answer_sha256'] != text_sha256(answer)):
                    raise ValueError(f'Training-view cache source differs: {row.get("sample_id")}')
                if cached['accepted']:
                    reformulated = cached['question']
            if reformulated is not None:
                expanded.append({'update': reformulated, 'ans': answer})
                fact_indices.append(fact_index)
    return expanded, fact_indices
