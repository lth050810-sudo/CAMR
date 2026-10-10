"""Shared event stream helpers without importing training or retrieval runners."""
from __future__ import annotations

import json

from baselines.common import group_by_stage


def select_rows(rows, max_stages=0, examples_per_stage=0):
    stages = group_by_stage(sorted(rows, key=lambda row: (int(row['stage']), str(row['sample_id']))))
    keys = list(stages)[:max_stages or None]
    return [row for key in keys for row in stages[key][:examples_per_stage or None]]


def make_events(rows, repeat_every=0):
    """Inject only old training QA, separately marked from natural arrivals."""
    events = []
    for index, row in enumerate(rows):
        events.append({'row': row, 'arrival_stage': row['stage'], 'diagnostic_repeat': False})
        if repeat_every and (index + 1) % repeat_every == 0:
            old = rows[max(0, index + 1 - repeat_every)]
            events.append({'row': old, 'arrival_stage': row['stage'], 'diagnostic_repeat': True})
    return events


def read_schedule(path, events, allow_ignore_replay=False, random_replay=False):
    """Match observed actions and counts, explicitly opting into IGNORE replay.

    ADD may include random history in v2. Positive IGNORE replay counts are
    accepted only when enabled and marked ``training_role='history_only'``;
    legacy IGNORE records still mean that no writer training took place.
    """
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != len(events):
        raise ValueError('replay schedule length differs from this event stream')
    for index, (record, event) in enumerate(zip(records, events)):
        if (record['event'] != index or record['sample_id'] != event['row']['sample_id']
                or record['arrival_stage'] != event['arrival_stage']
                or record['diagnostic_repeat'] != event['diagnostic_repeat']):
            raise ValueError(f'replay schedule stream mismatch at event {index}')
        count = record['replay_count']
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('scheduled replay count must be a nonnegative integer')
        action = record['action']
        if action not in {'add', 'merge', 'ignore'}:
            raise ValueError('invalid scheduled action')
        if random_replay and action != 'add':
            raise ValueError('random_replay schedules may contain ADD only; MERGE and IGNORE are policy actions')
        if action == 'merge' and count == 0:
            raise ValueError('scheduled merge requires at least one replay row')
        if action == 'ignore' and count != 0:
            if not allow_ignore_replay:
                raise ValueError('scheduled ignore must have no replay rows')
            if record.get('training_role') != 'history_only':
                raise ValueError("scheduled ignore replay requires training_role='history_only'")
    return records
