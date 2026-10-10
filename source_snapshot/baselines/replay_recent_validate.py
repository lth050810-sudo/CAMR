"""Validate a completed recent-replay run without loading a model."""
import argparse
import json
import math
from pathlib import Path


def validate(directory):
    config=json.loads((directory/'run_config.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    stream=json.loads((directory/'selected_training_stream.json').read_text())
    actions=[json.loads(line) for line in (directory/'actions.jsonl').read_text().splitlines()]
    exposures=json.loads((directory/'exposures.json').read_text())
    assert summary['status']=='completed'
    assert summary['totals']['optimizer_steps']==config['planned_optimizer_steps']
    assert sum(a['train']['optimizer_steps'] for a in actions)==config['planned_optimizer_steps']
    assert len(actions)==summary['events']
    assert summary['metrics'] and summary['metrics'][-1]['current_stage']==max(r['stage'] for r in stream)
    stages={r['sample_id']:r['stage'] for r in stream}
    seen=set()
    counts={}
    for n,action in enumerate(actions,1):
        assert action['event']==n and math.isfinite(action['train']['loss'])
        assert all(stages[i]<=action['stage'] for i in action['sample_ids'])
        if config['method'] in ('mir_full','srt_online'):
            assert set(action['replay_sample_ids'])<=seen
            seen.add(action['sample_ids'][0])
        for i in action['sample_ids']: counts[i]=counts.get(i,0)+action['train']['optimizer_steps']
    assert counts==exposures
    assert summary['unique_facts_trained']==len(counts)
    forbidden={'rephrase','personas','loc','loc_ans','mhop','mhop_ans'}
    assert not any(forbidden & set(r) for r in stream)
    if config['method']=='opr_ru':
        for line in (directory/'opr_rollouts.jsonl').read_text().splitlines():
            r=json.loads(line)
            assert stages[r['sample_id']]<=r['after_stage']
    print(json.dumps({'validated':True,'method':config['method'],'steps':config['planned_optimizer_steps'],
                      'unique_facts_trained':len(counts),'coverage':summary['coverage']}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    validate(p.parse_args().output_dir)
