"""Freeze unused screened facts, disjoint from test subjects, facts and all probes."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from baselines.wikibigedit_protocol import _normalized_key, _refresh_selection, validate_manifest, sha256_file

FIELDS=('update','rephrase','personas','loc','mhop')
IDENTITY_FIELDS=('sample_id','subject','subject_id','fact_fingerprint')
def probes(row):
    return {_normalized_key(row[k]) for k in FIELDS if row.get(k)}
def freeze(source, per_stage=200, seed=20260920):
    m=copy.deepcopy(source)
    test_ids=set(source['selection']['sample_ids'])
    test=[r for r in source['records'] if r['sample_id'] in test_ids]
    blocked={field: {_normalized_key(r[field]) for r in test if r.get(field)}
             for field in IDENTITY_FIELDS}
    text=set().union(*(probes(r) for r in test))
    clean=[r for r in m['records'] if r.get('base_screen',{}).get('status')=='unknown'
           and all(_normalized_key(r.get(field)) not in values for field, values in blocked.items())
           and not probes(r).intersection(text)]
    by_stage={}
    for stage in range(1,9):
        rows=sorted((r for r in clean if r['stage']==stage),key=lambda r:hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).digest())
        if len(rows)<per_stage: raise ValueError(f'insufficient independent dev facts at stage {stage}: {len(rows)}')
        by_stage[stage]=[r['sample_id'] for r in rows[:per_stage]]
    selected=set().union(*(set(v) for v in by_stage.values()))
    m['records']=[r for r in clean if r['sample_id'] in selected]
    for stage in m['stages']:
        ids=by_stage[stage['stage']]
        stage.update(candidate_sample_ids=ids,candidate_count=len(ids),target_count=len(ids))
    m['sampling'].update(seed=seed,total=per_stage*8,purpose='held-out Qwen3 configuration selection')
    _refresh_selection(m,by_stage,'subject/fact/all-probe disjoint base-unknown held-out dev',False)
    m['split_audit']={'role':'dev_only','parent_test_protocol_id':source['protocol_id'],
      'id_overlap':0,'subject_overlap':0,'subject_id_overlap':0,'fact_fingerprint_overlap':0,'probe_text_overlap':0,
      'normalization':'Unicode NFKC + casefold + collapsed whitespace; both subject names and IDs checked',
      'eligible_counts_by_stage':dict(Counter(r['stage'] for r in clean)),
      'selection_seed':seed,'selection_must_precede_dev_results':True,
      'base_screen_only':'Uses unused candidates screened on frozen base; no edited-model outcomes used.'}
    validate_manifest(m)
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--test-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=freeze(json.loads(a.test_manifest.read_text()));m['split_audit']['parent_test_sha256']=sha256_file(a.test_manifest)
    content=json.dumps(m,ensure_ascii=False,indent=2)+'\n'
    if a.output.exists() and a.output.read_text()!=content:raise FileExistsError('Frozen dev differs: refusing overwrite')
    a.output.write_text(content);print(m['protocol_id'],m['selection']['total'],m['split_audit'])
if __name__=='__main__': main()
