"""Fail closed on incomplete streams, missing axes, or future facts in new runs."""
import hashlib,json,math
from collections import Counter
from pathlib import Path
from baselines.report_expansion import NAMES,ROOT,RUNS,BATCH,EXPECTED,AXES
from baselines.wikibigedit_protocol import answer_scores,sha256_file

FIELDS={'update':('update','ans'),'rephrase':('rephrase','ans'),'persona':('personas','ans'),'locality':('loc','loc_ans'),'multihop':('mhop','mhop_ans')}

def digest(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def audit_one(name):
    p=RUNS/name
    if not (p/'completed.json').exists():return {'status':'not_complete'}
    done=json.loads((p/'completed.json').read_text())
    assert done.get('status')=='complete' and done.get('final_stage')==8, 'Invalid completion marker'
    assert done.get('num_facts',3000)==3000 and done.get('edits',3000)==3000
    m=json.loads((ROOT/'runs/protocols/wikibigedit_3k_seed42_qwen3_unknown.json').read_text())
    selected=set(m['selection']['sample_ids']);rows={r['sample_id']:r for r in m['records'] if r['sample_id'] in selected}
    c=json.loads((p/'run_config.json').read_text());metrics=json.loads((p/'metrics.json').read_text())
    assert c['manifest_protocol_id']==EXPECTED and c['num_facts']==3000
    assert c['manifest_sha256']==sha256_file(ROOT/'runs/protocols/wikibigedit_3k_seed42_qwen3_unknown.json')
    assert [v['current_stage'] for v in metrics]==list(range(1,9))
    for stage,v in enumerate(metrics,1):
        assert v['num_facts']==375*stage
        for a in AXES:
            assert v[a]['count']>0 and math.isfinite(v[a]['match_accuracy'])
    final_counts={a:metrics[-1][a]['count'] for a in AXES}
    assert final_counts==dict(update=3000,rephrase=3000,persona=3000,locality=2998,multihop=127)
    count=0;future=0;seen=set();observed=Counter();matches=Counter();exacts=Counter()
    with (p/'predictions.jsonl').open() as f:
        for line in f:
            r=json.loads(line);count+=1
            key=(r['checkpoint'],r['axis'],r['sample_id']);assert key not in seen;seen.add(key)
            assert r['sample_id'] in selected
            source=rows[r['sample_id']];q,a=FIELDS[r['axis']]
            assert r['question']==source[q] and r['gold']==source[a]
            assert r['fact_stage']==source['stage']
            exact,match=answer_scores(r['prediction'],r['gold'])
            assert r['exact']==exact and r['match']==match
            observed[(r['checkpoint'],r['axis'])]+=1
            matches[(r['checkpoint'],r['axis'])]+=match
            exacts[(r['checkpoint'],r['axis'])]+=exact
            if r['axis'] in ('update','rephrase','persona'):
                stage=int(r['checkpoint'].removeprefix('tau'))
                assert rows[r['sample_id']]['stage']<=stage
            if 'retrieved_sample_ids' in r:
                ids=r['retrieved_sample_ids'];stage=r['current_stage']
                assert len(ids)==len(set(ids))
                assert all(rows[i]['stage']<=stage for i in ids)
                assert [rows[i]['stage'] for i in ids]==r['retrieved_fact_stages']
                if c.get('recipe')=='wikibigedit':
                    context=' '.join(f"Q: {rows[i]['update'].strip()} A: {rows[i]['ans'].strip()}" for i in ids)
                    prompt=f"{context} Q: {r['question'].strip()} A:".strip()
                    assert r['reader_prompt_sha256']==digest(prompt)
    for metric in metrics:
        for axis in AXES:
            assert observed[(metric['checkpoint'],axis)]==metric[axis]['count']
            key=(metric['checkpoint'],axis);n=observed[key]
            assert matches[key]==metric[axis]['match_count'] and exacts[key]==metric[axis]['exact_count']
            assert math.isclose(matches[key]/n,metric[axis]['match_accuracy'],abs_tol=1e-12)
            assert math.isclose(exacts[key]/n,metric[axis]['exact_accuracy'],abs_tol=1e-12)
    trace=p/'edit_trace.jsonl'
    if name.startswith(('rome_','r_rome_','memit_nas_','nas_transfer_','horen_')):
        assert trace.is_file(), 'Missing required editor trace'
    if name.startswith(('memit_nas_','nas_transfer_')):
        assert (p/'nas_targets.jsonl').is_file(), 'Missing NAS target audit'
    if name.startswith('horen_'):
        assert (p/'routing_audit.json').is_file(), 'Missing HoReN routing audit'
    if name.startswith('olora_'):
        logs=json.loads((p/'training_log.json').read_text())
        assert [r['stage'] for r in logs]==list(range(1,9))
        assert all(r['current_count']==375 and r['frozen_backbone_verified'] and r['future_leak_count']==0 for r in logs)
        assert [r['stored_rank_per_module'] for r in logs]==list(range(16,129,16))
        assert (p/'adapters/olora.pt').is_file(), 'Missing retained O-LoRA adapters'
    trace_count=None
    if trace.exists():
        records=[json.loads(line) for line in trace.read_text().splitlines()]
        trace_count=len(records)
        expected_order=sorted(rows,key=lambda sid:(rows[sid]['stage'],sid))
        if records and 'sample_ids' in records[0]:
            assert [sid for r in records for sid in r['sample_ids']]==expected_order
            assert all(rows[sid]['stage']==r['stage'] for r in records for sid in r['sample_ids'])
        else:
            assert [r['sample_id'] for r in records]==expected_order
            assert [r['edit'] for r in records]==list(range(1,3001))
            assert all(r['stage']==rows[r['sample_id']]['stage'] for r in records)
        def finite_tree(value):
            if isinstance(value,float):return math.isfinite(value)
            if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
            if isinstance(value,list):return all(finite_tree(v) for v in value)
            return True
        assert all(r['finite'] and finite_tree(r) for r in records)
    if (p/'nas_targets.jsonl').exists():
        targets=[json.loads(line) for line in (p/'nas_targets.jsonl').read_text().splitlines()]
        assert [r['sample_id'] for r in targets]==expected_order
        assert all(math.isfinite(r['target_norm']) for r in targets)
    if (p/'routing_audit.json').exists():
        routes=json.loads((p/'routing_audit.json').read_text())
        assert [r['stage'] for r in routes]==list(range(1,9))
        assert all(r['future_leak_count']==0 and r['frozen_backbone_verified'] for r in routes)
    return {'status':'verified_complete_3k','stages':8,'final_counts':final_counts,'prediction_records':count,'scores_recomputed_from_predictions':True,'source_questions_and_gold_verified':True,'future_leaks':future,'trace_count':trace_count}

def main():
    result={}
    for _,name in NAMES:
        try:result[name]=audit_one(name)
        except Exception as e:result[name]={'status':'audit_failed','error':repr(e)}
    (BATCH/'completion_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v['status'] for k,v in result.items()})
    if any(v['status']=='audit_failed' for v in result.values()):raise SystemExit(1)
if __name__=='__main__':main()
