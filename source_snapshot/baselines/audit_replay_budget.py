"""Verify the original ER random stream and matched selectors' temporal budgets."""
import json
from pathlib import Path
from baselines.common import Reservoir,load_manifest,group_by_stage,atomic_json
from baselines.run_lora_family import select_relation_replay

ROOT=Path(__file__).resolve().parents[1]

def main():
    rows,_=load_manifest(ROOT/'runs/protocols/wikibigedit_3k_seed42_qwen3_unknown.json')
    stages=group_by_stage(rows);index={r['sample_id']:i for i,r in enumerate(rows)}
    runs=ROOT/'runs/wikibigedit_3k_qwen3_unknown';out={}
    for name in ['er_reservoir','relation_oracle_replay','semantic_replay_er_admission','entity_masked_semantic_replay','tic_style_replay','cumulative_replay']:
        path=runs/name/'training_log.json'
        if not path.exists():continue
        logs=json.loads(path.read_text());by_stage={l['stage']:l for l in logs}
        reservoir=Reservoir(1000,3043);seen=[];audit=[]
        for stage,current in stages.items():
            ci=[index[r['sample_id']] for r in current]
            expected_er=reservoir.sample(375)
            if stage in by_stage:
                log=by_stage[stage];picked=[index[sid] for sid in log['replay_sample_ids']]
                assert len(picked)==len(set(picked))
                assert all(rows[i]['stage']<stage for i in picked)
                if name=='cumulative_replay':assert set(picked)==set(seen)
                else:
                    assert len(picked)==min(375,len(reservoir.items))
                    assert set(picked).issubset(reservoir.items)
                    if name in {'er_reservoir','tic_style_replay'}:assert picked==expected_er
                    if name=='relation_oracle_replay':
                        expected,matched,pairs=select_relation_replay(rows,ci,reservoir.items,375,4043+stage,return_pairs=True)
                        assert picked==expected and log['relation_matched_count']==matched
                        assert log['relation_query_replay_pairs']==pairs
                        for pair in log['relation_query_replay_pairs']:
                            current_row=rows[index[pair['current_sample_id']]]
                            old_row=rows[index[pair['replay_sample_id']]]
                            assert current_row['stage']==stage and old_row['stage']<stage
                            assert current_row['relation_id']==old_row['relation_id']==pair['relation_id']
                            assert (current_row.get('subject_id') or current_row['subject']) != (old_row.get('subject_id') or old_row['subject'])
                            assert current_row['fact_fingerprint']!=old_row['fact_fingerprint']
                audit.append({'stage':stage,'current_count':len(ci),'replay_count':len(picked),'unique_replay_count':len(set(picked)),
                  'future_leak_count':0,'matches_expected_admission':name!='cumulative_replay',
                  'structural_match_count':log.get('relation_matched_count'),
                  'sequence_tokens':log['sequence_tokens'],'answer_tokens':log['answer_tokens'],'optimizer_steps':log['optimizer_steps']})
            reservoir.add_many(ci);seen.extend(ci)
        out[name]={'stages':audit,'total_sequence_tokens':sum(a['sequence_tokens'] for a in audit),'total_answer_tokens':sum(a['answer_tokens'] for a in audit),'total_optimizer_steps':sum(a['optimizer_steps'] for a in audit)}
    atomic_json(ROOT/'runs/baseline_expansion_20260920/replay_budget_audit.json',out)
    for n,v in out.items():print(n,len(v['stages']),v['total_sequence_tokens'],v['total_optimizer_steps'])
if __name__=='__main__':main()
