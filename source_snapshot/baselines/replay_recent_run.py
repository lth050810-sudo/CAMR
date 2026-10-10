#!/usr/bin/env python3
"""OPR-RU, SRT and full-parameter MIR under the shared Qwen3 WikiBigEdit protocol.

These are transparent task ports, not original-paper table reproductions.
OPR/SRT retain stage-level method semantics; MIR follows the online stream.
No held-out rephrase/persona/locality/multihop fields enter training state.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import random
import time
from pathlib import Path

from baselines.replay_recent_core import ReviewState, due_batch, balanced_top_rollouts


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['opr_ru','srt','srt_online','mir_full'],required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path)
    p.add_argument('--resume-checkpoint',type=Path)
    p.add_argument('--save-stage-checkpoints',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--save-final-checkpoint',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--model-dir',type=Path,default=Path('/root/engram/Qwen3-8B-Base'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--learning-rate',type=float,default=2e-6)
    p.add_argument('--weight-decay',type=float,default=0.)
    p.add_argument('--max-grad-norm',type=float,default=1.)
    p.add_argument('--steps-per-event',type=int,default=2)
    p.add_argument('--max-length',type=int,default=128)
    p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--eval-batch-size',type=int,default=4)
    p.add_argument('--activation-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--flash-attention',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--max-stages',type=int,default=0)
    p.add_argument('--examples-per-stage',type=int,default=0)
    p.add_argument('--stop-after-events',type=int,default=0)
    p.add_argument('--no-stage-eval',action='store_true')
    p.add_argument('--skip-base-eval',action='store_true')
    p.add_argument('--eval-axes',nargs='+',default=['update','rephrase','persona','locality','multihop'])
    p.add_argument('--locality-panel-size',type=int,default=512)
    p.add_argument('--multihop-panel-size',type=int,default=512)
    p.add_argument('--replay-k',type=int,default=1)
    p.add_argument('--opr-temperature',type=float,default=.1)
    p.add_argument('--srt-thresholds',type=float,nargs=5,default=[50.,100.,500.,1000.,5000.])
    p.add_argument('--srt-initial-stagger',type=int,default=4)
    p.add_argument('--mir-candidate-size',type=int,default=16)
    p.add_argument('--mir-memory-capacity',type=int,default=3000)
    a = p.parse_args(argv)
    if any(getattr(a,k)<1 for k in ['steps_per_event','max_length','max_new_tokens','eval_batch_size','replay_k','mir_candidate_size','mir_memory_capacity']):
        p.error('Step, batch, memory and token sizes must be positive')
    if sorted(set(a.srt_thresholds)) != a.srt_thresholds or a.srt_thresholds[0]<=0:
        p.error('SRT thresholds must be strictly increasing positive values')
    if a.checkpoint_dir is None:
        a.checkpoint_dir=Path('/mnt/yy2607004_hs/replay_recent_checkpoints')/a.output_dir.name
    return a


def source_metadata(method):
    return {
        'scope':'Qwen3-8B-Base WikiBigEdit common-protocol port; not paper-table reproduction',
        'opr_ru': {
            'paper':'https://arxiv.org/html/2605.29495v1',
            'official_code':'https://github.com/Yancey2024/OnPolicyReplay',
            'mechanism':'stage-boundary temperature=.1 top-p=1 generation on historical TRAIN prompts; exact-answer reward; per-stage top quotas; sampled answers are SFT targets',
            'changes':'Base completion prompt and answer-only loss; 64 rollout tokens; 100% replay budget; fixed online current-example order inside stage, paired with shuffled buffer; persistent AdamW; no first-stage replay',
        },
        'srt': {
            'paper':'https://arxiv.org/html/2608.17530v1',
            'official_code_verified':False,
            'mechanism':'old and new stage-visible due pools; pre-update per-example NLL from training forward; SM-2 Eqs7-10, ease floor1.3, initial2.5, intervals1/6/ceil(I*E)',
            'changes':'answer-only PPL instead of full-text PPL; old_fraction .5 vs paper .2; optimizer-step budget matched to 2*375/stage; current-stage pool available together; if all examples not due, advance scheduling clock; persistent AdamW',
            'threshold_uncertainty':'Paper explicitly gives endpoints50/5000 but referenced Table9 omits middle thresholds. Default100/500/1000 are declared, unvalidated transfer assumptions, never test-tuned.',
        },
        'srt_online': {
            'paper':'https://arxiv.org/html/2608.17530v1',
            'scope':'explicit online-arrival adaptation, NOT faithful paper SRT',
            'changes':'forces current incoming example and selects only due arrived history, falls back to earliest due to preserve replay budget; SM-2 state updated for every trained item',
        },
        'mir_full': {
            'paper':'https://arxiv.org/abs/1908.04742',
            'mechanism':'uniform reservoir candidates; answer-NLL increase after virtual full-parameter SGD; exact parameter restoration; real AdamW on incoming+top interference history',
            'changes':'answer-only language-model loss; candidate16, replay1, memory3000; CPU shard snapshots; same incoming stream and real-step budget as ours; extra virtual gradient is separately recorded',
        },
        'active_method':method,
        'comparison_limits':'Optimizer-step count does not imply token/FLOP equality; all selection/generation overhead and example exposure are reported. World size2 differs from ours6.',
    }


def run(args,driver):
    from baselines.common import (load_manifest,group_by_stage,manifest_panel,fixed_panel,
        atomic_json,append_jsonl,sha256_file,answer_scores)
    from baselines.engram_full_stream import select_rows
    from baselines.run_engram_full import training_row,evaluate
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new empty output directory')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw,metadata = load_manifest(args.manifest)
    rows = select_rows(raw,args.max_stages,args.examples_per_stage)
    train_rows = [training_row(r) for r in rows]
    stages = group_by_stage(train_rows)
    # This run intentionally targets frozen 3K new facts. Unreviewed versioned
    # raw-full streams need explicit retirement integration before admission.
    if any(r.get('supersedes_sample_ids') or r.get('supersedes') for r in rows):
        raise ValueError('Versioned replacement manifests are not supported by this 3K port')
    rng = random.Random(args.seed+7391)
    sampler_rng = random.Random(args.seed+3001)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    ignored={'output_dir','checkpoint_dir','resume_checkpoint','save_stage_checkpoints','save_final_checkpoint','stop_after_events'}
    signature={k:v for k,v in config.items() if k not in ignored}
    signature['manifest_sha256']=sha256_file(args.manifest)
    signature['sample_ids']=[r['sample_id'] for r in rows]
    config.update(method_provenance=source_metadata(args.method),backend=driver.backend.metadata,
        manifest_sha256=sha256_file(args.manifest),sample_ids=[r['sample_id'] for r in rows],
        num_facts=len(rows),planned_optimizer_steps=len(rows)*args.steps_per_event,
        closed_book_evaluation=True,full_parameter_update=True,trainable_fraction=1.,
        training_input_fields=['sample_id','stage','update','ans','subject','subject_id','relation','relation_id','object','object_id','tag','scope'],
        source_sha256={name:sha256_file(Path(__file__).parent/name) for name in
            ['replay_recent_run.py','replay_recent_backend.py','replay_recent_core.py','engram_full_backend.py','run_engram_full.py']})
    signature['source_sha256']=config['source_sha256']
    atomic_json(args.output_dir/'run_config.json',config)
    atomic_json(args.output_dir/'selected_training_stream.json',train_rows)
    locality = manifest_panel(metadata,'locality',limit=args.locality_panel_size)
    multihop = manifest_panel(metadata,'mhop',limit=args.multihop_panel_size)
    if locality is None: locality=fixed_panel(rows,'loc','loc_ans',args.locality_panel_size,args.seed+1001)
    if multihop is None: multihop=fixed_panel(rows,'mhop','mhop_ans',args.multihop_panel_size,args.seed+2001)
    metrics=[]
    action_records=[]
    prediction_records=[]
    rollout_records=[]
    exposures=Counter()
    totals=Counter()
    started=time.perf_counter()
    event=0

    def evaluate_at(checkpoint,stage):
        visible=rows if stage is None else [r for r in rows if r['stage']<=stage]
        result,records=evaluate(driver,visible,locality,multihop,args.eval_axes,checkpoint,stage)
        metrics.append(result)
        prediction_records.extend(records)
        atomic_json(args.output_dir/'metrics.json',metrics)
        append_jsonl(args.output_dir/'predictions.jsonl',records)

    def write_update(stage, selected, stats, detail):
        nonlocal event
        event+=1
        for r in selected: exposures[r['sample_id']]+=stats['optimizer_steps']
        for key in ['optimizer_steps','sequence_tokens','answer_tokens','physical_sequence_tokens','physical_answer_tokens','physical_padded_sequence_tokens']:
            totals[key]+=stats.get(key,0)
        record={'event':event,'stage':stage,'method':args.method,'sample_ids':[r['sample_id'] for r in selected],**detail,'train':stats}
        action_records.append(record)
        append_jsonl(args.output_dir/'actions.jsonl',[record])
        status={'status':'running','method':args.method,'stage':stage,'events':event,
            'totals':dict(totals),'unique_facts_trained':len(exposures),'wall_seconds':time.perf_counter()-started,
            'last_update':record,'forward_accounting':driver.backend.forward_accounting}
        atomic_json(args.output_dir/'status.json',status)
        if event%10==0 or event==1:
            print(json.dumps({'event':event,'stage':stage,'steps':totals['optimizer_steps'],'loss':stats['loss'],'peak_gib':stats['peak_vram_gb']}),flush=True)

    history=[]
    buffer=[]
    states={}
    by_id={r['sample_id']:r for r in train_rows}
    clock=0
    reservoir=[]
    seen=0
    halted=False
    completed_stage=0
    last_checkpoint=None
    prior_wall=0.
    if args.resume_checkpoint:
        saved=driver.load_checkpoint(args.resume_checkpoint)
        if saved['schema']!='recent-replay-stage/v1' or saved['signature']!=signature:
            raise ValueError('Checkpoint protocol/configuration mismatch')
        completed_stage=saved['completed_stage']
        if completed_stage not in stages:
            raise ValueError('Checkpoint completed stage is not in this stream')
        history=saved['history']
        buffer=saved['buffer']
        states={k:ReviewState(**v) for k,v in saved['states'].items()}
        reservoir=saved['reservoir']
        seen=saved['seen']
        event=saved['event']
        clock=saved['clock']
        rng.setstate(saved['rng'])
        sampler_rng.setstate(saved['sampler_rng'])
        totals.update(saved['totals'])
        exposures.update(saved['exposures'])
        metrics.extend(saved['metrics'])
        action_records.extend(saved['actions'])
        prediction_records.extend(saved['predictions'])
        rollout_records.extend(saved['rollouts'])
        prior_wall=saved['wall_seconds']
        last_checkpoint=str(args.resume_checkpoint)
        if len(action_records)!=event or driver.backend.total_optimizer_steps!=totals['optimizer_steps']:
            raise ValueError('Checkpoint model/controller step count mismatch')
        for name,records in [('actions.jsonl',action_records),('predictions.jsonl',prediction_records),('opr_rollouts.jsonl',rollout_records)]:
            if records: append_jsonl(args.output_dir/name,records)
        atomic_json(args.output_dir/'metrics.json',metrics)
        config['resumed_completed_stage']=completed_stage
        atomic_json(args.output_dir/'run_config.json',config)
    elif not args.skip_base_eval:
        evaluate_at('base',None)

    def checkpoint(stage):
        saved={'schema':'recent-replay-stage/v1','signature':signature,'completed_stage':stage,
            'history':history,'buffer':buffer,'states':{k:asdict(v) for k,v in states.items()},
            'reservoir':reservoir,'seen':seen,'event':event,'clock':clock,
            'rng':rng.getstate(),'sampler_rng':sampler_rng.getstate(),
            'totals':dict(totals),'exposures':dict(exposures),'metrics':metrics,
            'actions':action_records,'predictions':prediction_records,'rollouts':rollout_records,
            'wall_seconds':prior_wall+time.perf_counter()-started}
        path=args.checkpoint_dir/('tau'+str(stage))
        driver.save_checkpoint(path,saved)
        atomic_json(args.output_dir/'latest_checkpoint.json',{'path':str(path),'completed_stage':stage,
            'optimizer_steps':totals['optimizer_steps'],'resume_requires_same_world_size':True})
        return str(path)

    for stage, current in stages.items():
        if stage<=completed_stage: continue
        if args.method=='srt':
            old_ids=[r['sample_id'] for r in history]
            new_ids=[r['sample_id'] for r in current]
            for sample_id in new_ids:
                states[sample_id]=ReviewState(due=clock+rng.randrange(max(1,args.srt_initial_stagger)))
            for _ in range(len(current)*args.steps_per_event):
                ids,clock,num_old=due_batch(old_ids,new_ids,states,clock,1+args.replay_k,args.replay_k/(1+args.replay_k),rng)
                selected=[by_id[i] for i in ids]
                nlls,stats=driver.train_review(selected)
                quality=[states[i].review(nll,clock,args.srt_thresholds) for i,nll in zip(ids,nlls)]
                write_update(stage,selected,stats,{'clock':clock,'old_examples':num_old,'pre_update_nll':nlls,'quality':quality,'review_state':[asdict(states[i]) for i in ids]})
                clock+=1
                if args.stop_after_events and event>=args.stop_after_events: halted=True;break
        else:
            shuffled=list(buffer)
            rng.shuffle(shuffled)
            for index,row in enumerate(current):
                detail={}
                if args.method=='opr_ru':
                    chosen=shuffled[index*args.replay_k:(index+1)*args.replay_k]
                    detail={'replay_sample_ids':[r['sample_id'] for r in chosen],
                        'replay_targets':[r['ans'] for r in chosen],
                        'replay_rewards':[r['replay_reward'] for r in chosen]}
                elif args.method=='mir_full':
                    candidates=sampler_rng.sample(reservoir,min(len(reservoir),args.mir_candidate_size))
                    scores=driver.mir_scores([row],candidates,args.learning_rate) if candidates else []
                    ordering=sorted(range(len(scores)),key=lambda i:scores[i]['interference'],reverse=True)
                    chosen=[candidates[i] for i in ordering[:args.replay_k]]
                    detail={'replay_sample_ids':[r['sample_id'] for r in chosen],
                        'candidate_scores':[dict(sample_id=r['sample_id'],**s) for r,s in zip(candidates,scores)],
                        'virtual_gradient_steps':int(bool(candidates))}
                    totals['virtual_gradient_steps']+=int(bool(candidates))
                else:
                    due=[r for r in history if states[r['sample_id']].due<=clock]
                    if len(due)<min(args.replay_k,len(history)):
                        rest=sorted([r for r in history if r not in due],key=lambda r:states[r['sample_id']].due)
                        due+=rest[:args.replay_k-len(due)]
                    chosen=rng.sample(due,min(len(due),args.replay_k))
                    states[row['sample_id']]=ReviewState(due=clock)
                selected=[row]+chosen
                if args.method=='srt_online':
                    step_stats=[]
                    for _ in range(args.steps_per_event):
                        nlls,stats=driver.train_review(selected)
                        qualities=[states[r['sample_id']].review(nll,clock,args.srt_thresholds) for r,nll in zip(selected,nlls)]
                        step_stats.append(stats)
                        clock+=1
                    stats=dict(step_stats[-1])
                    for key in ['optimizer_steps','sequence_tokens','answer_tokens','physical_sequence_tokens','physical_answer_tokens','physical_padded_sequence_tokens','wall_seconds']:
                        stats[key]=sum(s[key] for s in step_stats)
                    detail={'replay_sample_ids':[r['sample_id'] for r in chosen],'last_pre_update_nll':nlls,'last_quality':qualities,'clock':clock}
                else:
                    stats=driver.train(selected,args.steps_per_event)
                write_update(stage,selected,stats,detail)
                if args.method in ('srt_online','mir_full'):
                    history.append(row)
                if args.method=='mir_full':
                    seen+=1
                    if len(reservoir)<args.mir_memory_capacity: reservoir.append(row)
                    else:
                        pos=sampler_rng.randrange(seen)
                        if pos<len(reservoir): reservoir[pos]=row
                if args.stop_after_events and event>=args.stop_after_events: halted=True;break
        if args.method in ('opr_ru','srt'): history.extend(current)
        if halted: break
        if not args.no_stage_eval:
            evaluate_at('tau'+str(stage),stage)
        atomic_json(args.output_dir/'exposures.json',dict(exposures))
        if args.method=='opr_ru' and stage != max(stages):
            rollouts=[]
            for start in range(0,len(history),args.eval_batch_size):
                batch=history[start:start+args.eval_batch_size]
                responses=driver.sample([r['update'] for r in batch],args.opr_temperature)
                for r,response in zip(batch,responses):
                    # Only training labels are used, never final-evaluation prompts.
                    score=float(answer_scores(response,r['ans'])['exact'])
                    rollouts.append({'row':r,'response':response,'score':score})
                new_rollouts=[
                    {'after_stage':stage,'sample_id':r['sample_id'],'train_prompt':r['update'],'train_gold':r['ans'],'response':response,'score':float(answer_scores(response,r['ans'])['exact'])}
                    for r,response in zip(batch,responses)]
                rollout_records.extend(new_rollouts)
                append_jsonl(args.output_dir/'opr_rollouts.jsonl',new_rollouts)
            next_stage=min(s for s in stages if s>stage)
            buffer=balanced_top_rollouts(rollouts,len(stages[next_stage])*args.replay_k,rng)
            atomic_json(args.output_dir/f'opr_buffer_after_tau{stage}.json',buffer)
        completed_stage=stage
        if args.save_stage_checkpoints or (stage==max(stages) and args.save_final_checkpoint):
            last_checkpoint=checkpoint(stage)
    if not halted and args.no_stage_eval: evaluate_at('final',max(stages))
    result={'status':'stopped_debug' if halted else 'completed','method':args.method,
        'events':event,'totals':dict(totals),'unique_facts_trained':len(exposures),'num_facts':len(rows),
        'coverage':len(exposures)/len(rows),'wall_seconds':prior_wall+time.perf_counter()-started,
        'forward_accounting':driver.backend.forward_accounting,'metrics':metrics,
        'checkpoint_saved':bool(last_checkpoint),'latest_checkpoint':last_checkpoint,
        'checkpoint_note':'Complete model/Adam/per-rank RNG and replay/controller state saved at stage boundaries on shared /mnt disk; resume requires same world size and new empty output directory.'}
    atomic_json(args.output_dir/'summary.json',result)
    atomic_json(args.output_dir/'exposures.json',dict(exposures))
    atomic_json(args.output_dir/'status.json',result)
    return result


def main():
    import torch.distributed as dist
    from baselines.replay_recent_backend import RecentReplayBackend,RecentCoordinator,serve_recent
    args=parse_args()
    backend=RecentReplayBackend(args)
    driver=RecentCoordinator(backend)
    if backend.rank==0:
        run(args,driver)
        driver.shutdown()
    else:
        serve_recent(backend)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__': main()
