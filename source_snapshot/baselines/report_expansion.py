"""Report whole-run completion separately from integration checks and blocked methods."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'runs/baseline_expansion_20260920'
RUNS=ROOT/'runs/wikibigedit_3k_qwen3_unknown'
EXPECTED='wikibigedit-3k-71674f2da6d0ba8a'
AXES=('update','rephrase','persona','locality','multihop')
NAMES=[
 ('MPNet Top-2 (shared prompt)','rag_dense_mpnet_top2'),
 ('MPNet Top-2 (paper recipe Qwen3 transfer)','rag_mpnet_top2_paper_recipe'),
 ('BM25 Top-2','rag_bm25_top2'),('Hybrid BM25 + MPNet RRF Top-2','rag_hybrid_rrf_top2'),
 ('Relation-Structure Oracle Replay + LoRA','relation_oracle_replay'),
 ('Cumulative Replay + LoRA','cumulative_replay'),
 ('ROME Qwen2.5→Qwen3 transfer','rome_qwen25_transfer_layer5'),
 ('R-ROME Qwen2.5→Qwen3 transfer','r_rome_qwen25_transfer_layer5'),
 ('Entity-masked Semantic Replay + LoRA','entity_masked_semantic_replay'),
 ('Semantic Replay, same ER admission control','semantic_replay_er_admission'),
 ('TiC-style Replay + LoRA','tic_style_replay'),
 ('MEMIT + NAS matched transfer','memit_nas_layers4_8_batch100_3k'),
 ('NAS Qwen2.5→Qwen3 transfer','nas_transfer_layers7_11_batch100_3k'),
 ('HoReN Qwen2.5→Qwen3 side memory','horen_qwen25_transfer_layer24'),
 ('O-LoRA shared all-linear r16 transfer','olora_shared_r16_alllinear')]

def read(p,default=None):
    return json.loads(p.read_text()) if p.exists() else default

def report():
    jobs=read(BATCH/'jobs.json',[])+read(BATCH/'queued_jobs.json',[])+read(BATCH/'additional_editor_jobs.json',[])
    jobs={j['name']:j for j in jobs}
    runs=[]
    for name,key in NAMES:
        p=RUNS/key;c=read(p/'run_config.json',{});done=read(p/'completed.json',{});ms=read(p/'metrics.json',[])
        last=ms[-1] if ms else {};trace=read(p/'progress.json',{})
        complete=(done.get('status')=='complete' and c.get('manifest_protocol_id')==EXPECTED and c.get('num_facts')==3000 and last.get('num_facts')==3000 and last.get('current_stage')==8 and all(last.get(a,{}).get('count',0)>0 for a in AXES))
        status='complete 3K' if complete else 'queued' if not c else 'running'
        j=jobs.get(key,{})
        if not c and j.get('status') in {'waiting_for_100k_moments','building_nullspace_projections','dev_integration'}:
            status=j['status']
        if not complete and j.get('pid'):
            stat=Path(f'/proc/{j["pid"]}/stat')
            if not stat.exists() or stat.read_text().split()[2]=='Z':status='failed/incomplete'
        stages=[m.get('current_stage') for m in ms]
        score={a:last.get(a,{}).get('match_accuracy') for a in AXES} if complete else {}
        runs.append(dict(name=name,run=key,status=status,evaluated_stages=stages,edited=trace.get('edits'),metrics=score))
    plan=read(BATCH/'execution_plan.json',{})
    payload={'generated_at':datetime.now(timezone.utc).isoformat(),'protocol_id':EXPECTED,'runs':runs,'dependency_blocks':plan.get('dependency_blocks',{}),'next_phase':plan.get('next_phase',{})}
    (BATCH/'coverage.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    lines=['# WikiBigEdit + Qwen3 补跑覆盖状态','',f"更新时间：{payload['generated_at']}",'',
      '本表只把完成 3,000 facts、τ8 和五轴评测的产物计为 complete 3K。接入检查不计入完成数。当前范围为 8×375 的冻结 3K 协议。按用户最新安排，完成本轮后先审阅模型/方法阵容，再启动全量实验；全量训练尚未启动。','',
      '| 方法 | 状态 | 已评测阶段 | 顺序编辑进度 | U / R / P / L / MH |','| --- | --- | --- | --- | --- |']
    for r in runs:
        scores=' / '.join(f"{100*r['metrics'][a]:.2f}" for a in AXES) if r['metrics'] else '—'
        edits=f"{r['edited']}/3000" if r['edited'] is not None else '—'
        lines.append(f"| {r['name']} | {r['status']} | {','.join(map(str,r['evaluated_stages'])) or '—'} | {edits} | {scores} |")
    lines+=['','完整结果与 locality 双口径见 `../wikibigedit_3k_qwen3_unknown/summary.md`。Relation Oracle 是结构参照，Cumulative 使用更多历史数据和算力，TiC-style 不是完整 TiC-LM 复现。','',
       '## 尚未完成的方法及实际依赖','', '| 方法 | 依赖 / 原因 |','| --- | --- |']
    lines += [f'| {name} | {reason} |' for name,reason in payload['dependency_blocks'].items()]
    lines+=['','独立 dev 已冻结：`../protocols/wikibigedit_dev1600_seed20260920_qwen3_unknown.json`，8×200。与 3K test 的 sample ID、subject、fact fingerprint 和五轴问题文本重叠均为零。','',
      '原 Dense top-1/top-3 和 Oracle 的独立 staged audit 已通过每条检索的时间检查及 τ8 字节一致性校验，并通过哈希绑定接入总表。','',
      '全量数据核验（仅准备）：`../full_dataset_preparation/README.md`。502,382 条原始记录来自完整八个时间区间，不能将 3K 结果称为全量结果。','']
    (BATCH/'coverage.md').write_text('\n'.join(lines))
    print('complete',sum(r['status']=='complete 3K' for r in runs),'/',len(runs))
if __name__=='__main__':report()
