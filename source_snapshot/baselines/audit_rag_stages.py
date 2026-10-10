"""Link independently completed staged RAG runs to their exact final-only artifacts."""
import argparse,json
from pathlib import Path
from baselines.common import atomic_json,sha256_file

def audit(main, staged):
    c=json.loads((main/'run_config.json').read_text());sc=json.loads((staged/'run_config.json').read_text())
    assert c['run_fingerprint']==sc['run_fingerprint'],'different RAG configuration'
    assert json.loads((staged/'completed.json').read_text())['status']=='complete'
    ms=json.loads((staged/'metrics.json').read_text());fm=json.loads((main/'metrics.json').read_text())[-1]
    assert [m['current_stage'] for m in ms]==list(range(1,9))
    assert ms[-1]==fm,'final metrics differ'
    main_tau8=main/'cache/evaluation/tau8/metrics.json'
    staged_tau8=staged/'cache/evaluation/tau8/metrics.json'
    assert main_tau8.read_bytes()==staged_tau8.read_bytes(),'tau8 metric bytes differ'
    assert all(m['retrieval']['future_leak_count']==0 for m in ms)
    final=(main/'predictions.jsonl').read_bytes()
    lines=(staged/'predictions.jsonl').read_bytes().splitlines(keepends=True)
    selected=[]
    for line in lines:
        r=json.loads(line)
        assert max(r['retrieved_fact_stages'],default=0)<=int(r['current_stage'])
        if r['checkpoint']=='tau8':selected.append(line)
    assert b''.join(selected)==final,'final predictions differ byte-for-byte'
    result={'schema_version':'rag-linked-staged-audit/v1','verified':True,'staged_run':str(staged.resolve()),
      'main_run_fingerprint':c['run_fingerprint'],'stages':list(range(1,9)),
      'index_counts':[m['retrieval']['index_count'] for m in ms],
      'future_leak_count':0,'tau8_metrics_identical':True,'tau8_metrics_byte_identical':True,'tau8_predictions_byte_identical':True,
      'files':{str(p.resolve()):sha256_file(p) for p in [main/'metrics.json',main/'predictions.jsonl',staged/'metrics.json',staged/'predictions.jsonl',staged/'completed.json',main_tau8,staged_tau8]}}
    atomic_json(main/'staged_audit.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-dir',type=Path,default=Path('runs/wikibigedit_3k_qwen3_unknown'));p.add_argument('--sweeps-dir',type=Path,default=Path('runs/sweeps'));a=p.parse_args()
    for main,staged in [('rag_dense_mpnet','rag_dense_mpnet_top3_staged'),('rag_dense_mpnet_top1','rag_dense_mpnet_top1_staged'),('rag_oracle','rag_oracle_staged')]:
        result=audit(a.runs_dir/main,a.sweeps_dir/staged);print(main,result['stages'],result['future_leak_count'])
if __name__=='__main__':main()
