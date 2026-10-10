"""CPU integration checks for scheduling, input boundaries and virtual updates."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from baselines import engram_full_backend as shared
from baselines.replay_recent_backend import RecentReplayBackend
from baselines.replay_recent_core import test_core
from baselines.replay_recent_run import parse_args,run
from baselines.replay_recent_validate import validate


@pytest.fixture
def tiny_backend():
    spec=importlib.util.spec_from_file_location('shared_backend_tests',Path(__file__).parents[1]/'tests/test_engram_full_backend.py')
    shared_tests=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared_tests)
    def builder(args,device):
        config=Qwen3Config(vocab_size=32,hidden_size=24,intermediate_size=48,num_hidden_layers=1,
            num_attention_heads=3,num_key_value_heads=1,head_dim=8,max_position_embeddings=256,
            attention_dropout=0.,eos_token_id=2,pad_token_id=0,tie_word_embeddings=False,use_cache=False)
        config._attn_implementation='eager'
        model=Qwen3ForCausalLM(config).float()
        return model,shared_tests.TinyTokenizer(),sum(p.numel() for p in model.parameters())
    torch.set_num_threads(1)
    args=SimpleNamespace(model_dir=shared.FIXED_MODEL_DIR,seed=42,learning_rate=2e-6,max_length=96,max_new_tokens=4,eval_batch_size=2)
    with patch.object(shared,'_runtime',lambda a:(0,1,0,torch.device('cpu'))),patch.object(shared,'_build_model',builder):
        yield RecentReplayBackend(args)


def test_review_loss_is_before_update_from_real_forward(tiny_backend):
    rows=[{'update':'Q1?','ans':'one'},{'update':'Q2?','ans':'two'}]
    before,_=tiny_backend.answer_nll(rows)
    actual,stats=tiny_backend.train_review(rows)
    after,_=tiny_backend.answer_nll(rows)
    assert actual==pytest.approx(before,abs=1e-6)
    assert max(abs(a-b) for a,b in zip(actual,after))>1e-6
    assert stats['optimizer_steps']==tiny_backend.total_optimizer_steps==1


def test_mir_restores_every_parameter_and_optimizer(tiny_backend):
    incoming=[{'update':'A?','ans':'one'}]
    candidates=[{'update':'B?','ans':'two'},{'update':'A?','ans':'three'}]
    tiny_backend.train(incoming,1)
    state=copy.deepcopy(tiny_backend.model.state_dict())
    optimizer=copy.deepcopy(tiny_backend.optimizer.state_dict())
    scores=tiny_backend.mir_scores(incoming,candidates,.01)
    assert len(scores)==2 and tiny_backend.total_optimizer_steps==1
    assert any(abs(s['interference'])>1e-6 for s in scores)
    assert all(torch.equal(state[k],v) for k,v in tiny_backend.model.state_dict().items())
    restored=tiny_backend.optimizer.state_dict()
    assert optimizer['param_groups']==restored['param_groups']
    for p,st in optimizer['state'].items():
        for k,v in st.items(): assert torch.equal(v,restored['state'][p][k])


def test_sampling_reproducible_count_and_padding(tiny_backend):
    torch.manual_seed(123)
    first=tiny_backend.sample(['A?','B?'],.1)
    torch.manual_seed(123)
    assert tiny_backend.sample(['A?','B?'],.1)==first
    assert len(first)==2 and tiny_backend.tokenizer.padding_side=='right'
    assert tiny_backend.forward_accounting['generation_calls']==2


class FakeDriver:
    checkpoints={}
    def __init__(self):
        self.backend=SimpleNamespace(metadata={'trainable_parameters':8190735360,'world_size':2},forward_accounting={},total_optimizer_steps=0)
        self.training=[]
        self.samples=[]
    def train(self,rows,steps):
        assert all(not set(r)&{'rephrase','personas','loc','loc_ans','mhop','mhop_ans'} for r in rows)
        self.training.extend(copy.deepcopy(rows))
        self.backend.total_optimizer_steps+=steps
        return {'loss':1.,'optimizer_steps':steps,'sequence_tokens':len(rows)*steps*5,
            'answer_tokens':len(rows)*steps,'physical_sequence_tokens':len(rows)*steps*10,
            'physical_answer_tokens':len(rows)*steps*2,'physical_padded_sequence_tokens':len(rows)*steps*10,
            'wall_seconds':.001,'peak_vram_gb':1.}
    def train_review(self,rows): return [0.]*len(rows),self.train(rows,1)
    def generate(self,questions): return ['fake']*len(questions)
    def sample(self,questions,temperature):
        self.samples.extend(questions)
        return ['SAMPLED-ANSWER']*len(questions)
    def mir_scores(self,incoming,candidates,virtual_lr):
        assert all(c['sample_id']!=incoming[0]['sample_id'] for c in candidates)
        return [{'before':1.,'after':1.+i,'interference':float(i)} for i,_ in enumerate(candidates)]
    def save_checkpoint(self,path,state):
        path.mkdir(parents=True)
        self.checkpoints[str(path)]=(copy.deepcopy(state),self.backend.total_optimizer_steps)
        return str(path)
    def load_checkpoint(self,path):
        state,steps=self.checkpoints[str(path)]
        self.backend.total_optimizer_steps=steps
        return copy.deepcopy(state)


@pytest.mark.parametrize('method',['opr_ru','srt','srt_online','mir_full'])
def test_real_manifest_protocol_input_boundary_and_budget(method,tmp_path):
    args=parse_args(['--method',method,'--manifest','runs/protocols/wikibigedit_3k_seed42_qwen3_unknown.json',
        '--output-dir',str(tmp_path/method),'--max-stages','2','--examples-per-stage','3',
        '--checkpoint-dir',str(tmp_path/'checkpoints'),
        '--skip-base-eval','--no-stage-eval'])
    driver=FakeDriver()
    result=run(args,driver)
    validate(args.output_dir)
    assert result['status']=='completed' and result['totals']['optimizer_steps']==12
    assert all(row['stage']<=2 for row in driver.training)
    config=json.loads((args.output_dir/'run_config.json').read_text())
    assert len(config['sample_ids'])==6
    assert config['full_parameter_update'] and config['closed_book_evaluation']
    actions=[json.loads(line) for line in (args.output_dir/'actions.jsonl').read_text().splitlines()]
    if method=='opr_ru':
        assert len(driver.samples)==3
        assert sum(row['ans']=='SAMPLED-ANSWER' for row in driver.training)==3
        assert all('replay_sample_ids' in a for a in actions)
    if method=='mir_full':
        seen=set()
        for action in actions:
            assert set(action['replay_sample_ids'])<=seen
            seen.add(action['sample_ids'][0])
        assert result['totals']['virtual_gradient_steps']==5
    resumed_args=parse_args(['--method',method,'--manifest',str(args.manifest),
        '--output-dir',str(tmp_path/(method+'_resumed')),'--max-stages','2','--examples-per-stage','3',
        '--checkpoint-dir',str(tmp_path/'resumed_checkpoints'),'--resume-checkpoint',str(tmp_path/'checkpoints/tau1'),
        '--skip-base-eval','--no-stage-eval'])
    resumed=run(resumed_args,FakeDriver())
    validate(resumed_args.output_dir)
    assert resumed['totals']==result['totals']
    assert resumed['unique_facts_trained']==result['unique_facts_trained']
    expected=(args.output_dir/'actions.jsonl').read_text()
    assert (resumed_args.output_dir/'actions.jsonl').read_text()==expected
    assert json.loads((resumed_args.output_dir/'exposures.json').read_text())==json.loads((args.output_dir/'exposures.json').read_text())
