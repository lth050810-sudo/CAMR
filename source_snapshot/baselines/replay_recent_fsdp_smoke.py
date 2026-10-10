#!/usr/bin/env python3
"""Explicit random-tiny-Qwen FSDP/NCCL protocol test; never loads 8B weights.

Only run when a supervisor has authorized the selected CUDA_VISIBLE_DEVICES.
The CUDA allocator is capped at 1.5% per process. CUDA context and NCCL memory
are external to this allocator cap and are reported as a practical limitation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision,ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import Qwen3Config,Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from baselines import engram_full_backend as shared
from baselines.replay_recent_backend import RecentReplayBackend,RecentCoordinator,serve_recent


def tiny_builder(args,device):
    spec=importlib.util.spec_from_file_location('tiny_tokenizer_source',Path(__file__).parents[1]/'tests/test_engram_full_backend.py')
    test_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    config=Qwen3Config(vocab_size=32,hidden_size=24,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=3,num_key_value_heads=1,head_dim=8,max_position_embeddings=256,
        attention_dropout=0.,eos_token_id=2,pad_token_id=0,tie_word_embeddings=False,use_cache=False)
    config._attn_implementation='eager'
    # All ranks start from identical random values, just as production ranks
    # load the same frozen pretrained checkpoint. No model weights are read.
    prior_rng=torch.get_rng_state()
    torch.manual_seed(987)
    model=Qwen3ForCausalLM(config).float().to(device)
    torch.set_rng_state(prior_rng)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    count=sum(p.numel() for p in model.parameters())
    model=FSDP(model,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={Qwen3DecoderLayer}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.bfloat16,buffer_dtype=torch.bfloat16),
        device_id=device,use_orig_params=False,sync_module_states=False,limit_all_gathers=True)
    return model,test_module.TinyTokenizer(),count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    args=parser.parse_args()
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local_rank)
    torch.cuda.set_per_process_memory_fraction(.015,local_rank)
    torch.set_num_threads(1)
    runtime=SimpleNamespace(model_dir=shared.FIXED_MODEL_DIR,seed=42,learning_rate=2e-6,
        max_length=96,max_new_tokens=4,eval_batch_size=2,max_grad_norm=1.,
        distributed_timeout_minutes=2)
    with patch.object(shared,'_build_model',tiny_builder):
        backend=RecentReplayBackend(runtime)
    backend.metadata['smoke_test_only']=True
    backend.metadata['actual_model']='random tiny Qwen3, 2 layers, hidden24, vocab32; no pretrained weights'
    driver=RecentCoordinator(backend)
    if backend.world_size!=2:
        raise ValueError('This smoke requires exactly two ranks')
    if backend.rank==0:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError('Smoke output must be new/empty')
        args.output_dir.mkdir(parents=True,exist_ok=True)
        rows=[{'sample_id':'tiny0','stage':1,'update':'A?','ans':'one'},
              {'sample_id':'tiny1','stage':1,'update':'B?','ans':'two'}]
        first=driver.train(rows,1)
        assert first['parameter_changed'] and first['optimizer_steps']==1
        nll,review=driver.train_review(rows)
        assert len(nll)==2 and review['optimizer_steps']==1
        driver.save_checkpoint(args.checkpoint_dir,{'smoke_only':True,'cursor':2})
        # The first branch runs a real MIR virtual step followed by Adam.
        sample_expected=driver.sample(['A?','B?'],.1)
        scores_expected=driver.mir_scores([rows[0]],[rows[1]],.01)
        result_expected=driver.train([rows[0]],1)
        restored=driver.load_checkpoint(args.checkpoint_dir)
        assert restored=={'smoke_only':True,'cursor':2}
        sample_actual=driver.sample(['A?','B?'],.1)
        scores_actual=driver.mir_scores([rows[0]],[rows[1]],.01)
        result_actual=driver.train([rows[0]],1)
        assert sample_actual==sample_expected,'Per-rank sampling RNG resume mismatch'
        assert scores_actual==scores_expected,'MIR scores changed after checkpoint resume'
        assert result_actual['parameter_fingerprint_after']==result_expected['parameter_fingerprint_after'], 'FSDP model+Adam next-step fingerprint mismatch'
        assert backend.total_optimizer_steps==3,'MIR advanced real optimizer counter'
        driver.shutdown()
        report={'status':'passed','test_only':True,'model':backend.metadata['actual_model'],
            'world_size':backend.world_size,'parameter_count':backend.parameter_count,
            'cuda_allocator_fraction_cap':.015,
            'allocator_cap_excludes':'CUDA context and NCCL allocations',
            'rank0_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'rank0_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'checked':['real two-rank FSDP train','train-forward review loss','collective categorical sampling',
                'full virtual MIR','sharded model+Adam+RNG checkpoint save/load','identical next-update global fingerprint'],
            'scope_limit':'Tiny-model collective smoke does not validate 8B memory peak or learning quality.'}
        (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    else:
        serve_recent(backend)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__': main()
