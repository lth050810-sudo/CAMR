#!/usr/bin/env python3
"""Official MEMOIR transfer to Qwen3-8B; untuned layer/threshold transfer.

Uses the upstream edit loss, optimizer, sparse masks and retrieval rule. The
inference mask is cached per question so autoregressive KV decoding preserves
the prompt-derived mask. Background features come from independent Wikipedia.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/recent_extra/MEMOIR-main"
CORPUS = Path("/mnt/yy2607004_hs/engram_corpora/wikipedia20220301_verified_clone_candidates/validation_seed1_1000.jsonl")


def load_official():
    # Only optional logging uses wandb; no SDK or authentication is required.
    if importlib.util.find_spec("wandb") is None:
        optional_logger = types.ModuleType("wandb")
        optional_logger.__spec__ = importlib.util.spec_from_loader("wandb", loader=None)
        optional_logger.run = None
        sys.modules["wandb"] = optional_logger
    package = types.ModuleType("recent_extra_memoir_official")
    package.__path__ = [str(SOURCE / "easyeditor/models/memoir")]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + ".MEMOIR",
        SOURCE / "easyeditor/models/memoir/MEMOIR.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    code=(SOURCE / "easyeditor/models/memoir/MEMOIR.py").read_text()
    if code.count(".to('cuda')") != 1:
        raise RuntimeError("unexpected upstream device-transfer code")
    # Exact single portability fix: background vectors follow the edited layer.
    exec(compile(code.replace(".to('cuda')", ".to(layer.weight.device)"), str(spec.origin), "exec"), module.__dict__)
    return module


def cache_inference_class(official):
    import torch
    import torch.nn.functional as F
    original = official.MEMOIRAdapter
    upstream_forward = original.new_weight_forward
    class PromptCachedMEMOIRAdapter(original):
        def new_weight_forward(self, x):
            if self.training:
                return upstream_forward(self, x)
            if not hasattr(self, "_generation_mask"):
                boundaries = self.last_prompt_token_loc_inference
                padding_counts = self.prompt_padding_counts_inference
                if len(boundaries) != x.shape[0] or len(padding_counts) != x.shape[0]:
                    raise ValueError("MEMOIR evaluation boundary batch does not match hidden-state batch")
                masks=[]
                for index in range(x.shape[0]):
                    features = self.aggregate_prompt_features(
                        x[index:index+1],int(boundaries[index]),int(padding_counts[index]))
                    selected = self.hasher.get_active_indices(features[0]).to(x.device)
                    mask = torch.zeros(x.shape[-1],dtype=torch.bool,device=x.device)
                    mask[selected] = True
                    if len(self.masks_for_edited_samples):
                        stored = self.masks_for_edited_samples
                        overlaps = mask.float() @ stored.T.float() / self.config.top_k
                        best_values,best_indices = torch.topk(overlaps,k=1)
                        mask = stored[best_indices[0]] if float(best_values[0]) >= self.config.irr_threshold else torch.zeros_like(mask)
                    else:
                        mask.zero_()
                    masks.append(mask)
                self._generation_mask = torch.stack(masks)
            if self._generation_mask.shape[0] != x.shape[0]:
                raise ValueError("MEMOIR cached mask batch changed during generation")
            return F.linear(x * self._generation_mask.unsqueeze(1), self.new_weight)
    # Retain the original class identity: its constructor explicitly uses
    # super(MEMOIRAdapter, self), so rebinding that global would break __init__.
    original.new_weight_forward = PromptCachedMEMOIRAdapter.new_weight_forward
    return original


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=ROOT / "Qwen3-8B-Base")
    p.add_argument("--background-corpus", type=Path, default=CORPUS)
    p.add_argument("--layer", type=int, default=30)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--memory-fraction", type=float, default=.4)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--edit-lr", type=float, default=1.)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--irr-threshold", type=float, default=.4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--locality-panel-size", type=int, default=512)
    p.add_argument("--multihop-panel-size", type=int, default=512)
    p.add_argument("--max-edits", type=int, default=3000)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-gpus", default="",
        help="Comma-separated logical CUDA devices used to shard evaluation; the training GPU must be first")
    p.add_argument("--eval-worker-state", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--eval-worker-output", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--eval-worker-index", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--eval-worker-count", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--eval-worker-checkpoint", default="", help=argparse.SUPPRESS)
    p.add_argument("--eval-only-state", type=Path,
        help="Evaluate an existing MEMOIR checkpoint without changing training artifacts")
    p.add_argument("--eval-only-tag", default="eval", help=argparse.SUPPRESS)
    return p.parse_args()


def install_generator(common, torch, adapter):
    @torch.inference_mode()
    def generate_answers(model,tokenizer,questions,device,*,batch_size,max_length,max_new_tokens):
        answers=[]
        model.eval()
        for start in range(0,len(questions),batch_size):
            chunk=questions[start:start+batch_size]
            batch=tokenizer([common.qa_prompt(question) for question in chunk],return_tensors="pt",
                padding=True,truncation=True,max_length=max_length).to(device)
            width=batch["input_ids"].shape[1]
            adapter.last_prompt_token_loc_inference=torch.full((len(chunk),),width-1,device=device,dtype=torch.long)
            adapter.prompt_padding_counts_inference=(batch["attention_mask"]==0).sum(dim=1)
            if hasattr(adapter,"_generation_mask"):del adapter._generation_mask
            try:
                output=model.generate(**batch,do_sample=False,use_cache=True,max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                answers.extend(tokenizer.batch_decode(output[:,width:],skip_special_tokens=True))
            finally:
                del adapter.last_prompt_token_loc_inference
                del adapter.prompt_padding_counts_inference
                if hasattr(adapter,"_generation_mask"):del adapter._generation_mask
        return answers
    common.generate_answers=generate_answers


def load_eval_model(args, common, torch, state):
    torch.cuda.set_device(args.gpu)
    device=torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    tok=common.load_tokenizer(args.model_dir)
    tok.model_max_length=args.max_length
    tok.padding_side="left"
    tok.truncation_side="left"
    model=common.load_model(args.model_dir,device,use_flash_attention=False)
    model.eval()
    for param in model.parameters():param.requires_grad_(False)
    background=args.output_dir/"background_features.pt"
    if not background.exists():
        raise FileNotFoundError(f"missing MEMOIR background features: {background}")
    official=load_official()
    hparams=types.SimpleNamespace(model_name="Qwen3-8B-Base",inner_params=[f"model.layers[{args.layer}].mlp.down_proj.weight"],
        top_k=args.top_k,irr_threshold=args.irr_threshold,prompt_feature_agg="mean_decentered",
        dir_background_features=str(background),RUN_SAVE_BACKGROUND_FEATURES=False,
        edit_lr=args.edit_lr,n_iter=args.iterations,objective_optimization="only_label",use_chat_template=False)
    cache_inference_class(official)
    wrapper=official.MEMOIR(hparams,model,device)
    adapter=wrapper.get_adapter_layer()
    with torch.no_grad():adapter.new_weight.copy_(state["new_weight"].to(device))
    adapter.masks_for_edited_samples=state["masks"].to(device)
    adapter.hasher.permutation=state["permutation"]
    install_generator(common,torch,adapter)
    return model,tok,device,adapter


def evaluate_shard(args, common, model, tok, device, rows, metadata, cursor, checkpoint):
    records=[]
    locality=common.manifest_panel(metadata,"locality",limit=args.locality_panel_size)
    multihop=common.manifest_panel(metadata,"mhop",limit=args.multihop_panel_size)
    for axis in common.AXES:
        axis_rows=rows[:cursor]
        if axis=="locality" and locality is not None:axis_rows=locality
        if axis=="multihop" and multihop is not None:axis_rows=multihop
        axis_rows=axis_rows[args.eval_worker_index::args.eval_worker_count]
        _,axis_records=common.evaluate_axis(model,tok,axis_rows,axis,device,batch_size=args.eval_batch_size,
            max_length=args.max_length,max_new_tokens=args.max_new_tokens,checkpoint=checkpoint)
        records.extend(axis_records)
    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    temp.write_text("")
    import baselines.common as common
    common.append_jsonl(temp,records)
    temp.replace(path)


def eval_worker(args):
    import torch
    import baselines.common as common
    rows,metadata=common.load_manifest(args.manifest)
    rows=rows[:args.max_edits]
    state=torch.load(args.eval_worker_state,map_location="cpu",weights_only=False)
    cursor=int(state["cursor"])
    model,tok,device,_=load_eval_model(args,common,torch,state)
    records=evaluate_shard(args,common,model,tok,device,rows,metadata,cursor,args.eval_worker_checkpoint)
    write_jsonl(args.eval_worker_output,records)


def metrics_from_records(checkpoint, num_facts, records):
    result={"checkpoint":checkpoint,"num_facts":num_facts}
    for axis in ("update","rephrase","persona","locality","multihop"):
        selected=[record for record in records if record["axis"]==axis]
        count=len(selected);exact=sum(bool(record["exact"]) for record in selected)
        match=sum(bool(record["match"]) for record in selected)
        result[axis]={"count":count,"exact_count":exact,"match_count":match,
            "exact_accuracy":exact/count if count else None,"match_accuracy":match/count if count else None}
    return result


def parallel_evaluate(args, common, torch, model, tok, device, adapter, rows, metadata, cursor, stage,
                      artifact_tag=None, checkpoint_label=None):
    gpu_ids=[int(item) for item in args.eval_gpus.split(",") if item.strip()]
    if len(gpu_ids)<2:
        return common.evaluate_checkpoint(model,tok,rows[:cursor],device,
            checkpoint=f"tau{stage}",axes=common.AXES,batch_size=args.eval_batch_size,max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            locality_panel=common.manifest_panel(metadata,"locality",limit=args.locality_panel_size),
            multihop_panel=common.manifest_panel(metadata,"mhop",limit=args.multihop_panel_size))
    if gpu_ids[0]!=args.gpu or len(set(gpu_ids))!=len(gpu_ids):
        raise ValueError("--eval-gpus must contain unique logical devices with the training --gpu first")
    artifact_tag=artifact_tag or f"tau{stage}"
    checkpoint_label=checkpoint_label or f"tau{stage}"
    stage_dir=args.output_dir/"parallel_eval"/artifact_tag
    stage_dir.mkdir(parents=True,exist_ok=True)
    state_path=stage_dir/"state.pt"
    state={"cursor":cursor,"new_weight":adapter.new_weight.detach().cpu(),
        "masks":adapter.masks_for_edited_samples.cpu(),"permutation":adapter.hasher.permutation.cpu()}
    temp=state_path.with_suffix(f".{os.getpid()}.tmp");torch.save(state,temp);temp.replace(state_path)
    processes=[];logs=[]
    base=[sys.executable,"-u","-m","baselines.recent_extra_memoir",
        "--model-dir",str(args.model_dir),"--manifest",str(args.manifest),"--output-dir",str(args.output_dir),
        "--seed",str(args.seed),"--layer",str(args.layer),"--iterations",str(args.iterations),
        "--edit-lr",str(args.edit_lr),"--top-k",str(args.top_k),"--irr-threshold",str(args.irr_threshold),
        "--max-length",str(args.max_length),"--max-new-tokens",str(args.max_new_tokens),
        "--locality-panel-size",str(args.locality_panel_size),"--multihop-panel-size",str(args.multihop_panel_size),
        "--max-edits",str(args.max_edits),"--memory-fraction",str(args.memory_fraction),
        "--eval-batch-size",str(args.eval_batch_size),
        "--eval-worker-state",str(state_path),"--eval-worker-count",str(len(gpu_ids)),
        "--eval-worker-checkpoint",checkpoint_label]
    try:
        for shard,gpu in enumerate(gpu_ids[1:],start=1):
            output=stage_dir/f"shard{shard}.jsonl";log_path=stage_dir/f"shard{shard}.log"
            log=log_path.open("a");logs.append(log)
            command=base+["--gpu",str(gpu),"--eval-worker-index",str(shard),"--eval-worker-output",str(output)]
            processes.append((shard,output,subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)))
        args.eval_worker_index=0;args.eval_worker_count=len(gpu_ids)
        local_records=evaluate_shard(args,common,model,tok,device,rows,metadata,cursor,checkpoint_label)
        write_jsonl(stage_dir/"shard0.jsonl",local_records)
        for shard,output,process in processes:
            code=process.wait()
            if code or not output.exists():
                raise RuntimeError(f"MEMOIR evaluation shard {shard} failed with exit code {code}")
    finally:
        for _,_,process in processes:
            if process.poll() is None:process.terminate()
        for log in logs:log.close()
    records=[]
    for shard in range(len(gpu_ids)):
        with (stage_dir/f"shard{shard}.jsonl").open() as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    axis_order={axis:index for index,axis in enumerate(common.AXES)}
    record_order={}
    locality=common.manifest_panel(metadata,"locality",limit=args.locality_panel_size)
    multihop=common.manifest_panel(metadata,"mhop",limit=args.multihop_panel_size)
    for axis in common.AXES:
        axis_rows=rows[:cursor]
        if axis=="locality" and locality is not None:axis_rows=locality
        if axis=="multihop" and multihop is not None:axis_rows=multihop
        record_order.update({(axis,str(row["sample_id"])):index for index,row in enumerate(axis_rows)})
    records.sort(key=lambda record:(axis_order[record["axis"]],record_order[(record["axis"],str(record["sample_id"]))]))
    common.atomic_json(stage_dir/"completed.json",{"status":"complete","checkpoint":checkpoint_label,
        "workers":len(gpu_ids),"logical_gpus":gpu_ids,"records":len(records),"max_new_tokens":args.max_new_tokens})
    return metrics_from_records(checkpoint_label,cursor,records),records


def eval_only(args):
    import torch
    import baselines.common as common
    from baselines.run_lora_family import summarize_checkpoint_records
    rows,metadata=common.load_manifest(args.manifest)
    rows=rows[:args.max_edits]
    state=torch.load(args.eval_only_state,map_location="cpu",weights_only=False)
    cursor=int(state["cursor"])
    if cursor<1 or cursor>len(rows):raise ValueError(f"invalid checkpoint cursor {cursor}")
    stage=int(rows[cursor-1]["stage"])
    model,tok,device,adapter=load_eval_model(args,common,torch,state)
    tag=args.eval_only_tag
    label=f"tau{stage}_{tag}"
    raw,records=parallel_evaluate(args,common,torch,model,tok,device,adapter,
        rows,metadata,cursor,stage,artifact_tag=tag,checkpoint_label=label)
    result={**raw,**summarize_checkpoint_records(records,stage)}
    predictions=args.output_dir/f"predictions_{tag}.jsonl"
    write_jsonl(predictions,records)
    common.atomic_json(args.output_dir/f"metrics_{tag}.json",result)
    common.atomic_json(args.output_dir/f"completed_{tag}.json",{"status":"complete","cursor":cursor,
        "checkpoint":label,"max_new_tokens":args.max_new_tokens,"eval_batch_size":args.eval_batch_size,
        "workers":len([x for x in args.eval_gpus.split(',') if x.strip()]),"records":len(records)})


def main():
    args=arguments()
    if args.eval_worker_state:
        eval_worker(args);return
    if args.eval_only_state:
        eval_only(args);return
    import torch
    import baselines.common as common
    from baselines.run_lora_family import summarize_checkpoint_records
    rows, metadata=common.load_manifest(args.manifest)
    if not 1 <= args.max_edits <= len(rows):
        raise ValueError("invalid max-edits")
    rows=rows[:args.max_edits]
    cfg={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()
         if k not in {"gpu","resume","dry_run","eval_gpus","eval_batch_size","eval_worker_state","eval_worker_output",
                      "eval_worker_index","eval_worker_count","eval_worker_checkpoint","eval_only_state","eval_only_tag"}}
    cfg.update(method="MEMOIR-Qwen3-untuned-transfer", num_facts=len(rows),
        manifest_sha256=common.sha256_file(args.manifest),
        background_corpus_sha256=common.sha256_file(args.background_corpus),
        upstream_repository="https://github.com/qym7/MEMOIR",
        upstream_archive_sha256=common.sha256_file(SOURCE.parent / "memoir.zip"),
        source_sha256={p.name:common.sha256_file(p) for p in (SOURCE/"easyeditor/models/memoir").glob("*.py")},
        transfer_note="Official Llama3 layer27/32 transferred by depth to Qwen3 layer30/36; top_k4096 and threshold0.4 untuned. Residual-memory editor, not full-parameter finetuning.",
        training_fields=["update","ans"], evaluation_fields_never_used_for_training=True,
        engineering_changes=["Independent Wikipedia background activations from this Qwen checkpoint (100 documents)",
          "Cache exact prompt-selected mask across KV generation; batch1 unpadded evaluation",
          "Background feature device follows edited layer instead of hardcoded default CUDA",
          "Disable optional wandb logging when SDK absent; suppress per-token upstream console logging",
          "128 token tokenizer cap; checkpoint every50 edits, fixed shared evaluation panels"],
        budget_note=f"Up to {args.iterations} SGD updates/edit, 11 official generated-context variants; early stop as upstream; not compute matched to Policy")
    official=load_official()
    if args.dry_run:
        print(json.dumps(cfg,indent=2));return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    configpath=args.output_dir/"run_config.json"
    if configpath.exists():
        if not args.resume or json.loads(configpath.read_text()) != cfg:
            raise ValueError("existing output requires resume with identical config")
    elif any(args.output_dir.iterdir()):
        raise FileExistsError("nonempty output without configuration")
    else:
        common.atomic_json(configpath,cfg)
    if (args.output_dir/"completed.json").exists():return
    torch.cuda.set_device(args.gpu)
    device=torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    tok=common.load_tokenizer(args.model_dir)
    tok.model_max_length=args.max_length
    tok.padding_side="left"
    tok.truncation_side="left"
    model=common.load_model(args.model_dir,device,use_flash_attention=False)
    model.eval()
    for param in model.parameters():param.requires_grad_(False)
    layer=model.model.layers[args.layer].mlp.down_proj
    background=args.output_dir/"background_features.pt"
    if not background.exists():
        captured=[]
        def hook(module,inputs):captured.append(inputs[0].detach().float().mean(1).cpu())
        handle=layer.register_forward_pre_hook(hook)
        documents=[]
        with args.background_corpus.open() as stream:
            for line in stream:
                row=json.loads(line)
                text=row.get("text", "").strip()
                if not text:continue
                documents.append({"source_row_index":row.get("source_row_index"),"id":row.get("id")})
                batch=tok(text,return_tensors="pt",truncation=True,max_length=args.max_length).to(device)
                with torch.no_grad():model(**batch,use_cache=False)
                if len(documents)==100:break
        handle.remove()
        if len(documents)!=100:raise ValueError("need100 independent background documents")
        features=torch.cat(captured)
        torch.save(features,background)
        common.atomic_json(args.output_dir/"background_provenance.json", {
            "corpus_sha256":cfg["background_corpus_sha256"],"documents":documents,
            "model_dir":str(args.model_dir),"layer":args.layer,"feature_shape":list(features.shape)})
        del captured,features
    from recent_extra_memoir_official.utils import get_context_templates, tokenize
    templates_path=args.output_dir/"context_templates.json"
    if templates_path.exists():templates=json.loads(templates_path.read_text())
    else:
        templates=get_context_templates(model,tok,[[5,5],[10,5]],device)
        # Generated braces are literal context, never additional format placeholders.
        templates=[s[:-3].replace("{","{{").replace("}","}}")+" {}" if s.endswith(" {}") else s for s in templates]
        common.atomic_json(templates_path,templates)
    hparams=types.SimpleNamespace(model_name="Qwen3-8B-Base",inner_params=[f"model.layers[{args.layer}].mlp.down_proj.weight"],
        top_k=args.top_k,irr_threshold=args.irr_threshold,prompt_feature_agg="mean_decentered",
        dir_background_features=str(background),RUN_SAVE_BACKGROUND_FEATURES=False,
        edit_lr=args.edit_lr,n_iter=args.iterations,objective_optimization="only_label",use_chat_template=False)
    cache_inference_class(official)
    wrapper=official.MEMOIR(hparams,model,device)
    adapter=wrapper.get_adapter_layer()
    checkpoint=args.output_dir/"checkpoint.pt"
    cursor=0;logs=[];metrics=[]
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if state["config"]!=cfg:raise ValueError("checkpoint identity mismatch")
        with torch.no_grad():adapter.new_weight.copy_(state["new_weight"].to(device))
        adapter.masks_for_edited_samples=state["masks"].to(device)
        adapter.hasher.permutation=state["permutation"]
        cursor=state["cursor"];logs=state["logs"];metrics=state["metrics"]
        torch.set_rng_state(state["torch_rng"]);torch.cuda.set_rng_state(state["cuda_rng"],device)
    locality=common.manifest_panel(metadata,"locality",limit=args.locality_panel_size)
    multihop=common.manifest_panel(metadata,"mhop",limit=args.multihop_panel_size)
    install_generator(common,torch,adapter)
    started=time.time()
    for index in range(cursor,len(rows)):
        row=rows[index];before=time.time()
        if hasattr(adapter,"_generation_mask"):del adapter._generation_mask
        request={"prompt":common.qa_prompt(row["update"]),"target_new":row["ans"].strip(),
                 "loc_prompt":"An unrelated passage describes ordinary daily activities."}
        tokens,_,_=tokenize([request],tokenizer=tok,device=device,context_templates=templates,hparams=hparams)
        torch.cuda.reset_peak_memory_stats(device)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            wrapper.edit(config=hparams,tokens=tokens)
        if not torch.isfinite(adapter.new_weight).all():
            raise FloatingPointError(f"nonfinite memory weights after edit {index}")
        log={"index":index,"sample_id":row["sample_id"],"stage":row["stage"],
             "optimizer_steps":captured.getvalue().count("Iteration "),
             "sequence_tokens_per_iteration":int(tokens["attention_mask"].sum()),
             "wall_seconds":time.time()-before,"peak_vram_gib":torch.cuda.max_memory_allocated(device)/2**30}
        logs.append(log);cursor=index+1
        del tokens
        stage_end=cursor==len(rows) or rows[cursor]["stage"]!=row["stage"]
        if stage_end and not args.skip_eval:
            raw,records=parallel_evaluate(args,common,torch,model,tok,device,adapter,
                rows,metadata,cursor,row["stage"])
            metrics.append({**raw,**summarize_checkpoint_records(records,row["stage"])})
            path=args.output_dir/f"predictions_tau{row['stage']}.jsonl"
            temp=path.with_suffix(".tmp");temp.write_text("")
            common.append_jsonl(temp,records);temp.replace(path)
        if cursor%args.checkpoint_every==0 or stage_end:
            common.atomic_json(args.output_dir/"training_log.json",logs)
            common.atomic_json(args.output_dir/"metrics.json",metrics)
            state={"config":cfg,"cursor":cursor,"new_weight":adapter.new_weight.detach().cpu(),
                "masks":adapter.masks_for_edited_samples.cpu(),"permutation":adapter.hasher.permutation.cpu(),
                "logs":logs,"metrics":metrics,"torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state(device)}
            temp=checkpoint.with_suffix(f".{os.getpid()}.tmp");torch.save(state,temp);temp.replace(checkpoint)
        if cursor%10==0 or cursor==1 or stage_end:
            progress={"status":"training","processed":cursor,"total":len(rows),"stage":row["stage"],
                "optimizer_steps":sum(r["optimizer_steps"] for r in logs),"memory_entries":len(adapter.masks_for_edited_samples),
                "elapsed_seconds":time.time()-started,"last_edit":log}
            common.atomic_json(args.output_dir/"progress.json",progress);print(json.dumps(progress),flush=True)
    common.atomic_json(args.output_dir/"completed.json",{"status":"complete","method":cfg["method"],
        "processed":cursor,"num_facts":len(rows),"smoke_only":args.skip_eval or len(rows)!=3000,
        "optimizer_steps":sum(r["optimizer_steps"] for r in logs),"extra_parameters":adapter.new_weight.numel(),
        "memory_mask_bytes":adapter.masks_for_edited_samples.numel(),"evaluation_stages":len(metrics)})


if __name__=="__main__":main()
