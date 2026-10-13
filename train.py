"""Eight-GPU CAMR more_learned, preserving the validated training algorithm."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source_snapshot'))
sys.path.insert(0, str(HERE))
import torch
import torch.distributed as dist
from baselines import run_engram_full as runner
from baselines.common import qa_prompt
from baselines.fullstream_scale_review import install
from trace_backend import TraceBackend
from trace_protocol import training_encoding
from suite_utils import CHECKPOINT_ROOT, AXES, atomic, read, checkpoint_inventory, audit_actions


class WikiBackend(TraceBackend):
    @lru_cache(maxsize=180000)
    def encoded(self, question, answer):
        return training_encoding(self.tokenizer, dict(update=question, ans=answer), 'WikiBigEdit')

    def generate(self, questions):
        return super().generate([qa_prompt(q) for q in questions])


def arguments(output, manifest, resume=None, stop=0, stress=False):
    values = [
        '--manifest', str(manifest), '--output-dir', str(output), '--checkpoint-dir', str(CHECKPOINT_ROOT),
        '--parameter-mode', 'full', '--mode', 'policy', '--fact-mode', 'structured',
        '--fsdp-sharding-strategy', 'shard_grad_op', '--learning-rate', '2e-6',
        '--steps-per-event', '1', '--seed', '42', '--max-length', '512', '--max-new-tokens', '64',
        '--eval-batch-size', '512', '--log-every', '100', '--skip-base-eval', '--no-stage-eval',
        '--eval-axes', 'update', 'rephrase', 'persona', 'locality',
        '--locality-panel-size', '0', '--multihop-panel-size', '0',
        '--save-every-events', '0' if stress else '25000',
        '--no-save-final-checkpoint' if stress else '--save-final-checkpoint',
        '--no-activation-checkpointing', '--training-loss', 'role_mean', '--training-views', '1',
        '--training-view-policy', 'accepted_only', '--history-only-loss-policy', 'follow_training_loss',
        '--current-loss-weight', '0.2', '--replay-k', '8', '--add-replay-k', '8', '--ignore-replay-k', '8',
        '--ignore-behavior', 'original', '--replay-selection', 'learned', '--merge-similar-k', '6',
        '--uniform-replay-fraction', '0.25', '--recent-replay-k', '0', '--reward-mode', 'normalized',
        '--gamma', '0', '--cost-weight', '0', '--probe-size', '8', '--probe-selection', 'stage_stratified',
        '--policy-lr', '3e-4', '--policy-hidden-dim', '32', '--rollout-steps', '32', '--max-grad-norm', '1.0',
        '--stop-after-events', str(stop),
    ]
    if resume:
        values.extend(['--resume-checkpoint', str(resume)])
    args = runner.parse_args(values)
    args.trace_task = 'WikiBigEdit'
    args.trace_micro_tokens = 16384
    args.trial_version = 'CAMR-K8-150K/v1'
    args.method_version = 'CAMR V5 more_learned 2026-10-08'
    args.external_final_evaluation = True
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    opts = parser.parse_args()
    manifest = HERE / ('stress_manifest.json' if opts.stress else 'manifest.json')
    payload = read(manifest)
    rows = payload['samples']
    args = arguments(opts.output, manifest, opts.resume, opts.stop_after, opts.stress)
    if opts.validate_only:
        print(json.dumps({'status': 'valid', 'events': len(rows), 'args': vars(args)}, default=str), flush=True)
        return
    if opts.output.exists() and any(opts.output.iterdir()):
        raise FileExistsError(opts.output)
    runner.load_manifest = lambda _: (rows, {k: v for k, v in payload.items() if k != 'samples'})
    runner.select_rows = lambda source, *unused: list(source)
    # No locality/rephrase/persona data enters memory, probes, or rewards.
    runner.fixed_panel = lambda *unused, **kwargs: []
    install(runner)
    torch.set_num_threads(4)
    backend = WikiBackend(args)
    assert backend.world_size == 8
    assert backend.metadata['trainable_parameters'] == backend.metadata['parameter_count'] == 8190735360
    backend.metadata.update(experiment='WikiBigEdit CAMR K8 150K',
        lm_template='Question: ...\nAnswer:', generation='greedy KV-cache, distributed examples',
        context_truncation=False, checkpoint_retention='Keep all experiment checkpoints')
    if backend.rank:
        runner.serve_backend(backend)
    else:
        driver = runner.BackendCoordinator(backend)
        start = time.monotonic()
        resume_cursor = int(opts.resume.name.removeprefix('event')) if opts.resume else 0
        progress_last = [start, resume_cursor]
        resume_expected = [None]
        resume_verified = [not opts.resume]
        original_append = runner.append_jsonl
        original_load = driver.load_checkpoint

        def load(path):
            state = original_load(path)
            previous = next((record.get('train') for record in reversed(state['actions'])
                             if record.get('train') is not None), None)
            if previous is None:
                raise ValueError('Resume checkpoint has no completed optimizer step')
            resume_expected[0] = previous['parameter_fingerprint_after']
            return state

        driver.load_checkpoint = load

        def append(path, records):
            original_append(path, records)
            if path.name != 'actions.jsonl' or not records:
                return
            last = records[-1]
            n = last['event'] + 1
            train = last.get('train')
            if opts.resume and n > resume_cursor and not resume_verified[0] and train is not None:
                assert train['parameter_fingerprint_before'] == resume_expected[0], 'Resume model fingerprint differs'
                resume_verified[0] = True
                atomic(HERE / 'resume_validation.json', dict(status='passed', resume_event=resume_cursor,
                    first_resumed_training_event=n, training_output=str(opts.output),
                    model_fingerprint_boundary_verified=True, optimizer_policy_rng_restored=True))
            if n <= resume_cursor or (n != 1 and n % 100):
                return
            now = time.monotonic()
            seconds = now - start
            done = n - resume_cursor
            rate = done / seconds if seconds > 0 else None
            train = train or {}
            atomic(opts.output / 'progress.json', dict(status='training', events=n, total_events=len(rows),
                resume_cursor=resume_cursor, elapsed_seconds=seconds, events_per_second=rate,
                remaining_training_seconds=(len(rows) - n) / rate if rate else None,
                recent_events_per_second=(n - progress_last[1]) / max(now - progress_last[0], 1e-6),
                peak_vram_gib=train.get('peak_vram_gb'), replay_count=last['replay_count'], world_size=8))
            progress_last[:] = [now, n]

        runner.append_jsonl = append
        original_save = driver.save_checkpoint

        def save(path, state):
            atomic(opts.output / 'progress.json', dict(status='saving_checkpoint', events=state['event_cursor'],
                total_events=len(rows), checkpoint=str(path), world_size=8))
            saved = original_save(path, state)
            inventory = checkpoint_inventory(path, state['event_cursor'], state['totals']['optimizer_steps'])
            atomic(opts.output / 'checkpoint_inventory' / (Path(path).name + '.json'), inventory)
            return saved

        driver.save_checkpoint = save
        # Exact load checks remain in the backend; the first resumed action
        # verifies the complete parameter fingerprint at the resume boundary.
        try:
            if opts.stress:
                stress_prompts = []
                for axis, (q, a) in AXES.items():
                    valid = [r[q] for r in rows if isinstance(r.get(q), str) and r[q].strip()
                             and isinstance(r.get(a), str) and r[a].strip()]
                    stress_prompts.extend(valid)
                # Exercise the final-evaluation global batch size before training.
                longest = sorted(stress_prompts, key=lambda q: len(backend.tokenizer(qa_prompt(q)).input_ids), reverse=True)
                batch = [longest[i % len(longest)] for i in range(512)]
                generated = driver.generate(batch)
                assert len(generated) == 512
                atomic(HERE / 'generation_stress.json', dict(status='passed', examples=512,
                    global_batch_size=512, all_four_axes_present=True, world_size=8))
            result = runner.run(args, driver)
            expected = opts.stop_after or len(rows)
            assert result['events'] == expected
            trace = audit_actions(opts.output / 'actions.jsonl', expected, rows)
            assert trace['optimizer_steps'] == result['optimizer_steps']
            assert trace['current_fact_presentations'] == result['totals']['current_fact_presentations']
            assert trace['ignore_events'] == result['action_counts'].get('ignore', 0)
            if opts.resume and not resume_verified[0]:
                raise RuntimeError('Resume completed without a post-resume optimizer step to verify')
            inventory = checkpoint_inventory(result['final_checkpoint'], expected, result['optimizer_steps']) if not opts.stress else None
            atomic(opts.output / 'audit.json', dict(status='passed', **trace, checkpoint=inventory,
                resumed_from=str(opts.resume) if opts.resume else None, all_parameters_trainable=True,
                trained_parameters=8190735360, world_size=8, method_version=args.method_version))
            atomic(opts.output / 'progress.json', dict(status='trained' if result['status'] == 'complete' else 'checkpointed_for_resume',
                events=expected, total_events=len(rows), checkpoint=result['final_checkpoint'],
                elapsed_seconds=time.monotonic() - start, world_size=8))
        finally:
            driver.shutdown()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
