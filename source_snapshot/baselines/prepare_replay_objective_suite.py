"""Prepare frozen replay-comparison commands; train only an explicitly selected job.

Preparation never imports torch, reserves GPUs, or starts a subprocess. The run
subcommand verifies all recorded inputs and acquires the existing GPU leases.
Historical queues, outputs and checkpoints are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

METHODS = ('random_full_k4', 'cumulative_lora', 'ours_full_k8')
SCHEMA = 'engram-replay-objective-suite/v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_jobs(a):
    """Use independent method branches, so random K4 cannot overwrite Ours K8."""
    root = a.source_root.resolve()
    suite = a.suite_dir.resolve()
    launcher = [str(a.python), '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8']
    common = ['--manifest', str(a.manifest.resolve()), '--model-dir', str(root / 'Qwen3-8B-Base'),
              '--seed', str(a.seed), '--eval-batch-size', '128', '--max-length', '128',
              '--max-new-tokens', '16', '--fsdp-sharding-strategy', 'full_shard',
              '--no-activation-checkpointing']
    jobs = {}
    for method in a.methods:
        name = f'{method}_tokenmean_seed{a.seed}' if method != 'ours_full_k8' else f'{method}_optimized_seed{a.seed}'
        output = suite / name
        checkpoint = a.checkpoint_root.resolve() / name
        paths = ['--output-dir', str(output), '--checkpoint-dir', str(checkpoint)]
        if method == 'cumulative_lora':
            args = ['-m', 'baselines.run_longstream_baselines'] + common + paths + [
                '--method', 'cumulative_lora', '--epochs', '1', '--global-batch-size', '128',
                '--max-padded-tokens', '4096', '--block-size', '2000', '--memory-capacity', '1000',
                '--target', str(a.target), '--control', str(suite / 'control.json'),
                '--learning-rate', '2e-4', '--training-loss', 'token_mean', '--training-views', '1']
            objective = 'ordinary mean cross entropy over all answer tokens in each batch'
        else:
            args = ['-m', 'baselines.run_engram_full'] + common + paths + [
                '--parameter-mode', 'full', '--steps-per-event', '1', '--learning-rate', '2e-6',
                '--longstream', '--longstream-control', str(suite / 'control.json'),
                '--log-every', '20', '--history-only-loss-policy', 'follow_training_loss']
            if method == 'random_full_k4':
                args += ['--mode', 'random_replay', '--replay-k', '4', '--add-replay-k', '4',
                         '--ignore-replay-k', '4', '--replay-selection', 'hybrid',
                         '--training-loss', 'token_mean', '--training-views', '1',
                         '--reward-mode', 'raw', '--probe-selection', 'uniform', '--probe-size', '4']
                objective = 'ordinary mean cross entropy over current plus four uniformly selected history facts'
            else:
                args += ['--mode', 'policy', '--replay-k', '8', '--add-replay-k', '8', '--ignore-replay-k', '8',
                         '--replay-selection', 'learned', '--merge-similar-k', '4',
                         '--uniform-replay-fraction', '0.5', '--recent-replay-k', '0',
                         '--training-loss', 'role_mean', '--current-loss-weight', '0.5',
                         '--training-views', '2', '--training-view-policy', 'accepted_only',
                         '--training-views-file', str(a.training_views_file.resolve()),
                         '--reward-mode', 'normalized_delayed', '--rollout-steps', '32',
                         '--probe-size', '8', '--probe-selection', 'stage_stratified']
                objective = 'role-balanced current/history CE with accepted training-only views'
        jobs[method] = {'name': name, 'command': launcher + args, 'output_dir': str(output),
                        'checkpoint_dir': str(checkpoint), 'objective': objective}
    return jobs


def prepare(a):
    a.source_root = a.source_root.resolve()
    a.suite_dir = a.suite_dir.resolve()
    if a.suite_dir.exists():
        raise FileExistsError(f'Choose a new suite directory: {a.suite_dir}')
    if not a.manifest.is_file():
        raise FileNotFoundError(a.manifest)
    if 'ours_full_k8' in a.methods and (a.training_views_file is None or not a.training_views_file.is_file()):
        raise ValueError('Ours requires the existing training-only accepted-view cache')
    jobs = build_jobs(a)
    for job in jobs.values():
        if Path(job['checkpoint_dir']).exists():
            raise FileExistsError(f"Choose new checkpoint destinations: {job['checkpoint_dir']}")
    sources = sorted((a.source_root / 'baselines').glob('*.py'))
    if not sources:
        raise ValueError('No baseline source files to fingerprint')
    source_hashes = {str(p.resolve()): sha256(p) for p in sources}
    inputs = {str(a.manifest.resolve()): sha256(a.manifest)}
    if 'ours_full_k8' in a.methods:
        inputs[str(a.training_views_file.resolve())] = sha256(a.training_views_file)
    plan = {'schema_version': SCHEMA, 'source_root': str(a.source_root),
            'python': str(a.python), 'gpus': a.gpus, 'seed': a.seed,
            'target_events': a.target, 'source_sha256': source_hashes,
            'input_sha256': inputs, 'jobs': jobs,
            'baseline_contract': 'token_mean, original view only; historical role_mean random results remain distinct',
            'probe_budget_note': 'Standard Random Replay retains four uniform history probes; Ours uses eight stage-stratified probes and delayed remeasurement. Probe costs differ and are recorded explicitly.',
            'cumulative_lora_note': 'The previous formal cumulative LoRA already used token_mean; this is an explicit reproduction, not a correction of its old scores.'}
    a.suite_dir.mkdir(parents=True)
    control = a.suite_dir / 'control.json'
    control.write_text(json.dumps({'target_events': a.target, 'cumulative_lora_block_size': 2000}, indent=2) + '\n', encoding='utf-8')
    plan['input_sha256'][str(control)] = sha256(control)
    path = a.suite_dir / 'experiment_plan.json'
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# Independent replay objective comparison', '',
             'Preparation only. No GPU job has been started. Each command below selects one method.', '',
             'Historical role-balanced Random Replay scores are retained separately. Cumulative LoRA already used ordinary token CE.', '',
             'Random Replay retains four uniform probes. Ours uses eight stage-stratified probes plus delayed remeasurement; report these additional forward costs.', '',
             'All source and input hashes are checked before a run. Source edits require a fresh plan.', '']
    for method in jobs:
        command = [str(a.python), '-m', 'baselines.prepare_replay_objective_suite', 'run', '--plan', str(path), '--method', method]
        lines += [f'## {method}', '', '```bash', shlex.join(command), '```', '']
    (a.suite_dir / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    return path


def verify_plan(path, method):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if plan.get('schema_version') != SCHEMA:
        raise ValueError('Unsupported experiment plan')
    if method not in plan['jobs']:
        raise ValueError('Requested method is absent from this plan')
    for fingerprints in ('source_sha256', 'input_sha256'):
        for file, expected in plan[fingerprints].items():
            if sha256(file) != expected:
                raise ValueError(f'Frozen file changed; prepare a new experiment: {file}')
    job = plan['jobs'][method]
    for key in ('output_dir', 'checkpoint_dir'):
        if Path(job[key]).exists():
            raise FileExistsError(f'Refusing to replace existing run artifacts: {job[key]}')
    return plan, job


def run_prepared(path, method):
    plan, job = verify_plan(path, method)
    command = [plan['python'], '-u', '-m', 'baselines.engram_gpu_lease', '--gpus', plan['gpus'],
               '--wait-seconds', '1', '--'] + job['command']
    log = Path(path).parent / f'{job["name"]}.log'
    env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    # Exclusive creation also prevents two concurrent launches of the same job.
    with log.open('x', encoding='utf-8') as handle:
        result = subprocess.run(command, cwd=plan['source_root'], env=env, stdout=handle, stderr=subprocess.STDOUT)
    return result.returncode


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('prepare', 'run'):
        argv.insert(0, 'prepare')
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='action', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--source-root', type=Path, default=Path('/root/engram'))
    prepare_parser.add_argument('--python', type=Path, default=Path('/root/engram/.venv-qwen3/bin/python'))
    prepare_parser.add_argument('--manifest', type=Path, required=True)
    prepare_parser.add_argument('--training-views-file', type=Path)
    prepare_parser.add_argument('--suite-dir', type=Path, required=True)
    prepare_parser.add_argument('--checkpoint-root', type=Path, required=True)
    prepare_parser.add_argument('--methods', choices=METHODS, nargs='+', default=list(METHODS))
    prepare_parser.add_argument('--target', type=int, choices=(50000, 100000), default=100000)
    prepare_parser.add_argument('--seed', type=int, default=42)
    prepare_parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--plan', type=Path, required=True)
    run_parser.add_argument('--method', choices=METHODS, required=True)
    a = p.parse_args(argv)
    if a.action == 'prepare':
        try:
            ids = [int(i) for i in a.gpus.split(',')]
        except ValueError:
            p.error('--gpus must contain eight comma-separated integer IDs')
        if len(ids) != 8 or len(set(ids)) != 8 or min(ids) < 0:
            p.error('These formal jobs require eight unique nonnegative GPU IDs')
        if len(set(a.methods)) != len(a.methods):
            p.error('Each method may occur only once')
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.action == 'prepare':
        print(prepare(a))
        return 0
    return run_prepared(a.plan, a.method)


if __name__ == '__main__':
    raise SystemExit(main())
