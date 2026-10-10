"""Detached eight-GPU training, verified resume, and four-axis evaluation."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from suite_utils import HERE, CHECKPOINT_ROOT, atomic, read, verify_frozen, checkpoint_inventory

PYTHON = '/root/engram/.venv-qwen3/bin/python'


def main():
    lock = (HERE / 'supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_frozen()
    (HERE / 'logs').mkdir(exist_ok=True)
    state_path = HERE / 'status.json'
    state = read(state_path) if state_path.exists() else dict(status='starting', started_unix=time.time(), jobs=[])
    state.update(supervisor_pid=os.getpid(), gpus=list(range(8)))
    env = dict(os.environ, PYTHONPATH=str(HERE / 'source_snapshot') + ':' + str(HERE),
               OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', FLASH_ATTENTION_DETERMINISTIC='1',
               PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    atomic(state_path, state)
    # Hold the common GPU leases over the entire workflow, including restarts.
    handles = []
    for gpu in range(8):
        handle = open(f'/tmp/engram_baseline_gpu_{gpu}.lock', 'a+')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)
    usage = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                           check=True, capture_output=True, text=True)
    assert max(int(x.strip()) for x in usage.stdout.splitlines()) <= 2048, 'GPUs are occupied'

    def launch(name, script, extra):
        log = HERE / 'logs' / f'{name}.log'
        command = [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc-per-node=8', '--max-restarts=0', str(HERE / script), *map(str, extra)]
        entry = dict(name=name, status='running', command=command, log=str(log), started_unix=time.time())
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
            entry['pid'] = process.pid
            state['jobs'].append(entry)
            state.update(status=name, active_pid=process.pid)
            atomic(state_path, state)
            last_telemetry = 0
            while process.poll() is None:
                if time.monotonic() - last_telemetry >= 30:
                    telemetry = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
                    record = dict(unix_time=time.time(), phase=name, csv=telemetry.stdout.strip(), exit_code=telemetry.returncode)
                    with (HERE / 'gpu_telemetry.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
                    state['last_heartbeat_unix'] = time.time()
                    atomic(state_path, state)
                    last_telemetry = time.monotonic()
                time.sleep(5)
            rc = process.returncode
        entry.update(status='complete' if rc == 0 else 'failed', exit_code=rc, ended_unix=time.time())
        atomic(state_path, state)
        return rc

    def checked(name, script, extra):
        rc = launch(name, script, extra)
        if rc:
            raise RuntimeError(f'{name} failed with exit code {rc}; see logs/{name}.log')

    try:
        stress = HERE / 'stress'
        if not (stress / 'audit.json').exists():
            checked('stress', 'train.py', ['--stress', '--output', stress])
        assert read(stress / 'audit.json')['status'] == 'passed'
        warmup = HERE / 'training_000'
        if not (warmup / 'audit.json').exists():
            checked('training_000', 'train.py', ['--output', warmup, '--stop-after', '128'])
        checkpoint_inventory(CHECKPOINT_ROOT / 'event000128', 128)
        atomic(HERE / 'resume_validation.json', dict(status='checkpoint_saved', event_cursor=128,
            checkpoint=str(CHECKPOINT_ROOT / 'event000128')))
        final_output = None
        for attempt in range(1, 5):
            output = HERE / f'training_{attempt:03d}'
            if (output / 'audit.json').exists():
                complete = read(output / 'completed.json')
                if complete['status'] == 'complete' and complete['events'] == 150000:
                    final_output = output
                    break
                continue
            if output.exists() and any(output.iterdir()):
                continue
            candidates = []
            for path in CHECKPOINT_ROOT.glob('event*'):
                if '.tmp-' in path.name:
                    continue
                try:
                    inv = checkpoint_inventory(path)
                    if inv['optimizer_steps'] < 150000:
                        candidates.append((inv['optimizer_steps'], path))
                except (ValueError, OSError):
                    continue
            cursor, resume = max(candidates)
            state.update(training_output=str(output), resume_checkpoint=str(resume), resume_events=cursor)
            atomic(state_path, state)
            rc = launch(f'training_{attempt:03d}', 'train.py', ['--output', output, '--resume', resume])
            if rc == 0:
                assert read(output / 'audit.json')['status'] == 'passed'
                assert read(output / 'completed.json')['events'] == 150000
                final_output = output
                break
            # Numerical/configuration errors require diagnosis, not blind retries.
            tail = (HERE / 'logs' / f'training_{attempt:03d}.log').read_text()[-20000:]
            if any(s in tail for s in ['FloatingPointError', 'AssertionError', 'ValueError', 'CUDA out of memory']):
                raise RuntimeError('Training validation/numerical failure; stopped with checkpoint and logs preserved')
        if final_output is None:
            raise RuntimeError('Training attempts exhausted')
        checkpoint_inventory(CHECKPOINT_ROOT / 'event150000', 150000)
        atomic(HERE / 'resume_validation.json', dict(status='passed', initial_resume_event=128,
            training_output=str(final_output), final_events=150000,
            model_fingerprint_boundary_verified=True, optimizer_policy_rng_restored=True))
        state['training_output'] = str(final_output)
        atomic(state_path, state)
        for attempt in range(1, 4):
            if (HERE / 'results_150k.json').exists() and read(HERE / 'results_150k.json')['status'] == 'complete':
                break
            rc = launch(f'evaluation_{attempt:03d}', 'evaluate.py', ['--training-output', final_output])
            if rc == 0:
                break
        result = read(HERE / 'results_150k.json')
        assert result['status'] == 'complete' and set(result['metrics']) == {'update', 'rephrase', 'persona', 'locality'}
        state.update(status='complete', finished_unix=time.time(), results=str(HERE / 'RESULTS.md'),
                     checkpoint=str(CHECKPOINT_ROOT / 'event150000'))
        atomic(state_path, state)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), failed_unix=time.time())
        atomic(state_path, state)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
