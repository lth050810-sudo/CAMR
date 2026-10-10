"""Run a bounded command while honoring the existing baseline GPU leases.

Uses the same /tmp/engram_baseline_gpu_N.lock protocol as run_baseline_queue.
Does not signal or terminate jobs owned by another process.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--wait-seconds', type=float, default=1800)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        gpu_ids = [int(value) for value in args.gpus.split(',')]
    except ValueError:
        parser.error('--gpus must be comma-separated integer device IDs')
    if not gpu_ids or min(gpu_ids) < 0 or len(set(gpu_ids)) != len(gpu_ids):
        parser.error('--gpus requires unique nonnegative device IDs')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.wait_seconds <= 0:
        parser.error('provide a command and positive wait-seconds')
    deadline = time.monotonic() + args.wait_seconds
    with ExitStack() as stack:
        for gpu in sorted(gpu_ids):
            handle = stack.enter_context(Path(f'/tmp/engram_baseline_gpu_{gpu}.lock').open('a+'))
            printed = False
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not printed:
                        print(f'Waiting for existing work on GPU {gpu} to release its lease.', flush=True)
                        printed = True
                    if time.monotonic() >= deadline:
                        raise TimeoutError('GPU lease wait expired; no job was launched')
                    time.sleep(2)
        # Non-cooperating jobs may still occupy a leased device. Wait rather
        # than start another model and force either job to OOM.
        printed = False
        while True:
            result = subprocess.run(['nvidia-smi', '--id=' + args.gpus, '--query-gpu=memory.used',
                                     '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
            usage = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            if len(usage) != len(gpu_ids):
                raise RuntimeError('nvidia-smi did not return every requested GPU')
            if max(usage) <= 2048:
                break
            if not printed:
                print('Leases held; waiting for other device allocations to finish.', flush=True)
                printed = True
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU memory wait expired; no job was launched')
            time.sleep(2)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
        print(f'GPU leases acquired: {args.gpus}', flush=True)
        result = subprocess.run(command, env=env)
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
