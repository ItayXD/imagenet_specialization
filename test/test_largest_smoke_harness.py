from collections import deque
from datetime import datetime
import os
import subprocess
import sys
import time

import pytest


PROGRESS_TOKENS = (
    'Running smoke test command:',
    'Using smoke base_dir:',
    'Found ',
    'Loading data...',
    'Running tasks...',
    'Task ',
    'Entering train function.',
    'Entering training loop...',
    'throughput tranches=',
    '--- Smoke Timing Summary ---',
)


def _failure_message(cmd, returncode, lines):
    last_progress_line = '<none>'
    for line in reversed(lines):
        if any(token in line for token in PROGRESS_TOKENS):
            last_progress_line = line
            break

    tail = '\n'.join(lines[-80:]) if lines else '<no output captured>'
    return (
        f'Largest smoke harness failed with exit code {returncode}.\n'
        f'Command: {" ".join(cmd)}\n'
        f'Last progress line: {last_progress_line}\n'
        f'Last output:\n{tail}'
    )


def _emit_timestamped_line(line, start_time):
    stamp = datetime.now().isoformat(timespec='milliseconds')
    elapsed_seconds = time.monotonic() - start_time
    print(f'[{stamp} +{elapsed_seconds:.3f}s] {line}', flush=True)


@pytest.mark.skipif(os.environ.get('RUN_LARGEST_SMOKE') != '1', reason='Largest smoke test is cluster-only and opt-in.')
def test_largest_smoke_harness_runs():
    experiment = os.environ.get('LARGEST_SMOKE_EXPERIMENT', 'exchangeability_w512_g0')
    max_tranches = os.environ.get('LARGEST_SMOKE_MAX_TRANCHES', '50')
    minibatch_size = os.environ.get('LARGEST_SMOKE_MINIBATCH_SIZE', '')
    microbatch_size = os.environ.get('LARGEST_SMOKE_MICROBATCH_SIZE', '')
    num_workers = os.environ.get('LARGEST_SMOKE_NUM_WORKERS', '')
    timing_source = os.environ.get('LARGEST_SMOKE_TIMING_SOURCE', '')

    cmd = [
        sys.executable,
        '-u',
        'scripts/run_largest_smoke.py',
        '--experiment',
        experiment,
        '--max-tranches',
        max_tranches,
    ]

    if minibatch_size:
        cmd.extend(['--minibatch-size', minibatch_size])
    if microbatch_size:
        cmd.extend(['--microbatch-size', microbatch_size])
    if num_workers:
        cmd.extend(['--num-workers', num_workers])
    if timing_source:
        cmd.extend(['--timing-source', timing_source])

    env = dict(os.environ)
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('HYDRA_FULL_ERROR', '1')

    start_time = time.monotonic()
    recent_lines = deque(maxlen=200)
    _emit_timestamped_line(f'largest smoke command: {" ".join(cmd)}', start_time)

    with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        ) as completed:
        assert completed.stdout is not None
        for raw_line in completed.stdout:
            line = raw_line.rstrip('\n')
            recent_lines.append(line)
            _emit_timestamped_line(line, start_time)
        returncode = completed.wait()

    if returncode != 0:
        pytest.fail(
            _failure_message(
                cmd=cmd,
                returncode=returncode,
                lines=list(recent_lines),
            )
        )
