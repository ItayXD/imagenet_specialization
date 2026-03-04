import os
import sys
import pytest


@pytest.mark.skipif(os.environ.get('RUN_LARGEST_SMOKE') != '1', reason='Largest smoke test is cluster-only and opt-in.')
def test_largest_smoke_harness_runs():
    import subprocess

    experiment = os.environ.get('LARGEST_SMOKE_EXPERIMENT', 'exchangeability_w512_g0')
    max_tranches = os.environ.get('LARGEST_SMOKE_MAX_TRANCHES', '50')
    minibatch_size = os.environ.get('LARGEST_SMOKE_MINIBATCH_SIZE', '')
    microbatch_size = os.environ.get('LARGEST_SMOKE_MICROBATCH_SIZE', '')
    num_workers = os.environ.get('LARGEST_SMOKE_NUM_WORKERS', '')

    cmd = [
        sys.executable,
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

    subprocess.run(cmd, check=True)
