import os
import pytest


@pytest.mark.skipif(os.environ.get('RUN_LARGEST_SMOKE') != '1', reason='Largest smoke test is cluster-only and opt-in.')
def test_largest_smoke_harness_runs():
    import subprocess

    cmd = [
        'python',
        'scripts/run_largest_smoke.py',
        '--experiment',
        'exchangeability_w512_g0',
        '--max-tranches',
        '50',
    ]
    subprocess.run(cmd, check=True)
