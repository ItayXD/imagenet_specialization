import csv
import subprocess
import sys

from scripts.run_largest_smoke import _build_main_cmd
from scripts.run_timing_manifest_row import main as run_timing_manifest_row_main


def test_build_main_cmd_uses_unbuffered_python():
    cmd = _build_main_cmd(
        experiment='exchangeability_w512_g0',
        smoke_run_id='run-123',
        max_tranches=50,
        target_images_seen=5_000_000,
        minibatch_size=1024,
        microbatch_size=128,
        num_workers=8,
        smoke_base_dir='/tmp/smoke',
    )

    assert cmd[:3] == [sys.executable, '-u', 'main.py']
    assert 'experiment=exchangeability_w512_g0' in cmd
    assert 'base_dir=/tmp/smoke' in cmd


def test_run_timing_manifest_row_uses_unbuffered_smoke_runner(tmp_path, monkeypatch):
    manifest_path = tmp_path / 'manifest.csv'
    summary_dir = tmp_path / 'summary'

    with open(manifest_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment_name', 'width', 'group_id'])
        writer.writeheader()
        writer.writerow(
            {
                'experiment_name': 'exchangeability_w32_g0',
                'width': '32',
                'group_id': '0',
            }
        )

    seen = {}

    def fake_run(cmd, check):
        seen['cmd'] = cmd
        seen['check'] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr('subprocess.run', fake_run)
    monkeypatch.setattr(
        'sys.argv',
        [
            'run_timing_manifest_row.py',
            '--manifest',
            str(manifest_path),
            '--index',
            '0',
            '--summary-dir',
            str(summary_dir),
        ],
    )

    run_timing_manifest_row_main()

    assert seen['check'] is True
    assert seen['cmd'][:3] == [sys.executable, '-u', 'scripts/run_largest_smoke.py']
