import os
from pathlib import Path
import subprocess
import sys


def test_analyze_exchangeability_import_does_not_require_torchvision(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fake_torchvision_dir = tmp_path / 'torchvision'
    fake_torchvision_dir.mkdir()
    (fake_torchvision_dir / '__init__.py').write_text(
        "raise RuntimeError('torchvision should not be imported during module import')\n",
        encoding='utf-8',
    )

    env = os.environ.copy()
    pythonpath_parts = [str(tmp_path), str(repo_root)]
    existing_pythonpath = env.get('PYTHONPATH', '')
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)

    proc = subprocess.run(
        [sys.executable, '-c', 'import scripts.analyze_exchangeability as module; print(module.__name__)'],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith('scripts.analyze_exchangeability')
