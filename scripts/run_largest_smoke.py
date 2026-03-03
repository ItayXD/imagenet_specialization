#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import time
import sys

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run largest-setting smoke test for memory/training progression.')
    parser.add_argument('--experiment', default='exchangeability_w512_g0', help='Hydra experiment config name')
    parser.add_argument('--max-tranches', type=int, default=50, help='Number of tranches to run')
    parser.add_argument('--target-images-seen', type=int, default=10_000_000, help='Target images seen override')
    parser.add_argument('--safety-factor', type=float, default=1.35, help='Multiplier for suggested SLURM time')
    parser.add_argument('--base-dir', default='', help='Optional explicit base_dir override for this smoke run')
    return parser.parse_args()


def _load_minibatch_size(experiment_name: str) -> int:
    cfg_path = f'conf/experiment/{experiment_name}.yaml'
    cfg = OmegaConf.load(cfg_path)
    return int(cfg.hyperparams.task_list[0].training_params.minibatch_size)


def _load_default_base_dir(experiment_name: str) -> str:
    cfg_path = f'conf/experiment/{experiment_name}.yaml'
    cfg = OmegaConf.load(cfg_path)
    return str(cfg.base_dir)


def main() -> None:
    args = parse_args()
    minibatch_size = _load_minibatch_size(args.experiment)
    smoke_images_seen = args.max_tranches * minibatch_size
    stamp = time.strftime('%Y%m%d-%H%M%S')
    default_base = _load_default_base_dir(args.experiment)
    smoke_base_dir = args.base_dir.strip() or os.path.join(default_base, 'smoke_runs', f'{stamp}-pid{os.getpid()}')

    cmd = [
        sys.executable,
        'main.py',
        f'experiment={args.experiment}',
        f'hyperparams.task_list.0.training_params.max_tranches={args.max_tranches}',
        f'hyperparams.task_list.0.training_params.target_images_seen={args.target_images_seen}',
        f'base_dir={smoke_base_dir}',
    ]

    print('Running smoke test command:')
    print(' '.join(cmd))
    print(f'Using smoke base_dir: {smoke_base_dir}')

    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed_s = time.time() - start

    if smoke_images_seen <= 0:
        print('Could not compute extrapolated time; smoke_images_seen <= 0.')
        return

    images_per_s = smoke_images_seen / elapsed_s
    est_full_s = args.target_images_seen / images_per_s
    est_full_h = est_full_s / 3600.0
    est_with_safety_h = est_full_h * args.safety_factor

    hh = int(est_with_safety_h)
    mm = int((est_with_safety_h - hh) * 60)
    ss = int((((est_with_safety_h - hh) * 60) - mm) * 60)

    print('\n--- Smoke Timing Summary ---')
    print(f'smoke_tranches={args.max_tranches}')
    print(f'smoke_images_seen={smoke_images_seen}')
    print(f'elapsed_seconds={elapsed_s:.2f}')
    print(f'images_per_second={images_per_s:.2f}')
    print(f'estimated_full_hours={est_full_h:.2f}')
    print(f'estimated_full_hours_with_safety={est_with_safety_h:.2f} (safety_factor={args.safety_factor})')
    print(f'suggested_sbatch_time={hh:02d}:{mm:02d}:{ss:02d}')


if __name__ == '__main__':
    main()
