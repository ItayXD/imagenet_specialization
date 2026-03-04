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
    parser.add_argument('--ensemble-subsets', type=int, default=0, help='Override ensemble_subsets for smoke run (0 = auto)')
    parser.add_argument('--minibatch-size', type=int, default=0, help='Optional minibatch_size override for smoke run')
    parser.add_argument('--microbatch-size', type=int, default=0, help='Optional microbatch_size override for smoke run')
    parser.add_argument('--num-workers', type=int, default=0, help='Optional DataLoader num_workers override for smoke run')
    return parser.parse_args()


def _load_experiment_cfg(experiment_name: str):
    cfg_path = f'conf/experiment/{experiment_name}.yaml'
    return OmegaConf.load(cfg_path)


def main() -> None:
    args = parse_args()
    cfg = _load_experiment_cfg(args.experiment)
    training_cfg = cfg.hyperparams.task_list[0].training_params
    model_cfg = cfg.hyperparams.task_list[0].model_params
    width = int(model_cfg.N)

    cfg_minibatch_size = int(training_cfg.minibatch_size)
    cfg_microbatch_size = int(training_cfg.microbatch_size)
    cfg_num_workers = int(training_cfg.num_workers)

    if args.minibatch_size > 0:
        minibatch_size = args.minibatch_size
    elif width >= 512 and cfg_minibatch_size > 256:
        minibatch_size = 256
    else:
        minibatch_size = cfg_minibatch_size

    if args.microbatch_size > 0:
        microbatch_size = args.microbatch_size
    elif width >= 512 and cfg_microbatch_size > 32:
        microbatch_size = 32
    else:
        microbatch_size = cfg_microbatch_size

    if args.num_workers > 0:
        num_workers = args.num_workers
    elif cfg_num_workers > 8:
        num_workers = 8
    else:
        num_workers = cfg_num_workers

    smoke_images_seen = args.max_tranches * minibatch_size
    stamp = time.strftime('%Y%m%d-%H%M%S')
    default_base = str(cfg.base_dir)
    smoke_base_dir = args.base_dir.strip() or os.path.join(default_base, 'smoke_runs', f'{stamp}-pid{os.getpid()}')
    ensemble_size = int(model_cfg.ensemble_size)
    ensemble_subsets = args.ensemble_subsets if args.ensemble_subsets > 0 else ensemble_size

    cmd = [
        sys.executable,
        'main.py',
        f'experiment={args.experiment}',
        f'hyperparams.task_list.0.training_params.max_tranches={args.max_tranches}',
        f'hyperparams.task_list.0.training_params.target_images_seen={args.target_images_seen}',
        f'hyperparams.task_list.0.training_params.ensemble_subsets={ensemble_subsets}',
        f'hyperparams.task_list.0.training_params.minibatch_size={minibatch_size}',
        f'hyperparams.task_list.0.training_params.microbatch_size={microbatch_size}',
        f'hyperparams.task_list.0.training_params.num_workers={num_workers}',
        f'base_dir={smoke_base_dir}',
    ]

    print('Running smoke test command:')
    print(' '.join(cmd))
    print(f'Using smoke base_dir: {smoke_base_dir}')
    print(f'Using smoke ensemble_subsets: {ensemble_subsets}')
    print(f'Using smoke minibatch_size: {minibatch_size}')
    print(f'Using smoke microbatch_size: {microbatch_size}')
    print(f'Using smoke num_workers: {num_workers}')

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
