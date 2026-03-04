#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import sys
import re

from omegaconf import OmegaConf


TRAIN_LOOP_RE = re.compile(r'\.\.\.exiting loop: elapsed time ([0-9]+(?:\.[0-9]+)?)s')
TASK_RE = re.compile(r'Task [0-9]+ completed\. Elapsed time \(s\): ([0-9]+(?:\.[0-9]+)?)')
THROUGHPUT_RE = re.compile(
    r'throughput tranches=(?P<tranches>[0-9]+) images_seen=(?P<images>[0-9]+) '
    r'ips_inst=(?P<ips_inst>[0-9]+(?:\.[0-9]+)?) ips_ema=(?P<ips_ema>[0-9]+(?:\.[0-9]+)?)'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run largest-setting smoke test for memory/training progression.')
    parser.add_argument('--experiment', default='exchangeability_w512_g0', help='Hydra experiment config name')
    parser.add_argument('--max-tranches', type=int, default=50, help='Number of tranches to run')
    parser.add_argument('--target-images-seen', type=int, default=10_000_000, help='Target images seen override')
    parser.add_argument('--safety-factor', type=float, default=1.35, help='Multiplier for suggested SLURM time')
    parser.add_argument('--base-dir', default='', help='Optional explicit base_dir override for this smoke run')
    parser.add_argument('--minibatch-size', type=int, default=0, help='Optional minibatch_size override for smoke run')
    parser.add_argument('--microbatch-size', type=int, default=0, help='Optional microbatch_size override for smoke run')
    parser.add_argument('--num-workers', type=int, default=0, help='Optional DataLoader num_workers override for smoke run')
    parser.add_argument('--summary-json', default='', help='Optional JSON path for writing timing summary')
    parser.add_argument(
        '--timing-source',
        choices=('auto', 'ema', 'train_loop', 'task', 'wall'),
        default='auto',
        help='Which timing source to use for extrapolation.',
    )
    return parser.parse_args()


def _load_experiment_cfg(experiment_name: str):
    cfg_path = f'conf/experiment/{experiment_name}.yaml'
    return OmegaConf.load(cfg_path)


def _choose_timing_source(
    requested: str,
    wall_seconds: float,
    task_seconds: float | None,
    train_loop_seconds: float | None,
    ema_ips: float | None,
) -> tuple[str, float]:
    if requested == 'ema':
        if ema_ips is None:
            raise RuntimeError('Requested timing_source=ema but no throughput EMA was logged.')
        return 'ema', ema_ips
    if requested == 'train_loop':
        if train_loop_seconds is None:
            raise RuntimeError('Requested timing_source=train_loop but train-loop timing was not found in logs.')
        return 'train_loop', train_loop_seconds
    if requested == 'task':
        if task_seconds is None:
            raise RuntimeError('Requested timing_source=task but task timing was not found in logs.')
        return 'task', task_seconds
    if requested == 'wall':
        return 'wall', wall_seconds

    # auto mode: prefer the least startup-biased source available
    if ema_ips is not None:
        return 'ema', ema_ips
    if train_loop_seconds is not None:
        return 'train_loop', train_loop_seconds
    if task_seconds is not None:
        return 'task', task_seconds
    return 'wall', wall_seconds


def main() -> None:
    args = parse_args()
    cfg = _load_experiment_cfg(args.experiment)
    training_cfg = cfg.hyperparams.task_list[0].training_params
    model_cfg = cfg.hyperparams.task_list[0].model_params
    width = int(model_cfg.N)

    cfg_minibatch_size = int(training_cfg.minibatch_size)
    cfg_microbatch_size = int(training_cfg.microbatch_size)
    cfg_num_workers = int(training_cfg.num_workers)

    minibatch_size = args.minibatch_size if args.minibatch_size > 0 else cfg_minibatch_size
    microbatch_size = args.microbatch_size if args.microbatch_size > 0 else cfg_microbatch_size
    num_workers = args.num_workers if args.num_workers > 0 else cfg_num_workers

    smoke_images_seen = args.max_tranches * minibatch_size
    stamp = time.strftime('%Y%m%d-%H%M%S')
    default_base = str(cfg.base_dir)
    smoke_base_dir = args.base_dir.strip() or os.path.join(default_base, 'smoke_runs', f'{stamp}-pid{os.getpid()}')
    ensemble_size = int(model_cfg.ensemble_size)
    ensemble_subsets = int(training_cfg.ensemble_subsets)

    cmd = [
        sys.executable,
        'main.py',
        f'experiment={args.experiment}',
        f'hyperparams.task_list.0.training_params.max_tranches={args.max_tranches}',
        f'hyperparams.task_list.0.training_params.target_images_seen={args.target_images_seen}',
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

    task_elapsed_s: float | None = None
    train_loop_elapsed_s: float | None = None
    last_ema_ips: float | None = None
    last_inst_ips: float | None = None
    last_throughput_images_seen: int | None = None
    last_throughput_tranches: int | None = None

    start = time.time()
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')

            m = TRAIN_LOOP_RE.search(line)
            if m:
                train_loop_elapsed_s = float(m.group(1))

            m = TASK_RE.search(line)
            if m:
                task_elapsed_s = float(m.group(1))

            m = THROUGHPUT_RE.search(line)
            if m:
                last_throughput_tranches = int(m.group('tranches'))
                last_throughput_images_seen = int(m.group('images'))
                last_inst_ips = float(m.group('ips_inst'))
                last_ema_ips = float(m.group('ips_ema'))

        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)

    elapsed_s = time.time() - start

    if smoke_images_seen <= 0:
        print('Could not compute extrapolated time; smoke_images_seen <= 0.')
        return

    timing_source, timing_value = _choose_timing_source(
        requested=args.timing_source,
        wall_seconds=elapsed_s,
        task_seconds=task_elapsed_s,
        train_loop_seconds=train_loop_elapsed_s,
        ema_ips=last_ema_ips,
    )

    if timing_source == 'ema':
        images_per_s = timing_value
        basis_seconds = smoke_images_seen / max(images_per_s, 1e-12)
    else:
        basis_seconds = timing_value
        images_per_s = smoke_images_seen / basis_seconds

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
    if task_elapsed_s is not None:
        print(f'task_elapsed_seconds={task_elapsed_s:.2f}')
    if train_loop_elapsed_s is not None:
        print(f'train_loop_elapsed_seconds={train_loop_elapsed_s:.2f}')
    if last_ema_ips is not None:
        print(f'last_throughput_ips_inst={last_inst_ips:.2f}')
        print(f'last_throughput_ips_ema={last_ema_ips:.2f}')
        print(
            f'last_throughput_point=tranches:{last_throughput_tranches},'
            f'images_seen:{last_throughput_images_seen}'
        )
    print(f'estimate_timing_source={timing_source}')
    if timing_source == 'ema':
        print(f'estimate_basis_images_per_second={images_per_s:.2f}')
    else:
        print(f'estimate_basis_elapsed_seconds={basis_seconds:.2f}')
    print(f'images_per_second={images_per_s:.2f}')
    print(f'estimated_full_hours={est_full_h:.2f}')
    print(f'estimated_full_hours_with_safety={est_with_safety_h:.2f} (safety_factor={args.safety_factor})')
    print(f'suggested_sbatch_time={hh:02d}:{mm:02d}:{ss:02d}')

    if args.summary_json:
        summary_dir = os.path.dirname(args.summary_json)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        summary = {
            'experiment': args.experiment,
            'width': width,
            'group_id': int(training_cfg.group_id),
            'ensemble_size': ensemble_size,
            'ensemble_subsets': ensemble_subsets,
            'max_tranches': args.max_tranches,
            'target_images_seen': args.target_images_seen,
            'minibatch_size': minibatch_size,
            'microbatch_size': microbatch_size,
            'num_workers': num_workers,
            'elapsed_seconds': elapsed_s,
            'task_elapsed_seconds': task_elapsed_s,
            'train_loop_elapsed_seconds': train_loop_elapsed_s,
            'last_throughput_ips_inst': last_inst_ips,
            'last_throughput_ips_ema': last_ema_ips,
            'last_throughput_images_seen': last_throughput_images_seen,
            'last_throughput_tranches': last_throughput_tranches,
            'estimate_timing_source': timing_source,
            'estimate_basis_elapsed_seconds': basis_seconds if timing_source != 'ema' else None,
            'smoke_images_seen': smoke_images_seen,
            'images_per_second': images_per_s,
            'estimated_full_hours': est_full_h,
            'estimated_full_hours_with_safety': est_with_safety_h,
            'safety_factor': args.safety_factor,
            'suggested_sbatch_time': f'{hh:02d}:{mm:02d}:{ss:02d}',
            'smoke_base_dir': smoke_base_dir,
        }
        with open(args.summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f'Wrote timing summary JSON: {args.summary_json}')


if __name__ == '__main__':
    main()
