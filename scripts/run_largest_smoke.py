#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import sys
import re
import uuid

from omegaconf import OmegaConf


TRAIN_LOOP_RE = re.compile(r'\.\.\.exiting loop: elapsed time ([0-9]+(?:\.[0-9]+)?)s')
TASK_RE = re.compile(r'Task [0-9]+ completed\. Elapsed time \(s\): ([0-9]+(?:\.[0-9]+)?)')
THROUGHPUT_RE = re.compile(
    r'throughput tranches=(?P<tranches>[0-9]+) images_seen=(?P<images>[0-9]+) '
    r'ips_inst=(?P<ips_inst>[0-9]+(?:\.[0-9]+)?) ips_ema=(?P<ips_ema>[0-9]+(?:\.[0-9]+)?)'
)
TIMING_METHODS = ('ema', 'train_loop', 'task', 'wall')


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


def _hms_from_hours(hours: float) -> tuple[int, int, int]:
    hh = int(hours)
    mm = int((hours - hh) * 60)
    ss = int((((hours - hh) * 60) - mm) * 60)
    return hh, mm, ss


def _estimate_from_ips(
    images_per_s: float,
    target_images_seen: int,
    safety_factor: float,
) -> dict[str, float | str]:
    est_full_s = target_images_seen / images_per_s
    est_full_h = est_full_s / 3600.0
    est_with_safety_h = est_full_h * safety_factor
    hh, mm, ss = _hms_from_hours(est_with_safety_h)
    return {
        'images_per_second': images_per_s,
        'estimated_full_hours': est_full_h,
        'estimated_full_hours_with_safety': est_with_safety_h,
        'suggested_sbatch_time': f'{hh:02d}:{mm:02d}:{ss:02d}',
    }


def _estimate_from_seconds(
    basis_seconds: float,
    smoke_images_seen: int,
    target_images_seen: int,
    safety_factor: float,
) -> dict[str, float | str]:
    images_per_s = smoke_images_seen / basis_seconds
    out = _estimate_from_ips(images_per_s, target_images_seen, safety_factor)
    out['basis_elapsed_seconds'] = basis_seconds
    return out


def _choose_timing_source(
    requested: str,
    method_estimates: dict[str, dict[str, float | str]],
) -> tuple[str, float]:
    if requested == 'ema':
        if 'ema' not in method_estimates:
            raise RuntimeError('Requested timing_source=ema but no throughput EMA was logged.')
        return 'ema', float(method_estimates['ema']['images_per_second'])
    if requested == 'train_loop':
        if 'train_loop' not in method_estimates:
            raise RuntimeError('Requested timing_source=train_loop but train-loop timing was not found in logs.')
        return 'train_loop', float(method_estimates['train_loop']['basis_elapsed_seconds'])
    if requested == 'task':
        if 'task' not in method_estimates:
            raise RuntimeError('Requested timing_source=task but task timing was not found in logs.')
        return 'task', float(method_estimates['task']['basis_elapsed_seconds'])
    if requested == 'wall':
        return 'wall', float(method_estimates['wall']['basis_elapsed_seconds'])

    # auto mode: prefer the least startup-biased source available
    for source in TIMING_METHODS:
        if source in method_estimates:
            if source == 'ema':
                return source, float(method_estimates[source]['images_per_second'])
            return source, float(method_estimates[source]['basis_elapsed_seconds'])
    raise RuntimeError('No timing method estimates were available.')


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
    base_run_id = str(training_cfg.run_id)
    smoke_run_id = f'{base_run_id}_smoke_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}'

    cmd = [
        sys.executable,
        'main.py',
        f'experiment={args.experiment}',
        f'hyperparams.task_list.0.training_params.run_id={smoke_run_id}',
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
    print(f'Using smoke run_id: {smoke_run_id}')
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

    method_estimates: dict[str, dict[str, float | str]] = {}
    if last_ema_ips is not None and last_ema_ips > 0:
        method_estimates['ema'] = _estimate_from_ips(last_ema_ips, args.target_images_seen, args.safety_factor)
    if train_loop_elapsed_s is not None and train_loop_elapsed_s > 0:
        method_estimates['train_loop'] = _estimate_from_seconds(
            train_loop_elapsed_s,
            smoke_images_seen,
            args.target_images_seen,
            args.safety_factor,
        )
    if task_elapsed_s is not None and task_elapsed_s > 0:
        method_estimates['task'] = _estimate_from_seconds(
            task_elapsed_s,
            smoke_images_seen,
            args.target_images_seen,
            args.safety_factor,
        )
    method_estimates['wall'] = _estimate_from_seconds(
        elapsed_s,
        smoke_images_seen,
        args.target_images_seen,
        args.safety_factor,
    )

    timing_source, _ = _choose_timing_source(
        requested=args.timing_source,
        method_estimates=method_estimates,
    )
    selected_estimate = method_estimates[timing_source]
    images_per_s = float(selected_estimate['images_per_second'])
    est_full_h = float(selected_estimate['estimated_full_hours'])
    est_with_safety_h = float(selected_estimate['estimated_full_hours_with_safety'])
    basis_seconds = (
        None if timing_source == 'ema' else float(selected_estimate['basis_elapsed_seconds'])
    )
    hh, mm, ss = _hms_from_hours(est_with_safety_h)

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
    print('all_timing_methods:')
    for method_name in TIMING_METHODS:
        est = method_estimates.get(method_name)
        if est is None:
            continue
        method_ips = float(est['images_per_second'])
        method_hours = float(est['estimated_full_hours_with_safety'])
        method_sbatch = str(est['suggested_sbatch_time'])
        method_basis_s = est.get('basis_elapsed_seconds')
        if method_basis_s is None:
            print(
                f'  method={method_name} ips={method_ips:.2f} '
                f'est_hours_with_safety={method_hours:.2f} suggested={method_sbatch}'
            )
        else:
            print(
                f'  method={method_name} ips={method_ips:.2f} '
                f'basis_elapsed_seconds={float(method_basis_s):.2f} '
                f'est_hours_with_safety={method_hours:.2f} suggested={method_sbatch}'
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
            'estimate_basis_elapsed_seconds': basis_seconds,
            'smoke_images_seen': smoke_images_seen,
            'images_per_second': images_per_s,
            'estimated_full_hours': est_full_h,
            'estimated_full_hours_with_safety': est_with_safety_h,
            'safety_factor': args.safety_factor,
            'suggested_sbatch_time': f'{hh:02d}:{mm:02d}:{ss:02d}',
            'timing_method_estimates': method_estimates,
            'smoke_base_dir': smoke_base_dir,
            'smoke_run_id': smoke_run_id,
        }
        for method_name in TIMING_METHODS:
            est = method_estimates.get(method_name)
            if est is None:
                summary[f'{method_name}_images_per_second'] = None
                summary[f'{method_name}_estimated_full_hours'] = None
                summary[f'{method_name}_estimated_full_hours_with_safety'] = None
                summary[f'{method_name}_suggested_sbatch_time'] = None
                summary[f'{method_name}_basis_elapsed_seconds'] = None
            else:
                summary[f'{method_name}_images_per_second'] = est['images_per_second']
                summary[f'{method_name}_estimated_full_hours'] = est['estimated_full_hours']
                summary[f'{method_name}_estimated_full_hours_with_safety'] = est['estimated_full_hours_with_safety']
                summary[f'{method_name}_suggested_sbatch_time'] = est['suggested_sbatch_time']
                summary[f'{method_name}_basis_elapsed_seconds'] = est.get('basis_elapsed_seconds')
        with open(args.summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f'Wrote timing summary JSON: {args.summary_json}')


if __name__ == '__main__':
    main()
