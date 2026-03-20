#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a short timing pilot for one manifest row.')
    parser.add_argument('--manifest', required=True, help='Manifest CSV path')
    parser.add_argument('--index', required=True, type=int, help='0-based manifest row index')
    parser.add_argument('--max-tranches', type=int, default=20, help='Short pilot length')
    parser.add_argument('--target-images-seen', type=int, default=10_000_000, help='Full run horizon for extrapolation')
    parser.add_argument('--safety-factor', type=float, default=1.35, help='Safety factor for time suggestion')
    parser.add_argument('--summary-dir', required=True, help='Directory for per-row summary JSON files')
    parser.add_argument('--num-workers', type=int, default=0, help='Optional DataLoader workers override')
    parser.add_argument('--minibatch-size', type=int, default=0, help='Optional minibatch override')
    parser.add_argument('--microbatch-size', type=int, default=0, help='Optional microbatch override')
    parser.add_argument(
        '--timing-source',
        choices=('auto', 'ema', 'train_loop', 'task', 'wall'),
        default='auto',
        help='Timing source passed through to run_largest_smoke.py',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if args.index < 0 or args.index >= len(rows):
        raise IndexError(f'Row index {args.index} out of bounds for {len(rows)} rows.')

    row = rows[args.index]
    experiment_name = row['experiment_name']
    width = int(row['width'])
    group_id = int(row['group_id'])

    os.makedirs(args.summary_dir, exist_ok=True)
    summary_json = os.path.join(
        args.summary_dir,
        f'timing_row_{args.index:03d}_w{width}_g{group_id}.json',
    )

    cmd = [
        sys.executable,
        '-u',
        'scripts/run_largest_smoke.py',
        '--experiment',
        experiment_name,
        '--max-tranches',
        str(args.max_tranches),
        '--target-images-seen',
        str(args.target_images_seen),
        '--safety-factor',
        str(args.safety_factor),
        '--summary-json',
        summary_json,
        '--timing-source',
        args.timing_source,
    ]

    if args.num_workers > 0:
        cmd.extend(['--num-workers', str(args.num_workers)])
    if args.minibatch_size > 0:
        cmd.extend(['--minibatch-size', str(args.minibatch_size)])
    if args.microbatch_size > 0:
        cmd.extend(['--microbatch-size', str(args.microbatch_size)])

    print(f'Running timing pilot for manifest row {args.index}: {experiment_name}', flush=True)
    print('Command:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
