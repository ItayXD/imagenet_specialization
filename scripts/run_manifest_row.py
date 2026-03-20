#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a single manifest row by index.')
    parser.add_argument('--manifest', required=True, help='Manifest CSV path')
    parser.add_argument('--index', required=True, type=int, help='0-based row index')
    parser.add_argument(
        '--run-id-suffix',
        default='',
        help='Optional suffix appended to manifest wandb_group for a unique training_params.run_id.',
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

    cmd = [sys.executable, 'main.py', f'experiment={experiment_name}']

    run_id_suffix = args.run_id_suffix.strip()
    if run_id_suffix:
        base_run_id = str(row.get('run_id', '')).strip() or str(row.get('wandb_group', '')).strip() or 'exchangeability'
        run_id = f'{base_run_id}_{run_id_suffix}'
        cmd.append(f'hyperparams.task_list.0.training_params.run_id={run_id}')
        print(f'Using run_id override: {run_id}')

        base_dir = str(row.get('base_dir', '')).strip()
        if base_dir:
            run_base_dir = f'{base_dir}_{run_id_suffix}'
            cmd.append(f'base_dir={run_base_dir}')
            print(f'Using base_dir override: {run_base_dir}')

    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
