#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a single manifest row by index.')
    parser.add_argument('--manifest', required=True, help='Manifest CSV path')
    parser.add_argument('--index', required=True, type=int, help='0-based row index')
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if args.index < 0 or args.index >= len(rows):
        raise IndexError(f'Row index {args.index} out of bounds for {len(rows)} rows.')

    row = rows[args.index]
    experiment_name = row['experiment_name']

    cmd = ['python', 'main.py', f'experiment={experiment_name}']
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
