#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from glob import glob

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize timing sweep JSON files.')
    parser.add_argument('--summary-dir', required=True, help='Directory with timing_row_*.json files')
    parser.add_argument('--output-csv', default='', help='Optional detailed per-row CSV output path')
    parser.add_argument('--output-width-csv', default='', help='Optional per-width summary CSV output path')
    return parser.parse_args()


def _to_hms(hours: float) -> str:
    total_seconds = int(np.ceil(max(0.0, float(hours)) * 3600.0))
    hh = total_seconds // 3600
    rem = total_seconds % 3600
    mm = rem // 60
    ss = rem % 60
    return f'{hh:02d}:{mm:02d}:{ss:02d}'


def _load_rows(summary_dir: str) -> list[dict]:
    paths = sorted(glob(os.path.join(summary_dir, 'timing_row_*.json')))
    rows = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            row = json.load(f)
        name = os.path.basename(path)
        try:
            row_index = int(name.split('_')[2])
        except Exception:
            row_index = -1
        row['row_index'] = row_index
        row['summary_json'] = path
        rows.append(row)
    return rows


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.summary_dir)
    if not rows:
        raise RuntimeError(f'No timing summary JSON files found in {args.summary_dir}')

    rows_sorted = sorted(rows, key=lambda r: (int(r['width']), int(r['group_id']), str(r['experiment'])))

    detail_csv = args.output_csv or os.path.join(args.summary_dir, 'timing_estimates.csv')
    detail_fields = [
        'row_index',
        'experiment',
        'width',
        'group_id',
        'ensemble_size',
        'ensemble_subsets',
        'max_tranches',
        'minibatch_size',
        'microbatch_size',
        'num_workers',
        'elapsed_seconds',
        'images_per_second',
        'estimated_full_hours',
        'estimated_full_hours_with_safety',
        'suggested_sbatch_time',
        'summary_json',
    ]
    _write_csv(detail_csv, rows_sorted, detail_fields)

    width_to_hours: dict[int, list[float]] = {}
    for row in rows_sorted:
        width = int(row['width'])
        width_to_hours.setdefault(width, []).append(float(row['estimated_full_hours_with_safety']))

    width_rows = []
    for width in sorted(width_to_hours):
        vals = np.array(width_to_hours[width], dtype=np.float64)
        max_hours = float(np.max(vals))
        p90_hours = float(np.percentile(vals, 90))
        median_hours = float(np.median(vals))
        width_rows.append(
            {
                'width': width,
                'num_jobs': int(vals.size),
                'median_estimated_hours_with_safety': round(median_hours, 3),
                'p90_estimated_hours_with_safety': round(p90_hours, 3),
                'max_estimated_hours_with_safety': round(max_hours, 3),
                'recommended_sbatch_time': _to_hms(max_hours),
            }
        )

    width_csv = args.output_width_csv or os.path.join(args.summary_dir, 'timing_by_width.csv')
    width_fields = [
        'width',
        'num_jobs',
        'median_estimated_hours_with_safety',
        'p90_estimated_hours_with_safety',
        'max_estimated_hours_with_safety',
        'recommended_sbatch_time',
    ]
    _write_csv(width_csv, width_rows, width_fields)

    print(f'Wrote per-row timing CSV: {detail_csv}')
    print(f'Wrote per-width timing CSV: {width_csv}')
    print('Width recommendations:')
    for row in width_rows:
        print(
            f"  width={row['width']}: "
            f"recommended_sbatch_time={row['recommended_sbatch_time']} "
            f"(max_hours={row['max_estimated_hours_with_safety']})"
        )


if __name__ == '__main__':
    main()
