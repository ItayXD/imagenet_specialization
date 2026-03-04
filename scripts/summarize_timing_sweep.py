#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from glob import glob

import numpy as np

ESTIMATE_METHODS = ('ema_plus_overhead', 'ema', 'train_loop', 'task', 'wall')


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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == '':
        return None
    try:
        return float(value)
    except Exception:
        return None


def _backfill_ema_plus_overhead(row: dict) -> None:
    existing = _as_float(row.get('ema_plus_overhead_estimated_full_hours_with_safety'))
    if existing is not None:
        return

    ema_ips = _as_float(row.get('ema_images_per_second'))
    smoke_images_seen = _as_float(row.get('smoke_images_seen'))
    target_images_seen = _as_float(row.get('target_images_seen'))
    safety_factor = _as_float(row.get('safety_factor'))
    if ema_ips is None or ema_ips <= 0:
        return
    if smoke_images_seen is None or smoke_images_seen <= 0:
        return
    if target_images_seen is None or target_images_seen <= 0:
        return
    if safety_factor is None or safety_factor <= 0:
        return

    steady_smoke_s = smoke_images_seen / ema_ips
    overhead_candidates = []
    for basis_key in ('train_loop_elapsed_seconds', 'task_elapsed_seconds', 'elapsed_seconds'):
        basis_s = _as_float(row.get(basis_key))
        if basis_s is None or basis_s <= 0:
            continue
        overhead_candidates.append(max(0.0, basis_s - steady_smoke_s))

    overhead_s = max(overhead_candidates) if overhead_candidates else 0.0
    est_full_h = ((target_images_seen / ema_ips) + overhead_s) / 3600.0
    est_full_h_safe = est_full_h * safety_factor

    row['ema_plus_overhead_images_per_second'] = ema_ips
    row['ema_plus_overhead_basis_elapsed_seconds'] = None
    row['ema_plus_overhead_basis_overhead_seconds'] = overhead_s
    row['ema_plus_overhead_estimated_full_hours'] = est_full_h
    row['ema_plus_overhead_estimated_full_hours_with_safety'] = est_full_h_safe
    row['ema_plus_overhead_suggested_sbatch_time'] = _to_hms(est_full_h_safe)


def _selected_hours_with_backfill(row: dict) -> float:
    source = str(row.get('estimate_timing_source', '')).strip()
    if source in ('ema', 'ema_plus_overhead'):
        v = _as_float(row.get('ema_plus_overhead_estimated_full_hours_with_safety'))
        if v is not None:
            return v
    v = _as_float(row.get('estimated_full_hours_with_safety'))
    if v is None:
        raise RuntimeError(f"Missing selected timing estimate for row: {row.get('summary_json', '<unknown>')}")
    return float(v)


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.summary_dir)
    if not rows:
        raise RuntimeError(f'No timing summary JSON files found in {args.summary_dir}')

    for row in rows:
        _backfill_ema_plus_overhead(row)

    rows_sorted = sorted(rows, key=lambda r: (int(r['width']), int(r['group_id']), str(r['experiment'])))

    detail_csv = args.output_csv or os.path.join(args.summary_dir, 'timing_estimates.csv')
    method_detail_fields = []
    for method in ESTIMATE_METHODS:
        method_detail_fields.extend(
            [
                f'{method}_images_per_second',
                f'{method}_basis_elapsed_seconds',
                f'{method}_estimated_full_hours',
                f'{method}_estimated_full_hours_with_safety',
                f'{method}_suggested_sbatch_time',
            ]
        )
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
        'estimate_timing_source',
        'estimate_basis_elapsed_seconds',
        'task_elapsed_seconds',
        'train_loop_elapsed_seconds',
        'last_throughput_ips_inst',
        'last_throughput_ips_ema',
        'last_throughput_images_seen',
        'last_throughput_tranches',
        *method_detail_fields,
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
        width_to_hours.setdefault(width, []).append(_selected_hours_with_backfill(row))

    width_rows = []
    for width in sorted(width_to_hours):
        out_row = {'width': width, 'num_jobs': 0}

        selected_vals = np.array(width_to_hours[width], dtype=np.float64)
        sel_max = float(np.max(selected_vals))
        sel_p90 = float(np.percentile(selected_vals, 90))
        sel_median = float(np.median(selected_vals))
        out_row['num_jobs'] = int(selected_vals.size)
        out_row['median_estimated_hours_with_safety'] = round(sel_median, 3)
        out_row['p90_estimated_hours_with_safety'] = round(sel_p90, 3)
        out_row['max_estimated_hours_with_safety'] = round(sel_max, 3)
        out_row['recommended_sbatch_time'] = _to_hms(sel_max)

        width_rows_only = [r for r in rows_sorted if int(r['width']) == width]
        for method in ESTIMATE_METHODS:
            method_key = f'{method}_estimated_full_hours_with_safety'
            method_vals = []
            for row in width_rows_only:
                v = _as_float(row.get(method_key))
                if v is not None:
                    method_vals.append(v)

            out_row[f'num_jobs_{method}'] = len(method_vals)
            if method_vals:
                vals = np.array(method_vals, dtype=np.float64)
                mmax = float(np.max(vals))
                mp90 = float(np.percentile(vals, 90))
                mmed = float(np.median(vals))
                out_row[f'median_estimated_hours_with_safety_{method}'] = round(mmed, 3)
                out_row[f'p90_estimated_hours_with_safety_{method}'] = round(mp90, 3)
                out_row[f'max_estimated_hours_with_safety_{method}'] = round(mmax, 3)
                out_row[f'recommended_sbatch_time_{method}'] = _to_hms(mmax)
            else:
                out_row[f'median_estimated_hours_with_safety_{method}'] = None
                out_row[f'p90_estimated_hours_with_safety_{method}'] = None
                out_row[f'max_estimated_hours_with_safety_{method}'] = None
                out_row[f'recommended_sbatch_time_{method}'] = None

        width_rows.append(out_row)

    width_csv = args.output_width_csv or os.path.join(args.summary_dir, 'timing_by_width.csv')
    width_fields = [
        'width',
        'num_jobs',
        'median_estimated_hours_with_safety',
        'p90_estimated_hours_with_safety',
        'max_estimated_hours_with_safety',
        'recommended_sbatch_time',
    ]
    for method in ESTIMATE_METHODS:
        width_fields.extend(
            [
                f'num_jobs_{method}',
                f'median_estimated_hours_with_safety_{method}',
                f'p90_estimated_hours_with_safety_{method}',
                f'max_estimated_hours_with_safety_{method}',
                f'recommended_sbatch_time_{method}',
            ]
        )
    _write_csv(width_csv, width_rows, width_fields)

    print(f'Wrote per-row timing CSV: {detail_csv}')
    print(f'Wrote per-width timing CSV: {width_csv}')
    print('Width recommendations:')
    for row in width_rows:
        summary_bits = [
            f"selected={row['recommended_sbatch_time']}",
            f"ema_plus_overhead={row.get('recommended_sbatch_time_ema_plus_overhead')}",
            f"ema={row.get('recommended_sbatch_time_ema')}",
            f"train_loop={row.get('recommended_sbatch_time_train_loop')}",
            f"task={row.get('recommended_sbatch_time_task')}",
            f"wall={row.get('recommended_sbatch_time_wall')}",
        ]
        print(f"  width={row['width']}: " + ', '.join(summary_bits))


if __name__ == '__main__':
    main()
