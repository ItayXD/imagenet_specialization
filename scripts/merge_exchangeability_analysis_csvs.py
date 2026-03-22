#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import glob
import os
from tempfile import NamedTemporaryFile


ANALYSIS_FIELDNAMES = [
    'dataset',
    'width',
    'source_run_id',
    'images_seen',
    'representation',
    'analysis_type',
    'shuffle_id',
    'ks_distance',
    'ks_p_raw',
    'ks_sigma_two_sided',
    'ks_p_empirical',
    'ks_sigma_empirical_two_sided',
    'w1_distance',
    'w1_p_empirical',
    'w1_sigma_empirical_two_sided',
    'train_loss',
    'val_loss',
    'train_error',
    'val_error',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Merge per-width exchangeability analysis CSVs into one deduplicated CSV.'
    )
    parser.add_argument(
        '--inputs-glob',
        action='append',
        default=[],
        help='Input CSV glob. Can be passed multiple times.',
    )
    parser.add_argument(
        '--base-save-dir',
        default=os.environ.get('BASE_SAVE_DIR', ''),
        help='Optional base save directory; defaults to BASE_SAVE_DIR if set.',
    )
    parser.add_argument(
        '--output',
        default='',
        help='Merged output CSV path.',
    )
    parser.add_argument(
        '--lock-file',
        default='',
        help='Optional lock file path (defaults to "<output>.lock").',
    )
    return parser.parse_args()


def _coerce_int(value: str, *, field: str, default_if_blank: int | None = None) -> int:
    text = '' if value is None else str(value).strip()
    if text == '' and default_if_blank is not None:
        return default_if_blank
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f'Invalid integer for {field}: {value!r}') from exc


def _row_identity(row: dict[str, str]) -> tuple[str, int, str, int, str, str, int]:
    return (
        str(row.get('dataset', 'imagenet') or 'imagenet'),
        _coerce_int(row.get('width', ''), field='width'),
        str(row.get('source_run_id', '')),
        _coerce_int(row.get('images_seen', ''), field='images_seen'),
        str(row['representation']),
        str(row['analysis_type']),
        _coerce_int(row.get('shuffle_id', ''), field='shuffle_id', default_if_blank=-1),
    )


def _logical_row_identity(row: dict[str, str]) -> tuple[str, int, int, str, str, int]:
    return (
        str(row.get('dataset', 'imagenet') or 'imagenet'),
        _coerce_int(row.get('width', ''), field='width'),
        _coerce_int(row.get('images_seen', ''), field='images_seen'),
        str(row['representation']),
        str(row['analysis_type']),
        _coerce_int(row.get('shuffle_id', ''), field='shuffle_id', default_if_blank=-1),
    )


def _row_sort_key(row: dict[str, str]) -> tuple[str, int, str, int, str, str, int]:
    return _row_identity(row)


def _discover_input_paths(input_globs: list[str]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for pattern in input_globs:
        for raw_path in sorted(glob.glob(pattern)):
            path = os.path.abspath(raw_path)
            if (not os.path.isfile(path)) or path in seen:
                continue
            seen.add(path)
            paths.append(path)
    if not paths:
        raise RuntimeError(
            'No input CSV files found. Pass --inputs-glob with a pattern that matches files.'
        )
    return paths


def _read_rows(path: str) -> list[dict[str, str]]:
    with open(path, 'r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        normalized = {field: row.get(field, '') for field in ANALYSIS_FIELDNAMES}
        normalized['dataset'] = str(normalized.get('dataset', '') or 'imagenet')
        normalized['shuffle_id'] = str(
            _coerce_int(normalized.get('shuffle_id', ''), field='shuffle_id', default_if_blank=-1)
        )
        normalized_rows.append(normalized)
    return normalized_rows


def _merge_rows(
    input_paths: list[str],
    *,
    output_path: str | None = None,
) -> list[dict[str, str]]:
    chosen_rows: dict[tuple[str, int, int, str, str, int], dict[str, str]] = {}
    chosen_paths: dict[tuple[str, int, int, str, str, int], str] = {}
    output_abs = os.path.abspath(output_path) if output_path else None

    for path in input_paths:
        path_abs = os.path.abspath(path)
        path_is_output = output_abs is not None and path_abs == output_abs
        rows = _read_rows(path)
        for row in rows:
            logical_key = _logical_row_identity(row)
            current_row = chosen_rows.get(logical_key)
            if current_row is None:
                chosen_rows[logical_key] = row
                chosen_paths[logical_key] = path_abs
                continue

            if _row_identity(current_row) == _row_identity(row):
                continue

            current_path = chosen_paths[logical_key]
            current_is_output = output_abs is not None and current_path == output_abs
            if current_is_output and (not path_is_output):
                chosen_rows[logical_key] = row
                chosen_paths[logical_key] = path_abs

    merged = list(chosen_rows.values())
    merged.sort(key=_row_sort_key)
    return merged


def _write_csv_atomic(rows: list[dict[str, str]], output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with NamedTemporaryFile('w', delete=False, dir=out_dir or '.', encoding='utf-8', newline='') as tmp:
        writer = csv.DictWriter(tmp, fieldnames=ANALYSIS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = tmp.name
    os.replace(temp_path, output_path)


def main() -> None:
    args = parse_args()
    input_globs = args.inputs_glob or []
    base_save_dir = os.path.abspath(args.base_save_dir) if args.base_save_dir else ''
    if not input_globs:
        if not base_save_dir:
            raise RuntimeError('Pass --inputs-glob or --base-save-dir.')
        input_globs = [os.path.join(base_save_dir, 'exchangeability_metrics_w*.csv')]
    if args.output:
        output_path = os.path.abspath(args.output)
    elif base_save_dir:
        output_path = os.path.join(base_save_dir, 'exchangeability_metrics.csv')
    else:
        raise RuntimeError('Pass --output or --base-save-dir.')
    lock_path = os.path.abspath(args.lock_file) if args.lock_file else f'{output_path}.lock'

    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    with open(lock_path, 'a+', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        input_paths = _discover_input_paths(input_globs)
        merged_rows = _merge_rows(input_paths, output_path=output_path)
        _write_csv_atomic(merged_rows, output_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    print(f'Merged {len(input_paths)} CSV files into {output_path} ({len(merged_rows)} rows).')


if __name__ == '__main__':
    main()
