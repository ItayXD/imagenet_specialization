#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import glob
import os
from tempfile import NamedTemporaryFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Merge per-width single-network subpart-vs-full W1 CSVs into one deduplicated CSV.'
    )
    parser.add_argument(
        '--mode',
        choices=['raw', 'summary'],
        required=True,
        help='Whether to merge per-member diagnostic rows or aggregated summary rows.',
    )
    parser.add_argument(
        '--inputs-glob',
        action='append',
        default=[],
        help='Input CSV glob. Can be passed multiple times.',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Merged output CSV path.',
    )
    parser.add_argument(
        '--lock-file',
        default='',
        help='Optional lock file path (defaults to "<output>.lock").',
    )
    return parser.parse_args()


def _discover_input_paths(input_globs: list[str]) -> list[str]:
    expanded: list[str] = []
    for pattern in input_globs:
        expanded.extend(glob.glob(pattern))
    paths = sorted({os.path.abspath(path) for path in expanded if os.path.isfile(path)})
    if not paths:
        raise RuntimeError(
            'No input CSV files found. Pass --inputs-glob with a pattern that matches files.'
        )
    return paths


def _coerce_int(value: str, *, field: str) -> int:
    text = '' if value is None else str(value).strip()
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f'Invalid integer for {field}: {value!r}') from exc


def _coerce_float(value: str, *, field: str) -> float:
    text = '' if value is None else str(value).strip()
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f'Invalid float for {field}: {value!r}') from exc


def _row_identity(row: dict[str, str], mode: str) -> tuple[object, ...]:
    base_key = (
        str(row.get('dataset', '')),
        _coerce_int(row.get('width', ''), field='width'),
        str(row.get('source_run_id', '')),
        _coerce_int(row.get('images_seen', ''), field='images_seen'),
        _coerce_float(row.get('fraction', ''), field='fraction'),
    )
    if mode == 'summary':
        return base_key
    return base_key + (
        _coerce_int(row.get('group_id', ''), field='group_id'),
        _coerce_int(row.get('member_index', ''), field='member_index'),
    )


def _row_sort_key(row: dict[str, str], mode: str) -> tuple[object, ...]:
    return _row_identity(row, mode)


def _read_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _merge_rows(input_paths: list[str], mode: str) -> tuple[list[str], list[dict[str, str]]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[object, ...]] = set()
    merged_fieldnames: list[str] = []

    for path in input_paths:
        fieldnames, rows = _read_rows(path)
        for field in fieldnames:
            if field not in merged_fieldnames:
                merged_fieldnames.append(field)
        for row in rows:
            normalized = {field: row.get(field, '') for field in merged_fieldnames}
            key = _row_identity(normalized, mode)
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)

    merged.sort(key=lambda row: _row_sort_key(row, mode))
    return merged_fieldnames, merged


def _write_csv_atomic(fieldnames: list[str], rows: list[dict[str, str]], output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with NamedTemporaryFile('w', delete=False, dir=out_dir or '.', encoding='utf-8', newline='') as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = tmp.name
    os.replace(temp_path, output_path)


def main() -> None:
    args = parse_args()
    if not args.inputs_glob:
        raise RuntimeError('At least one --inputs-glob is required.')

    output_path = os.path.abspath(args.output)
    lock_path = os.path.abspath(args.lock_file) if args.lock_file else f'{output_path}.lock'
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    with open(lock_path, 'a+', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        input_paths = _discover_input_paths(args.inputs_glob)
        fieldnames, merged_rows = _merge_rows(input_paths, args.mode)
        _write_csv_atomic(fieldnames, merged_rows, output_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    print(f'Merged {len(input_paths)} CSV files into {output_path} ({len(merged_rows)} rows).')


if __name__ == '__main__':
    main()
