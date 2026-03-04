#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import stat
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build width-specific manifests and SLURM submit scripts with baked time limits.'
    )
    parser.add_argument(
        '--manifest',
        default='conf/exchangeability_manifest.csv',
        help='Full exchangeability manifest CSV',
    )
    parser.add_argument(
        '--timing-width-csv',
        required=True,
        help='timing_by_width.csv produced by summarize_timing_sweep.py',
    )
    parser.add_argument(
        '--time-column',
        default='recommended_sbatch_time',
        help='Column in timing_by_width.csv to use for walltime',
    )
    parser.add_argument(
        '--manifest-output-dir',
        default='conf/manifests_by_width',
        help='Output directory for width-specific manifest CSVs',
    )
    parser.add_argument(
        '--slurm-output-dir',
        default='conf/slurm_jobs',
        help='Output directory for generated SLURM submit scripts',
    )
    parser.add_argument('--job-name-prefix', default='imgnet-exchg-w', help='SLURM job name prefix')
    parser.add_argument(
        '--account',
        default=os.environ.get('SBATCH_ACCOUNT', 'kempner_pehlevan_lab'),
        help='SLURM account',
    )
    parser.add_argument(
        '--partition',
        default=os.environ.get('SBATCH_PARTITION', 'kempner'),
        help='SLURM partition',
    )
    parser.add_argument('--gpus', type=int, default=1, help='GPUs per task')
    parser.add_argument('--cpus-per-task', type=int, default=24, help='CPUs per task')
    parser.add_argument('--mem', default='128G', help='Memory per task')
    return parser.parse_args()


def _load_manifest_rows(path: str) -> list[dict[str, str]]:
    with open(path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'Manifest has no rows: {path}')
    return rows


def _load_times(path: str, time_column: str) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            width = int(row['width'])
            time_value = (row.get(time_column) or '').strip()
            if not time_value:
                raise RuntimeError(
                    f'Missing {time_column} for width={width} in timing CSV: {path}'
                )
            out[width] = time_value
    if not out:
        raise RuntimeError(f'No width rows found in timing CSV: {path}')
    return out


def _group_by_width(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row['width'])].append(row)
    return dict(sorted(grouped.items()))


def _write_csv(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _script_text(
    *,
    width: int,
    time_limit: str,
    rows: int,
    account: str,
    partition: str,
    gpus: int,
    cpus_per_task: int,
    mem: str,
    manifest_relpath: str,
    job_name_prefix: str,
) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name_prefix}{width}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --array=0-{rows - 1}
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}

set -euo pipefail

if [[ -z "${{SLURM_JOB_ID:-}}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch conf/slurm_jobs/submit_exchangeability_w{width}.slurm" >&2
  exit 2
fi

ROOT_DIR="${{PROJECT_ROOT:-${{SLURM_SUBMIT_DIR:-$(pwd)}}}}"
if [[ ! -f "${{ROOT_DIR}}/pyproject.toml" ]]; then
  echo "Could not locate project root at ${{ROOT_DIR}}." >&2
  echo "Submit from repo root or set PROJECT_ROOT explicitly." >&2
  exit 2
fi

bash "${{ROOT_DIR}}/scripts/submit_exchangeability_slurm.sh" "${{ROOT_DIR}}/{manifest_relpath}"
"""


def _make_executable(path: str) -> None:
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> None:
    args = parse_args()
    manifest_rows = _load_manifest_rows(args.manifest)
    timing_by_width = _load_times(args.timing_width_csv, args.time_column)
    grouped = _group_by_width(manifest_rows)

    manifest_abs_dir = os.path.abspath(args.manifest_output_dir)
    slurm_abs_dir = os.path.abspath(args.slurm_output_dir)
    os.makedirs(manifest_abs_dir, exist_ok=True)
    os.makedirs(slurm_abs_dir, exist_ok=True)

    fieldnames = list(manifest_rows[0].keys())
    submit_paths: list[str] = []

    for width, rows in grouped.items():
        if width not in timing_by_width:
            raise RuntimeError(
                f'Missing width={width} in timing CSV {args.timing_width_csv}; cannot set walltime.'
            )

        manifest_name = f'exchangeability_manifest_w{width}.csv'
        manifest_abs_path = os.path.join(manifest_abs_dir, manifest_name)
        _write_csv(manifest_abs_path, rows, fieldnames)

        # Store paths in repo-relative form so generated scripts are portable.
        manifest_relpath = os.path.relpath(manifest_abs_path, os.getcwd())
        submit_name = f'submit_exchangeability_w{width}.slurm'
        submit_abs_path = os.path.join(slurm_abs_dir, submit_name)
        text = _script_text(
            width=width,
            time_limit=timing_by_width[width],
            rows=len(rows),
            account=args.account,
            partition=args.partition,
            gpus=args.gpus,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            manifest_relpath=manifest_relpath,
            job_name_prefix=args.job_name_prefix,
        )
        with open(submit_abs_path, 'w', encoding='utf-8') as f:
            f.write(text)

        submit_paths.append(os.path.relpath(submit_abs_path, os.getcwd()))

    submit_all_abs = os.path.join(slurm_abs_dir, 'submit_exchangeability_all_widths.sh')
    submit_all_rel = os.path.relpath(submit_all_abs, os.getcwd())
    with open(submit_all_abs, 'w', encoding='utf-8') as f:
        f.write('#!/usr/bin/env bash\n')
        f.write('set -euo pipefail\n\n')
        f.write('ROOT_DIR="${PROJECT_ROOT:-$(pwd)}"\n')
        f.write('if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then\n')
        f.write('  echo "Could not locate project root at ${ROOT_DIR}." >&2\n')
        f.write('  echo "Run from repo root or set PROJECT_ROOT explicitly." >&2\n')
        f.write('  exit 2\n')
        f.write('fi\n')
        f.write('cd "${ROOT_DIR}"\n\n')
        for submit_rel in submit_paths:
            f.write(f'sbatch "{submit_rel}"\n')

    _make_executable(submit_all_abs)

    print(f'Wrote {len(grouped)} width manifests to {args.manifest_output_dir}')
    print(f'Wrote {len(submit_paths)} width submit scripts to {args.slurm_output_dir}')
    print(f'Wrote submit-all helper: {submit_all_rel}')
    for submit_rel in submit_paths:
        print(f'  sbatch {submit_rel}')


if __name__ == '__main__':
    main()
