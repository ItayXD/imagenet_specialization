#!/usr/bin/env python3
"""Re-render the six classifier source/capacity figures from saved arrays.

Reads ``powerlaw_arrays.npz`` produced by ``scripts/analyze_classifier_powerlaw.py``
(from a ``--results-dir`` that has typically been rsync'd from the cluster) and writes
the figures locally, by default under ``artifacts/classifier_powerlaw/<run_label>/``.
The heavy computation lives in the compute script; this only plots.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from scripts.analyze_classifier_powerlaw import _render_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True,
                        help='Directory containing powerlaw_arrays.npz.')
    parser.add_argument('--output-dir', default='',
                        help='Figure output dir. Defaults to artifacts/classifier_powerlaw/<run_label>.')
    parser.add_argument('--format', choices=['pdf', 'png'], default='pdf')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = os.path.join(args.results_dir, 'powerlaw_arrays.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'Missing {npz_path}. Run analyze_classifier_powerlaw.py first.')
    data = np.load(npz_path, allow_pickle=True)
    arrays = {key: data[key] for key in data.files}
    run_label = str(arrays.get('run_label', 'classifier_powerlaw'))

    output_dir = args.output_dir or os.path.join('artifacts', 'classifier_powerlaw', run_label)
    output_dir = os.path.abspath(output_dir)
    written = _render_all_figures(arrays, output_dir, fmt=args.format)
    for path in written:
        print(f'wrote {path}')
    print(f'done; figures under {output_dir}')


if __name__ == '__main__':
    main()
