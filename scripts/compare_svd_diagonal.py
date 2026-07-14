#!/usr/bin/env python3
"""Compare the SVD diagonal-band / Haar descriptors across width and optimizer.

Scans a results root for */svd/diagonal_band_summary.json (produced by
analyze_svd_diagonal.py) and plots, per (dataset, optimizer) series vs width:

  * longitudinal exponent p (on-diagonal amplitude A(i) ~ i^-p);
  * structured-band size i_cross (floor-crossing mode = # modes aligned to the eigenbasis);
  * below-floor diagonal percentile (0.5 = Haar-random tail);
  * below-floor participation ratio / (D/3) (1.0 = Haar-random tail).

Only runs with enough singular modes are meaningful (ImageNet: C=1000); CIFAR-5M
(C=10) is skipped by the diagonal analysis and will simply be absent.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_RUN_RE = re.compile(r'(?P<dataset>[^/]+)/(?P<opt>sgd|adam|muon)_w(?P<width>\d+)/svd/right/')
_COLOR = {('imagenet', 'sgd'): 'tab:blue', ('imagenet', 'muon'): 'tab:orange',
          ('cifar5m', 'sgd'): 'tab:green', ('cifar5m', 'muon'): 'tab:red'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default='artifacts/classifier_powerlaw')
    parser.add_argument('--output-dir', default='artifacts/classifier_powerlaw/comparison')
    parser.add_argument('--format', choices=['pdf', 'png'], default='png')
    return parser.parse_args()


def _load(results_root: str) -> dict:
    series: dict = {}
    for path in sorted(glob.glob(os.path.join(results_root, '**', 'svd', 'right',
                                              'diagonal_band_summary.json'), recursive=True)):
        rel = os.path.relpath(path, results_root)
        m = _RUN_RE.search(rel.replace(os.sep, '/'))
        if not m:
            continue
        with open(path, encoding='utf-8') as h:
            d = json.load(h)
        haar = d.get('haar_below_floor', {})
        # sibling singular-value fit (one level up, under svd/, not right/)
        sing = {}
        sing_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'singular_powerlaw.json')
        if os.path.exists(sing_path):
            with open(sing_path, encoding='utf-8') as h:
                sing = json.load(h)
        key = (m['dataset'], m['opt'])
        series.setdefault(key, []).append({
            'width': int(m['width']),
            'p': d.get('longitudinal_exponent_p', np.nan),
            'p_r2': d.get('longitudinal_r2', np.nan),
            'q': d.get('transverse_length_exponent_q', np.nan),
            'shape': d.get('transverse_shape_verdict', ''),
            'ell_low': d.get('ell_low_exponent', np.nan),
            'ell_break': d.get('ell_breakpoint', np.nan),
            'shape_cross': d.get('shape_crossover_i', np.nan),
            'sing_c': sing.get('c', np.nan),
            'sing_r2': sing.get('r2', np.nan),
            'i_cross': haar.get('i_cross', np.nan),
            'D': d.get('num_features', np.nan),
            'diag_below': haar.get('diag_below_mean', np.nan),
            'diag_above': haar.get('diag_above_mean', np.nan),
            'pr_ratio': (haar.get('pr_below_mean', np.nan) / (d.get('num_features', np.nan) / 3.0)),
        })
    for key in series:
        series[key].sort(key=lambda r: r['width'])
    return dict(sorted(series.items()))


def main() -> None:
    args = parse_args()
    series = _load(args.results_root)
    if not series:
        raise SystemExit(f'No diagonal_band_summary.json found under {args.results_root}')
    print('series:', {f'{a}/{b}': [r['width'] for r in v] for (a, b), v in series.items()})

    panels = [
        ('p', r'longitudinal exponent $p$  ($A(i)\sim i^{-p}$)'),
        ('ell_low', r'transverse low-$i$ exponent  ($\ell\sim i^{q_\mathrm{low}}$)'),
        ('ell_break', r'$\ell(i)$ breakpoint $i$'),
        ('sing_c', r'singular-value exponent $c$  ($s_j\sim j^{-c}$)'),
        ('i_cross', r'structured-band size $i_\mathrm{cross}$ (# aligned modes)'),
        ('diag_below', r'below-floor diagonal percentile (0.5 = Haar)'),
        ('pr_ratio', r'below-floor PR / $(D/3)$  (1.0 = Haar)'),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20.0, 9.0))
    for (field, title), ax in zip(panels, axes.ravel()):
        for key, rows in series.items():
            widths = np.array([r['width'] for r in rows], dtype=float)
            ys = np.array([float(r[field]) if r[field] is not None else np.nan
                           for r in rows], dtype=float)
            ax.plot(widths, ys, marker='o', color=_COLOR.get(key), label=f'{key[0]}/{key[1]}')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('width (num_filters)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.25)
        if field == 'diag_below':
            ax.axhline(0.5, color='0.5', ls='--', lw=1)
        if field == 'pr_ratio':
            ax.axhline(1.0, color='0.5', ls='--', lw=1)
        if field == 'i_cross':
            ax.set_yscale('log')
    for ax in axes.ravel()[len(panels):]:
        ax.axis('off')
    axes.ravel()[0].legend(fontsize=9)
    fig.suptitle('SVD diagonal-band structure vs width', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)
    out = os.path.join(os.path.abspath(args.output_dir), f'compare_svd_diagonal.{args.format}')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)

    csv_path = os.path.join(os.path.abspath(args.output_dir), 'compare_svd_diagonal.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as h:
        cols = ['width', 'D', 'p', 'p_r2', 'q', 'ell_low', 'ell_break', 'shape', 'shape_cross',
                'sing_c', 'sing_r2', 'i_cross', 'diag_below', 'diag_above', 'pr_ratio']
        w = csv.DictWriter(h, fieldnames=['dataset', 'optimizer', *cols])
        w.writeheader()
        for (dataset, opt), rows in series.items():
            for r in rows:
                w.writerow({'dataset': dataset, 'optimizer': opt, **{k: r[k] for k in cols}})
    print(f'wrote {out}')
    print(f'wrote {csv_path}')


if __name__ == '__main__':
    main()
