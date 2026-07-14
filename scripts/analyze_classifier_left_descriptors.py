#!/usr/bin/env python3
"""Width-robust per-class descriptors from the classifier's left factor (What = L S R^T).

Leverage n_i = sum_j l_ij^2 is only the *total* left-mass of class i; it degenerates to 1
once D >= C (L becomes a full orthonormal basis, rows unit-norm). The width-robust object
is the *shape* of each class's mode distribution p_ij = l_ij^2 / n_i (a probability over
singular modes j, well-defined at any width). Its moments describe WHERE in the spectrum a
class's classifier lives:

  d1_i = sum_j l_ij^2 s_j^2 / mean(s^2)         (user d^(1); = E_i/mean(s^2))
  d2_i = sum_j l_ij^2 j    / mean(j)            (user d^(2))
  d1n_i = <s^2>_i / mean(s^2)  = d1_i / n_i     (shape only: mean sing.-value^2 the class sees)
  d2n_i = <j>_i   / mean(j)    = d2_i / n_i     (shape only: mean mode index)

and a natural [0,1] quantity that stays meaningful at large width:

  kappa_i = sum_j q_ij (j-1)/(p-1) in [0,1],   q_ij = s_j^2 l_ij^2 / E_i   (energy centroid:
            0 = all energy on the top singular mode, 1 = on the weakest)
  gtop_i  = <s^2>_i / s_1^2 in [0,1]            (spectral concentration: 1 = fully top mode)

We compute these per run, correlate each with class accuracy, and (across a results root)
show correlation-with-accuracy vs width -- leverage collapses at D>=C while the spectral
descriptors persist.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.analyze_classifier_powerlaw import _pearson_spearman  # noqa: E402


def compute_descriptors(what_hat: np.ndarray) -> dict:
    left, s, _ = np.linalg.svd(np.asarray(what_hat, dtype=np.float64), full_matrices=False)
    c, p = left.shape
    l2 = left ** 2
    s2 = s ** 2
    j = np.arange(1, p + 1, dtype=np.float64)
    n = l2.sum(axis=1)                      # leverage
    energy = l2 @ s2                        # E_i
    sbar2 = float(s2.mean())
    jbar = float(j.mean())
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = energy / sbar2
        d2 = (l2 @ j) / jbar
        d1n = np.where(n > 0, energy / (n * sbar2), np.nan)      # <s^2>/mean(s^2)
        d2n = np.where(n > 0, (l2 @ j) / (n * jbar), np.nan)     # <j>/mean(j)
        # energy-weighted normalized spectral centroid in [0,1]
        kappa = np.where(energy > 0, (l2 * s2) @ ((j - 1) / (p - 1)) / energy, np.nan)
        gtop = np.where(n > 0, (energy / n) / float(s2.max()), np.nan)  # <s^2>/s1^2 in [0,1]
        # mass-weighted centroid too (uses p_ij, no s-weighting)
        mass_centroid = np.where(n > 0, l2 @ ((j - 1) / (p - 1)) / n, np.nan)
    return {'C': c, 'p': p, 'leverage': n, 'energy': energy,
            'd1': d1, 'd2': d2, 'd1n': d1n, 'd2n': d2n,
            'kappa': kappa, 'gtop': gtop, 'mass_centroid': mass_centroid}


_DESCRIPTORS = ['leverage', 'd1n', 'd2n', 'kappa', 'gtop', 'mass_centroid']


def _process_run(run_dir: str) -> dict | None:
    npz = os.path.join(run_dir, 'powerlaw_arrays.npz')
    if not os.path.exists(npz):
        return None
    data = np.load(npz, allow_pickle=True)
    if 'W_hat' not in data.files:
        return None
    desc = compute_descriptors(np.asarray(data['W_hat'], dtype=np.float64))
    acc = np.asarray(data['acc_full'], dtype=np.float64) if 'acc_full' in data.files else None
    summary = json.load(open(os.path.join(run_dir, 'fit_summary.json'))) \
        if os.path.exists(os.path.join(run_dir, 'fit_summary.json')) else {}
    dataset = str(summary.get('dataset', 'na'))
    optimizer = str(summary.get('optimizer_key', 'na'))
    width = int(summary.get('width', 0))

    corr = {}
    if acc is not None:
        for key in _DESCRIPTORS:
            r, rho = _pearson_spearman(desc[key], acc)
            corr[key] = {'pearson': r, 'spearman': rho}

    # Per-run descriptor CSV under svd/left.
    out_dir = os.path.join(run_dir, 'svd', 'left')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'left_descriptors.csv'), 'w', newline='', encoding='utf-8') as h:
        fields = ['class_index'] + _DESCRIPTORS + ['energy', 'accuracy']
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for i in range(desc['C']):
            row = {'class_index': i, 'energy': float(desc['energy'][i]),
                   'accuracy': float(acc[i]) if acc is not None else float('nan')}
            for key in _DESCRIPTORS:
                row[key] = float(desc[key][i])
            w.writerow(row)

    return {'dataset': dataset, 'optimizer': optimizer, 'width': width,
            'C': desc['C'], 'p': desc['p'], 'corr': corr,
            'kappa': desc['kappa'], 'accuracy': acc, 'run_dir': run_dir}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default='artifacts/classifier_powerlaw')
    parser.add_argument('--datasets', nargs='*', default=['imagenet'],
                        help='Datasets to aggregate for the cross-width figure (cifar C=10 is noisy).')
    parser.add_argument('--output-dir', default='artifacts/classifier_powerlaw/comparison')
    parser.add_argument('--format', choices=['pdf', 'png'], default='png')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = []
    for npz in sorted(glob.glob(os.path.join(args.results_root, '**', 'powerlaw_arrays.npz'),
                                recursive=True)):
        res = _process_run(os.path.dirname(npz))
        if res is not None:
            runs.append(res)
    if not runs:
        raise SystemExit(f'No runs under {args.results_root}')

    # Cross-width figure: corr(descriptor, accuracy) vs width, per dataset/optimizer.
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)
    series = {}
    for r in runs:
        if r['dataset'] not in args.datasets or not r['corr']:
            continue
        series.setdefault((r['dataset'], r['optimizer']), []).append(r)
    for key in series:
        series[key].sort(key=lambda r: r['width'])

    fig, axes = plt.subplots(1, len(_DESCRIPTORS), figsize=(3.4 * len(_DESCRIPTORS), 4.2),
                             squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(series))))
    for ci, (skey, rlist) in enumerate(sorted(series.items())):
        widths = [r['width'] for r in rlist]
        label = f'{skey[0]}/{skey[1]}'
        for a, key in enumerate(_DESCRIPTORS):
            ax = axes[0][a]
            ys = [r['corr'].get(key, {}).get('pearson', np.nan) for r in rlist]
            ax.plot(widths, ys, marker='o', color=colors[ci], label=label)
    for a, key in enumerate(_DESCRIPTORS):
        ax = axes[0][a]
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('width')
        ax.set_ylabel('Pearson corr with accuracy')
        ax.set_title(key)
        ax.grid(True, which='both', alpha=0.25)
        ax.set_ylim(-0.75, 0.75)
    axes[0][0].legend(fontsize=7)
    fig.suptitle('Per-class descriptor vs accuracy: correlation across width '
                 '(leverage collapses at D>=C; spectral descriptors persist)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(os.path.abspath(args.output_dir), f'compare_left_descriptors_vs_width.{args.format}')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)

    # Table to stdout + json.
    print(f'{"run":22s} {"C":>5s} {"p":>5s} | ' + ' '.join(f'{k:>7s}' for k in _DESCRIPTORS))
    table = []
    for r in sorted(runs, key=lambda r: (r['dataset'], r['optimizer'], r['width'])):
        row = {'run': f'{r["dataset"]}/{r["optimizer"]}_w{r["width"]}', 'C': r['C'], 'p': r['p'],
               'corr': {k: r['corr'].get(k, {}).get('pearson', float('nan')) for k in _DESCRIPTORS}}
        table.append(row)
        if r['dataset'] in args.datasets:
            print(f'{row["run"]:22s} {r["C"]:5d} {r["p"]:5d} | '
                  + ' '.join(f'{row["corr"][k]:7.2f}' for k in _DESCRIPTORS))
    with open(os.path.join(os.path.abspath(args.output_dir), 'left_descriptors_corr.json'), 'w',
              encoding='utf-8') as h:
        json.dump(table, h, indent=2)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
