#!/usr/bin/env python3
"""Compare classifier source/capacity power-law results across width / dataset / optimizer.

Scans a results root for ``*/powerlaw_arrays.npz`` (each produced by
``scripts/analyze_classifier_powerlaw.py``) plus its sibling ``fit_summary.json`` and
produces cross-run comparison figures and a summary CSV:

  * scalars vs width: capacity exponent b, mean source exponent a, corr(a, accuracy),
    overall val accuracy;
  * feature spectra overlaid by width (one panel per dataset/optimizer);
  * PCA-truncation recovery overlaid by width (one panel per dataset/optimizer).

Each (dataset, optimizer) is a series; widths are the x-axis / color ramp.
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


# Learning-rate schedule per (dataset, optimizer), for honest legends.
_SCHEDULE = {
    ('imagenet', 'sgd'): 'const lr 0.4',
    ('imagenet', 'muon'): 'cosine, peak 0.006',
    ('imagenet', 'adam'): 'const lr 0.00625',
    ('cifar5m', 'sgd'): 'const lr 0.4',
    ('cifar5m', 'muon'): 'const lr 0.006',
    ('cifar5m', 'adam'): 'const lr 0.00625',
}
_SERIES_COLOR = {
    ('imagenet', 'sgd'): 'tab:blue',
    ('imagenet', 'muon'): 'tab:orange',
    ('cifar5m', 'sgd'): 'tab:green',
    ('cifar5m', 'muon'): 'tab:red',
    ('imagenet', 'adam'): 'tab:purple',
    ('cifar5m', 'adam'): 'tab:brown',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default='artifacts/classifier_powerlaw',
                        help='Directory scanned recursively for powerlaw_arrays.npz.')
    parser.add_argument('--output-dir', default='artifacts/classifier_powerlaw/comparison')
    parser.add_argument('--format', choices=['pdf', 'png'], default='png')
    return parser.parse_args()


def _series_label(dataset: str, optimizer: str) -> str:
    sched = _SCHEDULE.get((dataset, optimizer), '')
    return f'{dataset}/{optimizer}' + (f' ({sched})' if sched else '')


def _load_runs(results_root: str) -> list[dict]:
    runs: list[dict] = []
    for npz_path in sorted(glob.glob(os.path.join(results_root, '**', 'powerlaw_arrays.npz'),
                                     recursive=True)):
        run_dir = os.path.dirname(npz_path)
        summary_path = os.path.join(run_dir, 'fit_summary.json')
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, encoding='utf-8') as handle:
            summary = json.load(handle)
        runs.append({'dir': run_dir, 'npz': npz_path, 'summary': summary})
    return runs


def _grouped(runs: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        s = run['summary']
        key = (str(s['dataset']), str(s['optimizer_key']))
        groups.setdefault(key, []).append(run)
    for key in groups:
        groups[key].sort(key=lambda r: int(r['summary']['width']))
    return dict(sorted(groups.items()))


def _save(fig, output_dir: str, stem: str, fmt: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{stem}.{fmt}')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return path


def _plot_scalars_vs_width(groups, output_dir, fmt) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    panels = [
        ('capacity_exponent_b', 'capacity exponent $b$', axes[0, 0]),
        ('source_exponent_mean', 'mean source exponent $\\bar a$', axes[0, 1]),
        ('corr_a_vs_accuracy_pearson', r'corr($a_i$, accuracy) [Pearson]', axes[1, 0]),
        ('overall_val_accuracy', 'overall val accuracy', axes[1, 1]),
    ]
    for key, runs in groups.items():
        dataset, optimizer = key
        widths = np.array([int(r['summary']['width']) for r in runs], dtype=float)
        color = _SERIES_COLOR.get(key, None)
        label = _series_label(dataset, optimizer)
        for field, _title, ax in panels:
            ys = np.array([float(r['summary'].get(field, np.nan)) for r in runs])
            if field == 'capacity_exponent_b':
                yerr = np.array([float(r['summary'].get('capacity_b_sem', np.nan)) for r in runs])
                ax.errorbar(widths, ys, yerr=yerr, marker='o', color=color, label=label,
                            capsize=3, elinewidth=1.2)
            else:
                ax.plot(widths, ys, marker='o', color=color, label=label)
            if field == 'source_exponent_mean':
                stds = np.array([float(r['summary'].get('source_exponent_std', np.nan)) for r in runs])
                ax.fill_between(widths, ys - stds, ys + stds, color=color, alpha=0.15)
    for field, title, ax in panels:
        ax.set_xscale('log', base=2)
        ax.set_xlabel('width (num_filters)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc='best')
    fig.suptitle('Source/capacity structure vs width', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, output_dir, 'compare_scalars_vs_width', fmt)


def _panel_grid(n: int) -> tuple[int, int]:
    ncols = min(2, n) if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _plot_spectra_by_width(groups, output_dir, fmt) -> str:
    n = len(groups)
    nrows, ncols = _panel_grid(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), squeeze=False)
    for idx, (key, runs) in enumerate(groups.items()):
        ax = axes[idx // ncols][idx % ncols]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(runs)))
        for run, color in zip(runs, colors):
            data = np.load(run['npz'], allow_pickle=True)
            eig = np.asarray(data['eigenvalues'], dtype=np.float64)
            k = np.arange(1, eig.size + 1)
            width = int(run['summary']['width'])
            b = float(run['summary'].get('capacity_exponent_b', np.nan))
            ax.loglog(k, np.clip(eig, 1e-30, None), color=color, linewidth=1.3,
                      label=f'w{width} (b={b:.2f})')
        ax.set_title(_series_label(*key), fontsize=10)
        ax.set_xlabel('PC index $k$')
        ax.set_ylabel(r'$\lambda_k$')
        ax.grid(True, which='both', alpha=0.2)
        ax.legend(fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    fig.suptitle('Feature spectra by width', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, output_dir, 'compare_feature_spectra_by_width', fmt)


def _plot_recovery_by_width(groups, output_dir, fmt) -> str:
    n = len(groups)
    nrows, ncols = _panel_grid(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), squeeze=False)
    for idx, (key, runs) in enumerate(groups.items()):
        ax = axes[idx // ncols][idx % ncols]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(runs)))
        for run, color in zip(runs, colors):
            data = np.load(run['npz'], allow_pickle=True)
            trunc_m = np.asarray(data['trunc_m'], dtype=np.int64)
            acc_by_m = np.asarray(data['acc_by_m'], dtype=np.float64)
            width = int(run['summary']['width'])
            ax.plot(trunc_m, acc_by_m, marker='o', markersize=3, color=color, label=f'w{width}')
        ax.set_xscale('log')
        ax.set_title(_series_label(*key), fontsize=10)
        ax.set_xlabel('retained PCs $m$')
        ax.set_ylabel('accuracy')
        ax.grid(True, which='both', alpha=0.2)
        ax.legend(fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    fig.suptitle('PCA-truncation recovery by width', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, output_dir, 'compare_truncation_recovery_by_width', fmt)


def _write_summary_csv(runs, output_dir) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fields = [
        'dataset', 'optimizer_key', 'width', 'source_run_id', 'images_seen',
        'num_samples', 'num_features', 'num_classes', 'bulk_k_lo', 'bulk_k_hi',
        'capacity_exponent_b', 'capacity_b_sem', 'capacity_b_std', 'capacity_log_rmse', 'source_exponent_mean',
        'source_exponent_std', 'source_exponent_sem', 'source_negative_count', 'overall_val_accuracy',
        'overall_val_cross_entropy', 'corr_a_vs_accuracy_pearson',
        'corr_a_vs_accuracy_spearman', 'corr_a_vs_ce_pearson',
        'classifier_recon_max_abs_diff',
    ]
    path = os.path.join(output_dir, 'comparison_summary.csv')
    rows = sorted(
        (r['summary'] for r in runs),
        key=lambda s: (str(s['dataset']), str(s['optimizer_key']), int(s['width'])),
    )
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for s in rows:
            writer.writerow(s)
    return path


def main() -> None:
    args = parse_args()
    runs = _load_runs(args.results_root)
    if not runs:
        raise SystemExit(f'No powerlaw_arrays.npz found under {args.results_root}.')
    groups = _grouped(runs)
    print(f'found {len(runs)} runs in {len(groups)} series:')
    for key, series in groups.items():
        widths = [int(r['summary']['width']) for r in series]
        print(f'  {_series_label(*key)}: widths {widths}')

    output_dir = os.path.abspath(args.output_dir)
    written = [
        _write_summary_csv(runs, output_dir),
        _plot_scalars_vs_width(groups, output_dir, args.format),
        _plot_spectra_by_width(groups, output_dir, args.format),
        _plot_recovery_by_width(groups, output_dir, args.format),
    ]
    for path in written:
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
