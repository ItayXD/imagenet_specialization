#!/usr/bin/env python3
"""Recompute the (post-processing) fits for saved classifier-powerlaw runs.

The expensive part (features, PCA, What, truncation) is already saved in
powerlaw_arrays.npz. This re-runs only the fit layer with the current
robust_capacity_fit / fit_source_exponents (e.g. after changing the fitting
method), updates the npz/CSVs/JSON in place, and re-renders the figures. No GPU or
data needed. Point --results-root at a tree of run dirs (recursive).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from scripts.analyze_classifier_powerlaw import (
    _pearson_spearman,
    _render_all_figures,
    fit_source_exponents,
    robust_capacity_fit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True,
                        help='Directory scanned recursively for powerlaw_arrays.npz.')
    parser.add_argument('--fit-seeds', type=int, default=64)
    parser.add_argument('--fit-rmse-tol', type=float, default=0.10)
    parser.add_argument('--rng-seed', type=int, default=2423)
    parser.add_argument('--format', choices=['pdf', 'png'], default='png')
    return parser.parse_args()


def refit_run(run_dir: str, *, fit_seeds: int, fit_rmse_tol: float, rng_seed: int, fmt: str) -> dict:
    npz_path = os.path.join(run_dir, 'powerlaw_arrays.npz')
    data = {k: v for k, v in np.load(npz_path, allow_pickle=True).items()}
    eig = np.asarray(data['eigenvalues'], dtype=np.float64)
    what_hat = np.asarray(data['W_hat'], dtype=np.float64)
    what_sq = what_hat ** 2
    num_bins = int(data['num_bins']) if 'num_bins' in data else 24
    num_classes = int(data['num_classes'])

    cap = robust_capacity_fit(eig, num_bins, n_seeds=fit_seeds, rng_seed=rng_seed,
                              rmse_tol=fit_rmse_tol)
    k_lo, k_hi = cap['k_lo'], cap['k_hi']
    source = fit_source_exponents(what_sq, num_bins, k_lo=k_lo, k_hi=k_hi)
    a = source['a']
    finite_a = a[np.isfinite(a)]

    acc_full = np.asarray(data['acc_full'], dtype=np.float64)
    ce_full = np.asarray(data['ce_full'], dtype=np.float64)
    class_acc_by_m = np.asarray(data['class_acc_by_m'], dtype=np.float64)
    trunc_m = np.asarray(data['trunc_m'], dtype=np.int64)
    tail_by_m = np.asarray(data['tail_by_m'], dtype=np.float64)
    acc_by_m = np.asarray(data['acc_by_m'], dtype=np.float64)
    ce_by_m = np.asarray(data['ce_by_m'], dtype=np.float64)

    r_acc, rho_acc = _pearson_spearman(a, acc_full)
    r_ce, rho_ce = _pearson_spearman(a, ce_full)

    low_group_acc = np.full(trunc_m.size, np.nan)
    high_group_acc = np.full(trunc_m.size, np.nan)
    finite_idx = np.where(np.isfinite(a))[0]
    if finite_idx.size >= 20:
        order = finite_idx[np.argsort(a[finite_idx])]
        dec = max(1, order.size // 10)
        low_group_acc = np.nanmean(class_acc_by_m[order[:dec]], axis=0)
        high_group_acc = np.nanmean(class_acc_by_m[order[-dec:]], axis=0)

    # Update the fit-dependent npz fields; keep everything else.
    data['a'] = a.astype(np.float64)
    data['log_A2'] = source['log_A2'].astype(np.float64)
    data['class_fit_rmse'] = source['log_rmse'].astype(np.float64)
    data['capacity_exponent_b'] = np.float64(cap['b'])
    data['capacity_b_sem'] = np.float64(cap['b_sem'])
    data['capacity_b_std'] = np.float64(cap['b_std'])
    data['capacity_intercept'] = np.float64(cap['intercept'])
    data['capacity_log_rmse'] = np.float64(cap['log_rmse'])
    data['capacity_seed_slopes'] = np.asarray(cap['seed_slopes'], dtype=np.float64)
    data['capacity_exclusion_frac'] = np.asarray(cap['exclusion_frac'], dtype=np.float64)
    data['bulk_k_lo'] = np.int64(k_lo)
    data['bulk_k_hi'] = np.int64(k_hi)
    np.savez_compressed(npz_path, **data)

    # per_class_metrics.csv
    m_ref = 64 if 64 in set(int(x) for x in trunc_m) else int(trunc_m[len(trunc_m) // 2])
    ref_col = int(np.where(trunc_m == m_ref)[0][0])
    acc_rank = np.asarray(data['acc_rank'], dtype=np.float64) if 'acc_rank' in data \
        else np.full(num_classes, np.nan)
    with open(os.path.join(run_dir, 'per_class_metrics.csv'), 'w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=['class_index', 'a_i', 'log_A2', 'source_fit_rmse',
                                          'acc_full', 'ce_full', 'acc_rank', f'tail_at_m{m_ref}'])
        w.writeheader()
        for c in range(num_classes):
            w.writerow({'class_index': c, 'a_i': float(a[c]), 'log_A2': float(source['log_A2'][c]),
                        'source_fit_rmse': float(source['log_rmse'][c]), 'acc_full': float(acc_full[c]),
                        'ce_full': float(ce_full[c]), 'acc_rank': float(acc_rank[c]),
                        f'tail_at_m{m_ref}': float(tail_by_m[c, ref_col])})

    # truncation_curve.csv
    with open(os.path.join(run_dir, 'truncation_curve.csv'), 'w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=['m', 'overall_acc', 'overall_ce',
                                          'low_a_group_acc', 'high_a_group_acc'])
        w.writeheader()
        for j, m in enumerate(trunc_m):
            w.writerow({'m': int(m), 'overall_acc': float(acc_by_m[j]), 'overall_ce': float(ce_by_m[j]),
                        'low_a_group_acc': float(low_group_acc[j]),
                        'high_a_group_acc': float(high_group_acc[j])})

    # fit_summary.json (update fit fields, keep the rest)
    summary_path = os.path.join(run_dir, 'fit_summary.json')
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding='utf-8') as h:
            summary = json.load(h)
    summary.update({
        'bulk_k_lo': int(k_lo), 'bulk_k_hi': int(k_hi),
        'capacity_exponent_b': float(cap['b']), 'capacity_b_sem': float(cap['b_sem']),
        'capacity_b_std': float(cap['b_std']),
        'capacity_log_rmse': float(cap['log_rmse']), 'capacity_fit_seeds': int(cap['n_seeds_used']),
        'source_exponent_mean': float(np.mean(finite_a)) if finite_a.size else float('nan'),
        'source_exponent_std': float(np.std(finite_a)) if finite_a.size else float('nan'),
        'source_exponent_sem': float(np.std(finite_a) / np.sqrt(finite_a.size)) if finite_a.size else float('nan'),
        'source_classes_fit': int(finite_a.size),
        'source_negative_count': int(np.sum(np.isfinite(a) & (a < 0))),
        'corr_a_vs_accuracy_pearson': r_acc, 'corr_a_vs_accuracy_spearman': rho_acc,
        'corr_a_vs_ce_pearson': r_ce, 'corr_a_vs_ce_spearman': rho_ce,
    })
    with open(summary_path, 'w', encoding='utf-8') as h:
        json.dump(summary, h, indent=2)

    _render_all_figures(data, run_dir, fmt=fmt)
    return {'run': run_dir, 'b': cap['b'], 'b_sem': cap['b_sem'], 'k_lo': k_lo, 'k_hi': k_hi,
            'rmse': cap['log_rmse'], 'a_mean': float(np.mean(finite_a)) if finite_a.size else float('nan')}


def main() -> None:
    args = parse_args()
    npz_paths = sorted(glob.glob(os.path.join(args.results_root, '**', 'powerlaw_arrays.npz'),
                                 recursive=True))
    if not npz_paths:
        raise SystemExit(f'No powerlaw_arrays.npz under {args.results_root}')
    for npz_path in npz_paths:
        run_dir = os.path.dirname(npz_path)
        res = refit_run(run_dir, fit_seeds=args.fit_seeds, fit_rmse_tol=args.fit_rmse_tol,
                        rng_seed=args.rng_seed, fmt=args.format)
        print(f'{os.path.relpath(run_dir, args.results_root):24s} '
              f'b={res["b"]:.3f}+/-{res["b_sem"]:.3f} (SEM) consensus=[{res["k_lo"]},{res["k_hi"]}] '
              f'rmse={res["rmse"]:.3f} a_mean={res["a_mean"]:.3f}')


if __name__ == '__main__':
    main()
