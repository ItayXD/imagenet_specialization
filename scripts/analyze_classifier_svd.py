#!/usr/bin/env python3
"""SVD (singular-mode) view of the rotated classifier's source structure.

The per-class analysis (analyze_classifier_powerlaw.py) rotates the gauge-fixed
classifier into the feature-PCA basis, What = W V, and fits each *class* row
What_ik^2 ~ k^{-2 a_i}. This script instead decomposes What by its own SVD,

    What_ik = sum_j l_ij s_j r_jk        (What = L S R^T)

and characterizes each *singular mode* j: its right-singular-vector profile over the
PC index k, r_jk^2 ~ k^{-2 a_j} (fit over the same bulk window), giving a per-mode
exponent a_j, alongside the singular-value spectrum s_j ~ j^{-c}.

Runs locally on a saved powerlaw_arrays.npz (which stores What and the bulk window);
no GPU/data needed.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from scripts.analyze_classifier_powerlaw import (  # noqa: E402
    _pearson_spearman,
    fit_source_exponents,
    robust_capacity_fit,
    robust_loglog_slope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True,
                        help='Directory containing powerlaw_arrays.npz.')
    parser.add_argument('--output-dir', default='',
                        help='Figure/array output dir. Defaults to <results-dir>/svd.')
    parser.add_argument('--num-bins', type=int, default=0,
                        help='Log-spaced bins for the fits; 0 reuses the value in the npz.')
    parser.add_argument('--example-modes', type=int, nargs='*', default=None,
                        help='Mode indices j (1-based) to draw in the r_jk^2 figure. '
                             'Default is a log-spaced selection.')
    parser.add_argument('--format', choices=['pdf', 'png'], default='png')
    return parser.parse_args()


def _default_example_modes(num_modes: int) -> list[int]:
    candidates = [1, 4, 16, 64, 256, 512]
    return sorted({m for m in candidates if 1 <= m <= num_modes})


def compute_svd_source(what_hat: np.ndarray, num_bins: int, k_lo: int, k_hi: int) -> dict:
    """SVD of What: per-mode right-vector exponents a_j, per-class left-vector exponents
    c_i, and the singular-value spectrum s_j ~ j^{-c}."""
    what_hat = np.asarray(what_hat, dtype=np.float64)
    # What = L S R^T; rows of vt are the right singular vectors r_j over PC index k;
    # columns of u are the left singular vectors l_j over class index i.
    u, singular_values, vt = np.linalg.svd(what_hat, full_matrices=False)
    num_modes, num_features = vt.shape
    num_classes = u.shape[0]
    positions_k = np.arange(1, num_features + 1, dtype=np.float64)
    positions_j = np.arange(1, num_modes + 1, dtype=np.float64)
    right_sq = vt ** 2          # (p, D); r_jk^2
    left_sq = u ** 2            # (C, p); l_ij^2

    a_j = np.full(num_modes, np.nan)
    log_amp_j = np.full(num_modes, np.nan)
    rmse_j = np.full(num_modes, np.nan)
    for j in range(num_modes):
        fit = robust_loglog_slope(positions_k, right_sq[j], num_bins, k_lo=k_lo, k_hi=k_hi)
        a_j[j] = -0.5 * fit['slope']  # r_jk^2 ~ k^{-2 a_j}
        log_amp_j[j] = fit['intercept']
        rmse_j[j] = fit['log_rmse']

    # Singular-value spectrum s_j ~ j^{-c}, robust seed-and-grow over the mode axis.
    sj_cap = robust_capacity_fit(singular_values, num_bins)
    sj_lo, sj_hi = sj_cap['k_lo'], sj_cap['k_hi']

    # Per-class left-vector exponents c_i from l_ij^2 ~ B_i^2 j^{-2 c_i}, fit over the same
    # singular-mode bulk window (the meaningful, non-cliff modes).
    left = fit_source_exponents(left_sq, num_bins, k_lo=sj_lo, k_hi=sj_hi)

    return {
        'singular_values': singular_values,
        'right_sq': right_sq,
        'left_sq': left_sq,
        'num_classes': int(num_classes),
        'a_j': a_j,
        'log_amp_j': log_amp_j,
        'rmse_j': rmse_j,
        'c_i': left['a'],           # l_ij^2 ~ j^{-2 c_i}
        'log_B2_i': left['log_A2'],
        'c_rmse_i': left['log_rmse'],
        'k_lo': int(k_lo),
        'k_hi': int(k_hi),
        'sj_exponent_c': float(sj_cap['b']),
        'sj_intercept': float(sj_cap['intercept']),
        'sj_k_lo': int(sj_lo),
        'sj_k_hi': int(sj_hi),
    }


def _render(result: dict, run_label: str, example_modes: list[int], output_dir: str,
            fmt: str, class_accuracy: np.ndarray | None = None) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    s = result['singular_values']
    right_sq = result['right_sq']
    a_j = result['a_j']
    log_amp_j = result['log_amp_j']
    num_modes, num_features = right_sq.shape
    positions_k = np.arange(1, num_features + 1, dtype=np.float64)
    positions_j = np.arange(1, num_modes + 1, dtype=np.float64)
    k_lo, k_hi = result['k_lo'], result['k_hi']
    bulk_k = np.arange(k_lo, k_hi + 1, dtype=np.float64)
    written: list[str] = []

    def _save(fig, stem):
        path = os.path.join(output_dir, f'{stem}.{fmt}')
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        written.append(path)

    def _mark_bulk(ax):
        if k_lo > 1:
            ax.axvspan(0.9, k_lo, color='0.85', alpha=0.5, zorder=0)
        if k_hi < num_features:
            ax.axvspan(k_hi, num_features * 1.05, color='0.85', alpha=0.5, zorder=0)

    # 0. Heatmaps of r_jk^2 (rows j sorted by singular value, cols k sorted by eigenvalue).
    xt = max(1, num_features // 8)
    yt = max(1, num_modes // 8)
    for scale in ('linear', 'log'):
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        if scale == 'log':
            positive = right_sq[right_sq > 0]
            vmin = float(np.quantile(positive, 0.02)) if positive.size else 1e-12
            vmax = float(right_sq.max())
            norm = LogNorm(vmin=max(vmin, vmax * 1e-8), vmax=vmax)
            sns.heatmap(right_sq, cmap='magma', norm=norm, ax=ax,
                        xticklabels=xt, yticklabels=yt,
                        cbar_kws={'label': r'$r_{jk}^2$ (log color)'})
        else:
            vmax = float(np.quantile(right_sq, 0.999))
            sns.heatmap(right_sq, cmap='magma', vmin=0.0, vmax=vmax, ax=ax,
                        xticklabels=xt, yticklabels=yt,
                        cbar_kws={'label': r'$r_{jk}^2$'})
        ax.set_xlabel(r'PC index $k$ (sorted by eigenvalue $\lambda_k$)')
        ax.set_ylabel(r'singular mode $j$ (sorted by singular value $s_j$)')
        ax.set_title(f'$r_{{jk}}^2$ heatmap ({scale} color) — {run_label}')
        _save(fig, f'fig_svd_r2_heatmap_{scale}')

    # 1. Right-singular-vector profiles r_jk^2 vs k for example modes (fig2 analog).
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    _mark_bulk(ax)
    colors = plt.cm.viridis(np.linspace(0.1, 0.88, len(example_modes)))
    for mode, color in zip(example_modes, colors):
        j = int(mode) - 1
        ax.loglog(positions_k, np.clip(right_sq[j], 1e-30, None), marker='.', linestyle='none',
                  markersize=3, alpha=0.4, color=color,
                  label=f'j={mode} ($s={s[j]:.2g}$, $a_j={a_j[j]:.2f}$)')
        if np.isfinite(a_j[j]) and np.isfinite(log_amp_j[j]):
            ax.loglog(bulk_k, np.exp(log_amp_j[j]) * bulk_k ** (-2.0 * a_j[j]),
                      color=color, linewidth=2.2)
    ax.set_xlabel('PC index $k$')
    ax.set_ylabel(r'$r_{jk}^2$ (right singular vector)')
    ax.set_title(f'Singular-mode source profiles — {run_label}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, 'fig_svd_right_vector_profiles')

    # 2. a_j vs j.
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    finite = np.isfinite(a_j)
    ax.plot(positions_j[finite], a_j[finite], marker='.', linestyle='-', markersize=4,
            color='darkorange', alpha=0.8)
    ax.axhline(0.0, color='0.5', linewidth=0.8, linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('singular mode index $j$')
    ax.set_ylabel(r'source exponent $a_j$')
    ax.set_title(f'Per-mode exponent $a_j$ vs $j$ — {run_label}')
    ax.grid(True, which='both', alpha=0.25)
    _save(fig, 'fig_svd_a_vs_mode')

    # 3. Singular-value spectrum s_j vs j (fig1 analog).
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    sj_lo, sj_hi = result['sj_k_lo'], result['sj_k_hi']
    if sj_lo > 1:
        ax.axvspan(0.9, sj_lo, color='0.85', alpha=0.5, zorder=0)
    if sj_hi < num_modes:
        ax.axvspan(sj_hi, num_modes * 1.05, color='0.85', alpha=0.5, zorder=0)
    ax.loglog(positions_j, np.clip(s, 1e-30, None), marker='.', linestyle='none',
              markersize=3, alpha=0.7, label=r'$s_j$')
    c = result['sj_exponent_c']
    if np.isfinite(c):
        bulk_j = np.arange(sj_lo, sj_hi + 1, dtype=np.float64)
        ax.loglog(bulk_j, np.exp(result['sj_intercept']) * bulk_j ** (-c), color='crimson',
                  linewidth=2.4, label=fr'bulk fit $j^{{-c}}$, $c={c:.2f}$')
    ax.set_xlabel('singular mode index $j$')
    ax.set_ylabel(r'singular value $s_j$')
    ax.set_title(f'Classifier singular-value spectrum — {run_label}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    _save(fig, 'fig_svd_singular_values')

    # ---- Left singular vectors l_ij^2 : are classes the same, or easy/hard? ----
    left_sq = np.asarray(result['left_sq'], dtype=np.float64)   # (C, p)
    c_i = np.asarray(result['c_i'], dtype=np.float64)
    log_b2 = np.asarray(result['log_B2_i'], dtype=np.float64)
    c_rmse = np.asarray(result['c_rmse_i'], dtype=np.float64)
    num_classes = left_sq.shape[0]
    sj_lo, sj_hi = result['sj_k_lo'], result['sj_k_hi']
    bulk_j = np.arange(sj_lo, sj_hi + 1, dtype=np.float64)

    # Rows sorted by c_i so any easy/hard gradient across classes is visible.
    order = np.argsort(np.where(np.isfinite(c_i), c_i, np.inf))
    left_sorted = left_sq[order]
    xt_j = max(1, num_modes // 8)
    yt_i = max(1, num_classes // 8)
    for scale in ('linear', 'log'):
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        if scale == 'log':
            positive = left_sorted[left_sorted > 0]
            vmin = float(np.quantile(positive, 0.02)) if positive.size else 1e-12
            vmax = float(left_sorted.max())
            sns.heatmap(left_sorted, cmap='magma', norm=LogNorm(vmin=max(vmin, vmax * 1e-8), vmax=vmax),
                        ax=ax, xticklabels=xt_j, yticklabels=yt_i,
                        cbar_kws={'label': r'$l_{ij}^2$ (log color)'})
        else:
            sns.heatmap(left_sorted, cmap='magma', vmin=0.0, vmax=float(np.quantile(left_sorted, 0.999)),
                        ax=ax, xticklabels=xt_j, yticklabels=yt_i,
                        cbar_kws={'label': r'$l_{ij}^2$'})
        ax.set_xlabel(r'singular mode $j$ (sorted by singular value $s_j$)')
        ax.set_ylabel(r'class $i$ (sorted by exponent $c_i$)')
        ax.set_title(f'$l_{{ij}}^2$ heatmap ({scale} color) — {run_label}')
        _save(fig, f'fig_svd_l2_heatmap_{scale}')

    # Example class left-vector profiles l_ij^2 vs j with fits (are classes the same?).
    finite_c = np.where(np.isfinite(c_i))[0]
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    if sj_lo > 1:
        ax.axvspan(0.9, sj_lo, color='0.85', alpha=0.5, zorder=0)
    if sj_hi < num_modes:
        ax.axvspan(sj_hi, num_modes * 1.05, color='0.85', alpha=0.5, zorder=0)
    if finite_c.size:
        ranked = finite_c[np.argsort(c_i[finite_c])]
        examples = {'smallest $c_i$': int(ranked[0]),
                    'median $c_i$': int(ranked[ranked.size // 2]),
                    'largest $c_i$': int(ranked[-1])}
        colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(examples)))
        for (label, ci), color in zip(examples.items(), colors):
            ax.loglog(positions_j, np.clip(left_sq[ci], 1e-30, None), marker='.', linestyle='none',
                      markersize=3, alpha=0.4, color=color,
                      label=f'class {ci} ({label}={c_i[ci]:.2f})')
            if np.isfinite(c_i[ci]) and np.isfinite(log_b2[ci]):
                ax.loglog(bulk_j, np.exp(log_b2[ci]) * bulk_j ** (-2.0 * c_i[ci]),
                          color=color, linewidth=2.2)
    ax.set_xlabel('singular mode index $j$')
    ax.set_ylabel(r'$l_{ij}^2$ (left singular vector)')
    ax.set_title(f'Example class left-vector profiles — {run_label}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, 'fig_svd_left_vector_profiles')

    # Distribution of per-class exponents c_i and the fit quality (RMSE).
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    finite_ci = c_i[np.isfinite(c_i)]
    axes[0].hist(finite_ci, bins=40, color='steelblue', alpha=0.85)
    if finite_ci.size:
        axes[0].axvline(float(np.median(finite_ci)), color='crimson', linestyle='--',
                        label=f'median={np.median(finite_ci):.2f}')
        axes[0].legend()
    axes[0].set_xlabel(r'per-class left exponent $c_i$')
    axes[0].set_ylabel('number of classes')
    axes[0].set_title(fr'Distribution of $c_i$ (std={np.std(finite_ci):.2f})')
    axes[0].grid(True, alpha=0.25)
    finite_rmse = c_rmse[np.isfinite(c_rmse)]
    axes[1].hist(finite_rmse, bins=40, color='seagreen', alpha=0.85)
    if finite_rmse.size:
        axes[1].axvline(float(np.median(finite_rmse)), color='crimson', linestyle='--',
                        label=f'median={np.median(finite_rmse):.3f}')
        axes[1].legend()
    axes[1].set_xlabel('per-class fit log-RMSE')
    axes[1].set_ylabel('number of classes')
    axes[1].set_title('Left-vector fit quality')
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(f'Per-class left-vector power-law fits — {run_label}')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, 'fig_svd_c_distribution')

    # c_i vs class accuracy (easy vs hard), if accuracy is available.
    if class_accuracy is not None:
        acc = np.asarray(class_accuracy, dtype=np.float64)
        valid = np.isfinite(c_i) & np.isfinite(acc)
        if valid.sum() >= 3:
            r, rho = _pearson_spearman(c_i[valid], acc[valid])
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            ax.scatter(c_i[valid], acc[valid], s=8, alpha=0.5, color='indigo')
            ax.set_xlabel(r'per-class left exponent $c_i$')
            ax.set_ylabel('class accuracy')
            ax.set_title(fr'$c_i$ vs accuracy — Pearson $r={r:.2f}$, Spearman $\rho={rho:.2f}$')
            ax.grid(True, alpha=0.25)
            _save(fig, 'fig_svd_c_vs_accuracy')

    return written


def main() -> None:
    args = parse_args()
    npz_path = os.path.join(args.results_dir, 'powerlaw_arrays.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'Missing {npz_path}.')
    data = np.load(npz_path, allow_pickle=True)
    what_hat = np.asarray(data['W_hat'], dtype=np.float64)
    k_lo = int(data['bulk_k_lo']) if 'bulk_k_lo' in data.files else 1
    k_hi = int(data['bulk_k_hi']) if 'bulk_k_hi' in data.files else what_hat.shape[1]
    num_bins = int(args.num_bins) if args.num_bins > 0 else (
        int(data['num_bins']) if 'num_bins' in data.files else 24)
    run_label = str(data['run_label']) if 'run_label' in data.files else 'classifier_svd'

    result = compute_svd_source(what_hat, num_bins, k_lo, k_hi)
    num_modes = result['singular_values'].size
    example_modes = args.example_modes or _default_example_modes(num_modes)

    output_dir = os.path.abspath(args.output_dir) if args.output_dir \
        else os.path.join(os.path.abspath(args.results_dir), 'svd')
    os.makedirs(output_dir, exist_ok=True)

    class_accuracy = np.asarray(data['acc_full'], dtype=np.float64) if 'acc_full' in data.files else None
    finite_a = result['a_j'][np.isfinite(result['a_j'])]
    c_i = np.asarray(result['c_i'], dtype=np.float64)
    finite_c = c_i[np.isfinite(c_i)]
    c_rmse = np.asarray(result['c_rmse_i'], dtype=np.float64)
    finite_c_rmse = c_rmse[np.isfinite(c_rmse)]
    num_classes = result['num_classes']
    print(f'run={run_label} modes={num_modes} classes={num_classes} num_bins={num_bins}')
    print(f'singular-value exponent c={result["sj_exponent_c"]:.4f} '
          f'(bulk j=[{result["sj_k_lo"]},{result["sj_k_hi"]}])')
    print(f'a_j (right/per-mode): mean={finite_a.mean():.4f} std={finite_a.std():.4f}')
    print(f'c_i (left/per-class): mean={finite_c.mean():.4f} std={finite_c.std():.4f} '
          f'min={finite_c.min():.4f} max={finite_c.max():.4f} ({finite_c.size}/{num_classes} fit)')
    print(f'left fit quality: median log-RMSE={np.median(finite_c_rmse):.4f} '
          f'(p90={np.quantile(finite_c_rmse, 0.9):.4f})')
    if class_accuracy is not None:
        r, rho = _pearson_spearman(c_i, class_accuracy)
        print(f'corr(c_i, accuracy): pearson={r:.3f} spearman={rho:.3f}')

    # Save arrays + per-mode / per-class CSVs.
    np.savez_compressed(
        os.path.join(output_dir, 'svd_arrays.npz'),
        singular_values=result['singular_values'],
        a_j=result['a_j'],
        log_amp_j=result['log_amp_j'],
        rmse_j=result['rmse_j'],
        c_i=result['c_i'],
        log_B2_i=result['log_B2_i'],
        c_rmse_i=result['c_rmse_i'],
        sj_exponent_c=np.float64(result['sj_exponent_c']),
        bulk_k_lo=np.int64(k_lo), bulk_k_hi=np.int64(k_hi),
        sj_k_lo=np.int64(result['sj_k_lo']), sj_k_hi=np.int64(result['sj_k_hi']),
        run_label=run_label,
    )
    csv_path = os.path.join(output_dir, 'svd_per_mode.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['mode_j', 'singular_value', 'a_j',
                                                    'log_amp', 'fit_rmse'])
        writer.writeheader()
        for j in range(num_modes):
            writer.writerow({
                'mode_j': j + 1,
                'singular_value': float(result['singular_values'][j]),
                'a_j': float(result['a_j'][j]),
                'log_amp': float(result['log_amp_j'][j]),
                'fit_rmse': float(result['rmse_j'][j]),
            })
    class_csv = os.path.join(output_dir, 'svd_per_class.csv')
    with open(class_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['class_index', 'c_i', 'log_B2',
                                                    'fit_rmse', 'acc_full'])
        writer.writeheader()
        for i in range(num_classes):
            writer.writerow({
                'class_index': i,
                'c_i': float(result['c_i'][i]),
                'log_B2': float(result['log_B2_i'][i]),
                'fit_rmse': float(result['c_rmse_i'][i]),
                'acc_full': float(class_accuracy[i]) if class_accuracy is not None else float('nan'),
            })

    written = _render(result, run_label, example_modes, output_dir, args.format,
                      class_accuracy=class_accuracy)
    for path in [os.path.join(output_dir, 'svd_arrays.npz'), csv_path, class_csv, *written]:
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
