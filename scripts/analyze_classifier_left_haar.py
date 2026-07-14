#!/usr/bin/env python3
"""Test whether the classifier's left singular vectors L look Haar-random over classes.

What = L S R^T. If classes are exchangeable/structureless in the singular basis, L (a
C x p orthonormal frame) should be indistinguishable from a uniformly random (Haar)
p-frame in R^C. We compare several statistics of the empirical L against a Monte-Carlo
Haar null (Q from the QR of a Gaussian C x p matrix):

  1. Entry marginal:      sqrt(C) * l_ij  vs  N(0,1)         (KS statistic).
  2. Row norm^2:          n_i = sum_j l_ij^2                 (Haar: Beta(p/2,(C-p)/2)).
  3. Participation ratio: PR_i = n_i^2 / sum_j l_ij^4        (effective #modes per class).
  4. Class energy:        E_i = sum_j s_j^2 l_ij^2 = ||What_i.||^2  (uses s; tests L-s alignment).
  5. Pairwise row cosine: spread of <u_i,u_j> over class pairs (semantic clustering).

For (2)-(5) the diagnostic is the ACROSS-CLASS spread (std): if the empirical spread sits
inside the Haar null band, classes are Haar-like; if it exceeds it, classes are genuinely
heterogeneous (real easy/hard structure). Energy/PR are also correlated with accuracy.

Runs locally on a saved powerlaw_arrays.npz. imagenet sgd w64 by default.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.analyze_classifier_powerlaw import _pearson_spearman  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True,
                        help='Directory containing powerlaw_arrays.npz.')
    parser.add_argument('--output-dir', default='',
                        help='Output dir. Defaults to <results-dir>/svd.')
    parser.add_argument('--n-null', type=int, default=200, help='Haar Monte-Carlo draws.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--format', choices=['pdf', 'png'], default='png')
    return parser.parse_args()


def _row_stats(left: np.ndarray, s2: np.ndarray) -> dict:
    """Per-class statistics from a (C, p) orthonormal frame and squared singular values."""
    l2 = left ** 2
    n_i = l2.sum(axis=1)                                   # row norm^2
    pr_i = np.where(np.sum(l2 ** 2, axis=1) > 0,
                    n_i ** 2 / np.sum(l2 ** 2, axis=1), np.nan)  # participation ratio
    e_i = l2 @ s2                                          # class energy sum_j s_j^2 l_ij^2
    return {'n': n_i, 'pr': pr_i, 'e': e_i}


def _pairwise_cosine_std(left: np.ndarray, rng: np.random.Generator, max_pairs: int = 200000) -> float:
    """Std of cosine similarity between random pairs of (unit-normalized) class rows."""
    norms = np.linalg.norm(left, axis=1, keepdims=True)
    unit = left / np.where(norms > 0, norms, 1.0)
    c = left.shape[0]
    i = rng.integers(0, c, size=max_pairs)
    j = rng.integers(0, c, size=max_pairs)
    keep = i != j
    cos = np.sum(unit[i[keep]] * unit[j[keep]], axis=1)
    return float(np.std(cos))


def _haar_frame(c: int, p: int, rng: np.random.Generator, orthogonal_to_ones: bool = True) -> np.ndarray:
    """Uniform (Haar) orthonormal C x p frame via QR of a Gaussian matrix.

    Softmax-gauge removal (row-centering What) forces 1^T What = 0, so the empirical left
    vectors lie exactly in the (C-1)-dim subspace orthogonal to the all-ones vector. The
    matched Haar null draws the frame in that same subspace (center the Gaussian columns
    before QR), otherwise the null is mis-specified by one constrained dimension.
    """
    g = rng.standard_normal((c, p))
    if orthogonal_to_ones:
        g -= g.mean(axis=0, keepdims=True)  # each column orthogonal to the all-ones vector
    q, r = np.linalg.qr(g)
    q *= np.sign(np.diag(r))  # proper Haar column signs
    return q


def _zscore(empirical: float, null_samples: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(null_samples))
    sd = float(np.std(null_samples))
    z = (empirical - mu) / sd if sd > 0 else float('nan')
    # Two-sided empirical p-value.
    p = float((np.sum(np.abs(null_samples - mu) >= abs(empirical - mu)) + 1) / (null_samples.size + 1))
    return z, p


def main() -> None:
    args = parse_args()
    npz_path = os.path.join(args.results_dir, 'powerlaw_arrays.npz')
    data = np.load(npz_path, allow_pickle=True)
    what_hat = np.asarray(data['W_hat'], dtype=np.float64)
    acc = np.asarray(data['acc_full'], dtype=np.float64) if 'acc_full' in data.files else None
    run_label = str(data['run_label']) if 'run_label' in data.files else 'classifier'
    output_dir = os.path.abspath(args.output_dir) if args.output_dir \
        else os.path.join(os.path.abspath(args.results_dir), 'svd')
    os.makedirs(output_dir, exist_ok=True)

    left, s, _ = np.linalg.svd(what_hat, full_matrices=False)  # left = L (C, p)
    c, p = left.shape
    s2 = s ** 2
    rng = np.random.default_rng(args.seed)

    emp = _row_stats(left, s2)
    emp_cos_std = _pairwise_cosine_std(left, rng)
    emp_entry = (np.sqrt(c) * left).reshape(-1)

    # Haar Monte-Carlo null.
    null_std_n, null_std_pr, null_std_e, null_cos_std = [], [], [], []
    entry_null_sample = []
    for t in range(int(args.n_null)):
        q = _haar_frame(c, p, rng)
        rs = _row_stats(q, s2)
        null_std_n.append(np.std(rs['n']))
        null_std_pr.append(np.nanstd(rs['pr']))
        null_std_e.append(np.std(rs['e']))
        if t < 20:
            null_cos_std.append(_pairwise_cosine_std(q, rng, max_pairs=50000))
        if t < 5:
            entry_null_sample.append((np.sqrt(c) * q).reshape(-1))
    null_std_n = np.asarray(null_std_n)
    null_std_pr = np.asarray(null_std_pr)
    null_std_e = np.asarray(null_std_e)
    null_cos_std = np.asarray(null_cos_std)

    # KS of entries vs N(0,1).
    try:
        from scipy.stats import kstest
        ks_stat, ks_p = kstest(emp_entry, 'norm')
        ks_stat, ks_p = float(ks_stat), float(ks_p)
    except Exception:
        ks_stat, ks_p = float('nan'), float('nan')

    z_n, p_n = _zscore(float(np.std(emp['n'])), null_std_n)
    z_pr, p_pr = _zscore(float(np.nanstd(emp['pr'])), null_std_pr)
    z_e, p_e = _zscore(float(np.std(emp['e'])), null_std_e)
    z_cos, p_cos = _zscore(emp_cos_std, null_cos_std)

    corr = {}
    if acc is not None:
        for key in ('n', 'pr', 'e'):
            r, rho = _pearson_spearman(emp[key], acc)
            corr[key] = {'pearson': r, 'spearman': rho}

    summary = {
        'run_label': run_label, 'num_classes': int(c), 'num_modes': int(p),
        'n_null': int(args.n_null),
        'entry_ks_vs_normal': {'stat': ks_stat, 'p': ks_p},
        'row_norm2': {'emp_mean': float(np.mean(emp['n'])), 'haar_mean': float(p / c),
                      'emp_across_class_std': float(np.std(emp['n'])),
                      'haar_std_mean': float(np.mean(null_std_n)), 'z': z_n, 'p': p_n},
        'participation_ratio': {'emp_mean': float(np.nanmean(emp['pr'])),
                                'emp_across_class_std': float(np.nanstd(emp['pr'])),
                                'haar_std_mean': float(np.mean(null_std_pr)), 'z': z_pr, 'p': p_pr},
        'class_energy': {'emp_mean': float(np.mean(emp['e'])),
                         'emp_across_class_std': float(np.std(emp['e'])),
                         'haar_std_mean': float(np.mean(null_std_e)), 'z': z_e, 'p': p_e},
        'pairwise_cosine_std': {'emp': emp_cos_std, 'haar_mean': float(np.mean(null_cos_std)),
                                'haar_1_over_sqrt_p': float(1.0 / np.sqrt(p)), 'z': z_cos, 'p': p_cos},
        'corr_with_accuracy': corr,
    }
    with open(os.path.join(output_dir, 'left_haar_test.json'), 'w', encoding='utf-8') as h:
        json.dump(summary, h, indent=2)

    # ---- Figures ----
    def _save(fig, stem):
        path = os.path.join(output_dir, f'{stem}.{args.format}')
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        return path

    written = []

    # 1. Entry marginal vs N(0,1).
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.hist(emp_entry, bins=120, density=True, alpha=0.6, color='steelblue', label=r'empirical $\sqrt{C}\,l_{ij}$')
    xs = np.linspace(-4, 4, 400)
    ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), 'k-', lw=2, label=r'$N(0,1)$ (Haar limit)')
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r'$\sqrt{C}\, l_{ij}$')
    ax.set_ylabel('density')
    ax.set_title(f'Left-vector entries vs Gaussian (KS={ks_stat:.3f}) — {run_label}')
    ax.legend()
    written.append(_save(fig, 'fig_haar_entry_marginal'))

    # 2-4. Empirical across-class spread vs Haar null band, for n_i, PR_i, E_i.
    panels = [('row norm$^2$ $n_i$', emp['n'], null_std_n, np.std(emp['n']), z_n, p_n),
              ('participation ratio $PR_i$', emp['pr'], null_std_pr, np.nanstd(emp['pr']), z_pr, p_pr),
              ('class energy $E_i$', emp['e'], null_std_e, np.std(emp['e']), z_e, p_e)]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for ax, (title, emp_vals, null_stds, emp_std, z, pval) in zip(axes, panels):
        ax.hist(null_stds, bins=30, color='0.7', alpha=0.9, label='Haar null (across-class std)')
        ax.axvline(emp_std, color='crimson', lw=2.5, label=f'empirical (z={z:.1f}, p={pval:.3f})')
        ax.set_xlabel(f'across-class std of {title}')
        ax.set_ylabel('Haar draws')
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle(f'Empirical class heterogeneity vs Haar null — {run_label}')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    written.append(_save(fig, 'fig_haar_heterogeneity'))

    # 5. Class energy vs accuracy (the easy/hard-relevant statistic).
    if acc is not None:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        valid = np.isfinite(emp['e']) & np.isfinite(acc)
        r, rho = _pearson_spearman(emp['e'][valid], acc[valid])
        ax.scatter(emp['e'][valid], acc[valid], s=8, alpha=0.5, color='indigo')
        ax.set_xlabel(r'class energy $E_i=\|\widehat W_{i\cdot}\|^2$')
        ax.set_ylabel('class accuracy')
        ax.set_title(fr'Class energy vs accuracy — Pearson $r={r:.2f}$, Spearman $\rho={rho:.2f}$')
        ax.grid(True, alpha=0.25)
        written.append(_save(fig, 'fig_haar_energy_vs_accuracy'))

    print(f'run={run_label} C={c} p={p} n_null={args.n_null}')
    print(f'entry KS vs N(0,1): stat={ks_stat:.4f} p={ks_p:.3g}')
    print(f'row-norm^2:  emp_std={np.std(emp["n"]):.4g} haar={np.mean(null_std_n):.4g} z={z_n:.2f} p={p_n:.3f}')
    print(f'part.ratio:  emp_std={np.nanstd(emp["pr"]):.4g} haar={np.mean(null_std_pr):.4g} z={z_pr:.2f} p={p_pr:.3f}')
    print(f'class energy:emp_std={np.std(emp["e"]):.4g} haar={np.mean(null_std_e):.4g} z={z_e:.2f} p={p_e:.3f}')
    print(f'pair cosine: emp={emp_cos_std:.4g} haar={np.mean(null_cos_std):.4g} (1/sqrt(p)={1/np.sqrt(p):.4g}) z={z_cos:.2f}')
    if corr:
        print(f'corr(E_i, acc): pearson={corr["e"]["pearson"]:.3f} spearman={corr["e"]["spearman"]:.3f}')
        print(f'corr(PR_i,acc): pearson={corr["pr"]["pearson"]:.3f}  corr(n_i,acc): pearson={corr["n"]["pearson"]:.3f}')
    for path in [os.path.join(output_dir, 'left_haar_test.json'), *written]:
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
