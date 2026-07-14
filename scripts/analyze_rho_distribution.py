#!/usr/bin/env python3
"""Distribution of the per-class spectral location rho_i = <j>_i and rho_i/p.

rho_i = sum_j j * p_ij with p_ij = l_ij^2 / n_i (the mean singular-mode rank of class i's
left mass; What = L S R^T). We fit lognormal to rho_i and Beta to rho_i/p in (0,1), report
KS goodness, and check whether the distribution is consistent across widths and optimizers.
imagenet only (cifar p=10 is degenerate). Reads saved powerlaw_arrays.npz.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402


def _rho(what_hat: np.ndarray) -> tuple[np.ndarray, int]:
    left, _s, _ = np.linalg.svd(np.asarray(what_hat, dtype=np.float64), full_matrices=False)
    c, p = left.shape
    l2 = left ** 2
    n = l2.sum(axis=1)
    j = np.arange(1, p + 1, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.where(n > 0, (l2 @ j) / n, np.nan)
    return rho[np.isfinite(rho)], p


def _fits(rho: np.ndarray, p: int) -> dict:
    x = rho / p
    x = np.clip(x, 1e-6, 1 - 1e-6)
    out = {'mean_rho_over_p': float(np.mean(rho / p)), 'std_rho_over_p': float(np.std(rho / p)),
           'cv': float(np.std(rho) / np.mean(rho)),
           'skew_rho': float(sps.skew(rho)), 'skew_log_rho': float(sps.skew(np.log(rho)))}
    # lognormal on rho
    shp, loc, scale = sps.lognorm.fit(rho, floc=0)
    out['lognorm_sigma'] = float(shp)
    out['lognorm_mu'] = float(np.log(scale))
    out['ks_lognorm'] = float(sps.kstest(rho, 'lognorm', args=(shp, 0, scale)).pvalue)
    # Beta on rho/p
    a, b, _, _ = sps.beta.fit(x, floc=0, fscale=1)
    out['beta_a'] = float(a)
    out['beta_b'] = float(b)
    out['ks_beta'] = float(sps.kstest(x, 'beta', args=(a, b, 0, 1)).pvalue)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-root', default='artifacts/classifier_powerlaw')
    ap.add_argument('--datasets', nargs='*', default=['imagenet'])
    ap.add_argument('--output-dir', default='artifacts/classifier_powerlaw/comparison')
    ap.add_argument('--format', choices=['pdf', 'png'], default='png')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runs = []
    for npz in sorted(glob.glob(os.path.join(args.results_root, '**', 'powerlaw_arrays.npz'),
                                recursive=True)):
        d = np.load(npz, allow_pickle=True)
        if 'W_hat' not in d.files:
            continue
        s = json.load(open(os.path.join(os.path.dirname(npz), 'fit_summary.json'))) \
            if os.path.exists(os.path.join(os.path.dirname(npz), 'fit_summary.json')) else {}
        ds, opt, w = str(s.get('dataset', 'na')), str(s.get('optimizer_key', 'na')), int(s.get('width', 0))
        if ds not in args.datasets:
            continue
        rho, p = _rho(np.asarray(d['W_hat'], dtype=np.float64))
        runs.append({'dataset': ds, 'optimizer': opt, 'width': w, 'p': p,
                     'rho': rho, 'fits': _fits(rho, p)})
    if not runs:
        raise SystemExit('no imagenet runs found')
    runs.sort(key=lambda r: (r['optimizer'], r['width']))
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)

    optimizers = sorted({r['optimizer'] for r in runs})

    # Fig 1: consistency of rho/p across widths, one panel per optimizer, + standardized overlay.
    fig, axes = plt.subplots(2, len(optimizers), figsize=(6.2 * len(optimizers), 8.4), squeeze=False)
    for oi, opt in enumerate(optimizers):
        rs = [r for r in runs if r['optimizer'] == opt]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rs)))
        ax = axes[0][oi]
        for r, col in zip(rs, colors):
            x = r['rho'] / r['p']
            ax.hist(x, bins=40, density=True, histtype='step', color=col, lw=1.8,
                    label=f"w{r['width']} (a={r['fits']['beta_a']:.0f},b={r['fits']['beta_b']:.0f})")
        ax.axvline(0.5, color='0.5', ls=':')
        ax.set_xlabel(r'$\rho_i/p=\langle j\rangle_i/p$'); ax.set_ylabel('density')
        ax.set_title(f'{opt}: $\\rho_i/p$ across widths'); ax.legend(fontsize=7)
        # standardized log(rho): overlay to check a common shape family
        ax = axes[1][oi]
        for r, col in zip(rs, colors):
            z = (np.log(r['rho']) - np.log(r['rho']).mean()) / np.log(r['rho']).std()
            ax.hist(z, bins=40, density=True, histtype='step', color=col, lw=1.6, label=f"w{r['width']}")
        xs = np.linspace(-4, 4, 200)
        ax.plot(xs, sps.norm.pdf(xs), 'k--', lw=1.5, label='N(0,1)')
        ax.set_xlabel(r'standardized $\log\rho_i$'); ax.set_ylabel('density')
        ax.set_title(f'{opt}: shape of $\\log\\rho_i$ (standardized)'); ax.legend(fontsize=7)
    fig.suptitle('Consistency of the spectral-location distribution across width', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = os.path.join(os.path.abspath(args.output_dir), f'rho_distribution_consistency.{args.format}')
    fig.savefig(p1, bbox_inches='tight', dpi=200); plt.close(fig)

    # Fig 2: fitted parameters vs width.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for opt in optimizers:
        rs = [r for r in runs if r['optimizer'] == opt]
        ws = [r['width'] for r in rs]
        axes[0].plot(ws, [r['fits']['beta_a'] for r in rs], marker='o', label=f'{opt} a')
        axes[0].plot(ws, [r['fits']['beta_b'] for r in rs], marker='s', ls='--', label=f'{opt} b')
        axes[1].plot(ws, [r['fits']['lognorm_sigma'] for r in rs], marker='o', label=opt)
        axes[2].plot(ws, [r['fits']['std_rho_over_p'] for r in rs], marker='o', label=opt)
    axes[0].set_title(r'Beta$(a,b)$ of $\rho_i/p$'); axes[0].set_ylabel('param')
    axes[1].set_title(r'lognormal $\sigma$ of $\rho_i$')
    axes[2].set_title(r'std$(\rho_i/p)$')
    for ax in axes:
        ax.set_xscale('log', base=2); ax.set_xlabel('width'); ax.grid(True, which='both', alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle('Location distribution: fitted parameters vs width', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p2 = os.path.join(os.path.abspath(args.output_dir), f'rho_distribution_params.{args.format}')
    fig.savefig(p2, bbox_inches='tight', dpi=200); plt.close(fig)

    print(f'{"run":20s} {"p":>5s} {"mean":>6s} {"std":>6s} | {"beta_a":>7s} {"beta_b":>7s} {"KSbeta":>7s} '
          f'| {"logN_s":>7s} {"KSlogN":>7s} {"skewR":>6s} {"skewLR":>6s}')
    for r in runs:
        f = r['fits']
        print(f"{r['optimizer']+'_w'+str(r['width']):20s} {r['p']:5d} {f['mean_rho_over_p']:6.3f} "
              f"{f['std_rho_over_p']:6.3f} | {f['beta_a']:7.1f} {f['beta_b']:7.1f} {f['ks_beta']:7.3f} "
              f"| {f['lognorm_sigma']:7.3f} {f['ks_lognorm']:7.3f} {f['skew_rho']:6.2f} {f['skew_log_rho']:6.2f}")
    with open(os.path.join(os.path.abspath(args.output_dir), 'rho_distribution.json'), 'w', encoding='utf-8') as h:
        json.dump([{'run': f"{r['optimizer']}_w{r['width']}", **r['fits']} for r in runs], h, indent=2)
    print(f'wrote {p1}\nwrote {p2}')


if __name__ == '__main__':
    main()
