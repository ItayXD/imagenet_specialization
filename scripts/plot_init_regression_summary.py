#!/usr/bin/env python3
"""Cross-width / cross-lambda summary of the at-init ridge-regression power-law analysis.

Scans an init_regression_powerlaw output tree (dirs init_reg_w<width>_lam<rel>/, each with
fit_summary.json, svd/singular_powerlaw.json, truncation_curve.csv) and renders a single
multi-panel figure + a tidy CSV comparing:

  * capacity exponent b (feature covariance lambda_k ~ k^-b) vs width;
  * source exponent mean a_i vs ridge lambda, per width (crossing a=0 = source condition
    restored by regularization);
  * classifier singular-value exponent c (s_j ~ j^-c) vs ridge lambda, per width;
  * Spearman corr(a_i, class val accuracy) vs width, per lambda;
  * PCA-truncation val-accuracy recovery vs #PCs, per width (at one reference lambda);
  * generalization gap (train - val accuracy) vs ridge lambda, per width.

Pure post-processing of saved JSON/CSV; no GPU/data.
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

_DIR_RE = re.compile(r'init_reg_w(\d+)_lam([0-9.eE+-]+)$')


def _load_runs(root: str) -> list[dict]:
    runs = []
    for summary_path in sorted(glob.glob(os.path.join(root, 'init_reg_w*_lam*', 'fit_summary.json'))):
        run_dir = os.path.dirname(summary_path)
        m = _DIR_RE.search(os.path.basename(run_dir))
        if not m:
            continue
        d = json.load(open(summary_path))
        rec = {
            'width': int(d.get('width', int(m.group(1)))),
            'rel_lambda': float(d.get('ridge_rel_lambda', float(m.group(2)))),
            'num_features': int(d.get('num_features', 0)),
            'capacity_b': float(d.get('capacity_exponent_b', float('nan'))),
            'capacity_b_std': float(d.get('capacity_b_std', float('nan'))),
            'a_mean': float(d.get('source_exponent_mean', float('nan'))),
            'a_std': float(d.get('source_exponent_std', float('nan'))),
            'dof': float(d.get('ridge_effective_dof', float('nan'))),
            'train_acc': float(d.get('train_accuracy', float('nan'))),
            'val_acc': float(d.get('overall_val_accuracy', float('nan'))),
            'rho_a_acc': float(d.get('corr_a_vs_accuracy_spearman', float('nan'))),
            'r_a_acc': float(d.get('corr_a_vs_accuracy_pearson', float('nan'))),
        }
        sing = os.path.join(run_dir, 'svd', 'singular_powerlaw.json')
        rec['sing_c'] = float(json.load(open(sing)).get('c', float('nan'))) if os.path.exists(sing) else float('nan')
        trunc = os.path.join(run_dir, 'truncation_curve.csv')
        if os.path.exists(trunc):
            ms, accs = [], []
            for row in csv.DictReader(open(trunc)):
                ms.append(int(row['m']))
                accs.append(float(row['overall_acc']))
            rec['trunc_m'] = np.asarray(ms)
            rec['trunc_acc'] = np.asarray(accs)
        runs.append(rec)
    return runs


def _widths(runs):
    return sorted({r['width'] for r in runs})


def _lambdas(runs):
    return sorted({r['rel_lambda'] for r in runs})


def _by(runs, **f):
    return sorted([r for r in runs if all(r[k] == v for k, v in f.items())],
                  key=lambda r: (r['width'], r['rel_lambda']))


def _lambda_label(rel: float) -> str:
    return r'$\lambda_{\mathrm{rel}}=0$ (min-norm)' if rel == 0 else fr'$\lambda_{{\mathrm{{rel}}}}={rel:g}$'


def render(runs: list[dict], out_path: str, ref_lambda: float) -> None:
    widths = _widths(runs)
    lambdas = _lambdas(runs)
    wcolors = plt.cm.viridis(np.linspace(0.1, 0.9, len(widths)))
    wcolor = {w: c for w, c in zip(widths, wcolors)}
    lcolors = plt.cm.plasma(np.linspace(0.1, 0.85, len(lambdas)))
    lcolor = {l: c for l, c in zip(lambdas, lcolors)}

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))

    # (a) capacity exponent b vs width (b is lambda-independent -> use lambda=0 rows).
    ax = axes[0, 0]
    base = _by(runs, rel_lambda=0.0) or _by(runs, rel_lambda=lambdas[0])
    xs = [r['width'] for r in base]
    ax.errorbar(xs, [r['capacity_b'] for r in base], yerr=[r['capacity_b_std'] for r in base],
                marker='o', ms=6, lw=1.8, capsize=3, color='steelblue')
    ax.set_xscale('log', base=2)
    ax.set_xticks(widths); ax.set_xticklabels(widths)
    ax.set_xlabel('width (num_filters)')
    ax.set_ylabel(r'capacity exponent $b$  ($\lambda_k\sim k^{-b}$)')
    ax.set_title('(a) Feature-spectrum capacity vs width')
    ax.grid(True, which='both', alpha=0.25)

    # (b) source exponent mean vs ridge lambda, per width.
    ax = axes[0, 1]
    for w in widths:
        rr = _by(runs, width=w)
        xs = [max(r['rel_lambda'], 1e-4) for r in rr]  # 0 shown at 1e-4 on log axis
        ax.plot(xs, [r['a_mean'] for r in rr], marker='o', ms=5, lw=1.8,
                color=wcolor[w], label=f'w{w}')
    ax.axhline(0.0, color='0.4', ls='--', lw=1.2)
    ax.text(0.02, 0.03, 'a>0: source condition holds', transform=ax.transAxes, fontsize=8, color='0.3')
    ax.set_xscale('log')
    ax.set_xlabel(r'ridge $\lambda_{\mathrm{rel}}$ (0 plotted at $10^{-4}$)')
    ax.set_ylabel(r'mean source exponent $\bar a_i$')
    ax.set_title('(b) Source condition restored by ridge')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    # (c) singular-value exponent c vs ridge lambda, per width.
    ax = axes[0, 2]
    for w in widths:
        rr = _by(runs, width=w)
        xs = [max(r['rel_lambda'], 1e-4) for r in rr]
        ax.plot(xs, [r['sing_c'] for r in rr], marker='s', ms=5, lw=1.8,
                color=wcolor[w], label=f'w{w}')
    ax.set_xscale('log')
    ax.set_xlabel(r'ridge $\lambda_{\mathrm{rel}}$ (0 at $10^{-4}$)')
    ax.set_ylabel(r'singular-value exponent $c$  ($s_j\sim j^{-c}$)')
    ax.set_title('(c) Classifier singular spectrum vs ridge')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    # (d) Spearman corr(a, val accuracy) vs width, per lambda.
    ax = axes[1, 0]
    for l in lambdas:
        rr = _by(runs, rel_lambda=l)
        ax.plot([r['width'] for r in rr], [r['rho_a_acc'] for r in rr], marker='o', ms=5,
                lw=1.8, color=lcolor[l], label=_lambda_label(l))
    ax.set_xscale('log', base=2)
    ax.set_xticks(widths); ax.set_xticklabels(widths)
    ax.set_xlabel('width (num_filters)')
    ax.set_ylabel(r'Spearman $\rho(a_i,\ \mathrm{class\ acc})$')
    ax.set_title('(d) Source exponent predicts class accuracy')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=7)

    # (e) PCA-truncation recovery at the reference lambda, per width.
    ax = axes[1, 1]
    ref = ref_lambda if ref_lambda in lambdas else lambdas[len(lambdas) // 2]
    for w in widths:
        rr = _by(runs, width=w, rel_lambda=ref)
        if rr and 'trunc_m' in rr[0]:
            ax.plot(rr[0]['trunc_m'], rr[0]['trunc_acc'], marker='.', lw=1.8,
                    color=wcolor[w], label=f'w{w}')
    ax.set_xscale('log')
    ax.set_xlabel('retained PCs $m$')
    ax.set_ylabel('val accuracy')
    ax.set_title(f'(e) Truncation recovery ({_lambda_label(ref)})')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    # (f) generalization gap (train - val acc) vs ridge lambda, per width.
    ax = axes[1, 2]
    for w in widths:
        rr = _by(runs, width=w)
        xs = [max(r['rel_lambda'], 1e-4) for r in rr]
        ax.plot(xs, [r['train_acc'] - r['val_acc'] for r in rr], marker='o', ms=5, lw=1.8,
                color=wcolor[w], label=f'w{w}')
    ax.set_xscale('log')
    ax.set_xlabel(r'ridge $\lambda_{\mathrm{rel}}$ (0 at $10^{-4}$)')
    ax.set_ylabel('train acc $-$ val acc')
    ax.set_title('(f) Overfitting gap shrinks with ridge')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle('ResNet18 at init + ridge readout — capacity / source / SVD structure (ImageNet)',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def write_table(runs: list[dict], csv_path: str) -> None:
    cols = ['width', 'rel_lambda', 'num_features', 'dof', 'capacity_b', 'a_mean', 'a_std',
            'sing_c', 'train_acc', 'val_acc', 'r_a_acc', 'rho_a_acc']
    with open(csv_path, 'w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for r in sorted(runs, key=lambda r: (r['width'], r['rel_lambda'])):
            w.writerow({c: r.get(c, '') for c in cols})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='artifacts/init_regression_powerlaw/imagenet',
                   help='Dir containing init_reg_w<width>_lam<rel>/ subdirs.')
    p.add_argument('--output-dir', default='', help='Defaults to <root>/../summary.')
    p.add_argument('--ref-lambda', type=float, default=0.1,
                   help='Reference ridge lambda for the truncation-recovery panel.')
    p.add_argument('--format', choices=['png', 'pdf'], default='png')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs = _load_runs(args.root)
    if not runs:
        raise SystemExit(f'No runs found under {args.root}.')
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.root)), 'summary')
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f'init_regression_summary.{args.format}')
    csv_path = os.path.join(out_dir, 'init_regression_summary.csv')
    render(runs, fig_path, args.ref_lambda)
    write_table(runs, csv_path)
    print(f'{len(runs)} runs, widths={_widths(runs)}, lambdas={_lambdas(runs)}')
    print(f'wrote {fig_path}')
    print(f'wrote {csv_path}')


if __name__ == '__main__':
    main()
