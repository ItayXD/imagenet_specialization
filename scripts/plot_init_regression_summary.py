#!/usr/bin/env python3
"""Cross-width / cross-lambda / variant summary of the at-init ridge-regression analysis.

Scans an init_regression_powerlaw output tree (dirs init_reg_w<width>_lam<rel>[_<tags>], each
with fit_summary.json, svd/singular_powerlaw.json, truncation_curve.csv) and renders one
multi-panel figure + a tidy CSV. Runs are keyed by (width, variant) where variant combines the
normalization and pooling read from fit_summary.json:

  bn-gap  = BatchNorm + global-average-pool   (baseline)
  ln-gap  = LayerNorm (GroupNorm 1-group) + GAP
  bn-vec  = BatchNorm + flattened spatial map (CNN-VEC)
  ...

Color encodes width; linestyle+marker encode the variant. Panels:
  (a) capacity exponent b vs width;      (b) source exponent mean a_i vs ridge;
  (c) singular-value exponent c vs ridge;(d) Spearman corr(a_i, class acc) vs ridge;
  (e) PCA-truncation val-accuracy recovery vs #PCs (ref lambda);
  (f) train (dotted) & val (solid) accuracy vs ridge (log-y).
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
from matplotlib.lines import Line2D  # noqa: E402

_DIR_RE = re.compile(r'init_reg_w(\d+)_lam([0-9.eE+-]+?)(?:_[a-z0-9_]+)?$')

# variant -> (linestyle, marker), in a sensible plot order.
_VAR_STYLE = {
    'bn-gap': ('-', 'o'),
    'ln-gap': ('--', 's'),
    'bn-vec': (':', '^'),
    'ln-vec': ('-.', 'D'),
}


def _variant(norm: str, pool: str) -> str:
    return f"{'ln' if norm == 'layernorm' else 'bn'}-{pool}"


def _load_runs(root: str) -> list[dict]:
    runs = []
    for summary_path in sorted(glob.glob(os.path.join(root, 'init_reg_w*_lam*', 'fit_summary.json'))):
        run_dir = os.path.dirname(summary_path)
        if not _DIR_RE.search(os.path.basename(run_dir)):
            continue
        d = json.load(open(summary_path))
        pool = str(d.get('pool_mode', 'gap'))
        norm = str(d.get('norm_type', 'batchnorm'))
        rec = {
            'width': int(d['width']),
            'rel_lambda': float(d['ridge_rel_lambda']),
            'pool': pool, 'norm': norm, 'variant': _variant(norm, pool),
            'num_features': int(d.get('num_features', 0)),
            'capacity_b': float(d.get('capacity_exponent_b', float('nan'))),
            'capacity_b_std': float(d.get('capacity_b_std', float('nan'))),
            'a_mean': float(d.get('source_exponent_mean', float('nan'))),
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


def _variants(runs):
    order = list(_VAR_STYLE)
    return sorted({r['variant'] for r in runs}, key=lambda v: (order.index(v) if v in order else 99, v))


def _series(runs, width, variant):
    return sorted([r for r in runs if r['width'] == width and r['variant'] == variant],
                  key=lambda r: r['rel_lambda'])


def _xlam(rel):
    return max(rel, 1e-4)  # 0 shown at 1e-4 on a log axis.


def _ls(v):
    return _VAR_STYLE.get(v, ('-', 'x'))[0]


def _mk(v):
    return _VAR_STYLE.get(v, ('-', 'x'))[1]


def render(runs: list[dict], out_path: str, ref_lambda: float, dataset_label: str = '') -> None:
    widths = _widths(runs)
    variants = _variants(runs)
    wcolors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(widths), 1)))
    wcolor = {w: c for w, c in zip(widths, wcolors)}

    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.8))

    def each_series():
        for w in widths:
            for v in variants:
                s = _series(runs, w, v)
                if s:
                    yield w, v, s

    # (a) capacity exponent b vs width (b ~ lambda-independent -> smallest-lambda row).
    ax = axes[0, 0]
    for v in variants:
        pts = []
        for w in widths:
            s = _series(runs, w, v)
            if s:
                pts.append((w, s[0]['capacity_b'], s[0]['capacity_b_std']))
        if pts:
            xs, ys, es = zip(*pts)
            ax.errorbar(xs, ys, yerr=es, marker=_mk(v), ls=_ls(v), color='0.3', capsize=3, label=v)
    ax.set_xscale('log', base=2)
    ax.set_xticks(widths); ax.set_xticklabels(widths)
    ax.set_xlabel('width (num_filters)'); ax.set_ylabel(r'capacity exponent $b$')
    ax.set_title('(a) Feature-spectrum capacity vs width')
    ax.grid(True, which='both', alpha=0.25); ax.legend(fontsize=8)

    def _lam_panel(ax, key, ylabel, title):
        for w, v, s in each_series():
            ax.plot([_xlam(r['rel_lambda']) for r in s], [r[key] for r in s],
                    marker=_mk(v), ms=4, ls=_ls(v), color=wcolor[w])
        ax.set_xscale('log')
        ax.set_xlabel(r'ridge $\lambda_{\mathrm{rel}}$ (0 at $10^{-4}$)')
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, which='both', alpha=0.25)

    ax = axes[0, 1]; _lam_panel(ax, 'a_mean', r'mean source exponent $\bar a_i$', '(b) Source condition vs ridge')
    ax.axhline(0.0, color='0.4', ls=':', lw=1.2)
    _lam_panel(axes[0, 2], 'sing_c', r'singular-value exponent $c$', '(c) Classifier singular spectrum vs ridge')
    _lam_panel(axes[1, 0], 'rho_a_acc', r'Spearman $\rho(a_i,\ \mathrm{acc})$', '(d) Source exponent predicts class accuracy')

    # (e) PCA-truncation recovery at the reference lambda.
    ax = axes[1, 1]
    all_lams = sorted({r['rel_lambda'] for r in runs})
    ref = ref_lambda if ref_lambda in all_lams else (all_lams[len(all_lams) // 2] if all_lams else 0.1)
    for w, v, s in each_series():
        sel = [r for r in s if r['rel_lambda'] == ref and 'trunc_m' in r]
        if sel:
            ax.plot(sel[0]['trunc_m'], sel[0]['trunc_acc'], marker='.', ls=_ls(v), color=wcolor[w])
    ax.set_xscale('log')
    ax.set_xlabel('retained PCs $m$'); ax.set_ylabel('val accuracy')
    ax.set_title(f'(e) Truncation recovery ($\\lambda_{{\\mathrm{{rel}}}}={ref:g}$)')
    ax.grid(True, which='both', alpha=0.25)

    # (f) train (dotted) & val (solid) accuracy vs ridge, log-y.
    ax = axes[1, 2]
    clip = lambda vals: np.clip(np.asarray(vals, float), 1e-4, None)  # noqa: E731
    for w, v, s in each_series():
        xs = [_xlam(r['rel_lambda']) for r in s]
        ax.plot(xs, clip([r['val_acc'] for r in s]), marker=_mk(v), ms=4, ls='-', color=wcolor[w])
        ax.plot(xs, clip([r['train_acc'] for r in s]), marker=_mk(v), ms=3, ls=':', color=wcolor[w], alpha=0.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'ridge $\lambda_{\mathrm{rel}}$ (0 at $10^{-4}$)')
    ax.set_ylabel('accuracy (log)')
    ax.set_title('(f) Train (dotted) & val (solid) accuracy vs ridge')
    ax.grid(True, which='both', alpha=0.25)

    handles = [Line2D([0], [0], color=wcolor[w], lw=2.4, label=f'w{w}') for w in widths]
    handles += [Line2D([0], [0], color='0.3', lw=2.0, ls=_ls(v), marker=_mk(v), label=v) for v in variants]
    handles += [Line2D([0], [0], color='0.3', lw=2.0, ls='-', label='val (f)'),
                Line2D([0], [0], color='0.3', lw=1.5, ls=':', label='train (f)')]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 10), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    suffix = f' ({dataset_label})' if dataset_label else ''
    fig.suptitle(f'ResNet18 at init + ridge readout — capacity / source / SVD / accuracy{suffix}',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def write_table(runs: list[dict], csv_path: str) -> None:
    cols = ['width', 'variant', 'norm', 'pool', 'rel_lambda', 'num_features', 'dof',
            'capacity_b', 'a_mean', 'sing_c', 'train_acc', 'val_acc', 'r_a_acc', 'rho_a_acc']
    with open(csv_path, 'w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for r in sorted(runs, key=lambda r: (r['width'], r['variant'], r['rel_lambda'])):
            w.writerow({c: r.get(c, '') for c in cols})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='artifacts/init_regression_powerlaw/imagenet')
    p.add_argument('--output-dir', default='')
    p.add_argument('--ref-lambda', type=float, default=0.1)
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
    render(runs, fig_path, args.ref_lambda, dataset_label=os.path.basename(os.path.normpath(args.root)))
    write_table(runs, csv_path)
    print(f'{len(runs)} runs, widths={_widths(runs)}, variants={_variants(runs)}')
    print(f'wrote {fig_path}')


if __name__ == '__main__':
    main()
