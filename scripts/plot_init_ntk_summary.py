#!/usr/bin/env python3
"""Summary figure for the linearized/tangent (NTK) random-feature readout at init.

Reads the ntk_summary.json + ntk_arrays.npz written by analyze_init_ntk_regression.py for
one or more (width, m) runs and renders:

  (a) NTK-sketch eigenspectrum mu_j (log-log) with a power-law slope, per m -- the capacity
      spectrum of the tangent random features;
  (b) train (dotted) & val (solid) accuracy vs ridge lambda, per m;
  (c) best val accuracy vs number of random tangent directions m (the sketch-size scaling),
      with an optional reference line for the last-layer (conjugate) readout accuracy.
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


def _loglog_slope(y: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, float)
    y = y[y > 0]
    if y.size < 4:
        return float('nan'), float('nan')
    j = np.arange(1, y.size + 1, dtype=float)
    lo, hi = int(0.05 * y.size), int(0.6 * y.size)  # fit the bulk, skip head/tail
    lo = max(lo, 1)
    lx, ly = np.log(j[lo:hi]), np.log(y[lo:hi])
    if lx.size < 2:
        return float('nan'), float('nan')
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)


def _load(root: str) -> list[dict]:
    runs = []
    for f in sorted(glob.glob(os.path.join(root, '**', 'ntk_summary.json'), recursive=True)):
        d = json.load(open(f))
        npz = os.path.join(os.path.dirname(f), 'ntk_arrays.npz')
        d['_eig'] = np.sort(np.load(npz)['ntk_eigenvalues'])[::-1] if os.path.exists(npz) else None
        runs.append(d)
    return runs


def render(runs: list[dict], out_path: str, ref_acc: float | None, ref_label: str) -> None:
    runs = sorted(runs, key=lambda d: d['num_features_m'])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(runs), 1)))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

    # (a) NTK eigenspectrum.
    ax = axes[0]
    for d, c in zip(runs, colors):
        eig = d.get('_eig')
        if eig is None:
            continue
        eig = eig[eig > 0]
        j = np.arange(1, eig.size + 1)
        slope, intercept = _loglog_slope(eig)
        ax.loglog(j, eig, '.', ms=3, color=c,
                  label=f"m={d['num_features_m']} (eff_rank {d['ntk_eff_rank']:.0f}, "
                        f"$\\mu\\sim j^{{{slope:.2f}}}$)")
        if np.isfinite(slope):
            xs = np.array([1, eig.size], float)
            ax.loglog(xs, np.exp(intercept) * xs ** slope, color=c, lw=1.5, alpha=0.6)
    ax.set_xlabel('index $j$')
    ax.set_ylabel(r'NTK-sketch eigenvalue $\mu_j$')
    ax.set_title('(a) Tangent-feature (NTK) spectrum')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8)

    # (b) train/val accuracy vs ridge lambda.
    ax = axes[1]
    for d, c in zip(runs, colors):
        paths = sorted(d['paths'], key=lambda p: p['rel_lambda'])
        xs = [max(p['rel_lambda'], 1e-4) for p in paths]
        ax.plot(xs, [p['val_acc'] for p in paths], 'o-', ms=4, color=c,
                label=f"m={d['num_features_m']}")
        ax.plot(xs, [p['train_acc'] for p in paths], ':', ms=3, color=c, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel(r'ridge $\lambda_{\mathrm{rel}}$ (0 at $10^{-4}$)')
    ax.set_ylabel('accuracy')
    ax.set_title('(b) Train (dotted) & val (solid) vs ridge')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8)

    # (c) best val accuracy vs m (sketch-size scaling).
    ax = axes[2]
    ms = [d['num_features_m'] for d in runs]
    best = [max(p['val_acc'] for p in d['paths']) for d in runs]
    ax.plot(ms, best, 'o-', ms=6, color='steelblue', label='NTK tangent RF (best $\\lambda$)')
    if ref_acc is not None:
        ax.axhline(ref_acc, color='crimson', ls='--', lw=1.6, label=ref_label)
    ax.set_xscale('log')
    ax.set_xlabel('number of random tangent directions $m$')
    ax.set_ylabel('best val accuracy')
    ax.set_title('(c) Sketch-size scaling')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8)

    ds = runs[0].get('dataset', '') if runs else ''
    w = runs[0].get('width', '') if runs else ''
    fig.suptitle(f'ResNet18 at init — linearized (NTK) tangent random-feature readout '
                 f'({ds} w{w})', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='artifacts/init_ntk_regression/cifar5m')
    p.add_argument('--output-dir', default='')
    p.add_argument('--ref-acc', type=float, default=None,
                   help='Reference accuracy line (e.g. last-layer conjugate readout).')
    p.add_argument('--ref-label', default='last-layer (conjugate)')
    p.add_argument('--format', choices=['png', 'pdf'], default='png')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs = _load(args.root)
    if not runs:
        raise SystemExit(f'No ntk_summary.json found under {args.root}.')
    out_dir = args.output_dir or os.path.join(args.root, 'summary')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'ntk_summary.{args.format}')
    render(runs, out_path, args.ref_acc, args.ref_label)
    print(f'{len(runs)} NTK runs: m={[d["num_features_m"] for d in runs]}')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
