#!/usr/bin/env python3
"""Width-robust per-class descriptors from the classifier's left factor (What = L S R^T).

Leverage n_i = sum_j l_ij^2 is only the *total* left-mass of class i; it degenerates to 1
once D >= C (L becomes a full orthonormal basis, rows unit-norm). The width-robust object
is the class's mode distribution p_ij = l_ij^2 / n_i (a probability over singular modes j,
well-defined at any width). Two independent things summarize it:

  * LOCATION  rho_i = <j>_i = sum_j j p_ij            (mean mode rank; norm rho_i/p in (0,1])
              -> where in the spectrum class i's weight sits (top modes vs weak tail).
  * SPREAD    perplexity_i = exp(H_i), H_i = -sum_j p_ij log p_ij   (effective #modes)
              -> how many modes it spreads over. (This tracks the participation ratio.)

Location and spread are independent: a distribution can slide its mean while keeping the
same width, so a constant participation ratio (spread) is compatible with a varying
centroid (location). We also keep the user's moments d1=E_i/mean(s^2), d2=<j>/mean(j) and
the s^2-weighted concentration d1n=<s^2>_i/mean(s^2).

Each descriptor is correlated with BOTH per-class accuracy and per-class cross-entropy
('entropy'), across the grid, and correlation-vs-width is plotted.
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
    n = l2.sum(axis=1)                          # leverage (total left-mass)
    energy = l2 @ s2                            # E_i
    sbar2 = float(s2.mean())
    with np.errstate(divide='ignore', invalid='ignore'):
        pij = l2 / n[:, None]                   # per-class mode distribution (sums to 1)
        mean_rank = pij @ j                     # rho_i = <j>_i  (LOCATION)
        norm_mean_rank = mean_rank / p          # in (0,1]
        ent = -np.sum(np.where(pij > 0, pij * np.log(pij), 0.0), axis=1)  # H_i (SPREAD)
        perplexity = np.exp(ent)               # effective # modes
        perplexity_frac = perplexity / p       # effective fraction of modes
        d1n = np.where(n > 0, energy / (n * sbar2), np.nan)   # <s^2>_i/mean(s^2) (s^2-weighted)
    return {'C': c, 'p': p, 'leverage': n, 'energy': energy, 'mean_rank': mean_rank,
            'norm_mean_rank': norm_mean_rank, 'mode_entropy': ent,
            'perplexity_frac': perplexity_frac, 'd1n': d1n}


# Featured descriptors (LOCATION rho, SPREAD perplexity, s^2-weighted contrast, leverage).
_DESCRIPTORS = ['leverage', 'norm_mean_rank', 'perplexity_frac', 'd1n']
_TARGETS = ['accuracy', 'cross_entropy']


def _process_run(run_dir: str) -> dict | None:
    npz = os.path.join(run_dir, 'powerlaw_arrays.npz')
    if not os.path.exists(npz):
        return None
    data = np.load(npz, allow_pickle=True)
    if 'W_hat' not in data.files:
        return None
    desc = compute_descriptors(np.asarray(data['W_hat'], dtype=np.float64))
    targets = {}
    if 'acc_full' in data.files:
        targets['accuracy'] = np.asarray(data['acc_full'], dtype=np.float64)
    if 'ce_full' in data.files:
        targets['cross_entropy'] = np.asarray(data['ce_full'], dtype=np.float64)
    summary = json.load(open(os.path.join(run_dir, 'fit_summary.json'))) \
        if os.path.exists(os.path.join(run_dir, 'fit_summary.json')) else {}
    dataset = str(summary.get('dataset', 'na'))
    optimizer = str(summary.get('optimizer_key', 'na'))
    width = int(summary.get('width', 0))

    corr = {t: {} for t in _TARGETS}
    for t, y in targets.items():
        for key in _DESCRIPTORS + ['mean_rank', 'mode_entropy']:
            r, rho = _pearson_spearman(desc[key], y)
            corr[t][key] = {'pearson': r, 'spearman': rho}

    out_dir = os.path.join(run_dir, 'svd', 'left')
    os.makedirs(out_dir, exist_ok=True)
    cols = ['leverage', 'mean_rank', 'norm_mean_rank', 'mode_entropy', 'perplexity_frac', 'd1n', 'energy']
    with open(os.path.join(out_dir, 'left_descriptors.csv'), 'w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=['class_index'] + cols + list(targets))
        w.writeheader()
        for i in range(desc['C']):
            row = {'class_index': i}
            for k in cols:
                row[k] = float(desc[k][i])
            for t, y in targets.items():
                row[t] = float(y[i])
            w.writerow(row)

    return {'dataset': dataset, 'optimizer': optimizer, 'width': width, 'C': desc['C'],
            'p': desc['p'], 'corr': corr,
            'spread_cv': float(np.nanstd(desc['perplexity_frac']) / np.nanmean(desc['perplexity_frac'])),
            'loc_cv': float(np.nanstd(desc['norm_mean_rank']) / np.nanmean(desc['norm_mean_rank']))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default='artifacts/classifier_powerlaw')
    parser.add_argument('--datasets', nargs='*', default=['imagenet'])
    parser.add_argument('--target', choices=_TARGETS, default='cross_entropy',
                        help='Difficulty variable for the cross-width correlation figure.')
    parser.add_argument('--output-dir', default='artifacts/classifier_powerlaw/comparison')
    parser.add_argument('--format', choices=['pdf', 'png'], default='png')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [r for r in (_process_run(os.path.dirname(n)) for n in
                        sorted(glob.glob(os.path.join(args.results_root, '**', 'powerlaw_arrays.npz'),
                                         recursive=True))) if r is not None]
    if not runs:
        raise SystemExit(f'No runs under {args.results_root}')
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)

    series = {}
    for r in runs:
        if r['dataset'] in args.datasets:
            series.setdefault((r['dataset'], r['optimizer']), []).append(r)
    for key in series:
        series[key].sort(key=lambda r: r['width'])

    keys = ['leverage', 'norm_mean_rank', 'perplexity_frac', 'd1n']
    titles = {'leverage': 'leverage $n_i$ (total mass)',
              'norm_mean_rank': r'location $\langle j\rangle_i/p$',
              'perplexity_frac': 'spread (perplexity/$p$)',
              'd1n': r'$s^2$-weighted $\langle s^2\rangle_i/\overline{s^2}$'}
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 4.2), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(series))))
    for ci, (skey, rlist) in enumerate(sorted(series.items())):
        widths = [r['width'] for r in rlist]
        for a, key in enumerate(keys):
            ys = [r['corr'][args.target].get(key, {}).get('pearson', np.nan) for r in rlist]
            axes[0][a].plot(widths, ys, marker='o', color=colors[ci], label=f'{skey[0]}/{skey[1]}')
    for a, key in enumerate(keys):
        ax = axes[0][a]
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('width'); ax.set_ylabel(f'corr with {args.target}')
        ax.set_title(titles[key]); ax.grid(True, which='both', alpha=0.25); ax.set_ylim(-0.75, 0.75)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f'Per-class left descriptors: correlation with {args.target} across width', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(os.path.abspath(args.output_dir),
                        f'compare_left_descriptors_vs_{args.target}.{args.format}')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)

    print(f'target = {args.target}')
    print(f'{"run":22s} {"C":>5s} {"p":>5s} | {"lev":>6s} {"loc":>6s} {"spread":>6s} {"d1n":>6s} '
          f'| {"locCV":>6s} {"sprdCV":>6s}')
    table = []
    for r in sorted(runs, key=lambda r: (r['dataset'], r['optimizer'], r['width'])):
        cr = r['corr'][args.target]
        rowname = f'{r["dataset"]}/{r["optimizer"]}_w{r["width"]}'
        table.append({'run': rowname, 'C': r['C'], 'p': r['p'],
                      'corr': {t: r['corr'][t] for t in _TARGETS},
                      'loc_cv': r['loc_cv'], 'spread_cv': r['spread_cv']})
        if r['dataset'] in args.datasets:
            g = lambda k: cr.get(k, {}).get('pearson', float('nan'))  # noqa: E731
            print(f'{rowname:22s} {r["C"]:5d} {r["p"]:5d} | {g("leverage"):6.2f} '
                  f'{g("norm_mean_rank"):6.2f} {g("perplexity_frac"):6.2f} {g("d1n"):6.2f} '
                  f'| {r["loc_cv"]:6.3f} {r["spread_cv"]:6.3f}')
    with open(os.path.join(os.path.abspath(args.output_dir), 'left_descriptors_corr.json'), 'w',
              encoding='utf-8') as h:
        json.dump(table, h, indent=2)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
