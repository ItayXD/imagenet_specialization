#!/usr/bin/env python3
"""Structure tests for the classifier's left singular vectors L (What = L S R^T).

Two tests, on top of the Haar-null summary in analyze_classifier_left_haar.py:

A. Eigenvector delocalization / IPR. For each left singular vector l_j (a unit vector
   over the C classes) the inverse participation ratio IPR_j = sum_i l_ij^4 measures
   localization; N_eff(j) = 1/IPR_j is the effective number of classes the mode touches.
   A Haar column is delocalized (N_eff ~ C/3). Modes with N_eff far below the Haar band
   are "specialist" directions localized on a few classes -> read off which classes.

B. External-covariate exchangeability. If classes were exchangeable, per-class summaries
   (leverage n_i, energy E_i, participation ratio PR_i, source exponent a_i) would be
   independent of any meaningful class covariate. We test each against (1) accuracy
   (correlation with a label-permutation null) and (2) WordNet superclass (a one-way
   ANOVA eta^2 with a label-permutation null). Rejecting independence == non-exchangeable.

Local, imagenet sgd w64 by default. Superclasses need the class-name file + nltk/wordnet;
without them, part B runs the accuracy test only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.analyze_classifier_left_haar import _haar_frame, _row_stats  # noqa: E402
from scripts.analyze_classifier_powerlaw import _pearson_spearman  # noqa: E402


# WordNet coarse superclasses, most-specific first (first ancestor match wins).
_SUPERCLASS_ANCHORS = [
    ('dog', 'dog.n.01'), ('feline', 'feline.n.01'), ('bird', 'bird.n.01'),
    ('reptile', 'reptile.n.01'), ('amphibian', 'amphibian.n.03'), ('fish', 'fish.n.01'),
    ('insect', 'insect.n.01'), ('other mammal', 'mammal.n.01'),
    ('other animal', 'animal.n.01'),
    ('vehicle', 'vehicle.n.01'), ('musical instrument', 'musical_instrument.n.01'),
    ('clothing', 'garment.n.01'), ('container', 'container.n.01'),
    ('food', 'food.n.01'), ('plant/produce', 'plant.n.02'), ('structure', 'structure.n.01'),
    ('device', 'device.n.01'), ('other artifact', 'artifact.n.01'),
]


def _load_class_names(path: str) -> list[str] | None:
    """Parse 'class_0000__tench__tinca_tinca' lines into per-index synonym lists."""
    if not path or not os.path.exists(path):
        return None
    names: dict[int, str] = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        m = re.match(r'class_(\d+)__(.+)', line)
        if m:
            names[int(m.group(1))] = m.group(2)
    if not names:
        return None
    return [names.get(i, '') for i in range(max(names) + 1)]


def _superclasses(class_names: list[str]) -> tuple[np.ndarray, dict]:
    """Map each class to a coarse WordNet superclass label (array of str), 'other' fallback."""
    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        from nltk.corpus import wordnet as wn
    except Exception:
        return np.array(['<no-wordnet>'] * len(class_names)), {}

    anchor_syn = {}
    for label, syn in _SUPERCLASS_ANCHORS:
        try:
            anchor_syn[label] = wn.synset(syn)
        except Exception:
            pass

    labels = []
    for raw in class_names:
        assigned = 'other'
        candidates = [tok for tok in raw.split('__') if tok]
        synset = None
        for cand in candidates:
            got = wn.synsets(cand, pos='n')
            if got:
                synset = got[0]
                break
        if synset is not None:
            ancestors = set()
            for pathway in synset.hypernym_paths():
                ancestors.update(pathway)
            for label, _syn in _SUPERCLASS_ANCHORS:
                if label in anchor_syn and anchor_syn[label] in ancestors:
                    assigned = label
                    break
        labels.append(assigned)
    labels = np.array(labels)
    counts = {lab: int(np.sum(labels == lab)) for lab in sorted(set(labels))}
    return labels, counts


def _perm_corr_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """Pearson r of (x, y) and a two-sided label-permutation p-value."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 5:
        return float('nan'), float('nan')
    r_obs = float(np.corrcoef(x, y)[0, 1])
    count = 0
    for _ in range(n_perm):
        if abs(float(np.corrcoef(rng.permutation(x), y)[0, 1])) >= abs(r_obs):
            count += 1
    return r_obs, (count + 1) / (n_perm + 1)


def _eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA effect size eta^2 = between-group SS / total SS."""
    finite = np.isfinite(values)
    values, groups = values[finite], groups[finite]
    grand = values.mean()
    ss_tot = np.sum((values - grand) ** 2)
    if ss_tot <= 0:
        return float('nan')
    ss_between = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        ss_between += v.size * (v.mean() - grand) ** 2
    return float(ss_between / ss_tot)


def _perm_eta_pvalue(values: np.ndarray, groups: np.ndarray, n_perm: int,
                     rng: np.random.Generator) -> tuple[float, float]:
    finite = np.isfinite(values)
    values, groups = values[finite], groups[finite]
    eta_obs = _eta_squared(values, groups)
    count = sum(1 for _ in range(n_perm)
                if _eta_squared(values, rng.permutation(groups)) >= eta_obs)
    return eta_obs, (count + 1) / (n_perm + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--class-names', default='artifacts/classifier_powerlaw/imagenet_class_order.txt')
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--n-null', type=int, default=100, help='Haar draws for the IPR null.')
    parser.add_argument('--n-perm', type=int, default=2000, help='Label permutations for part B.')
    parser.add_argument('--top-classes', type=int, default=12, help='Classes to list per localized mode.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--format', choices=['pdf', 'png'], default='png')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(os.path.join(args.results_dir, 'powerlaw_arrays.npz'), allow_pickle=True)
    what_hat = np.asarray(data['W_hat'], dtype=np.float64)
    acc = np.asarray(data['acc_full'], dtype=np.float64) if 'acc_full' in data.files else None
    a_src = np.asarray(data['a'], dtype=np.float64) if 'a' in data.files else None
    run_label = str(data['run_label']) if 'run_label' in data.files else 'classifier'
    output_dir = os.path.abspath(args.output_dir) if args.output_dir \
        else os.path.join(os.path.abspath(args.results_dir), 'svd')
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    left, s, _ = np.linalg.svd(what_hat, full_matrices=False)   # left = L (C, p)
    c, p = left.shape
    s2 = s ** 2
    l2 = left ** 2
    stats = _row_stats(left, s2)  # n (leverage), pr, e (energy)

    class_names = _load_class_names(args.class_names)
    if class_names is not None and len(class_names) < c:
        class_names = None
    superclass, super_counts = (_superclasses(class_names) if class_names else (None, {}))

    written = []

    def _save(fig, stem):
        path = os.path.join(output_dir, f'{stem}.{args.format}')
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        written.append(path)
        return path

    # ================= Part A: IPR / delocalization (columns over classes) =============
    ipr = np.sum(l2 ** 2, axis=0)          # IPR_j = sum_i l_ij^4  (columns are unit norm)
    n_eff = 1.0 / ipr                      # effective number of classes per mode
    null_ipr = []
    for _ in range(int(args.n_null)):
        q = _haar_frame(c, p, rng)
        null_ipr.append(np.sum(q ** 4, axis=0))
    null_ipr = np.concatenate(null_ipr)
    ipr_hi = float(np.quantile(null_ipr, 0.99))       # localized if IPR above this
    neff_lo = 1.0 / ipr_hi
    haar_neff = float(np.median(1.0 / null_ipr))
    localized = np.where(ipr > ipr_hi)[0]
    print(f'run={run_label} C={c} p={p}')
    print(f'IPR/delocalization: Haar N_eff~{haar_neff:.0f} classes; '
          f'localized modes (N_eff<{neff_lo:.0f}): {localized.size}/{p}')

    # List the top classes for the most localized modes.
    localized_rows = []
    order_loc = localized[np.argsort(-ipr[localized])] if localized.size else np.array([], dtype=int)
    for j in order_loc[:15]:
        top = np.argsort(-l2[:, j])[:args.top_classes]
        top_names = [class_names[t].split('__')[0] if class_names else str(t) for t in top]
        localized_rows.append({'mode_j': int(j), 'singular_value': float(s[j]),
                               'N_eff': float(n_eff[j]),
                               'top_classes': ', '.join(f'{t}:{class_names[t].split("__")[0]}'
                                                        if class_names else str(t) for t in top)})
        print(f'  mode j={int(j):3d} s={s[j]:.2g} N_eff={n_eff[j]:.0f}: ' + ', '.join(top_names[:8]))

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.axhspan(float(np.quantile(1.0 / null_ipr, 0.01)), float(np.quantile(1.0 / null_ipr, 0.99)),
               color='0.85', label='Haar 1-99% band')
    ax.axhline(haar_neff, color='0.5', ls='--', lw=1, label=f'Haar median ~{haar_neff:.0f}')
    ax.scatter(np.arange(1, p + 1), n_eff, s=8, alpha=0.5, color='steelblue')
    if localized.size:
        ax.scatter(localized + 1, n_eff[localized], s=20, color='crimson', label='localized modes', zorder=5)
    ax.set_xscale('log')
    ax.set_xlabel('singular mode index $j$')
    ax.set_ylabel(r'$N_\mathrm{eff}(j)=1/\mathrm{IPR}_j$ (effective #classes)')
    ax.set_title(f'Left-vector delocalization vs Haar — {run_label}')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.25)
    _save(fig, 'fig_left_delocalization')

    # ================= Part B: external-covariate exchangeability =====================
    per_class = {'leverage n_i': stats['n'], 'energy E_i': stats['e'],
                 'participation PR_i': stats['pr']}
    if a_src is not None:
        per_class['source a_i'] = a_src

    acc_rows = []
    if acc is not None:
        for name, vals in per_class.items():
            r, pval = _perm_corr_pvalue(vals, acc, args.n_perm, rng)
            _, rho = _pearson_spearman(vals, acc)
            acc_rows.append({'statistic': name, 'pearson_r': r, 'spearman': rho, 'perm_p': pval})
            print(f'[acc] corr({name}, accuracy): r={r:+.3f} (spearman {rho:+.3f}) perm_p={pval:.4f}')

    super_rows = []
    if superclass is not None and '<no-wordnet>' not in set(superclass):
        # Restrict to reasonably populated superclasses for a stable ANOVA.
        keep = np.array([lab for lab, n in super_counts.items() if n >= 10])
        mask = np.isin(superclass, keep)
        for name, vals in {**per_class, **({'accuracy': acc} if acc is not None else {})}.items():
            eta, pval = _perm_eta_pvalue(vals[mask], superclass[mask], args.n_perm, rng)
            super_rows.append({'statistic': name, 'eta_squared': eta, 'perm_p': pval})
            print(f'[superclass] {name}: eta^2={eta:.3f} perm_p={pval:.4f}')

    # Figure: correlation-with-accuracy bars (+ significance).
    if acc_rows:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        labels = [r['statistic'] for r in acc_rows]
        rs = [r['pearson_r'] for r in acc_rows]
        colors = ['crimson' if r['perm_p'] < 0.05 else '0.6' for r in acc_rows]
        ax.barh(labels, rs, color=colors)
        for i, r in enumerate(acc_rows):
            ax.text(r['pearson_r'], i, f"  p={r['perm_p']:.3f}", va='center',
                    ha='left' if r['pearson_r'] >= 0 else 'right', fontsize=8)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlabel('Pearson corr with class accuracy')
        ax.set_title(f'Per-class stats vs accuracy (red: perm p<0.05) — {run_label}')
        ax.grid(True, axis='x', alpha=0.25)
        _save(fig, 'fig_left_corr_accuracy')

    # Figure: leverage & accuracy by superclass.
    if super_rows:
        keep = sorted([lab for lab, n in super_counts.items() if n >= 10],
                      key=lambda lab: np.nanmean(stats['n'][superclass == lab]))
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
        for ax, (vals, title) in zip(axes, [(stats['n'], 'leverage $n_i$'),
                                            (acc if acc is not None else stats['e'],
                                             'accuracy' if acc is not None else 'energy $E_i$')]):
            box = [vals[superclass == lab] for lab in keep]
            ax.boxplot(box, vert=False, tick_labels=[f'{lab} ({super_counts[lab]})' for lab in keep],
                       showfliers=False)
            ax.set_xlabel(title)
            ax.grid(True, axis='x', alpha=0.25)
        fig.suptitle(f'Per-class structure by WordNet superclass — {run_label}')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        _save(fig, 'fig_left_by_superclass')

    # ---- save summary + tables ----
    summary = {
        'run_label': run_label, 'num_classes': int(c), 'num_modes': int(p),
        'ipr': {'haar_median_neff': haar_neff, 'localized_mode_count': int(localized.size),
                'neff_localized_threshold': neff_lo},
        'accuracy_covariate': acc_rows,
        'superclass_covariate': super_rows,
        'superclass_counts': super_counts,
    }
    with open(os.path.join(output_dir, 'left_structure.json'), 'w', encoding='utf-8') as h:
        json.dump(summary, h, indent=2)
    if localized_rows:
        with open(os.path.join(output_dir, 'left_localized_modes.csv'), 'w', newline='', encoding='utf-8') as h:
            w = csv.DictWriter(h, fieldnames=['mode_j', 'singular_value', 'N_eff', 'top_classes'])
            w.writeheader()
            w.writerows(localized_rows)
    for path in [os.path.join(output_dir, 'left_structure.json'), *written]:
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
