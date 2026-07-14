#!/usr/bin/env python3
"""Quantify the diagonal band of r_jk^2 (SVD of the rotated classifier).

The heatmap of r_jk^2 (j = singular mode sorted by s_j, k = PC index sorted by
lambda_k) shows a diagonal band: the classifier's right singular vectors approximately
coincide with the feature eigenvectors, mode j ~ PC j. This script quantifies that band:

  * longitudinal amplitude A(i): the transverse-smoothed on-diagonal value r_ii^2 (minus
    the 1/D background floor), fit as a power law A(i) ~ i^{-p_long};
  * transverse width w(i): RMS width of the band cross-section at diagonal position i,
    fit as w(i) ~ i^{q};
  * transverse shape: at several i, is the floor-subtracted cross-section P(|d|),
    d = k - j, Gaussian (~exp(-d^2)), exponential (~exp(-|d|)), or power law (~|d|^{-r})?
    Chosen by comparing log-space R^2 of the three linearizations.

Runs locally on a saved powerlaw_arrays.npz. No GPU/data needed.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True, help='Dir with powerlaw_arrays.npz.')
    parser.add_argument('--output-dir', default='', help='Defaults to <results-dir>/svd.')
    parser.add_argument('--max-offset', type=int, default=0,
                        help='Transverse half-window M (|d|<=M). 0 => min(n//2, 120).')
    parser.add_argument('--transverse-smooth', type=int, default=2,
                        help='Half-width (in d) for smoothing the on-diagonal amplitude.')
    parser.add_argument('--n-null', type=int, default=400, help='Haar Monte-Carlo draws.')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for the Haar null.')
    parser.add_argument('--format', choices=['pdf', 'png'], default='png')
    return parser.parse_args()


def _band_matrix(right_sq: np.ndarray, max_offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Return B[i, d] = r^2_{i, i+d} for i in [0, n) and d in [-M, M] (NaN off-grid)."""
    num_modes, num_features = right_sq.shape
    n = min(num_modes, num_features)
    offsets = np.arange(-max_offset, max_offset + 1)
    band = np.full((n, offsets.size), np.nan)
    for col, d in enumerate(offsets):
        k = np.arange(n) + d
        valid = (k >= 0) & (k < num_features)
        band[valid, col] = right_sq[np.arange(n)[valid], k[valid]]
    return band, offsets


def _loglog_fit(x: np.ndarray, y: np.ndarray) -> dict:
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(m.sum()) < 3:
        return {'slope': float('nan'), 'intercept': float('nan'), 'r2': float('nan'), 'n': int(m.sum())}
    lx, ly = np.log(x[m]), np.log(y[m])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'slope': float(slope), 'intercept': float(intercept), 'r2': float(r2), 'n': int(m.sum())}


def _linear_r2(x: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares y = a*x + b; return slope, intercept, R^2."""
    if x.size < 3:
        return {'slope': float('nan'), 'intercept': float('nan'), 'r2': float('nan')}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'slope': float(slope), 'intercept': float(intercept), 'r2': float(r2)}


def _symmetric_excess(offsets: np.ndarray, profile: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Symmetrize over +/-d and subtract a per-profile floor (median over the far half)."""
    abs_d = np.abs(offsets.astype(np.float64))
    far = abs_d > 0.5 * float(abs_d.max())
    floor = float(np.nanmedian(profile[far])) if np.any(far) else float(np.nanmedian(profile))
    uniq = np.unique(abs_d)
    sym = np.array([np.nanmean(profile[abs_d == u]) for u in uniq])
    return uniq, sym - floor, floor


def _core_mask(excess: np.ndarray, floor: float) -> np.ndarray:
    """Contiguous core from |d|=0 while excess stays clearly above the floor noise."""
    peak = float(np.nanmax(excess)) if excess.size else 0.0
    thresh = max(0.05 * peak, 0.5 * floor)
    good = np.isfinite(excess) & (excess > thresh)
    if not good.any():
        return good
    first_bad = int(np.argmax(~good)) if (~good).any() else good.size
    core = np.zeros_like(good)
    core[:max(first_bad, 1)] = True
    return good & core


def _transverse_shape(offsets: np.ndarray, profile: np.ndarray) -> dict:
    """Classify the transverse cross-section shape on its above-floor core.

    Compare how well the (symmetrized, floor-subtracted) excess is linearized by log(e)
    vs d^2 (Gaussian), vs |d| (exponential), vs log|d| (power law).
    """
    abs_d, excess, floor = _symmetric_excess(offsets, profile)
    core = _core_mask(excess, floor)
    ud, ue = abs_d[core], excess[core]
    result = {'floor': float(floor), 'n_points': int(ud.size)}
    if ud.size < 4:
        result.update({'gaussian_r2': float('nan'), 'exponential_r2': float('nan'),
                       'power_r2': float('nan'), 'best': 'undetermined',
                       'gaussian_sigma': float('nan'), 'exponential_ell': float('nan'),
                       'power_exponent': float('nan')})
        return result
    log_e = np.log(ue)
    gauss = _linear_r2(ud ** 2, log_e)
    expo = _linear_r2(ud, log_e)
    nz = ud > 0
    power = _linear_r2(np.log(ud[nz]), log_e[nz]) if int(nz.sum()) >= 3 else {'r2': float('nan'), 'slope': float('nan')}
    r2s = {'gaussian': gauss['r2'], 'exponential': expo['r2'], 'power': power['r2']}
    best = max(r2s, key=lambda k: (r2s[k] if np.isfinite(r2s[k]) else -np.inf))
    result.update({
        'gaussian_r2': gauss['r2'], 'exponential_r2': expo['r2'], 'power_r2': power['r2'],
        'gaussian_sigma': float(np.sqrt(-0.5 / gauss['slope'])) if gauss['slope'] < 0 else float('nan'),
        'exponential_ell': float(-1.0 / expo['slope']) if expo['slope'] < 0 else float('nan'),
        'power_exponent': -float(power['slope']),
        'best': best,
    })
    return result


def _log_binned_median(x: np.ndarray, y: np.ndarray, num_bins: int = 22) -> tuple[np.ndarray, np.ndarray]:
    """Median of y within log-spaced bins of x>0 (denoises noisy per-i quantities)."""
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[m], y[m]
    if x.size < 2:
        return np.zeros(0), np.zeros(0)
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), num_bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    idx = np.digitize(x, edges) - 1
    cx, cy = [], []
    for b in range(num_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        cx.append(float(np.exp(np.mean(np.log(x[sel])))))
        cy.append(float(np.median(y[sel])))
    return np.asarray(cx), np.asarray(cy)


def _hwhm(offsets: np.ndarray, profile: np.ndarray) -> float:
    """Half-width at half-maximum of the floor-subtracted transverse core (shape-agnostic)."""
    abs_d, excess, _ = _symmetric_excess(offsets, profile)
    order = np.argsort(abs_d)
    abs_d, excess = abs_d[order], excess[order]
    peak = float(excess[0]) if excess.size else 0.0
    if peak <= 0:
        return float('nan')
    half = 0.5 * peak
    for i in range(1, abs_d.size):
        if excess[i] <= half:
            x0, x1, y0, y1 = abs_d[i - 1], abs_d[i], excess[i - 1], excess[i]
            return float(x1 if y0 == y1 else x0 + (half - y0) * (x1 - x0) / (y1 - y0))
    return float(abs_d[-1])


def _profile_over(band: np.ndarray, lo: int, hi: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(band[lo:hi, :], axis=0)


def analyze(right_sq: np.ndarray, *, max_offset: int, transverse_smooth: int) -> dict:
    num_modes, num_features = right_sq.shape
    n = min(num_modes, num_features)
    if max_offset <= 0:
        max_offset = int(min(n // 2, 120))
    band, offsets = _band_matrix(right_sq, max_offset)
    zero = int(np.where(offsets == 0)[0][0])

    # Background floor: median r^2 far from the diagonal (random ~ 1/D).
    far = np.abs(offsets) > max_offset // 2
    floor = float(np.nanmedian(band[:, far]))

    # Longitudinal amplitude A(i): transverse-smoothed on-diagonal value.
    s = int(transverse_smooth)
    core = slice(max(0, zero - s), zero + s + 1)
    amp = np.nanmean(band[:, core], axis=1)
    positions = np.arange(1, n + 1, dtype=np.float64)
    excess_amp = amp - floor

    # Longitudinal power law of the band excess, over the range where it is above noise.
    # Fit log-binned medians (the per-i amplitudes are individually very noisy).
    above = np.where(excess_amp > 3.0 * floor)[0]
    if above.size:
        i_lo = max(1, int(above[0]) + 1)
        i_hi = int(above[-1]) + 1
    else:
        i_lo, i_hi = 2, n
    long_mask = (positions >= i_lo) & (positions <= i_hi)
    amp_bx, amp_by = _log_binned_median(positions[long_mask], excess_amp[long_mask])
    long_fit = _loglog_fit(amp_bx, amp_by)

    # Densely probe the transverse cross-section at EVERY diagonal position i (sliding
    # longitudinal window; no log-spaced sampling), from a small i up to where the
    # on-diagonal amplitude fades into the floor (only there is a band resolvable). Fit the
    # exponential length ell(i) ~ i^q from the dense set.
    cross = np.where(amp_by > 2.0 * floor)[0]
    i_cross = int(amp_bx[cross[-1]]) if cross.size else i_hi
    i_cross = int(min(i_cross, n // 3))
    i_probe_lo = 6
    probe_positions = list(range(i_probe_lo, max(i_probe_lo + 1, i_cross + 1)))
    ell_by_i = np.full(n + 1, np.nan)
    width_bx_list, ell_i = [], []
    shape_profiles, shapes = {}, {}
    highlight = {int(round(x)) for x in np.linspace(i_probe_lo, max(i_probe_lo, i_cross), 5)}
    all_verdicts = []
    for c in probe_positions:
        w_bin = max(6, int(0.25 * c))
        prof = _profile_over(band, max(0, c - w_bin), min(n, c + w_bin + 1))
        sh = _transverse_shape(offsets, prof)
        if c in highlight:
            shape_profiles[c] = prof
            shapes[c] = sh
        if np.isfinite(sh.get('exponential_ell', np.nan)) and sh['best'] != 'undetermined':
            width_bx_list.append(float(c))
            ell_i.append(float(sh['exponential_ell']))
            ell_by_i[c] = float(sh['exponential_ell'])
            all_verdicts.append(sh['best'])
    width_bx = np.asarray(width_bx_list)
    width_by = np.asarray(ell_i)  # transverse "width" = exponential length ell(i), dense in i
    width_fit = _loglog_fit(width_bx, width_by)

    # Majority transverse-shape verdict across ALL resolvable dense probes.
    shape_verdict = max(set(all_verdicts), key=all_verdicts.count) if all_verdicts else 'undetermined'

    return {
        'offsets': offsets, 'floor': floor, 'positions': positions,
        'amp': amp, 'excess_amp': excess_amp,
        'amp_bx': amp_bx, 'amp_by': amp_by, 'width_bx': width_bx, 'width_by': width_by,
        'long_fit': long_fit, 'width_fit': width_fit, 'shape_verdict': shape_verdict,
        'long_range': (i_lo, i_hi), 'i_cross': int(i_cross),
        'shape_profiles': shape_profiles, 'shapes': shapes,
        'n': n, 'num_features': num_features,
    }


def _render(res: dict, run_label: str, output_dir: str, fmt: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    written = []

    def _save(fig, stem):
        path = os.path.join(output_dir, f'{stem}.{fmt}')
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        written.append(path)

    positions = res['positions']
    floor = res['floor']
    offsets = res['offsets']
    i_lo, i_hi = res['long_range']

    # 1. Longitudinal amplitude A(i) - floor vs i, log-log, with power-law fit.
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    ax.loglog(positions, np.clip(res['excess_amp'], 1e-30, None), '.', ms=2, alpha=0.2,
              color='0.6', label=r'$A(i)-$floor (raw)')
    ax.loglog(res['amp_bx'], np.clip(res['amp_by'], 1e-30, None), 'o', ms=5,
              color='steelblue', label='log-binned median')
    lf = res['long_fit']
    if np.isfinite(lf['slope']):
        xs = np.arange(i_lo, i_hi + 1, dtype=np.float64)
        ax.loglog(xs, np.exp(lf['intercept']) * xs ** lf['slope'], color='crimson', lw=2.2,
                  label=fr"fit $i^{{-p}}$, $p={-lf['slope']:.2f}$ ($R^2={lf['r2']:.3f}$)")
    ax.axhline(floor, color='0.6', ls='--', lw=1, label=f'floor 1/D~{floor:.2e}')
    ax.axvspan(0.9, i_lo, color='0.9', zorder=0)
    ax.set_xlabel('diagonal position $i$')
    ax.set_ylabel(r'on-diagonal amplitude $r_{ii}^2-$floor')
    ax.set_title(f'Longitudinal decay along the band — {run_label}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, 'fig_svd_diag_longitudinal')

    # 2. Transverse width w(i) (HWHM, log-binned) vs i, log-log, with power-law fit.
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    ax.loglog(res['width_bx'], np.clip(res['width_by'], 1e-30, None), '.', ms=4,
              color='seagreen', label=r'exponential length $\ell(i)$ (every $i$)')
    wf = res['width_fit']
    if np.isfinite(wf['slope']) and res['width_bx'].size:
        xs = np.linspace(res['width_bx'].min(), res['width_bx'].max(), 50)
        ax.loglog(xs, np.exp(wf['intercept']) * xs ** wf['slope'], color='crimson', lw=2.2,
                  label=fr"fit $i^{{q}}$, $q={wf['slope']:.2f}$ ($R^2={wf['r2']:.3f}$)")
    ax.set_xlabel('diagonal position $i$')
    ax.set_ylabel(r'transverse length $\ell(i)$')
    ax.set_title(f'Transverse band width vs position — {run_label}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, 'fig_svd_diag_width')

    # 3. Transverse core cross-sections: semilog-y vs |d| (exp->line) and vs d^2 (gauss->line).
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    prof_items = sorted(res['shape_profiles'].items())
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(prof_items)))
    core_max = 1
    for (c, prof), color in zip(prof_items, colors):
        abs_d, excess, _floor = _symmetric_excess(offsets, prof)
        core = _core_mask(excess, _floor)
        if not core.any():
            continue
        core_max = max(core_max, int(abs_d[core].max()))
        sh = res['shapes'][c]
        best = sh.get('best', 'undetermined')
        r2 = sh.get(f'{best}_r2', float('nan'))
        lbl = f"i~{c} ({best[:3]}, R2={r2:.2f})" if np.isfinite(r2) else f'i~{c}'
        axes[0].plot(abs_d[core], np.clip(excess[core], 1e-30, None), 'o-', ms=3, color=color, label=lbl)
        axes[1].plot(abs_d[core] ** 2, np.clip(excess[core], 1e-30, None), 'o-', ms=3, color=color)
    for ax in axes:
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.25)
    axes[0].set_xlabel('|d| = |k - j|')
    axes[0].set_ylabel(r'excess $r^2$ (log)')
    axes[0].set_title('exponential test (line = exp)')
    axes[0].set_xlim(0, core_max * 1.05)
    axes[1].set_xlabel(r'$d^2$')
    axes[1].set_title('Gaussian test (line = Gaussian)')
    axes[1].set_xlim(0, (core_max * 1.05) ** 2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Transverse core cross-section shape — {run_label}')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, 'fig_svd_diag_transverse')

    return written


def haar_below_floor(vt: np.ndarray, i_cross: int, *, n_null: int = 400, rng_seed: int = 0) -> dict:
    """Test whether the right singular vectors below the noise floor are Haar-random.

    Above the floor (modes j <= i_cross) each r_j is aligned with PC j (its on-diagonal
    component dominates). Below the floor (j > i_cross) we test whether r_j is
    indistinguishable from a uniform random unit vector in R^D via:

      * participation ratio  PR_j = 1 / sum_k r_jk^4  (Haar ~ D/3; localized << that);
      * diagonal percentile  frac of components with r_jk^2 < r_jj^2 (Haar ~ Uniform,
        mean 0.5; aligned ~ 1);
      * pooled component marginal sqrt(D) r_jk vs N(0,1) (KS).

    Empirical below-floor statistics are compared to a Monte-Carlo Haar null (random unit
    vectors in R^D) via z-scores.
    """
    p, dim = vt.shape
    m = min(p, dim)
    r2 = vt[:m] ** 2
    modes = np.arange(1, m + 1)
    pr = 1.0 / np.sum(r2 ** 2, axis=1)
    diag_pct = np.array([float(np.mean(r2[j] < r2[j, j])) for j in range(m)])

    below = modes > i_cross
    above = modes <= i_cross

    rng = np.random.default_rng(rng_seed)
    g = rng.standard_normal((n_null, dim))
    u2 = (g / np.linalg.norm(g, axis=1, keepdims=True)) ** 2
    pr_null = 1.0 / np.sum(u2 ** 2, axis=1)
    pr_mu, pr_sd = float(pr_null.mean()), float(pr_null.std())
    diag_mu, diag_sd = 0.5, float(1.0 / np.sqrt(12.0))  # Uniform order-statistic

    def _z(vals, mu, sd):
        vals = vals[np.isfinite(vals)]
        if vals.size == 0 or sd == 0:
            return float('nan')
        return float((vals.mean() - mu) / (sd / np.sqrt(vals.size)))

    ks_stat = ks_p = float('nan')
    if below.sum() > 1:
        try:
            from scipy.stats import kstest
            z = (np.sqrt(dim) * vt[:m][below]).ravel()
            ks_stat, ks_p = (float(v) for v in kstest(z, 'norm'))
        except Exception:
            pass

    return {
        'modes': modes, 'pr': pr, 'diag_pct': diag_pct, 'below': below, 'above': above,
        'i_cross': int(i_cross), 'dim': int(dim),
        'pr_null_mu': pr_mu, 'pr_null_sd': pr_sd, 'diag_null_mu': diag_mu, 'diag_null_sd': diag_sd,
        'pr_below_mean': float(np.nanmean(pr[below])) if below.any() else float('nan'),
        'pr_above_mean': float(np.nanmean(pr[above])) if above.any() else float('nan'),
        'diag_below_mean': float(np.nanmean(diag_pct[below])) if below.any() else float('nan'),
        'diag_above_mean': float(np.nanmean(diag_pct[above])) if above.any() else float('nan'),
        'pr_below_z': _z(pr[below], pr_mu, pr_sd),
        'diag_below_z': _z(diag_pct[below], diag_mu, diag_sd),
        'ks_stat': ks_stat, 'ks_p': ks_p,
    }


def _render_haar(h: dict, run_label: str, output_dir: str, fmt: str) -> str:
    modes = h['modes']
    below, above = h['below'], h['above']
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    # Participation ratio vs mode.
    ax = axes[0]
    ax.plot(modes[above], h['pr'][above], '.', ms=3, color='indianred', label='above floor')
    ax.plot(modes[below], h['pr'][below], '.', ms=3, color='steelblue', label='below floor')
    lo = h['pr_null_mu'] - 2 * h['pr_null_sd']
    hi = h['pr_null_mu'] + 2 * h['pr_null_sd']
    ax.axhspan(lo, hi, color='0.7', alpha=0.4, label='Haar null (±2σ)')
    ax.axhline(h['dim'] / 3.0, color='0.4', ls=':', lw=1, label='D/3')
    ax.axvline(h['i_cross'], color='k', ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('singular mode $j$')
    ax.set_ylabel(r'participation ratio $1/\sum_k r_{jk}^4$')
    ax.set_title('participation ratio vs Haar')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.25)
    # Diagonal percentile vs mode.
    ax = axes[1]
    ax.plot(modes[above], h['diag_pct'][above], '.', ms=3, color='indianred', label='above floor')
    ax.plot(modes[below], h['diag_pct'][below], '.', ms=3, color='steelblue', label='below floor')
    ax.axhspan(0.5 - 2 * h['diag_null_sd'], 0.5 + 2 * h['diag_null_sd'], color='0.7', alpha=0.4,
               label='Haar null (±2σ)')
    ax.axhline(0.5, color='0.4', ls=':', lw=1)
    ax.axvline(h['i_cross'], color='k', ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('singular mode $j$')
    ax.set_ylabel(r'diagonal percentile of $r_{jj}^2$')
    ax.set_title('diagonal dominance vs Haar')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.25)
    fig.suptitle(f'Below-floor Haar test (right singular vectors) — {run_label}')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(output_dir, f'fig_svd_haar_below_floor.{fmt}')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    npz_path = os.path.join(args.results_dir, 'powerlaw_arrays.npz')
    data = np.load(npz_path, allow_pickle=True)
    what_hat = np.asarray(data['W_hat'], dtype=np.float64)
    run_label = str(data['run_label']) if 'run_label' in data.files else 'classifier_svd'
    _, _, vt = np.linalg.svd(what_hat, full_matrices=False)
    right_sq = vt ** 2

    res = analyze(right_sq, max_offset=args.max_offset, transverse_smooth=args.transverse_smooth)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir \
        else os.path.join(os.path.abspath(args.results_dir), 'svd')

    lf, wf = res['long_fit'], res['width_fit']
    mean_floor_1_over_D = 1.0 / res['num_features']
    print(f'run={run_label}  n={res["n"]}  D={res["num_features"]}  '
          f'median floor={res["floor"]:.3e}  (mean 1/D={mean_floor_1_over_D:.3e})')
    print(f'longitudinal A(i)-floor ~ i^-p : p={-lf["slope"]:.3f} (R^2={lf["r2"]:.3f}, '
          f'range i in [{res["long_range"][0]},{res["long_range"][1]}])')
    print(f'transverse shape (majority)    : {res["shape_verdict"]}')
    print(f'transverse length ell(i) ~ i^q : q={wf["slope"]:.3f} (R^2={wf["r2"]:.3f}, '
          f'n_probes={res["width_bx"].size})')
    for c, sh in sorted(res['shapes'].items()):
        print(f'  transverse shape @ i~{c:4d}: best={sh["best"]:11s} '
              f'R2[gauss/exp/pow]={sh["gaussian_r2"]:.3f}/{sh["exponential_r2"]:.3f}/{sh["power_r2"]:.3f} '
              f'(sigma={sh.get("gaussian_sigma", float("nan")):.2f}, ell={sh.get("exponential_ell", float("nan")):.2f})')

    summary = {
        'run_label': run_label, 'n': int(res['n']), 'num_features': int(res['num_features']),
        'median_floor': res['floor'], 'mean_floor_1_over_D': mean_floor_1_over_D,
        'longitudinal_exponent_p': -lf['slope'], 'longitudinal_r2': lf['r2'],
        'longitudinal_range': list(res['long_range']),
        'transverse_shape_verdict': res['shape_verdict'],
        'transverse_length_exponent_q': wf['slope'], 'transverse_length_r2': wf['r2'],
        'transverse_shapes': {int(c): sh for c, sh in res['shapes'].items()},
    }
    # Haar test: below the floor, are the right singular vectors random unit vectors?
    haar = haar_below_floor(vt, res['i_cross'], n_null=args.n_null, rng_seed=args.seed)
    print(f'--- Haar test (floor crossing at mode i~{res["i_cross"]}) ---')
    print(f'participation ratio: below={haar["pr_below_mean"]:.1f} above={haar["pr_above_mean"]:.1f} '
          f'Haar D/3={res["num_features"]/3:.1f} (null {haar["pr_null_mu"]:.1f}+/-{haar["pr_null_sd"]:.1f}); '
          f'below z={haar["pr_below_z"]:.2f}')
    print(f'diagonal percentile: below={haar["diag_below_mean"]:.3f} above={haar["diag_above_mean"]:.3f} '
          f'(Haar 0.5); below z={haar["diag_below_z"]:.2f}')
    print(f'pooled below-floor sqrt(D) r_jk vs N(0,1): KS={haar["ks_stat"]:.4f} p={haar["ks_p"]:.3g}')

    summary['haar_below_floor'] = {k: (v if not isinstance(v, np.ndarray) else None)
                                   for k, v in haar.items()}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'diagonal_band_summary.json'), 'w', encoding='utf-8') as h:
        json.dump(summary, h, indent=2)
    written = _render(res, run_label, output_dir, args.format)
    written.append(_render_haar(haar, run_label, output_dir, args.format))
    for p in [os.path.join(output_dir, 'diagonal_band_summary.json'), *written]:
        print(f'wrote {p}')


if __name__ == '__main__':
    main()
