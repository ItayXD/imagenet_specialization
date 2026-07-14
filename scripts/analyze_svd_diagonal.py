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
from matplotlib.colors import LogNorm  # noqa: E402

from scripts.analyze_classifier_powerlaw import robust_capacity_fit  # noqa: E402


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


def _rms_width(offsets: np.ndarray, profile: np.ndarray) -> float:
    """Shape-agnostic transverse width: RMS second moment over the above-floor core.

    w = sqrt(sum d^2 e(d) / sum e(d)) with e = floor-subtracted excess. For an exponential
    core w = sqrt(2) * ell; for a Gaussian core w = sigma. So it tracks the true scale
    smoothly across the exponential->Gaussian regime change, unlike a fixed-shape fit.
    """
    abs_d, excess, floor = _symmetric_excess(offsets, profile)
    core = _core_mask(excess, floor)
    d = abs_d[core]
    e = np.clip(excess[core], 0.0, None)
    total = float(e.sum())
    if total <= 0 or d.size < 2:
        return float('nan')
    return float(np.sqrt(np.sum(d ** 2 * e) / total))


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


def _piecewise_powerlaw(x: np.ndarray, y: np.ndarray, *, min_seg: int = 8) -> tuple:
    """Continuous segmented (broken-power-law) fit in log-log: two lines that MEET at the
    break. Grid-search the breakpoint minimizing the total residual; because the fit is
    continuous, the break lands at the genuine change of slope rather than early (noise) or
    late (as a disjoint two-line split does). Returns (breakpoint_x, low_fit, high_fit) with
    a shared model R^2 on each side.
    """
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if x.size < 2 * min_seg + 1:
        return None, _loglog_fit(x, y), {'slope': float('nan'), 'r2': float('nan')}
    order = np.argsort(x)
    lx, ly = np.log(x[order]), np.log(y[order])
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))

    best = None  # (sse, xb, a, b1, b2)
    for bi in range(min_seg, lx.size - min_seg):
        xb = lx[bi]
        hinge = np.maximum(0.0, lx - xb)
        A = np.column_stack([np.ones_like(lx), lx, hinge])
        coef, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
        sse = float(np.sum((A @ coef - ly) ** 2))
        if best is None or sse < best[0]:
            best = (sse, xb, coef[0], coef[1], coef[1] + coef[2])
    sse, xb, a, b1, b2 = best
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float('nan')
    low = {'slope': float(b1), 'intercept': float(a), 'r2': float(r2)}
    high = {'slope': float(b2), 'intercept': float(a + b1 * xb - b2 * xb), 'r2': float(r2)}
    return float(np.exp(xb)), low, high


def _shape_crossover(x: np.ndarray, exp_r2: np.ndarray, gauss_r2: np.ndarray) -> float:
    """First position where the Gaussian fit overtakes the exponential fit (smoothed).

    Uses the sign of a lightly-smoothed (gaussian_r2 - exponential_r2) along i; returns the
    x where it first turns positive and stays positive, or NaN if exp stays preferred.
    """
    m = np.isfinite(x) & np.isfinite(exp_r2) & np.isfinite(gauss_r2)
    x, diff = x[m], (gauss_r2[m] - exp_r2[m])
    if x.size < 5:
        return float('nan')
    order = np.argsort(x)
    x, diff = x[order], diff[order]
    k = min(7, diff.size)
    smooth = np.convolve(diff, np.ones(k) / k, mode='same')
    # First upcrossing: exponential is preferred (diff<0) below, Gaussian (diff>0) above.
    for i in range(1, smooth.size):
        if smooth[i - 1] <= 0.0 < smooth[i]:
            return float(x[i])
    return float('nan')


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
    width_bx_list, ell_i, exp_r2_i, gauss_r2_i = [], [], [], []
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
        if sh['best'] != 'undetermined':
            # Shape-AGNOSTIC width: RMS second moment of the floor-subtracted core. Unlike
            # the exponential length, this stays meaningful when the cross-section turns
            # from exponential to Gaussian, so w(i) is not corrupted by a fixed-shape fit.
            wdt = _rms_width(offsets, prof)
            if np.isfinite(wdt) and wdt > 0:
                width_bx_list.append(float(c))
                ell_i.append(float(wdt))
                exp_r2_i.append(float(sh['exponential_r2']))
                gauss_r2_i.append(float(sh['gaussian_r2']))
                all_verdicts.append(sh['best'])
    width_bx = np.asarray(width_bx_list)
    width_by = np.asarray(ell_i)  # transverse "width" = HWHM w(i), shape-agnostic, dense in i
    exp_r2_arr = np.asarray(exp_r2_i)
    gauss_r2_arr = np.asarray(gauss_r2_i)
    width_fit = _loglog_fit(width_bx, width_by)

    # (1) Is w(i) a power law only up to a break, then something else? Fit the break on
    # LOG-BINNED medians so high-i scatter (band dissolving into the floor) does not jerk
    # the breakpoint around; the raw points are still shown in the figure.
    wbin_x, wbin_y = _log_binned_median(width_bx, width_by, num_bins=20)
    breakpoint, low_fit, high_fit = _piecewise_powerlaw(wbin_x, wbin_y)
    # (2) Shape crossover: the i at which the cross-section stops being exponential-preferred
    # and becomes Gaussian-preferred (a sharp change of transverse shape along the diagonal).
    shape_cross = _shape_crossover(width_bx, exp_r2_arr, gauss_r2_arr)

    shape_verdict = max(set(all_verdicts), key=all_verdicts.count) if all_verdicts else 'undetermined'

    return {
        'offsets': offsets, 'floor': floor, 'positions': positions,
        'amp': amp, 'excess_amp': excess_amp,
        'amp_bx': amp_bx, 'amp_by': amp_by, 'width_bx': width_bx, 'width_by': width_by,
        'wbin_x': wbin_x, 'wbin_y': wbin_y,
        'exp_r2': exp_r2_arr, 'gauss_r2': gauss_r2_arr,
        'long_fit': long_fit, 'width_fit': width_fit, 'shape_verdict': shape_verdict,
        'breakpoint': breakpoint, 'low_fit': low_fit, 'high_fit': high_fit,
        'shape_crossover': shape_cross,
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

    # 2. Transverse length ell(i): single fit + two-segment (breakpoint) fit + shape crossover.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    ax = axes[0]
    ax.loglog(res['width_bx'], np.clip(res['width_by'], 1e-30, None), '.', ms=3, alpha=0.35,
              color='0.6', label=r'RMS $w(i)$ (every $i$)')
    if res['wbin_x'].size:
        ax.loglog(res['wbin_x'], np.clip(res['wbin_y'], 1e-30, None), 'o', ms=5,
                  color='seagreen', label='log-binned median (fit to this)')
    bp, lo_fit, hi_fit = res['breakpoint'], res['low_fit'], res['high_fit']
    if bp is not None and np.isfinite(lo_fit['slope']):
        xlo = np.linspace(res['width_bx'].min(), bp, 40)
        ax.loglog(xlo, np.exp(lo_fit['intercept']) * xlo ** lo_fit['slope'], color='crimson', lw=2.4,
                  label=fr"low: $i^{{{lo_fit['slope']:.2f}}}$ ($R^2$={lo_fit['r2']:.3f})")
        xhi = np.linspace(bp, res['width_bx'].max(), 40)
        if np.isfinite(hi_fit['slope']):
            ax.loglog(xhi, np.exp(hi_fit['intercept']) * xhi ** hi_fit['slope'], color='purple',
                      lw=2.4, label=fr"high: $i^{{{hi_fit['slope']:.2f}}}$ ($R^2$={hi_fit['r2']:.3f})")
        ax.axvline(bp, color='k', ls='--', lw=1.2, label=f'break i~{bp:.0f}')
    if np.isfinite(res['shape_crossover']):
        ax.axvline(res['shape_crossover'], color='darkorange', ls=':', lw=1.6,
                   label=f"exp→gauss i~{res['shape_crossover']:.0f}")
    ax.set_xlabel('diagonal position $i$')
    ax.set_ylabel(r'transverse RMS width $w(i)$ (shape-agnostic)')
    ax.set_title('width vs position (continuous segmented fit)')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8)
    # Right panel: transverse-shape preference (Gaussian R^2 - exponential R^2) vs i.
    ax = axes[1]
    diff = res['gauss_r2'] - res['exp_r2']
    ax.plot(res['width_bx'], diff, '.', ms=4, color='slateblue')
    ax.axhline(0.0, color='0.5', ls='--', lw=1)
    if np.isfinite(res['shape_crossover']):
        ax.axvline(res['shape_crossover'], color='darkorange', ls=':', lw=1.6,
                   label=f"crossover i~{res['shape_crossover']:.0f}")
        ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('diagonal position $i$')
    ax.set_ylabel(r'$R^2_\mathrm{gauss} - R^2_\mathrm{exp}$  (>0: Gaussian better)')
    ax.set_title('transverse shape preference vs position')
    ax.grid(True, which='both', alpha=0.25)
    fig.suptitle(f'Transverse band width & shape vs position — {run_label}')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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


def _render_r2_heatmap(right_sq: np.ndarray, run_label: str, output_dir: str, fmt: str) -> list[str]:
    """r_jk^2 heatmap with square cells (aspect='equal'), so the diagonal has slope 1."""
    written = []
    p, dim = right_sq.shape
    height = 6.0
    width = float(np.clip(height * dim / max(p, 1), 4.0, 22.0))
    for scale in ('linear', 'log'):
        fig, ax = plt.subplots(figsize=(width + 1.3, height))
        if scale == 'log':
            pos = right_sq[right_sq > 0]
            vmax = float(right_sq.max())
            vmin = float(np.quantile(pos, 0.02)) if pos.size else vmax * 1e-6
            im = ax.imshow(right_sq, aspect='equal', origin='upper', cmap='magma',
                           norm=LogNorm(vmin=max(vmin, vmax * 1e-8), vmax=vmax),
                           interpolation='nearest')
        else:
            im = ax.imshow(right_sq, aspect='equal', origin='upper', cmap='magma',
                           vmin=0.0, vmax=float(np.quantile(right_sq, 0.999)),
                           interpolation='nearest')
        ax.set_xlabel(r'PC index $k$ (by eigenvalue $\lambda_k$)')
        ax.set_ylabel(r'singular mode $j$ (by singular value $s_j$)')
        ax.set_title(f'$r_{{jk}}^2$ heatmap ({scale}, square cells) — {run_label}')
        fig.colorbar(im, ax=ax, fraction=0.046 * p / max(dim, 1) + 0.02,
                     label=r'$r_{jk}^2$' + (' (log)' if scale == 'log' else ''))
        path = os.path.join(output_dir, f'fig_svd_r2_heatmap_{scale}.{fmt}')
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        written.append(path)
    return written


def _cliff_onset(ly: np.ndarray) -> int:
    """Index j_hi (1-based) of the last mode BEFORE the finite-dimension cliff.

    The cliff is a contiguous run of trailing modes whose per-step log-drop is far larger
    than the bulk (a sharp collapse, e.g. CIFAR's rank-deficient last mode). Only the tail
    is cut -- the head is always kept.
    """
    p = ly.size
    if p < 6:
        return p
    drop = -np.diff(ly)  # drop[i] = ly[i]-ly[i+1] >= ~0 in the bulk
    mid = drop[p // 4: max(p // 4 + 1, 3 * p // 4)]
    med = float(np.median(mid))
    mad = float(np.median(np.abs(mid - med))) or 1e-9
    hi = p  # keep [1..hi]
    for i in range(p - 2, -1, -1):  # step i links mode i+1 and i+2 (1-based)
        if drop[i] > med + 6.0 * 1.4826 * mad and drop[i] > 2.5 * max(med, 1e-9):
            hi = i + 1  # exclude modes after i+1
        else:
            break
    return max(hi, max(6, p // 2))  # never trim more than the tail half


def robust_singular_powerlaw(s: np.ndarray, num_bins: int = 24) -> dict:
    """Power-law fit s_j ~ j^{-c} keeping the head, excluding ONLY the sharp finite-
    dimension cliff at the tail.

    The fit is over LOG-SPACED-BINNED (j, s) on [1, j_hi]: binning gives every decade
    equal weight, so the numerous mid/tail modes don't dominate and steepen the slope past
    the (kept) head. The line then hugs the whole curve from head to cliff; residual
    curvature shows up as a lower R^2.
    """
    s = np.asarray(s, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    p = s.size
    j = np.arange(1, p + 1, dtype=np.float64)
    lx, ly = np.log(j), np.log(s)
    if p < 4:
        sl, ic = (np.polyfit(lx, ly, 1) if p >= 2 else (float('nan'), float('nan')))
        return {'c': -float(sl), 'intercept': float(ic), 'r2': float('nan'),
                'j_lo': 1, 'j_hi': p, 'n_excluded': 0, 'method': 'OLS', 'j': j, 's': s}

    j_hi = _cliff_onset(ly)
    bx, by = _log_binned_median(j[:j_hi], s[:j_hi], num_bins=min(num_bins, max(4, j_hi)))
    if bx.size >= 2:
        lbx, lby = np.log(bx), np.log(by)
        sl, ic = np.polyfit(lbx, lby, 1)
        pred = sl * lbx + ic
        ss_res = float(np.sum((lby - pred) ** 2))
        ss_tot = float(np.sum((lby - lby.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    else:  # too few bins (e.g. CIFAR): plain OLS on the kept modes
        sl, ic = np.polyfit(lx[:j_hi], ly[:j_hi], 1)
        r2 = float('nan')
    return {'c': -float(sl), 'intercept': float(ic), 'r2': r2,
            'j_lo': 1, 'j_hi': int(j_hi), 'n_excluded': int(p - j_hi),
            'method': 'log-binned OLS, sharp-tail trimmed (head kept)', 'j': j, 's': s}


def _render_singular(sfit: dict, run_label: str, output_dir: str, fmt: str) -> str:
    j, s = sfit['j'], sfit['s']
    j_lo, j_hi = sfit['j_lo'], sfit['j_hi']
    inside = (j >= j_lo) & (j <= j_hi)
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    if j_lo > 1:
        ax.axvspan(0.8, j_lo, color='0.85', alpha=0.5, zorder=0)
    if j_hi < j[-1]:
        ax.axvspan(j_hi, j[-1] * 1.05, color='0.85', alpha=0.5, zorder=0)
    ax.loglog(j[inside], s[inside], 'o', ms=4, color='steelblue', label=f'fit modes [{j_lo},{j_hi}]')
    ax.loglog(j[~inside], s[~inside], 'x', ms=7, color='crimson', label='excluded (head/cliff)')
    if np.isfinite(sfit['c']):
        xs = np.linspace(j_lo, j_hi, 60)
        ax.loglog(xs, np.exp(sfit['intercept']) * xs ** (-sfit['c']), color='crimson', lw=2.4,
                  label=fr"$j^{{-c}}$, $c={sfit['c']:.2f}$ ($R^2={sfit['r2']:.3f}$, {sfit['method']})")
    ax.set_xlabel('singular mode index $j$')
    ax.set_ylabel(r'singular value $s_j$')
    ax.set_title(f'Classifier singular-value spectrum (robust) — {run_label}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8)
    path = os.path.join(output_dir, f'fig_svd_singular_values_robust.{fmt}')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    npz_path = os.path.join(args.results_dir, 'powerlaw_arrays.npz')
    data = np.load(npz_path, allow_pickle=True)
    what_hat = np.asarray(data['W_hat'], dtype=np.float64)
    run_label = str(data['run_label']) if 'run_label' in data.files else 'classifier_svd'
    _, s_values, vt = np.linalg.svd(what_hat, full_matrices=False)
    right_sq = vt ** 2
    p_modes = int(min(vt.shape))

    svd_dir = os.path.abspath(args.output_dir) if args.output_dir \
        else os.path.join(os.path.abspath(args.results_dir), 'svd')
    output_dir = os.path.join(svd_dir, 'right')  # all R-related outputs live under svd/right/
    os.makedirs(svd_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    written = []

    # (4) Robust singular-value power law s_j ~ j^-c (Theil-Sen, cliff-excluded). Always run
    # (this is the one non-R quantity requested); lives under svd/ (NOT right/).
    sfit = robust_singular_powerlaw(s_values)
    print(f'run={run_label}  D={right_sq.shape[1]}  modes p={p_modes}')
    print(f'singular values s_j ~ j^-c   : c={sfit["c"]:.3f} (R^2={sfit["r2"]:.3f}, '
          f'j in [{sfit["j_lo"]},{sfit["j_hi"]}], excluded {sfit["n_excluded"]} cliff modes)')
    with open(os.path.join(svd_dir, 'singular_powerlaw.json'), 'w', encoding='utf-8') as h:
        json.dump({k: sfit[k] for k in ('c', 'intercept', 'r2', 'j_lo', 'j_hi', 'n_excluded')}, h, indent=2)
    sing_path = _render_singular(sfit, run_label, svd_dir, args.format)
    written.append(sing_path)

    # The diagonal-band / Haar analysis of R needs many singular modes; skip when too few
    # (e.g. CIFAR-5M has C=10 -> 10 modes). Only R data is generated, all under svd/right/.
    if p_modes < 32:
        print(f'only {p_modes} singular modes; skipping the R diagonal-band/Haar analysis '
              f'(singular-value fit + r^2 heatmap still written).')
        # The r^2 heatmap is still an informative R visualization even with few modes.
        written.extend(_render_r2_heatmap(right_sq, run_label, output_dir, args.format))
        for pth in written:
            print(f'wrote {pth}')
        return

    res = analyze(right_sq, max_offset=args.max_offset, transverse_smooth=args.transverse_smooth)
    lf, wf = res['long_fit'], res['width_fit']
    mean_floor_1_over_D = 1.0 / res['num_features']
    print(f'  n={res["n"]}  median floor={res["floor"]:.3e}  (mean 1/D={mean_floor_1_over_D:.3e})')
    print(f'longitudinal A(i)-floor ~ i^-p : p={-lf["slope"]:.3f} (R^2={lf["r2"]:.3f})')
    print(f'transverse shape (majority)    : {res["shape_verdict"]}')
    print(f'transverse length ell(i) ~ i^q : q={wf["slope"]:.3f} (R^2={wf["r2"]:.3f}, '
          f'n_probes={res["width_bx"].size})')
    bp, lo_fit, hi_fit = res['breakpoint'], res['low_fit'], res['high_fit']
    if bp is not None:
        print(f'  piecewise ell(i): break at i~{bp:.0f}; '
              f'low i^{lo_fit["slope"]:.2f} (R^2={lo_fit["r2"]:.3f}), '
              f'high i^{hi_fit["slope"]:.2f} (R^2={hi_fit["r2"]:.3f})')
    print(f'  transverse shape crossover (exp->gauss) at i~{res["shape_crossover"]:.0f}')

    haar = haar_below_floor(vt, res['i_cross'], n_null=args.n_null, rng_seed=args.seed)
    print(f'--- Haar test (floor crossing at mode i~{res["i_cross"]}) ---')
    print(f'participation ratio: below={haar["pr_below_mean"]:.1f} above={haar["pr_above_mean"]:.1f} '
          f'Haar D/3={res["num_features"]/3:.1f}; below z={haar["pr_below_z"]:.2f}')
    print(f'diagonal percentile: below={haar["diag_below_mean"]:.3f} above={haar["diag_above_mean"]:.3f} '
          f'(Haar 0.5); below z={haar["diag_below_z"]:.2f}')

    summary = {
        'run_label': run_label, 'n': int(res['n']), 'num_features': int(res['num_features']),
        'median_floor': res['floor'], 'mean_floor_1_over_D': mean_floor_1_over_D,
        'longitudinal_exponent_p': -lf['slope'], 'longitudinal_r2': lf['r2'],
        'longitudinal_range': list(res['long_range']),
        'transverse_shape_verdict': res['shape_verdict'],
        'transverse_length_exponent_q': wf['slope'], 'transverse_length_r2': wf['r2'],
        'ell_breakpoint': bp, 'ell_low_exponent': lo_fit['slope'], 'ell_low_r2': lo_fit['r2'],
        'ell_high_exponent': hi_fit['slope'], 'ell_high_r2': hi_fit['r2'],
        'shape_crossover_i': res['shape_crossover'],
        'transverse_shapes': {int(c): sh for c, sh in res['shapes'].items()},
        'haar_below_floor': {k: (v if not isinstance(v, np.ndarray) else None) for k, v in haar.items()},
        'singular_powerlaw': {k: sfit[k] for k in ('c', 'r2', 'j_lo', 'j_hi', 'n_excluded')},
    }

    # Remove this script's earlier outputs that used to live directly under svd/ (now right/).
    for stale in ('fig_svd_diag_longitudinal', 'fig_svd_diag_width', 'fig_svd_diag_transverse',
                  'fig_svd_haar_below_floor', 'diagonal_band_summary'):
        for ext in ('png', 'pdf', 'json'):
            old = os.path.join(svd_dir, f'{stale}.{ext}')
            if os.path.exists(old):
                os.remove(old)

    with open(os.path.join(output_dir, 'diagonal_band_summary.json'), 'w', encoding='utf-8') as h:
        json.dump(summary, h, indent=2)
    written.extend(_render(res, run_label, output_dir, args.format))
    written.append(_render_haar(haar, run_label, output_dir, args.format))
    written.extend(_render_r2_heatmap(right_sq, run_label, output_dir, args.format))
    written.append(os.path.join(output_dir, 'diagonal_band_summary.json'))
    for pth in written:
        print(f'wrote {pth}')


if __name__ == '__main__':
    main()
