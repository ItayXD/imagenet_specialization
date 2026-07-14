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

    # Transverse width via HWHM of denoised, log-spaced longitudinal bins (the per-i
    # cross-sections are individually too noisy). Each bin averages many rows before HWHM.
    band_lo = max(2, i_lo)
    band_hi = min(n, i_hi)
    bin_edges = np.unique(np.round(np.logspace(np.log10(band_lo), np.log10(band_hi), 16)).astype(int))
    width_centers, width_vals = [], []
    for e0, e1 in zip(bin_edges[:-1], bin_edges[1:]):
        prof = _profile_over(band, e0 - 1, e1)
        w = _hwhm(offsets, prof)
        if np.isfinite(w) and w > 0:
            width_centers.append(float(np.sqrt(e0 * e1)))  # geometric bin center
            width_vals.append(w)
    width_bx = np.asarray(width_centers)
    width_by = np.asarray(width_vals)
    width_fit = _loglog_fit(width_bx, width_by)

    # Transverse shape at a few diagonal positions (longitudinally binned to denoise).
    probe_centers = [c for c in (16, 32, 64, 128, 256) if c < n - 4]
    shape_profiles = {}
    shapes = {}
    for c in probe_centers:
        w_bin = max(6, int(0.25 * c))
        prof = _profile_over(band, max(0, c - w_bin), min(n, c + w_bin + 1))
        shape_profiles[c] = prof
        shapes[c] = _transverse_shape(offsets, prof)

    return {
        'offsets': offsets, 'floor': floor, 'positions': positions,
        'amp': amp, 'excess_amp': excess_amp,
        'amp_bx': amp_bx, 'amp_by': amp_by, 'width_bx': width_bx, 'width_by': width_by,
        'long_fit': long_fit, 'width_fit': width_fit,
        'long_range': (i_lo, i_hi), 'shape_profiles': shape_profiles, 'shapes': shapes,
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
    ax.loglog(res['width_bx'], np.clip(res['width_by'], 1e-30, None), 'o', ms=5,
              color='seagreen', label='HWHM (log-binned)')
    wf = res['width_fit']
    if np.isfinite(wf['slope']) and res['width_bx'].size:
        xs = np.linspace(res['width_bx'].min(), res['width_bx'].max(), 50)
        ax.loglog(xs, np.exp(wf['intercept']) * xs ** wf['slope'], color='crimson', lw=2.2,
                  label=fr"fit $i^{{q}}$, $q={wf['slope']:.2f}$ ($R^2={wf['r2']:.3f}$)")
    ax.set_xlabel('diagonal position $i$')
    ax.set_ylabel(r'transverse width HWHM $w(i)$')
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
    print(f'run={run_label}  n={res["n"]}  floor(1/D)={res["floor"]:.3e}')
    print(f'longitudinal A(i)-floor ~ i^-p : p={-lf["slope"]:.3f} (R^2={lf["r2"]:.3f}, '
          f'range i in [{res["long_range"][0]},{res["long_range"][1]}])')
    print(f'transverse width w(i) ~ i^q    : q={wf["slope"]:.3f} (R^2={wf["r2"]:.3f})')
    for c, sh in sorted(res['shapes'].items()):
        print(f'  transverse shape @ i~{c:4d}: best={sh["best"]:11s} '
              f'R2[gauss/exp/pow]={sh["gaussian_r2"]:.3f}/{sh["exponential_r2"]:.3f}/{sh["power_r2"]:.3f} '
              f'(sigma={sh.get("gaussian_sigma", float("nan")):.2f}, ell={sh.get("exponential_ell", float("nan")):.2f})')

    summary = {
        'run_label': run_label, 'n': int(res['n']), 'floor': res['floor'],
        'longitudinal_exponent_p': -lf['slope'], 'longitudinal_r2': lf['r2'],
        'longitudinal_range': list(res['long_range']),
        'width_exponent_q': wf['slope'], 'width_r2': wf['r2'],
        'transverse_shapes': {int(c): sh for c, sh in res['shapes'].items()},
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'diagonal_band_summary.json'), 'w', encoding='utf-8') as h:
        json.dump(summary, h, indent=2)
    written = _render(res, run_label, output_dir, args.format)
    for p in [os.path.join(output_dir, 'diagonal_band_summary.json'), *written]:
        print(f'wrote {p}')


if __name__ == '__main__':
    main()
