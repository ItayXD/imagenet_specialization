#!/usr/bin/env python3
"""Per-class source/capacity structure of a trained ResNet18 classifier.

Freezes the penultimate features h(x) of a trained ResNet18, diagonalizes their
centered covariance to obtain the data eigenbasis V (eigenvalues lambda_k), rotates
the softmax-gauge-fixed classifier into that basis (What = W V), and studies:

  * capacity condition   lambda_k        ~ k^{-b}
  * source condition     What_ik^2       ~ A_i^2 k^{-2 a_i}   (per class i)
  * residual error mass  sum_{k>m} lambda_k What_ik^2         (per class i)

It then tests the functional prediction that classes with smaller source exponent
a_i (heavier tails relative to the data spectrum) recover accuracy more slowly under
PCA truncation of the features to the top-m principal components.

This is a data-modeling / spectral-learning (source & capacity) analysis; a_i and b
describe the data geometry and the class-target alignment, not weight optimization.

Reuses checkpoint restore / run resolution helpers from analyze_exchangeability.py
and the model/run conventions from eval_singular_ablation.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# --------------------------------------------------------------------------------------
# Pure-numpy analysis helpers (no JAX / no data; unit-tested in test_classifier_powerlaw)
# --------------------------------------------------------------------------------------
def row_center_classifier(weight: np.ndarray) -> np.ndarray:
    """Remove the softmax gauge: W <- W - (1/C) 1 1^T W (subtract per-column class mean).

    weight has shape (C, D); the returned matrix has zero mean along the class axis for
    every feature/PC column, which leaves softmax outputs (and hence loss/accuracy)
    unchanged while fixing the otherwise-free additive gauge of the logits.
    """
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError(f'Expected a 2D classifier weight (C, D); got shape {weight.shape}.')
    return weight - weight.mean(axis=0, keepdims=True)


def feature_pca(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center features and diagonalize their covariance.

    Returns (feature_mean (D,), eigenvalues (D,) descending & clipped >=0,
    eigenvectors V (D, D) with columns ordered to match the eigenvalues).
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f'Expected features of shape (N, D); got {features.shape}.')
    num_samples = int(features.shape[0])
    if num_samples < 2:
        raise ValueError('Need at least two samples to estimate a feature covariance.')
    feature_mean = features.mean(axis=0)
    centered = features - feature_mean
    covariance = (centered.T @ centered) / float(num_samples)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # ascending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    return feature_mean, eigenvalues, eigenvectors


def loglog_binned_slope(
    positions: np.ndarray,
    values: np.ndarray,
    num_bins: int,
) -> dict[str, Any]:
    """Fit log(binmean(values)) ~= const + slope * log(position) over log-spaced bins.

    positions are 1-indexed PC indices k (1..D); values are non-negative quantities
    (e.g. lambda_k or What_ik^2). Bins are log-spaced over the position range; within
    each bin the mean of the positive values is taken and placed at the geometric-mean
    position. Bins with no positive value are skipped. Returns slope, intercept, the
    log-space RMSE of the fit, and the bin centers/means used (for plotting).
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if positions.shape != values.shape:
        raise ValueError('positions and values must have the same shape.')
    num_bins = int(num_bins)
    if num_bins < 2:
        raise ValueError('num_bins must be at least 2.')

    positive = positions > 0
    positions = positions[positive]
    values = values[positive]
    if positions.size == 0:
        raise ValueError('No positive positions available for the log-log fit.')

    lo = float(positions.min())
    hi = float(positions.max())
    if hi <= lo:
        raise ValueError('Need a nontrivial position range for log-spaced binning.')
    edges = np.logspace(np.log10(lo), np.log10(hi), num_bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)  # make the last bin right-inclusive
    bin_index = np.digitize(positions, edges) - 1

    centers: list[float] = []
    means: list[float] = []
    for b in range(num_bins):
        mask = bin_index == b
        if not np.any(mask):
            continue
        bin_values = values[mask]
        finite_positive = bin_values[np.isfinite(bin_values) & (bin_values > 0.0)]
        if finite_positive.size == 0:
            continue
        centers.append(float(np.exp(np.mean(np.log(positions[mask])))))
        means.append(float(np.mean(finite_positive)))

    if len(centers) < 2:
        return {
            'slope': float('nan'),
            'intercept': float('nan'),
            'log_rmse': float('nan'),
            'bin_centers': np.asarray(centers, dtype=np.float64),
            'bin_means': np.asarray(means, dtype=np.float64),
            'num_points': len(centers),
        }

    log_centers = np.log(np.asarray(centers, dtype=np.float64))
    log_means = np.log(np.asarray(means, dtype=np.float64))
    slope, intercept = np.polyfit(log_centers, log_means, deg=1)
    residuals = log_means - (slope * log_centers + intercept)
    log_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'log_rmse': log_rmse,
        'bin_centers': np.asarray(centers, dtype=np.float64),
        'bin_means': np.asarray(means, dtype=np.float64),
        'num_points': len(centers),
    }


def _local_loglog_slopes(log_k: np.ndarray, log_v: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window local log-log slope at every index (NaN where undefined)."""
    n = log_k.size
    half = max(2, window // 2)
    slopes = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        xs = log_k[lo:hi]
        ys = log_v[lo:hi]
        good = np.isfinite(ys)
        if int(good.sum()) >= 3 and np.ptp(xs[good]) > 0:
            slopes[i] = np.polyfit(xs[good], ys[good], deg=1)[0]
    return slopes


def select_bulk_window(
    values: np.ndarray,
    *,
    local_window: int | None = None,
    slope_tol: float = 0.5,
    min_window: int = 8,
) -> tuple[int, int, dict[str, Any]]:
    """Locate the asymptotic power-law (bulk) regime of a spectrum, 1-indexed [k_lo, k_hi].

    A spectrum like lambda_k typically has three regimes: an early/head law (small k),
    the asymptotic bulk power law, and a finite-dimension drop where the last PCs fall
    off a cliff. This estimates a bulk slope from the central region, then expands a
    contiguous window outward from the center while the sliding local slope stays within
    ``slope_tol`` of that bulk slope, so the head and the finite-dimension tail (both of
    which have markedly different local slopes) are excluded.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.size
    log_k = np.log(np.arange(1, n + 1, dtype=np.float64))
    log_v = np.where(values > 0, np.log(np.where(values > 0, values, 1.0)), np.nan)
    if local_window is None:
        local_window = max(5, n // 15)

    slopes = _local_loglog_slopes(log_k, log_v, local_window)
    finite_idx = np.where(np.isfinite(slopes))[0]
    if finite_idx.size < max(min_window, 4):
        return 1, n, {'b_bulk': float('nan'), 'local_slopes': slopes, 'fallback': True}

    # Bulk slope from the central 20-60% of the (finite) index range.
    lo_c = finite_idx[int(0.20 * (finite_idx.size - 1))]
    hi_c = finite_idx[int(0.60 * (finite_idx.size - 1))]
    central = slopes[lo_c:hi_c + 1]
    central = central[np.isfinite(central)]
    b_bulk = float(np.median(central)) if central.size else float(np.nanmedian(slopes))
    center = (lo_c + hi_c) // 2

    left = center
    while left - 1 >= 0 and np.isfinite(slopes[left - 1]) and abs(slopes[left - 1] - b_bulk) <= slope_tol:
        left -= 1
    right = center
    while right + 1 < n and np.isfinite(slopes[right + 1]) and abs(slopes[right + 1] - b_bulk) <= slope_tol:
        right += 1

    k_lo, k_hi = left + 1, right + 1  # 1-indexed
    if k_hi - k_lo + 1 < min_window:
        # Widen symmetrically around the center to guarantee a usable window.
        pad = (min_window - (k_hi - k_lo + 1) + 1) // 2
        k_lo = max(1, k_lo - pad)
        k_hi = min(n, k_hi + pad)
    return int(k_lo), int(k_hi), {'b_bulk': b_bulk, 'local_slopes': slopes, 'fallback': False}


def _robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust slope/intercept via Theil-Sen (median of pairwise slopes); OLS fallback."""
    if x.size < 2:
        return float('nan'), float('nan')
    try:
        from scipy.stats import theilslopes

        slope, intercept, _, _ = theilslopes(y, x)
        return float(slope), float(intercept)
    except Exception:
        slope, intercept = np.polyfit(x, y, deg=1)
        return float(slope), float(intercept)


def robust_loglog_slope(
    positions: np.ndarray,
    values: np.ndarray,
    num_bins: int,
    *,
    k_lo: int,
    k_hi: int,
) -> dict[str, Any]:
    """Robust log-log slope over the window [k_lo, k_hi].

    Two robustness layers stacked on top of restricting to the bulk window: within each
    log-spaced bin the *median* of the values is used (robust to per-bin outliers), and
    the slope across bins is fit with Theil-Sen (robust to a bad bin). Returns slope,
    intercept, log-RMSE, and the bin centers/values used.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    sel = (positions >= k_lo) & (positions <= k_hi) & np.isfinite(values) & (values > 0)
    x = positions[sel]
    y = values[sel]
    nan_result = {
        'slope': float('nan'), 'intercept': float('nan'), 'log_rmse': float('nan'),
        'bin_centers': np.zeros(0), 'bin_values': np.zeros(0), 'num_points': 0,
    }
    if x.size < 3:
        return nan_result

    edges = np.logspace(np.log10(float(k_lo)), np.log10(float(k_hi)), int(num_bins) + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    bin_index = np.digitize(x, edges) - 1
    centers: list[float] = []
    reps: list[float] = []
    for b in range(int(num_bins)):
        mask = bin_index == b
        if not np.any(mask):
            continue
        centers.append(float(np.exp(np.mean(np.log(x[mask])))))
        reps.append(float(np.median(y[mask])))  # robust within-bin representative
    centers_arr = np.asarray(centers, dtype=np.float64)
    reps_arr = np.asarray(reps, dtype=np.float64)
    good = reps_arr > 0
    if int(good.sum()) < 2:
        return nan_result

    log_c = np.log(centers_arr[good])
    log_r = np.log(reps_arr[good])
    slope, intercept = _robust_line(log_c, log_r)
    residuals = log_r - (slope * log_c + intercept)
    return {
        'slope': slope,
        'intercept': intercept,
        'log_rmse': float(np.sqrt(np.mean(residuals ** 2))),
        'bin_centers': centers_arr[good],
        'bin_values': reps_arr[good],
        'num_points': int(good.sum()),
    }


def _ols_slope(log_x: np.ndarray, log_y: np.ndarray) -> float:
    if log_x.size < 2 or np.ptp(log_x) <= 0:
        return float('nan')
    return float(np.polyfit(log_x, log_y, deg=1)[0])


def _grow_window_slope(
    fit_fn,
    x_min: float,
    x_max: float,
    anchor: float,
    span: float,
    *,
    init_frac: float,
    step_frac: float,
    rmse_tol: float,
) -> tuple[float, float, float, int]:
    """Grow a fit window outward from an anchor while it still fits a single line well.

    ``fit_fn(log_lo, log_hi) -> (slope, rmse)`` fits over [exp(log_lo), exp(log_hi)] using
    the same binned-robust estimator used for plotting/fig1. Start from a small window
    (fraction ``init_frac`` of the log-range ``span``) around ``anchor`` and repeatedly
    extend whichever side keeps the binned RMSE lowest, but only while it stays below
    ``rmse_tol``. When extending either side would push RMSE past the tolerance (head/tail
    curvature bends the line), stop. Returns (log_lo, log_hi, slope, n_steps).
    """
    step = max(step_frac * span, 1e-6)
    lo = max(x_min, anchor - init_frac * span)
    hi = min(x_max, anchor + init_frac * span)

    slope, _ = fit_fn(lo, hi)
    while not np.isfinite(slope) and (lo > x_min or hi < x_max):
        lo = max(x_min, lo - step)
        hi = min(x_max, hi + step)
        slope, _ = fit_fn(lo, hi)
    if not np.isfinite(slope):
        return lo, hi, float('nan'), 0

    steps = 0
    while True:
        options = []
        if hi < x_max:
            new_hi = min(x_max, hi + step)
            s, r = fit_fn(lo, new_hi)
            if np.isfinite(r) and r <= rmse_tol:
                options.append((r, lo, new_hi, s))
        if lo > x_min:
            new_lo = max(x_min, lo - step)
            s, r = fit_fn(new_lo, hi)
            if np.isfinite(r) and r <= rmse_tol:
                options.append((r, new_lo, hi, s))
        if not options:
            break  # extending either side breaks the single-line fit -> regime edge
        options.sort(key=lambda o: o[0])
        _, lo, hi, slope = options[0]
        steps += 1
        if lo <= x_min and hi >= x_max:
            break
    return lo, hi, slope, steps


def detect_head_cliff(values: np.ndarray, *, grid: int = 200, smooth_frac: float = 0.04,
                      curvature_tol: float = 0.05) -> tuple[int, int]:
    """Data-driven [k_lo, k_hi] trim of a spectrum's early head and finite-dimension cliff.

    Works in log-log on a uniform log-k grid (so it does not assume the bulk sits at any
    particular index position). The head is the sharpest concave-up knee in the low-k
    third; the cliff is the sharpest downward bend in the high-k portion. Each trim is
    applied only if its knee curvature exceeds ``curvature_tol`` (so a clean single power
    law, whose curvature is ~0, is not trimmed). Returns 1-based inclusive [k_lo, k_hi].
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.size
    k = np.arange(1, n + 1, dtype=np.float64)
    mask = values > 0
    if int(mask.sum()) < 8:
        return 1, n
    log_k = np.log(k[mask])
    log_v = np.log(values[mask])
    g = np.linspace(log_k[0], log_k[-1], grid)
    vg = np.interp(g, log_k, log_v)
    win = max(3, int(smooth_frac * grid))
    kern = np.ones(win) / win
    vs = np.convolve(vg, kern, mode='same')
    d2 = np.gradient(np.gradient(vs, g), g)

    lo_i = int(0.03 * grid)
    head_region = d2[lo_i:int(0.45 * grid)]
    cliff_region = d2[int(0.55 * grid):]
    k_lo, k_hi = 1, n
    if head_region.size:
        h = lo_i + int(np.argmax(head_region))
        if d2[h] > curvature_tol:  # a real concave-up head knee
            k_lo = max(1, int(round(np.exp(g[h]))))
    if cliff_region.size:
        c = int(0.55 * grid) + int(np.argmin(cliff_region))
        if d2[c] < -curvature_tol:  # a real downward cliff bend
            k_hi = min(n, int(round(np.exp(g[c]))))
    if k_hi - k_lo < max(8, n // 50):  # guard against an over-aggressive trim
        return 1, n
    return k_lo, k_hi


def robust_powerlaw_fit(
    positions: np.ndarray,
    values: np.ndarray,
    num_bins: int,
    *,
    n_seeds: int = 64,
    rng_seed: int = 0,
    init_frac: float = 0.06,
    step_frac: float = 0.02,
    rmse_tol: float = 0.12,
    min_points: int = 10,
    head_floor: float = 30.0,
) -> dict[str, Any]:
    """Seed-and-grow power-law fit of the clean regime, climbing toward large eigenvalues.

    The head (the top ~``head_floor`` eigenvalues) is a hard lower bound; the finite-
    dimension collapse (detect_head_cliff) is the upper bound. Anchors are planted in the
    lower-mid band and each window grows BOTH ways while the binned-robust log-log RMSE
    stays <= rmse_tol: downward it climbs toward the large eigenvalues and stops at either
    the head floor or where the head-transition curvature raises RMSE (whichever comes
    first); upward it extends toward the collapse. Anchoring in the mid band (not the tail)
    prevents seeds getting stuck in the small-eigenvalue tail. Reports the MEDIAN seed
    slope with the across-seed std as the error, plus a per-k exclusion frequency for the
    graded fig1 shading.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = (positions > 0) & (values > 0) & np.isfinite(values)
    log_x = np.log(positions[mask])
    order = np.argsort(log_x)
    log_x = log_x[order]

    def fit_fn(log_lo: float, log_hi: float) -> tuple[float, float]:
        k_lo = max(1, int(round(np.exp(log_lo))))
        k_hi = int(round(np.exp(log_hi)))
        in_win = int(np.sum((positions >= k_lo) & (positions <= k_hi) & (values > 0)))
        if in_win < min_points or k_hi <= k_lo:
            return float('nan'), float('nan')
        r = robust_loglog_slope(positions, values, num_bins, k_lo=k_lo, k_hi=k_hi)
        return r['slope'], r['log_rmse']

    if log_x.size < max(2 * min_points, 4):
        s, _ = fit_fn(log_x[0], log_x[-1]) if log_x.size >= 2 else (float('nan'), None)
        lo_k = int(round(np.exp(log_x[0]))) if log_x.size else 1
        hi_k = int(round(np.exp(log_x[-1]))) if log_x.size else 1
        return {'slope': s, 'slope_sem': float('nan'), 'slope_std': float('nan'),
                'k_lo': max(1, lo_k), 'k_hi': max(1, hi_k),
                'exclusion_frac': np.zeros(values.size, dtype=np.float64),
                'seed_slopes': np.asarray([s]), 'n_seeds_used': 1}

    # Head floor (hard lower bound: top ~head_floor eigenvalues), capped for tiny D; the
    # finite-dimension collapse is the upper bound. Anchor in the lower-mid band [floor,
    # sqrt(floor*cliff)] and grow both ways (downward toward the large eigenvalues, upward
    # toward the collapse) so no seed gets stuck in the small-eigenvalue tail.
    n_positions = int(positions.max())
    floor = float(min(head_floor, max(1, n_positions // 10)))
    _, cliff = detect_head_cliff(values)
    cliff = float(max(floor + 1.0, cliff))
    x_min = float(np.log(floor))
    x_max = float(np.log(cliff))
    span = x_max - x_min
    rng = np.random.default_rng(int(rng_seed))
    anchors = np.log(rng.uniform(floor, float(np.sqrt(floor * cliff)), size=int(n_seeds)))

    slopes: list[float] = []
    los: list[float] = []
    his: list[float] = []
    for anchor in anchors:
        lo, hi, slope, _ = _grow_window_slope(
            fit_fn, x_min, x_max, float(anchor), span,
            init_frac=init_frac, step_frac=step_frac, rmse_tol=rmse_tol,
        )
        if np.isfinite(slope):
            slopes.append(slope)
            los.append(lo)
            his.append(hi)
    if not slopes:
        s, _ = fit_fn(log_x[0], log_x[-1])
        return {'slope': s, 'slope_sem': float('nan'), 'slope_std': float('nan'),
                'k_lo': int(max(1, round(np.exp(log_x[0])))),
                'k_hi': int(round(np.exp(log_x[-1]))),
                'exclusion_frac': np.zeros(values.size, dtype=np.float64),
                'seed_slopes': np.asarray([s]), 'n_seeds_used': 0}

    slopes_arr = np.asarray(slopes, dtype=np.float64)
    los_arr = np.asarray(los)
    his_arr = np.asarray(his)
    n = slopes_arr.size

    # Point estimate = MEDIAN across seeds (robust to a seed that strays into a steeper
    # sub-regime of a curved bulk); error = standard error of the mean between seeds.
    # Two uncertainties: within-window (the fit) and about-the-window (which k each seed
    # keeps). The latter is summarized per-k as the fraction of seeds that EXCLUDED k ->
    # exclusion_frac (0 = every seed kept it, 1 = every seed dropped it), for the graded
    # fig1 shading. The consensus window (kept by a majority of seeds) is used only as a
    # single [k_lo, k_hi] for the per-class source fits.
    slope_med = float(np.median(slopes_arr))
    slope_sem = float(np.std(slopes_arr, ddof=1) / np.sqrt(n)) if n > 1 else float('nan')

    log_k_all = np.log(np.arange(1, values.size + 1, dtype=np.float64))
    inclusion = np.zeros(values.size, dtype=np.float64)
    for lo, hi in zip(los_arr, his_arr):
        inclusion += (log_k_all >= lo) & (log_k_all <= hi)
    exclusion_frac = 1.0 - inclusion / float(n)
    majority = np.where(inclusion >= (n / 2.0))[0]
    if majority.size:
        k_lo = int(majority[0] + 1)
        k_hi = int(majority[-1] + 1)
    else:
        k_lo = int(max(1, round(np.exp(float(np.median(los_arr))))))
        k_hi = int(round(np.exp(float(np.median(his_arr)))))
    return {
        'slope': slope_med,
        'slope_sem': slope_sem,
        'slope_std': float(np.std(slopes_arr, ddof=1)) if n > 1 else float('nan'),
        'k_lo': k_lo,
        'k_hi': k_hi,
        'exclusion_frac': exclusion_frac,
        'seed_slopes': slopes_arr,
        'n_seeds_used': int(n),
    }


def robust_capacity_fit(eigenvalues: np.ndarray, num_bins: int, **seed_kwargs) -> dict[str, Any]:
    """Capacity exponent b (lambda_k ~ k^{-b}) via seed-and-grow, with an error bar.

    The window comes from robust_powerlaw_fit; the plotted line's slope/intercept/rmse
    come from a binned robust fit over that window; the error bar is the across-seed std.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    positions = np.arange(1, eigenvalues.size + 1, dtype=np.float64)
    seed = robust_powerlaw_fit(positions, eigenvalues, num_bins, **seed_kwargs)
    k_lo, k_hi = seed['k_lo'], seed['k_hi']
    # Line drawn with the seed-mean slope, intercept least-squares matched over the
    # consensus window so it sits on the data.
    b = float(-seed['slope'])
    consensus = (positions >= k_lo) & (positions <= k_hi) & (eigenvalues > 0)
    if int(consensus.sum()) >= 2:
        log_k = np.log(positions[consensus])
        log_v = np.log(eigenvalues[consensus])
        intercept = float(np.mean(log_v + b * log_k))  # mean residual with slope fixed at -b
        resid = log_v - (-b * log_k + intercept)
        log_rmse = float(np.sqrt(np.mean(resid ** 2)))
    else:
        intercept, log_rmse = float('nan'), float('nan')
    return {
        'b': b,
        'b_sem': float(seed['slope_sem']),
        'b_std': float(seed['slope_std']),
        'intercept': intercept,
        'log_rmse': log_rmse,
        'k_lo': int(k_lo),
        'k_hi': int(k_hi),
        'exclusion_frac': np.asarray(seed['exclusion_frac'], dtype=np.float64),
        'seed_slopes': seed['seed_slopes'],
        'n_seeds_used': seed['n_seeds_used'],
    }


def fit_source_exponents(
    what_squared: np.ndarray,
    num_bins: int,
    *,
    k_lo: int,
    k_hi: int,
) -> dict[str, np.ndarray]:
    """Fit per-class source exponents a_i from What_ik^2 ~ A_i^2 k^{-2 a_i}.

    The fit is restricted to the bulk window [k_lo, k_hi] (excluding the head and the
    finite-dimension tail) and uses the robust log-log estimator. what_squared is (C, D).
    Returns arrays a (C,), log_A2 (C,), log_rmse (C,).
    """
    what_squared = np.asarray(what_squared, dtype=np.float64)
    num_classes, num_features = what_squared.shape
    positions = np.arange(1, num_features + 1, dtype=np.float64)
    a = np.full(num_classes, np.nan, dtype=np.float64)
    log_a2 = np.full(num_classes, np.nan, dtype=np.float64)
    log_rmse = np.full(num_classes, np.nan, dtype=np.float64)
    for i in range(num_classes):
        fit = robust_loglog_slope(positions, what_squared[i], num_bins, k_lo=k_lo, k_hi=k_hi)
        a[i] = -0.5 * fit['slope']  # slope = -2 a_i
        log_a2[i] = fit['intercept']
        log_rmse[i] = fit['log_rmse']
    return {'a': a, 'log_A2': log_a2, 'log_rmse': log_rmse}


def fit_capacity_exponent(
    eigenvalues: np.ndarray,
    num_bins: int,
    *,
    k_lo: int,
    k_hi: int,
) -> dict[str, Any]:
    """Fit the capacity exponent b from lambda_k ~ k^{-b} over the bulk window."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    positions = np.arange(1, eigenvalues.size + 1, dtype=np.float64)
    fit = robust_loglog_slope(positions, eigenvalues, num_bins, k_lo=k_lo, k_hi=k_hi)
    return {
        'b': -float(fit['slope']),  # slope = -b
        'intercept': float(fit['intercept']),
        'log_rmse': float(fit['log_rmse']),
        'bin_centers': fit['bin_centers'],
        'bin_values': fit['bin_values'],
    }


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def per_class_accuracy_and_ce(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class accuracy and cross-entropy grouped by true label.

    Classes with no example present are reported as NaN.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = logits.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    log_probs = _log_softmax(logits)
    ce = -log_probs[np.arange(labels.size), labels]

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    correct_sum = np.bincount(labels, weights=correct, minlength=num_classes)
    ce_sum = np.bincount(labels, weights=ce, minlength=num_classes)
    with np.errstate(invalid='ignore', divide='ignore'):
        acc_i = np.where(counts > 0, correct_sum / counts, np.nan)
        ce_i = np.where(counts > 0, ce_sum / counts, np.nan)
    return acc_i, ce_i


def truncated_logits(
    coeffs: np.ndarray,
    what_eff: np.ndarray,
    base_logits: np.ndarray,
    m: int,
) -> np.ndarray:
    """Logits from features projected onto the top-m PCs.

    Uses the identity logits_m = base + coeffs[:, :m] @ What_eff[:, :m]^T, where
    coeffs = (h - h_mean) V, What_eff = W_eff V, and base = h_mean @ W_eff^T + b.
    """
    m = int(m)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    what_eff = np.asarray(what_eff, dtype=np.float64)
    base_logits = np.asarray(base_logits, dtype=np.float64)
    if m <= 0:
        return np.broadcast_to(base_logits, (coeffs.shape[0], base_logits.shape[0])).copy()
    m = min(m, coeffs.shape[1])
    return base_logits[None, :] + coeffs[:, :m] @ what_eff[:, :m].T


def residual_tail_mass(
    eigenvalues: np.ndarray,
    what_squared: np.ndarray,
    trunc_m: np.ndarray,
) -> np.ndarray:
    """Per-class residual mass sum_{k>m} lambda_k What_ik^2 for each m in trunc_m.

    Returns an array of shape (C, len(trunc_m)).
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    what_squared = np.asarray(what_squared, dtype=np.float64)
    trunc_m = np.asarray(trunc_m, dtype=np.int64).reshape(-1)
    weighted = what_squared * eigenvalues[None, :]  # (C, D)
    total = weighted.sum(axis=1)  # (C,)
    prefix = np.cumsum(weighted, axis=1)  # (C, D); prefix[:, k-1] = sum_{j<=k}
    num_features = weighted.shape[1]
    tail = np.empty((weighted.shape[0], trunc_m.size), dtype=np.float64)
    for j, m in enumerate(trunc_m):
        m = int(m)
        if m <= 0:
            tail[:, j] = total
        elif m >= num_features:
            tail[:, j] = 0.0
        else:
            tail[:, j] = total - prefix[:, m - 1]
    return tail


def default_trunc_list(num_features: int) -> list[int]:
    candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    values = sorted({m for m in candidates if 1 <= m < int(num_features)})
    values.append(int(num_features))
    return sorted(set(values))


# --------------------------------------------------------------------------------------
# Shared plotting (imported by scripts/plot_classifier_powerlaw.py)
# --------------------------------------------------------------------------------------
def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan'), float('nan')
    x, y = x[mask], y[mask]
    try:
        from scipy.stats import pearsonr, spearmanr

        return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])
    except Exception:
        pearson = float(np.corrcoef(x, y)[0, 1])
        rank_x = np.argsort(np.argsort(x))
        rank_y = np.argsort(np.argsort(y))
        spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])
        return pearson, spearman


def _render_all_figures(arrays: dict[str, Any], output_dir: str, fmt: str = 'pdf') -> list[str]:
    """Render the six figures from a dict of saved arrays. Returns written paths."""
    os.makedirs(output_dir, exist_ok=True)
    eigenvalues = np.asarray(arrays['eigenvalues'], dtype=np.float64)
    what_hat = np.asarray(arrays['W_hat'], dtype=np.float64)
    a = np.asarray(arrays['a'], dtype=np.float64)
    capacity_b = float(np.asarray(arrays['capacity_exponent_b']))
    capacity_b_std = float(np.asarray(arrays['capacity_b_std'])) if 'capacity_b_std' in arrays else float('nan')
    capacity_intercept = float(np.asarray(arrays['capacity_intercept']))
    exclusion_frac = np.asarray(arrays['capacity_exclusion_frac'], dtype=np.float64) \
        if 'capacity_exclusion_frac' in arrays else None
    acc_full = np.asarray(arrays['acc_full'], dtype=np.float64)
    ce_full = np.asarray(arrays['ce_full'], dtype=np.float64)
    trunc_m = np.asarray(arrays['trunc_m'], dtype=np.int64)
    acc_by_m = np.asarray(arrays['acc_by_m'], dtype=np.float64)
    class_acc_by_m = np.asarray(arrays['class_acc_by_m'], dtype=np.float64)
    log_a2 = np.asarray(arrays['log_A2'], dtype=np.float64)
    num_features = eigenvalues.size
    positions = np.arange(1, num_features + 1, dtype=np.float64)
    run_label = str(arrays.get('run_label', 'classifier_powerlaw'))
    k_lo = int(arrays['bulk_k_lo']) if 'bulk_k_lo' in arrays else 1
    k_hi = int(arrays['bulk_k_hi']) if 'bulk_k_hi' in arrays else num_features
    bulk_k = np.arange(k_lo, k_hi + 1, dtype=np.float64)  # fit-line domain (bulk only)
    written: list[str] = []

    def _mark_bulk(ax) -> None:
        # Shade the excluded head and finite-dimension tail so the fit region is explicit.
        if k_lo > 1:
            ax.axvspan(0.9, k_lo, color='0.85', alpha=0.5, zorder=0)
        if k_hi < num_features:
            ax.axvspan(k_hi, num_features * 1.05, color='0.85', alpha=0.5, zorder=0)

    def _save(fig, stem: str) -> None:
        path = os.path.join(output_dir, f'{stem}.{fmt}')
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        written.append(path)

    # 1. Feature eigenvalue spectrum + seed-grow capacity fit. Graded gray shading =
    # fraction of seeds that excluded each k (white = every seed kept it, dark = every
    # seed dropped it) -> the "uncertainty about the window".
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.loglog(positions, np.clip(eigenvalues, 1e-30, None), marker='.', linestyle='none',
              markersize=3, alpha=0.7, zorder=3, label=r'$\lambda_k$')
    if exclusion_frac is not None and exclusion_frac.size == positions.size:
        edges = np.empty(positions.size + 1)
        edges[1:-1] = np.sqrt(positions[:-1] * positions[1:])
        edges[0] = positions[0] * (positions[0] / edges[1])
        edges[-1] = positions[-1] * (positions[-1] / edges[-2])
        ylo, yhi = ax.get_ylim()
        ax.pcolormesh(edges, np.array([ylo, yhi]), exclusion_frac[None, :],
                      cmap='Greys', vmin=0.0, vmax=1.0, alpha=0.55, zorder=0,
                      shading='flat', rasterized=True)
        ax.set_ylim(ylo, yhi)
    if np.isfinite(capacity_b):
        fit_line = np.exp(capacity_intercept) * bulk_k ** (-capacity_b)
        b_label = fr'$b={capacity_b:.2f}\pm{capacity_b_std:.2f}$' if np.isfinite(capacity_b_std) \
            else fr'$b={capacity_b:.2f}$'
        ax.loglog(bulk_k, fit_line, color='crimson', linewidth=2.4, zorder=4,
                  label=fr'seed-grow fit $k^{{-b}}$, {b_label} (median$\pm$std)')
    ax.set_xlabel('PC index $k$')
    ax.set_ylabel(r'eigenvalue $\lambda_k$')
    ax.set_title(f'Feature (data) spectrum — {run_label}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    _save(fig, 'fig1_feature_spectrum')

    # 2. Example class source spectra with fitted power laws (bulk regime only).
    finite_a = np.where(np.isfinite(a))[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    _mark_bulk(ax)
    if finite_a.size:
        order = finite_a[np.argsort(a[finite_a])]
        examples = {
            'smallest $a_i$': int(order[0]),
            'median $a_i$': int(order[order.size // 2]),
            'largest $a_i$': int(order[-1]),
        }
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(examples)))
        for (label, ci), color in zip(examples.items(), colors):
            ax.loglog(positions, np.clip(what_hat[ci] ** 2, 1e-30, None), marker='.',
                      linestyle='none', markersize=3, alpha=0.4, color=color,
                      label=f'class {ci} ({label}, $a={a[ci]:.2f}$)')
            if np.isfinite(a[ci]) and np.isfinite(log_a2[ci]):
                fit_line = np.exp(log_a2[ci]) * bulk_k ** (-2.0 * a[ci])
                ax.loglog(bulk_k, fit_line, color=color, linewidth=2.2)
    ax.set_xlabel('PC index $k$')
    ax.set_ylabel(r'$\widehat{W}_{ik}^2$')
    ax.set_title(f'Example class source spectra — {run_label}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, 'fig2_example_class_spectra')

    # 3. Histogram of fitted source exponents a_i.
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(a[np.isfinite(a)], bins=40, color='steelblue', alpha=0.85)
    ax.set_xlabel(r'source exponent $a_i$')
    ax.set_ylabel('number of classes')
    ax.set_title(f'Distribution of $a_i$ — {run_label}')
    ax.grid(True, alpha=0.25)
    _save(fig, 'fig3_source_exponent_histogram')

    # 4. a_i against the empirical per-class accuracy rank.
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    valid = np.isfinite(a) & np.isfinite(acc_full)
    # rank 0 = hardest (lowest accuracy).
    acc_rank = np.full(acc_full.shape, np.nan)
    order = np.argsort(acc_full[valid])
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    acc_rank[np.where(valid)[0]] = ranks
    ax.scatter(acc_rank[valid], a[valid], s=8, alpha=0.5, color='darkorange')
    ax.set_xlabel('class rank by accuracy (0 = hardest)')
    ax.set_ylabel(r'source exponent $a_i$')
    ax.set_title(f'$a_i$ vs empirical accuracy rank — {run_label}')
    ax.grid(True, alpha=0.25)
    _save(fig, 'fig4_a_vs_accuracy_rank')

    # 5. PCA-truncation recovery curves: low-a vs high-a groups vs overall.
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(trunc_m, acc_by_m, marker='o', color='black', linewidth=2.0, label='overall')
    if finite_a.size >= 20:
        sorted_idx = finite_a[np.argsort(a[finite_a])]
        decile = max(1, sorted_idx.size // 10)
        low_group = sorted_idx[:decile]
        high_group = sorted_idx[-decile:]
        low_curve = np.nanmean(class_acc_by_m[low_group], axis=0)
        high_curve = np.nanmean(class_acc_by_m[high_group], axis=0)
        ax.plot(trunc_m, low_curve, marker='s', color='crimson', linewidth=1.8,
                label='low-$a$ decile (heavy tail)')
        ax.plot(trunc_m, high_curve, marker='^', color='seagreen', linewidth=1.8,
                label='high-$a$ decile (light tail)')
    ax.set_xscale('log')
    ax.set_xlabel('number of retained PCs $m$')
    ax.set_ylabel('accuracy')
    ax.set_title(f'PCA-truncation recovery — {run_label}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    _save(fig, 'fig5_truncation_recovery')

    # 6. a_i vs class accuracy (and loss) with correlations.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    r_acc, rho_acc = _pearson_spearman(a, acc_full)
    axes[0].scatter(a[valid], acc_full[valid], s=8, alpha=0.5, color='steelblue')
    axes[0].set_xlabel(r'source exponent $a_i$')
    axes[0].set_ylabel('class accuracy')
    axes[0].set_title(fr'Pearson $r={r_acc:.2f}$, Spearman $\rho={rho_acc:.2f}$')
    axes[0].grid(True, alpha=0.25)
    valid_ce = np.isfinite(a) & np.isfinite(ce_full)
    r_ce, rho_ce = _pearson_spearman(a, ce_full)
    axes[1].scatter(a[valid_ce], ce_full[valid_ce], s=8, alpha=0.5, color='indianred')
    axes[1].set_xlabel(r'source exponent $a_i$')
    axes[1].set_ylabel('class cross-entropy')
    axes[1].set_title(fr'Pearson $r={r_ce:.2f}$, Spearman $\rho={rho_ce:.2f}$')
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(f'$a_i$ vs class-wise metrics — {run_label}')
    _save(fig, 'fig6_a_vs_class_metrics')

    return written


# --------------------------------------------------------------------------------------
# Checkpoint / model / data plumbing (JAX + torch; runs on the cluster)
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['imagenet', 'cifar5m'], default='imagenet')
    parser.add_argument('--optimizer-key', choices=['sgd', 'adam', 'muon'], default='sgd')
    parser.add_argument('--width', type=int, default=64, help='Single width (used if --widths is unset).')
    parser.add_argument('--widths', type=int, nargs='*', default=None,
                        help='Multiple widths to analyze in one process (data is loaded once). '
                             'Each width writes to <output-root>/<dataset>/<optimizer>_w<width>.')
    parser.add_argument('--base-save-dir', default='', help='Override dataset save root.')
    parser.add_argument('--run-id', default='', help='Optional explicit run id override.')
    parser.add_argument('--run-id-resolution', choices=['exact', 'latest_prefix', 'auto'],
                        default='exact')
    parser.add_argument('--images-seen', type=int, default=0,
                        help='Checkpoint step to evaluate; 0 selects the latest common step.')
    parser.add_argument('--num-images', type=int, default=50000,
                        help='Number of val images for PCA + truncation eval (<=0 or >=N uses all).')
    parser.add_argument('--seed', type=int, default=2423, help='Seed for the val subset selection.')
    parser.add_argument('--eval-batch-size', type=int, default=250)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-bins', type=int, default=24, help='Log-spaced bins over k for fits.')
    parser.add_argument('--fit-seeds', type=int, default=64,
                        help='Number of seed anchors for the robust seed-and-grow capacity fit.')
    parser.add_argument('--fit-rmse-tol', type=float, default=0.10,
                        help='Max binned log-log RMSE while growing a seed window.')
    parser.add_argument('--compute-dtype', choices=['float32', 'bfloat16'], default='float32',
                        help='Forward-pass precision. float32 (default) evaluates the trained '
                             'weights with ~1e-5 reconstruction error and a clean covariance tail; '
                             'bfloat16 reproduces the model\'s training-time inference noise.')
    parser.add_argument('--trunc-list', type=int, nargs='*', default=None,
                        help='Retained-PC counts m. Default is a log-spaced schedule.')
    parser.add_argument('--output-dir', default='',
                        help='Output dir for a single width. Ignored when --widths is set.')
    parser.add_argument('--output-root', default='',
                        help='Parent dir for multi-width runs; defaults to '
                             '<base-save-dir>/classifier_powerlaw.')
    parser.add_argument('--no-plots', action='store_true', help='Skip rendering figures.')
    args = parser.parse_args()
    if not args.widths and not args.output_dir:
        parser.error('Provide --output-dir for a single width, or --widths for a multi-width run.')
    return args


def _default_base_save_dir(dataset: str) -> str:
    if str(dataset).strip().lower() == 'cifar5m':
        return os.environ.get('CIFAR5M_BASE_SAVE_DIR',
                              '/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_cifar5m')
    return os.environ.get('IMAGENET_BASE_SAVE_DIR',
                          '/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_imagenet')


def _chw_to_hwc(tensor):
    """Channels-first -> channels-last (module-level so it is picklable under spawn)."""
    return tensor.permute(1, 2, 0)


def _make_loader(dataset_obj, batch_size: int, num_workers: int):
    """DataLoader that uses a spawn context when workers>0 (JAX + os.fork() deadlock-safe)."""
    from torch.utils.data import DataLoader

    kwargs: dict[str, Any] = dict(batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, drop_last=False)
    if num_workers > 0:
        import multiprocessing as mp

        kwargs['multiprocessing_context'] = mp.get_context('spawn')
        kwargs['persistent_workers'] = True
    return DataLoader(dataset_obj, **kwargs)


def _build_eval_loader(dataset: str, num_images: int, seed: int, batch_size: int, num_workers: int):
    """DataLoader over the val split (full set or a seeded subset), channels-last NHWC."""
    from torch.utils.data import Subset

    if str(dataset).strip().lower() == 'cifar5m':
        # Reuse the repo's CIFAR-5M probe subset builder (already seeded).
        from scripts.analyze_exchangeability import _load_cifar5m_probe_subset_builder
        from src.run.constants import CIFAR5M_FOLDER

        if CIFAR5M_FOLDER is None:
            raise ValueError('CIFAR5M_FOLDER must be set for cifar5m analysis.')
        size = num_images if num_images > 0 else 50000
        subset = _load_cifar5m_probe_subset_builder()(CIFAR5M_FOLDER, size, seed)
        return _make_loader(subset, batch_size, num_workers)

    from scripts.analyze_exchangeability import _load_imagenet_torchvision
    from src.run.constants import IMAGENET_FOLDER

    if IMAGENET_FOLDER is None:
        raise ValueError('IMAGENET_FOLDER must be set for imagenet analysis.')
    ImageFolder, ImageNet, transforms = _load_imagenet_torchvision()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    channels_last = transforms.Lambda(_chw_to_hwc)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        channels_last,
    ])
    val_dir = os.path.join(IMAGENET_FOLDER, 'val')
    if os.path.isdir(val_dir):
        base_dataset = ImageFolder(val_dir, transform=val_transform)
    else:
        base_dataset = ImageNet(IMAGENET_FOLDER, 'val', transform=val_transform)

    total = len(base_dataset)
    if num_images <= 0 or num_images >= total:
        dataset_obj = base_dataset
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=num_images, replace=False)
        dataset_obj = Subset(base_dataset, indices.tolist())
    return _make_loader(dataset_obj, batch_size, num_workers)


def _extract_features(model, variables, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over the loader, returning (features h, model logits, labels)."""
    import jax
    import jax.numpy as jnp

    from src.experiment.model.flax_mup.resnet import ResNetBlock

    def _is_resnet_block(module, method_name):
        del method_name
        return isinstance(module, ResNetBlock)

    _block_re = re.compile(r'^ResNetBlock_(\d+)$')

    @jax.jit
    def forward(variables, x):
        logits, state = model.apply(
            variables, x, train=False, capture_intermediates=_is_resnet_block,
        )
        intermediates = state['intermediates']
        block_keys = [k for k in intermediates if _block_re.match(k)]
        if not block_keys:
            raise RuntimeError('capture_intermediates did not record any ResNetBlock outputs.')
        last_key = max(block_keys, key=lambda s: int(_block_re.match(s).group(1)))
        feature_map = intermediates[last_key]['__call__'][0]
        h = jnp.mean(feature_map, axis=(1, 2))
        return logits, h

    features: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        x = jnp.asarray(np.asarray(batch_x))
        logits, h = forward(variables, x)
        logits_all.append(np.asarray(logits, dtype=np.float32))
        features.append(np.asarray(h, dtype=np.float32))
        labels_all.append(np.asarray(batch_y).reshape(-1).astype(np.int64))
    return (
        np.concatenate(features, axis=0),
        np.concatenate(logits_all, axis=0),
        np.concatenate(labels_all, axis=0),
    )


def _classifier_weight_and_bias(member: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Return effective classifier W (C, D), bias b (C,), and the muP divisor.

    logits = (h / divisor) @ W_kernel + b, so W_eff = (W_kernel / divisor)^T.
    """
    dense = member['params']['Readout_0']['Dense_0']
    kernel = np.asarray(dense['kernel'], dtype=np.float64)  # (D, C)
    bias = np.asarray(dense['bias'], dtype=np.float64)  # (C,)
    divisor = float(np.asarray(member['mup']['Readout_0']['divisor']))
    weight_eff = (kernel / divisor).T  # (C, D)
    return weight_eff, bias, divisor


def _cast_tree(tree, dtype):
    """Cast floating-point leaves of a variable tree to dtype; leave integers untouched."""
    import jax
    import jax.numpy as jnp

    def _cast(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(dtype) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf

    return jax.tree_util.tree_map(_cast, tree)


def run_powerlaw_analysis(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    weight_eff: np.ndarray,
    bias: np.ndarray,
    num_classes: int,
    run_label: str,
    output_dir: str,
    num_bins: int = 24,
    fit_seeds: int = 64,
    fit_rmse_tol: float = 0.10,
    seed: int = 2423,
    trunc_list: list[int] | None = None,
    no_plots: bool = False,
    model_logits: np.ndarray | None = None,
    source_arrays: dict[str, Any] | None = None,
    source_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Source/capacity/truncation analysis for a linear classifier over fixed features.

    Given features h(x) (N, D), a linear classifier weight_eff (C, D) + bias (C,), and
    labels, this diagonalizes the centered feature covariance to get the data eigenbasis V,
    rotates the softmax-gauge-fixed classifier into it (What = W V), fits the capacity
    exponent b (lambda_k ~ k^-b) and per-class source exponents a_i (What_ik^2 ~ k^-2a_i),
    measures PCA-truncation recovery, and writes powerlaw_arrays.npz + per_class_metrics.csv
    + truncation_curve.csv + fit_summary.json (+ figures) under output_dir.

    model_logits defaults to features @ weight_eff.T + bias (the classifier's own
    predictions); pass the model's actual forward-pass logits to reproduce those exactly.
    source_arrays / source_summary are merged into the saved npz / JSON for provenance
    (e.g. checkpoint id and muP divisor for a trained readout, or the ridge lambda and
    residual-scale init for an at-init regression classifier). Returns the arrays dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    weight_eff = np.asarray(weight_eff, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    num_samples, num_features = features.shape
    if model_logits is None:
        model_logits = features @ weight_eff.T + bias
    model_logits = np.asarray(model_logits, dtype=np.float64)

    # Softmax-gauge removal, feature PCA, and rotation into the data eigenbasis.
    weight_gauge = row_center_classifier(weight_eff)  # (C, D)
    feature_mean, eigenvalues, eigenvectors = feature_pca(features)
    what_gauge = weight_gauge @ eigenvectors  # (C, D) — analyzed target
    what_eff = weight_eff @ eigenvectors  # (C, D) — for exact truncation logits
    what_gauge_sq = what_gauge ** 2

    # Capacity fit via seed-and-grow: robust window + across-seed error bar. The bulk
    # window [k_lo, k_hi] is reused for the per-class source fits.
    capacity = robust_capacity_fit(
        eigenvalues, num_bins, n_seeds=fit_seeds, rng_seed=int(seed), rmse_tol=fit_rmse_tol,
    )
    k_lo, k_hi = capacity['k_lo'], capacity['k_hi']
    print(f'bulk_window=[{k_lo}, {k_hi}] of {num_features} PCs '
          f'(b={capacity["b"]:.3f}+/-{capacity["b_std"]:.3f}, rmse={capacity["log_rmse"]:.3f}, '
          f'seeds={capacity["n_seeds_used"]})')

    source = fit_source_exponents(what_gauge_sq, num_bins, k_lo=k_lo, k_hi=k_hi)
    a = source['a']
    num_negative_a = int(np.sum(np.isfinite(a) & (a < 0)))
    if num_negative_a:
        print(f'WARNING: {num_negative_a} classes still have negative source exponent a_i.')
    finite_a = a[np.isfinite(a)]
    print(f'source_exponent a: mean={np.mean(finite_a):.4f} std={np.std(finite_a):.4f} '
          f'({finite_a.size}/{num_classes} classes fit)')

    # Empirical per-class metrics (the classifier's own logits).
    acc_full, ce_full = per_class_accuracy_and_ce(model_logits, labels, num_classes)
    overall_acc_full = float(np.mean(model_logits.argmax(1) == labels))
    overall_ce_full = float(np.mean(-_log_softmax(model_logits)[np.arange(num_samples), labels]))
    print(f'overall val accuracy={overall_acc_full:.4f} cross-entropy={overall_ce_full:.4f}')

    # PCA-truncation eval.
    trunc_m = np.asarray(trunc_list if trunc_list else default_trunc_list(num_features),
                         dtype=np.int64)
    trunc_m = np.asarray(sorted({int(min(max(m, 1), num_features)) for m in trunc_m}), dtype=np.int64)
    coeffs = (features - feature_mean) @ eigenvectors  # (N, D)
    base_logits = feature_mean @ weight_eff.T + bias  # (C,)
    acc_by_m = np.empty(trunc_m.size, dtype=np.float64)
    ce_by_m = np.empty(trunc_m.size, dtype=np.float64)
    class_acc_by_m = np.empty((num_classes, trunc_m.size), dtype=np.float64)
    class_ce_by_m = np.empty((num_classes, trunc_m.size), dtype=np.float64)
    for j, m in enumerate(trunc_m):
        logits_m = truncated_logits(coeffs, what_eff, base_logits, int(m))
        acc_by_m[j] = float(np.mean(logits_m.argmax(1) == labels))
        ce_by_m[j] = float(np.mean(-_log_softmax(logits_m)[np.arange(num_samples), labels]))
        class_acc, class_ce = per_class_accuracy_and_ce(logits_m, labels, num_classes)
        class_acc_by_m[:, j] = class_acc
        class_ce_by_m[:, j] = class_ce
        print(f'  m={int(m):5d}: acc={acc_by_m[j]:.4f} ce={ce_by_m[j]:.4f}')
    tail_by_m = residual_tail_mass(eigenvalues, what_gauge_sq, trunc_m)

    # Low-/high-a decile group curves for the truncation-curve CSV.
    low_group_acc = np.full(trunc_m.size, np.nan)
    high_group_acc = np.full(trunc_m.size, np.nan)
    finite_idx = np.where(np.isfinite(a))[0]
    if finite_idx.size >= 20:
        sorted_idx = finite_idx[np.argsort(a[finite_idx])]
        decile = max(1, sorted_idx.size // 10)
        low_group_acc = np.nanmean(class_acc_by_m[sorted_idx[:decile]], axis=0)
        high_group_acc = np.nanmean(class_acc_by_m[sorted_idx[-decile:]], axis=0)

    # Accuracy-rank ordering (0 = hardest), reported as an empirical measurement.
    acc_rank = np.full(num_classes, np.nan)
    valid_acc = np.where(np.isfinite(acc_full))[0]
    order = np.argsort(acc_full[valid_acc])
    ranks = np.empty(order.size)
    ranks[order] = np.arange(order.size)
    acc_rank[valid_acc] = ranks

    r_acc, rho_acc = _pearson_spearman(a, acc_full)
    r_ce, rho_ce = _pearson_spearman(a, ce_full)

    # --- Save raw arrays ---
    arrays: dict[str, Any] = {
        'eigenvalues': eigenvalues.astype(np.float64),
        'W_hat': what_gauge.astype(np.float32),
        'a': a.astype(np.float64),
        'log_A2': source['log_A2'].astype(np.float64),
        'class_fit_rmse': source['log_rmse'].astype(np.float64),
        'capacity_exponent_b': np.float64(capacity['b']),
        'capacity_b_sem': np.float64(capacity['b_sem']),
        'capacity_b_std': np.float64(capacity['b_std']),
        'capacity_intercept': np.float64(capacity['intercept']),
        'capacity_log_rmse': np.float64(capacity['log_rmse']),
        'capacity_seed_slopes': np.asarray(capacity['seed_slopes'], dtype=np.float64),
        'capacity_exclusion_frac': np.asarray(capacity['exclusion_frac'], dtype=np.float64),
        'bulk_k_lo': np.int64(k_lo),
        'bulk_k_hi': np.int64(k_hi),
        'acc_full': acc_full.astype(np.float64),
        'ce_full': ce_full.astype(np.float64),
        'acc_rank': acc_rank.astype(np.float64),
        'trunc_m': trunc_m.astype(np.int64),
        'acc_by_m': acc_by_m.astype(np.float64),
        'ce_by_m': ce_by_m.astype(np.float64),
        'class_acc_by_m': class_acc_by_m.astype(np.float64),
        'class_ce_by_m': class_ce_by_m.astype(np.float64),
        'tail_by_m': tail_by_m.astype(np.float64),
        'feature_mean': feature_mean.astype(np.float32),
        'run_label': run_label,
        'num_samples': np.int64(num_samples),
        'num_features': np.int64(num_features),
        'num_classes': np.int64(num_classes),
        'num_bins': np.int64(num_bins),
    }
    arrays.update(source_arrays or {})
    npz_path = os.path.join(output_dir, 'powerlaw_arrays.npz')
    np.savez_compressed(npz_path, **arrays)
    print(f'wrote {npz_path}')

    # --- Per-class metrics CSV ---
    m_ref = 64 if 64 in set(int(x) for x in trunc_m) else int(trunc_m[min(len(trunc_m) - 1, len(trunc_m) // 2)])
    ref_col = int(np.where(trunc_m == m_ref)[0][0])
    per_class_path = os.path.join(output_dir, 'per_class_metrics.csv')
    with open(per_class_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'class_index', 'a_i', 'log_A2', 'source_fit_rmse', 'acc_full', 'ce_full',
            'acc_rank', f'tail_at_m{m_ref}',
        ])
        writer.writeheader()
        for c in range(num_classes):
            writer.writerow({
                'class_index': c,
                'a_i': float(a[c]),
                'log_A2': float(source['log_A2'][c]),
                'source_fit_rmse': float(source['log_rmse'][c]),
                'acc_full': float(acc_full[c]),
                'ce_full': float(ce_full[c]),
                'acc_rank': float(acc_rank[c]),
                f'tail_at_m{m_ref}': float(tail_by_m[c, ref_col]),
            })
    print(f'wrote {per_class_path}')

    # --- Truncation curve CSV ---
    trunc_path = os.path.join(output_dir, 'truncation_curve.csv')
    with open(trunc_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'm', 'overall_acc', 'overall_ce', 'low_a_group_acc', 'high_a_group_acc',
        ])
        writer.writeheader()
        for j, m in enumerate(trunc_m):
            writer.writerow({
                'm': int(m),
                'overall_acc': float(acc_by_m[j]),
                'overall_ce': float(ce_by_m[j]),
                'low_a_group_acc': float(low_group_acc[j]),
                'high_a_group_acc': float(high_group_acc[j]),
            })
    print(f'wrote {trunc_path}')

    # --- Fit summary JSON ---
    summary: dict[str, Any] = {
        'run_label': run_label,
        'num_samples': int(num_samples),
        'num_features': int(num_features),
        'num_classes': int(num_classes),
        'num_bins': int(num_bins),
        'bulk_k_lo': int(k_lo),
        'bulk_k_hi': int(k_hi),
        'capacity_exponent_b': float(capacity['b']),
        'capacity_b_sem': float(capacity['b_sem']),
        'capacity_b_std': float(capacity['b_std']),
        'capacity_log_rmse': float(capacity['log_rmse']),
        'capacity_fit_seeds': int(capacity['n_seeds_used']),
        'source_exponent_mean': float(np.mean(finite_a)) if finite_a.size else float('nan'),
        'source_exponent_std': float(np.std(finite_a)) if finite_a.size else float('nan'),
        'source_classes_fit': int(finite_a.size),
        'source_negative_count': num_negative_a,
        'overall_val_accuracy': overall_acc_full,
        'overall_val_cross_entropy': overall_ce_full,
        'accuracy_at_full_m': float(acc_by_m[-1]),
        'corr_a_vs_accuracy_pearson': r_acc,
        'corr_a_vs_accuracy_spearman': rho_acc,
        'corr_a_vs_ce_pearson': r_ce,
        'corr_a_vs_ce_spearman': rho_ce,
        'trunc_m': [int(m) for m in trunc_m],
    }
    summary.update(source_summary or {})
    summary_path = os.path.join(output_dir, 'fit_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print(f'wrote {summary_path}')
    print(f'corr(a, accuracy): pearson={r_acc:.3f} spearman={rho_acc:.3f}')
    print(f'corr(a, cross-entropy): pearson={r_ce:.3f} spearman={rho_ce:.3f}')

    # --- Figures ---
    if not no_plots:
        plot_arrays = {k: (np.asarray(v) if not isinstance(v, str) else v) for k, v in arrays.items()}
        for path in _render_all_figures(plot_arrays, output_dir, fmt='pdf'):
            print(f'wrote {path}')

    print(f'done; outputs under {output_dir}')
    return arrays


def _analyze_width(*, width, args, dataset, num_classes, compute_dtype, spec,
                   base_save_dir, run_spec, run_id_resolution, cached_batches) -> None:
    """Restore one width, extract features from the cached batches, fit, and save."""
    from scripts.analyze_exchangeability import (
        _collect_target_steps,
        _list_group_dirs,
        _member_variables_from_state,
        _resolve_width_dirs,
        _restore_state_checkpoint,
    )
    from src.experiment.model.flax_mup.resnet import ResNet18

    width = int(width)
    run_id = args.run_id or run_spec.width_to_run_id[width]
    run_id_resolution = run_id_resolution
    if args.output_dir and not args.widths:
        output_dir = os.path.abspath(args.output_dir)
    else:
        root = os.path.abspath(args.output_root) if args.output_root \
            else os.path.join(base_save_dir, 'classifier_powerlaw')
        output_dir = os.path.join(root, dataset, f'{args.optimizer_key}_w{width}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'\n=== dataset={dataset} optimizer={run_spec.legend_label} width={width} '
          f'run_id={run_id} -> {output_dir} ===')

    width_dirs, width_sources = _resolve_width_dirs(
        base_save_dir=base_save_dir,
        run_id=run_id,
        resolution_mode=run_id_resolution,
        requested_widths=[width],
    )
    if width not in width_dirs:
        raise RuntimeError(f'Could not resolve width {width} under run_id={run_id}.')
    width_dir = width_dirs[width]
    source_run_id = width_sources[width]
    group_dirs = _list_group_dirs(width_dir)
    if not group_dirs:
        raise RuntimeError(f'No group directories under {width_dir}.')

    images_seen = int(args.images_seen)
    if images_seen <= 0:
        common_steps = _collect_target_steps(group_dirs)
        if not common_steps:
            raise RuntimeError('No common checkpoint steps found for the selected width.')
        images_seen = int(common_steps[-1])
    print(f'source_run_id={source_run_id} images_seen={images_seen} groups={len(group_dirs)}')

    state_dir = os.path.join(group_dirs[0], 'state_ckpts')
    state_obj = _restore_state_checkpoint(state_dir, images_seen)
    members = _member_variables_from_state(state_obj)
    member = members[0]
    print(f'restored {len(members)} member(s); using member 0 from {state_dir}')

    model = ResNet18(
        num_classes=num_classes,
        num_filters=width,
        param_dtype=compute_dtype,
        stem_type=spec.stem_type,
    )
    variables = {
        'params': _cast_tree(member['params'], compute_dtype),
        'batch_stats': _cast_tree(member['batch_stats'], compute_dtype),
        'mup': _cast_tree(member['mup'], compute_dtype),
    }

    print('extracting features...')
    features, model_logits, labels = _extract_features(model, variables, cached_batches)
    num_samples, num_features = features.shape
    print(f'features={features.shape} logits={model_logits.shape} labels={labels.shape}')

    weight_eff, bias, divisor = _classifier_weight_and_bias(member)
    # Self-check: reconstructed logits should match the model's forward pass.
    recon = features.astype(np.float64) @ weight_eff.T + bias
    recon_max_diff = float(np.max(np.abs(recon - model_logits.astype(np.float64))))
    print(f'divisor={divisor:.6g} classifier_recon_max_abs_diff={recon_max_diff:.4g}')

    run_label = f'{source_run_id}_w{width}'
    run_powerlaw_analysis(
        features=features.astype(np.float64),
        labels=labels,
        weight_eff=weight_eff,
        bias=bias,
        num_classes=num_classes,
        run_label=run_label,
        output_dir=output_dir,
        num_bins=args.num_bins,
        fit_seeds=args.fit_seeds,
        fit_rmse_tol=args.fit_rmse_tol,
        seed=int(args.seed),
        trunc_list=args.trunc_list,
        no_plots=args.no_plots,
        model_logits=model_logits,  # the model's own forward-pass logits
        source_arrays={
            'dataset': dataset,
            'optimizer_key': args.optimizer_key,
            'width': np.int64(width),
            'source_run_id': source_run_id,
            'images_seen': np.int64(images_seen),
        },
        source_summary={
            'dataset': dataset,
            'optimizer_key': args.optimizer_key,
            'width': width,
            'source_run_id': source_run_id,
            'images_seen': images_seen,
            'muP_divisor': divisor,
            'classifier_recon_max_abs_diff': recon_max_diff,
        },
    )


def main() -> None:
    args = parse_args()
    import jax.numpy as jnp

    from scripts.eval_singular_ablation import default_run_specs
    from src.experiment.dataset_specs import get_dataset_spec

    dataset = args.dataset
    widths = [int(w) for w in args.widths] if args.widths else [int(args.width)]
    base_save_dir = args.base_save_dir or _default_base_save_dir(dataset)
    run_spec = default_run_specs(dataset)[args.optimizer_key]
    spec = get_dataset_spec(dataset)
    num_classes = spec.num_classes
    compute_dtype = jnp.float32 if args.compute_dtype == 'float32' else jnp.bfloat16
    print(f'dataset={dataset} optimizer={run_spec.legend_label} widths={widths}')
    print(f'base_save_dir={base_save_dir} compute_dtype={args.compute_dtype}')

    # Load the eval data once: it is identical across widths (and optimizers) for a
    # dataset, so all widths in this job reuse the same cached batches.
    print(f'loading {dataset} eval data once '
          f'(num_images={args.num_images}, workers={args.num_workers})...')
    loader = _build_eval_loader(
        dataset=dataset,
        num_images=int(args.num_images),
        seed=int(args.seed),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    cached_batches = [(np.asarray(bx), np.asarray(by)) for bx, by in loader]
    del loader
    n_imgs = int(sum(np.asarray(by).reshape(-1).shape[0] for _, by in cached_batches))
    print(f'cached {len(cached_batches)} batches ({n_imgs} images)')

    failures: list[tuple[int, str]] = []
    for width in widths:
        try:
            _analyze_width(
                width=width, args=args, dataset=dataset, num_classes=num_classes,
                compute_dtype=compute_dtype, spec=spec, base_save_dir=base_save_dir,
                run_spec=run_spec, run_id_resolution=args.run_id_resolution,
                cached_batches=cached_batches,
            )
        except Exception as exc:  # keep going so one bad width doesn't lose the rest
            import traceback

            traceback.print_exc()
            failures.append((int(width), repr(exc)))

    if failures:
        print(f'FAILURES ({len(failures)}): {failures}')
        raise SystemExit(1)
    print(f'all done for widths {widths}')


if __name__ == '__main__':
    main()
