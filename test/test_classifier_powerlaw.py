import numpy as np

from scripts.analyze_classifier_powerlaw import (
    _log_softmax,
    default_trunc_list,
    feature_pca,
    fit_capacity_exponent,
    fit_source_exponents,
    loglog_binned_slope,
    per_class_accuracy_and_ce,
    residual_tail_mass,
    robust_loglog_slope,
    row_center_classifier,
    select_bulk_window,
    truncated_logits,
)


def test_loglog_binned_slope_recovers_known_exponent():
    positions = np.arange(1, 513, dtype=np.float64)
    amplitude, exponent = 3.0, 1.5
    values = amplitude * positions ** (-exponent)
    fit = loglog_binned_slope(positions, values, num_bins=24)
    assert abs(fit['slope'] - (-exponent)) < 0.02
    assert abs(np.exp(fit['intercept']) - amplitude) < 0.1
    # Near-perfect fit; small residual comes from linear-space averaging within bins.
    assert fit['log_rmse'] < 0.01


def test_fit_capacity_exponent_recovers_b():
    b_true = 1.2
    eigenvalues = np.arange(1, 401, dtype=np.float64) ** (-b_true)
    fit = fit_capacity_exponent(eigenvalues, num_bins=20, k_lo=1, k_hi=400)
    assert abs(fit['b'] - b_true) < 0.03


def test_fit_source_exponents_recovers_per_class_a():
    num_features = 512
    positions = np.arange(1, num_features + 1, dtype=np.float64)
    a_true = np.array([0.5, 1.0, 1.7])
    log_a2_true = np.array([0.0, np.log(4.0), np.log(0.25)])
    what_squared = np.stack([
        np.exp(log_a2_true[i]) * positions ** (-2.0 * a_true[i]) for i in range(len(a_true))
    ], axis=0)
    fit = fit_source_exponents(what_squared, num_bins=24, k_lo=1, k_hi=num_features)
    assert np.allclose(fit['a'], a_true, atol=0.03)
    assert np.allclose(fit['log_A2'], log_a2_true, atol=0.08)


def _three_regime_spectrum(n=400):
    """Flat head (k<=7), k^-1 bulk (8..300), steep k^-5 finite-dimension tail (>300)."""
    k = np.arange(1, n + 1, dtype=np.float64)
    lam = np.empty(n)
    head = k <= 7
    tail = k > 300
    bulk = ~head & ~tail
    lam[head] = 1.0
    lam[bulk] = 8.0 / k[bulk]
    lam[tail] = (8.0 / 300.0) * (300.0 / k[tail]) ** 5
    return lam


def test_select_bulk_window_excludes_head_and_tail():
    lam = _three_regime_spectrum(400)
    k_lo, k_hi, info = select_bulk_window(lam, slope_tol=0.5)
    # Head (flat, slope 0) and the steep tail (slope -5) must be excluded.
    assert k_lo >= 5
    assert k_hi <= 320
    assert k_hi - k_lo > 100  # a substantial bulk remains
    # The bulk slope is ~ -1, so the capacity fit over the detected window recovers b~1.
    fit = fit_capacity_exponent(lam, num_bins=16, k_lo=k_lo, k_hi=k_hi)
    assert abs(fit['b'] - 1.0) < 0.15


def test_bulk_restriction_makes_source_exponent_positive():
    # k^-2 bulk (a=1) with a RISING noise-floor tail beyond k=200; a full-range fit is
    # biased toward zero/negative, but restricting to the bulk recovers the true slope.
    n = 512
    k = np.arange(1, n + 1, dtype=np.float64)
    what_sq = k ** (-2.0)
    tail = k > 200
    what_sq[tail] = (200.0 ** -2.0) * (k[tail] / 200.0) ** 1.0  # rising tail
    full = robust_loglog_slope(k, what_sq, num_bins=24, k_lo=1, k_hi=n)
    bulk = robust_loglog_slope(k, what_sq, num_bins=24, k_lo=5, k_hi=180)
    assert -0.5 * bulk['slope'] > 0.8  # a_i ~ 1, clearly positive
    assert -0.5 * full['slope'] < -0.5 * bulk['slope']  # full-range fit is biased upward


def test_row_center_classifier_zeroes_class_mean_and_preserves_softmax():
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(10, 6))
    centered = row_center_classifier(weight)
    assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-12)

    h = rng.normal(size=(4, 6))
    logits_full = h @ weight.T
    logits_gauge = h @ centered.T
    # Softmax (hence loss / accuracy) is invariant to the removed additive gauge.
    assert np.allclose(_log_softmax(logits_full), _log_softmax(logits_gauge), atol=1e-10)


def test_feature_pca_orthonormal_descending_and_reconstructs_covariance():
    rng = np.random.default_rng(1)
    num_samples, dim = 5000, 8
    factor = rng.normal(size=(dim, dim))
    features = rng.normal(size=(num_samples, dim)) @ factor.T + 5.0
    feature_mean, eigenvalues, eigenvectors = feature_pca(features)

    assert np.allclose(feature_mean, features.mean(axis=0))
    assert np.all(np.diff(eigenvalues) <= 1e-9)  # descending
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(dim), atol=1e-8)

    centered = features - feature_mean
    covariance = (centered.T @ centered) / num_samples
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    assert np.allclose(reconstructed, covariance, atol=1e-8)


def test_truncated_logits_exact_at_full_rank_and_base_at_zero():
    rng = np.random.default_rng(2)
    num_samples, dim, num_classes = 200, 8, 5
    features = rng.normal(size=(num_samples, dim))
    weight = rng.normal(size=(num_classes, dim))
    bias = rng.normal(size=(num_classes,))

    feature_mean, _, eigenvectors = feature_pca(features)
    coeffs = (features - feature_mean) @ eigenvectors
    what_eff = weight @ eigenvectors
    base_logits = feature_mean @ weight.T + bias

    full = truncated_logits(coeffs, what_eff, base_logits, dim)
    assert np.allclose(full, features @ weight.T + bias, atol=1e-8)

    zero = truncated_logits(coeffs, what_eff, base_logits, 0)
    assert np.allclose(zero, np.broadcast_to(base_logits, (num_samples, num_classes)), atol=1e-12)


def test_residual_tail_mass_monotone_and_endpoints():
    rng = np.random.default_rng(3)
    num_classes, dim = 4, 32
    eigenvalues = np.sort(rng.uniform(0.1, 1.0, size=dim))[::-1]
    what_squared = rng.uniform(0.0, 1.0, size=(num_classes, dim))
    trunc = np.array([0, 1, 4, 8, 16, dim])
    tail = residual_tail_mass(eigenvalues, what_squared, trunc)

    total = (what_squared * eigenvalues[None, :]).sum(axis=1)
    assert np.allclose(tail[:, 0], total)  # m=0 -> full mass
    assert np.allclose(tail[:, -1], 0.0)  # m=dim -> no residual
    assert np.all(np.diff(tail, axis=1) <= 1e-12)  # non-increasing in m


def test_tail_mass_larger_for_weight_on_later_pcs():
    dim = 64
    eigenvalues = np.arange(1, dim + 1, dtype=np.float64) ** (-1.0)
    early = np.zeros(dim)
    early[0] = 1.0  # target aligned with the leading PC (high a, light tail)
    late = np.zeros(dim)
    late[dim - 1] = 1.0  # target on the last PC (low a, heavy tail)
    what_squared = np.stack([early, late], axis=0)
    trunc = np.array([1, 8, 32])
    tail = residual_tail_mass(eigenvalues, what_squared, trunc)
    # The late-PC class always carries more residual mass past m.
    assert np.all(tail[1] >= tail[0])
    assert tail[0, 0] == 0.0 and tail[1, 0] > 0.0


def test_per_class_accuracy_and_ce_toy():
    # Two classes, four examples; class 0 perfect, class 1 half correct.
    logits = np.array([
        [5.0, 0.0],  # label 0 -> correct
        [4.0, 0.0],  # label 0 -> correct
        [0.0, 3.0],  # label 1 -> correct
        [2.0, 0.0],  # label 1 -> wrong
    ])
    labels = np.array([0, 0, 1, 1])
    acc, ce = per_class_accuracy_and_ce(logits, labels, num_classes=2)
    assert np.isclose(acc[0], 1.0)
    assert np.isclose(acc[1], 0.5)
    assert np.all(ce >= 0.0)
    # Missing class reports NaN.
    acc3, ce3 = per_class_accuracy_and_ce(logits, labels, num_classes=3)
    assert np.isnan(acc3[2]) and np.isnan(ce3[2])


def test_compute_svd_source_matches_svd():
    from scripts.analyze_classifier_svd import compute_svd_source

    rng = np.random.default_rng(0)
    what = rng.normal(size=(50, 120))
    result = compute_svd_source(what, num_bins=12, k_lo=5, k_hi=100)
    s_ref = np.linalg.svd(what, compute_uv=False)
    assert np.allclose(result['singular_values'], s_ref, atol=1e-8)
    assert np.all(np.diff(result['singular_values']) <= 1e-9)  # descending
    assert result['a_j'].shape[0] == min(what.shape)
    assert np.isfinite(result['sj_exponent_c'])


def test_default_trunc_list_includes_full_rank_and_is_sorted():
    values = default_trunc_list(512)
    assert values == sorted(values)
    assert values[-1] == 512
    assert all(1 <= m <= 512 for m in values)
    assert 64 in values
