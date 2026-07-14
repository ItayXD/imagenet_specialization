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
    row_center_classifier,
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
    fit = fit_capacity_exponent(eigenvalues, num_bins=20)
    assert abs(fit['b'] - b_true) < 0.02


def test_fit_source_exponents_recovers_per_class_a():
    num_features = 512
    positions = np.arange(1, num_features + 1, dtype=np.float64)
    a_true = np.array([0.5, 1.0, 1.7])
    log_a2_true = np.array([0.0, np.log(4.0), np.log(0.25)])
    what_squared = np.stack([
        np.exp(log_a2_true[i]) * positions ** (-2.0 * a_true[i]) for i in range(len(a_true))
    ], axis=0)
    fit = fit_source_exponents(what_squared, num_bins=24)
    assert np.allclose(fit['a'], a_true, atol=0.02)
    assert np.allclose(fit['log_A2'], log_a2_true, atol=0.05)


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


def test_default_trunc_list_includes_full_rank_and_is_sorted():
    values = default_trunc_list(512)
    assert values == sorted(values)
    assert values[-1] == 512
    assert all(1 <= m <= 512 for m in values)
    assert 64 in values
