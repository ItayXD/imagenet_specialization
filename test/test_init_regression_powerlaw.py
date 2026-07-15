import numpy as np

from scripts.analyze_init_regression_powerlaw import (
    activate_residual_scales,
    fit_ridge_paths,
)


def _toy_params():
    """Nested params mimicking Flax: BN 'scale'/'bias' leaves, conv 'kernel', the residual
    last-BN scales zero-initialized (all-zero), every other scale = ones."""
    return {
        'bn_init': {'scale': np.ones(4), 'bias': np.zeros(4)},
        'conv_init': {'kernel': np.ones((3, 3, 3, 4))},
        'ResNetBlock_0': {
            'BatchNorm_0': {'scale': np.ones(4), 'bias': np.zeros(4)},
            'BatchNorm_1': {'scale': np.zeros(4), 'bias': np.zeros(4)},  # residual last BN
        },
        'ResNetBlock_1': {
            'BatchNorm_0': {'scale': np.ones(8), 'bias': np.zeros(8)},
            'BatchNorm_1': {'scale': np.zeros(8), 'bias': np.zeros(8)},  # residual last BN
            'norm_proj': {'scale': np.ones(8), 'bias': np.zeros(8)},
        },
    }


def test_activate_residual_scales_ones_targets_only_zero_scales():
    params = _toy_params()
    new, n = activate_residual_scales(params, 'ones')
    assert n == 2  # exactly the two all-zero residual-last-BN scales
    assert np.allclose(new['ResNetBlock_0']['BatchNorm_1']['scale'], 1.0)
    assert np.allclose(new['ResNetBlock_1']['BatchNorm_1']['scale'], 1.0)
    # Non-zero scales, biases and conv kernels are untouched.
    assert np.allclose(new['bn_init']['scale'], 1.0)
    assert np.allclose(new['ResNetBlock_1']['norm_proj']['scale'], 1.0)
    assert np.allclose(new['ResNetBlock_0']['BatchNorm_1']['bias'], 0.0)
    assert new['conv_init']['kernel'].shape == (3, 3, 3, 4)
    # Input dict is not mutated in place.
    assert np.allclose(params['ResNetBlock_0']['BatchNorm_1']['scale'], 0.0)


def test_activate_residual_scales_zeros_is_noop():
    params = _toy_params()
    new, n = activate_residual_scales(params, 'zeros')
    assert n == 0
    assert np.allclose(new['ResNetBlock_0']['BatchNorm_1']['scale'], 0.0)


def test_activate_residual_scales_random_positive_and_counted():
    params = _toy_params()
    rng = np.random.default_rng(0)
    new, n = activate_residual_scales(params, 'random', rng=rng, std=0.3)
    assert n == 2
    for block in ('ResNetBlock_0', 'ResNetBlock_1'):
        vals = new[block]['BatchNorm_1']['scale']
        assert np.all(vals > 0)
        assert not np.allclose(vals, 1.0)  # actually randomized


def test_fit_ridge_paths_recovers_linearly_separable_targets():
    # Build features whose top directions linearly determine the class; ridge at lambda->0
    # should classify the (in-sample) training data near-perfectly.
    rng = np.random.default_rng(1)
    num_classes, dim, n = 5, 12, 400
    centers = rng.normal(size=(num_classes, dim)) * 3.0
    labels = rng.integers(0, num_classes, size=n)
    features = centers[labels] + rng.normal(size=(n, dim)) * 0.1
    paths = fit_ridge_paths(features, labels, num_classes, [0.0, 1e-3, 1.0])
    assert [p['rel_lambda'] for p in paths] == [0.0, 1e-3, 1.0]
    # min-norm LS fit
    w0, b0 = paths[0]['W'], paths[0]['b']
    assert w0.shape == (num_classes, dim) and b0.shape == (num_classes,)
    acc0 = np.mean((features @ w0.T + b0).argmax(1) == labels)
    assert acc0 > 0.98
    # Effective DOF decreases monotonically as lambda grows.
    dofs = [p['dof'] for p in paths]
    assert dofs[0] >= dofs[1] >= dofs[2]
    assert dofs[0] <= dim + 1e-6


def test_fit_ridge_paths_intercept_matches_centered_solution():
    # The returned (W, b) must reproduce logits = H_c @ W_c + Ybar exactly.
    rng = np.random.default_rng(2)
    num_classes, dim, n = 4, 6, 200
    features = rng.normal(size=(n, dim))
    labels = rng.integers(0, num_classes, size=n)
    lam = 0.5
    p = fit_ridge_paths(features, labels, num_classes, [lam])[0]
    logits = features @ p['W'].T + p['b']
    # Recompute via the centered normal equations independently.
    hbar = features.mean(0)
    hc = features - hbar
    onehot = np.eye(num_classes)[labels]
    ybar = onehot.mean(0)
    yc = onehot - ybar
    scale = np.mean(np.clip(np.linalg.eigvalsh(hc.T @ hc), 0, None))
    wc = np.linalg.solve(hc.T @ hc + lam * scale * np.eye(dim), hc.T @ yc)  # (D, C)
    ref = hc @ wc + ybar
    assert np.allclose(logits, ref, atol=1e-8)
