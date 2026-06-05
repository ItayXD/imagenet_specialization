import jax.numpy as jnp
import pytest

from src.experiment.training.online_momentum import (
    _build_optimizer,
    _flatten_conv_kernel_for_muon,
    _label_muon_params,
    _restore_conv_kernel_from_muon,
)


def test_build_optimizer_supports_sgd_with_constant_lr():
    training_params = {
        'eta_0': 0.01,
        'optimizer': 'sgd',
        'use_warmup_cosine_decay': False,
    }

    optimizer_name, _, lr_schedule = _build_optimizer(training_params, total_micro_steps=100)

    assert optimizer_name == 'sgd'
    assert lr_schedule(0) == 0.01
    assert lr_schedule(50) == 0.01


def test_build_optimizer_supports_muon_with_constant_lr():
    training_params = {
        'eta_0': 0.01,
        'optimizer': 'muon',
        'use_warmup_cosine_decay': False,
    }

    optimizer_name, _, lr_schedule = _build_optimizer(training_params, total_micro_steps=100)

    assert optimizer_name == 'muon'
    assert lr_schedule(0) == 0.01
    assert lr_schedule(50) == 0.01


def test_muon_labels_only_non_stem_conv_kernels():
    params = {
        'conv_init': {'kernel': jnp.zeros((3, 3, 3, 32))},
        'ResNetBlock_0': {
            'Conv_0': {'kernel': jnp.zeros((3, 3, 32, 32))},
            'BatchNorm_0': {'scale': jnp.zeros((32,))},
        },
        'Readout_0': {'Dense_0': {'kernel': jnp.zeros((256, 10))}},
    }

    labels = _label_muon_params(params)

    assert labels['conv_init']['kernel'] == 'adam'
    assert labels['ResNetBlock_0']['Conv_0']['kernel'] == 'muon'
    assert labels['ResNetBlock_0']['BatchNorm_0']['scale'] == 'adam'
    assert labels['Readout_0']['Dense_0']['kernel'] == 'adam'


def test_muon_conv_flatten_round_trip_preserves_shape():
    kernel = jnp.arange(3 * 3 * 32 * 64, dtype=jnp.float32).reshape((3, 3, 32, 64))

    flat = _flatten_conv_kernel_for_muon(kernel)
    restored = _restore_conv_kernel_from_muon(flat, kernel.shape)

    assert flat.shape == (64, 3 * 3 * 32)
    assert restored.shape == kernel.shape
    assert jnp.array_equal(restored, kernel)


def test_build_optimizer_rejects_unknown_name():
    training_params = {
        'eta_0': 0.01,
        'optimizer': 'rmsprop',
        'use_warmup_cosine_decay': False,
    }

    with pytest.raises(ValueError, match='Unsupported optimizer'):
        _build_optimizer(training_params, total_micro_steps=100)
