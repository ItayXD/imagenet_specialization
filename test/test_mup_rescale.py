import numpy as np
import jax.numpy as jnp

from src.experiment.model.flax_mup.mup import Mup


def test_rescale_parameters_handles_missing_mup_collection():
    variables = {
        'params': {
            'Readout_0': {
                'Dense_0': {
                    'kernel': jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
                },
            },
        },
    }
    width_mults = {
        'params': {
            'Readout_0': {
                'Dense_0': {
                    'kernel': 2.0,
                },
            },
        },
    }

    out = Mup.rescale_parameters(variables, width_mults, readout_zero_init=True)
    np.testing.assert_allclose(np.asarray(out['params']['Readout_0']['Dense_0']['kernel']), 0.0)


def test_rescale_parameters_updates_mup_divisor_when_present():
    variables = {
        'params': {
            'Readout_0': {
                'Dense_0': {
                    'kernel': jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
                },
            },
        },
        'mup': {
            'Readout_0': {
                'divisor': jnp.array(1.0, dtype=jnp.float32),
            },
        },
    }
    width_mults = {
        'params': {
            'Readout_0': {
                'Dense_0': {
                    'kernel': 8.0,
                },
            },
        },
    }

    out = Mup.rescale_parameters(variables, width_mults, readout_zero_init=True)
    np.testing.assert_allclose(np.asarray(out['mup']['Readout_0']['divisor']), 8.0)
    np.testing.assert_allclose(np.asarray(out['params']['Readout_0']['Dense_0']['kernel']), 0.0)
