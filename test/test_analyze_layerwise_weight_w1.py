import numpy as np

from scripts.analyze_layerwise_weight_w1 import (
    _collect_conv_layer_paths,
    _extract_across_values_blocked,
    _extract_within_values_blocked,
    _flatten_kernel_filters,
    _row_normalize,
    _stack_layer_filters,
)


def test_collect_conv_layer_paths_uses_natural_depth_order():
    params = {
        'conv_init': {
            'kernel': np.zeros((7, 7, 3, 32), dtype=np.float32),
        },
        'ResNetBlock_2': {
            'Conv_0': {'kernel': np.zeros((3, 3, 32, 32), dtype=np.float32)},
        },
        'ResNetBlock_10': {
            'Conv_0': {'kernel': np.zeros((3, 3, 32, 64), dtype=np.float32)},
        },
        'ResNetBlock_1': {
            'Conv_0': {'kernel': np.zeros((3, 3, 32, 32), dtype=np.float32)},
            'BatchNorm_0': {'scale': np.ones((32,), dtype=np.float32)},
        },
        'Readout_0': {
            'kernel': np.zeros((32, 1000), dtype=np.float32),
        },
    }

    got = _collect_conv_layer_paths(params)

    assert got == [
        ('conv_init',),
        ('ResNetBlock_1', 'Conv_0'),
        ('ResNetBlock_2', 'Conv_0'),
        ('ResNetBlock_10', 'Conv_0'),
    ]


def test_flatten_kernel_filters_outputs_rows_per_output_channel():
    kernel = np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)

    got = _flatten_kernel_filters(kernel)

    assert got.shape == (4, 12)
    assert np.allclose(got[0], kernel[..., 0].reshape(-1))
    assert np.allclose(got[3], kernel[..., 3].reshape(-1))


def test_stack_layer_filters_concatenates_members_in_member_order():
    member_params = [
        {'block': {'kernel': np.arange(2 * 2 * 1 * 3, dtype=np.float32).reshape(2, 2, 1, 3)}},
        {'block': {'kernel': (100 + np.arange(2 * 2 * 1 * 3, dtype=np.float32)).reshape(2, 2, 1, 3)}},
    ]

    stacked, member_width, kernel_numel = _stack_layer_filters(member_params, ('block',))

    assert member_width == 3
    assert kernel_numel == 4
    assert stacked.shape == (6, 4)
    assert np.allclose(stacked[:3], _flatten_kernel_filters(member_params[0]['block']['kernel']))
    assert np.allclose(stacked[3:], _flatten_kernel_filters(member_params[1]['block']['kernel']))


def test_row_normalize_scales_each_row_independently():
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
            [np.nan, 4.0, 8.0],
        ],
        dtype=np.float64,
    )

    got = _row_normalize(matrix)

    assert np.allclose(got[0], [0.0, 0.5, 1.0], equal_nan=True)
    assert np.allclose(got[1], [0.5, 0.5, 0.5], equal_nan=True)
    assert np.allclose(got[2], [np.nan, 0.0, 1.0], equal_nan=True)


def test_blocked_pair_extraction_matches_expected_layout():
    num_members = 3
    member_width = 2
    total = num_members * member_width
    sim = np.arange(total * total, dtype=np.float64).reshape(total, total)
    sim = (sim + sim.T) / 2.0

    across = _extract_across_values_blocked(sim, num_members, member_width)
    within = _extract_within_values_blocked(sim, num_members, member_width)

    expected_across = []
    expected_within = []
    for left_member in range(num_members):
        left_start = left_member * member_width
        left_end = left_start + member_width
        expected_within.extend(sim[left_start:left_end, left_start:left_end][np.triu_indices(member_width, k=1)])
        for right_member in range(left_member + 1, num_members):
            right_start = right_member * member_width
            right_end = right_start + member_width
            expected_across.extend(sim[left_start:left_end, right_start:right_end].reshape(-1))

    assert np.allclose(across, np.asarray(expected_across))
    assert np.allclose(within, np.asarray(expected_within))
