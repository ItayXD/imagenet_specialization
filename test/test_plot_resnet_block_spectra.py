import numpy as np
from pathlib import Path

from scripts.plot_resnet_block_spectra import WidthSpectrumBundle
from scripts.plot_resnet_block_spectra import (
    _available_steps_and_indices,
    _clipped_density_histogram,
    _legacy_saved_theoretical_mp_scale_sq,
    _load_saved_step_spectra,
    _member_spectrum_summary,
    _mp_density,
    _mp_support,
    _normalize_singular_values,
    _reference_edge_normalized,
    _resolve_layer_path,
    _square_unfold_kernel,
    _select_first_block_second_conv_path,
)


def test_select_first_block_second_conv_path_picks_earliest_resnet_block():
    layer_paths = [
        ('conv_init',),
        ('ResNetBlock_1', 'Conv_0'),
        ('ResNetBlock_1', 'Conv_1'),
        ('ResNetBlock_0', 'Conv_0'),
        ('ResNetBlock_0', 'Conv_1'),
        ('ResNetBlock_3', 'Conv_1'),
    ]

    got = _select_first_block_second_conv_path(layer_paths)

    assert got == ('ResNetBlock_0', 'Conv_1')


def test_resolve_layer_path_can_pick_conv_init():
    layer_paths = [
        ('conv_init',),
        ('ResNetBlock_0', 'Conv_0'),
        ('ResNetBlock_0', 'Conv_1'),
    ]

    got = _resolve_layer_path(layer_paths, 'conv_init')

    assert got == ('conv_init',)


def test_square_unfold_kernel_groups_one_spatial_axis_with_output_channels():
    kernel = np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(2, 3, 2, 4)

    got = _square_unfold_kernel(kernel)

    assert got.shape == (2 * 4, 3 * 2)
    expected = np.transpose(kernel, (0, 3, 1, 2)).reshape(8, 6)
    assert np.allclose(got, expected)


def test_member_spectrum_summary_returns_expected_singular_values_and_mp_edge():
    kernel = np.zeros((1, 1, 2, 2), dtype=np.float32)
    kernel[0, 0, 0, 0] = 3.0
    kernel[0, 0, 1, 1] = 4.0

    singular_values, normalized, aspect_ratio, mp_edge_normalized, unfolded_shape = _member_spectrum_summary(
        kernel,
        unfolding_mode='square',
    )

    assert np.allclose(singular_values, [3.0, 4.0])
    assert np.allclose(normalized, [3.0 / np.sqrt(12.5), 4.0 / np.sqrt(12.5)], atol=1e-6)
    assert np.isclose(aspect_ratio, 1.0)
    assert np.isclose(mp_edge_normalized, 2.0)
    assert unfolded_shape == (2, 2)


def test_member_spectrum_summary_uses_smaller_gram_side_for_rectangular_kernel():
    kernel = np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(2, 3, 2, 4)

    singular_values, normalized, aspect_ratio, mp_edge_normalized, unfolded_shape = _member_spectrum_summary(
        kernel,
        unfolding_mode='square',
    )

    assert singular_values.shape == (6,)
    assert normalized.shape == (6,)
    assert np.isclose(aspect_ratio, 8.0 / 6.0)
    assert np.isclose(mp_edge_normalized, 1.0 + np.sqrt(8.0 / 6.0))
    assert unfolded_shape == (8, 6)


def test_member_spectrum_summary_conv_init_uses_square_unfolding():
    kernel = np.arange(3 * 3 * 3 * 8, dtype=np.float32).reshape(3, 3, 3, 8)

    singular_values, normalized, aspect_ratio, mp_edge_normalized, unfolded_shape = _member_spectrum_summary(
        kernel,
        unfolding_mode='square',
    )

    assert singular_values.shape == (9,)
    assert normalized.shape == (9,)
    assert np.isclose(aspect_ratio, 24.0 / 9.0)
    assert np.isclose(mp_edge_normalized, 1.0 + np.sqrt(24.0 / 9.0))
    assert unfolded_shape == (24, 9)


def test_normalize_singular_values_matches_mp_scaling_for_rectangular_matrix():
    singular_values = np.array([4.0, 3.0], dtype=np.float32)

    normalized, aspect_ratio, mp_edge_normalized = _normalize_singular_values(
        singular_values,
        (4, 2),
        normalization_mode='empirical',
    )

    assert np.allclose(normalized, [1.6, 1.2], atol=1e-6)
    assert np.isclose(aspect_ratio, 2.0)
    assert np.isclose(mp_edge_normalized, 1.0 + np.sqrt(2.0))


def test_normalize_singular_values_init_theory_uses_supplied_scale():
    singular_values = np.array([4.0, 3.0], dtype=np.float32)

    normalized, aspect_ratio, mp_edge_normalized = _normalize_singular_values(
        singular_values,
        (4, 2),
        normalization_mode='init_theory',
        theoretical_mp_scale_sq=2.0,
    )

    assert np.allclose(normalized, [4.0 / np.sqrt(2.0), 3.0 / np.sqrt(2.0)], atol=1e-6)
    assert np.isclose(aspect_ratio, 2.0)
    assert np.isclose(mp_edge_normalized, 1.0 + np.sqrt(2.0))


def test_reference_edge_is_always_the_analytic_mp_edge():
    unfolded_shape = (24, 9)
    aspect_ratio = 24.0 / 9.0

    assert np.isclose(
        _reference_edge_normalized(unfolded_shape, aspect_ratio),
        1.0 + np.sqrt(aspect_ratio),
    )


def test_mp_density_is_zero_outside_support_and_positive_inside():
    aspect_ratio = 0.25
    s_minus, s_plus = _mp_support(aspect_ratio)
    xs = np.array([s_minus - 1e-3, 0.5 * (s_minus + s_plus), s_plus + 1e-3], dtype=np.float64)

    density = _mp_density(xs, aspect_ratio)

    assert np.isclose(density[0], 0.0)
    assert density[1] > 0.0
    assert np.isclose(density[2], 0.0)


def test_mp_density_integrates_to_one_for_tall_unfolded_matrix():
    aspect_ratio = 8.0 / 3.0
    s_minus, s_plus = _mp_support(aspect_ratio)
    xs = np.linspace(s_minus + 1e-5, s_plus - 1e-5, 20000, dtype=np.float64)

    density = _mp_density(xs, aspect_ratio)
    area = np.trapezoid(density, xs)

    assert np.isclose(area, 1.0, atol=2e-3)


def test_clipped_density_histogram_preserves_mass_lost_beyond_bulk_window():
    values = np.array([0.2, 0.4, 1.8, 5.0], dtype=np.float32)
    bin_edges = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    hist = _clipped_density_histogram(values, bin_edges)

    assert np.allclose(hist, [0.5, 0.25])
    assert np.isclose(np.sum(hist * np.diff(bin_edges)), 0.75)


def test_available_steps_and_indices_uses_union_and_marks_missing_width_steps():
    bundle_a = WidthSpectrumBundle(
        width=32,
        source_run_id='run_a',
        layer_path=('ResNetBlock_0', 'Conv_1'),
        unfolded_shape=(96, 9),
        aspect_ratio=1.0,
        mp_edge_normalized=2.0,
        reference_edge_normalized=2.0,
        common_steps=[1000, 10000, 100000],
        member_labels=['member_0'],
        normalized_by_step=[np.ones((1, 2), dtype=np.float32) for _ in range(3)],
        tail_by_step=[np.ones(1, dtype=np.float32) for _ in range(3)],
        bulk_max=2.5,
        tail_max=3.5,
    )
    bundle_b = WidthSpectrumBundle(
        width=64,
        source_run_id='run_b',
        layer_path=('ResNetBlock_0', 'Conv_1'),
        unfolded_shape=(192, 9),
        aspect_ratio=1.0,
        mp_edge_normalized=2.0,
        reference_edge_normalized=2.0,
        common_steps=[10000, 100000, 1000000],
        member_labels=['member_0'],
        normalized_by_step=[np.ones((1, 2), dtype=np.float32) for _ in range(3)],
        tail_by_step=[np.ones(1, dtype=np.float32) for _ in range(3)],
        bulk_max=2.5,
        tail_max=3.5,
    )

    got = _available_steps_and_indices([bundle_a, bundle_b])

    assert got == [(1000, [0, None]), (10000, [1, 0]), (100000, [2, 1]), (1000000, [None, 2])]


def test_load_saved_step_spectra_reuses_saved_normalized_values(tmp_path: Path):
    spectra_dir = tmp_path / "spectra"
    width_dir = spectra_dir / "width_64"
    width_dir.mkdir(parents=True)
    np.savez_compressed(
        width_dir / "step_10000.npz",
        width=np.int32(64),
        images_seen=np.int64(10000),
        source_run_id=np.asarray("run_a"),
        unfolding_mode=np.asarray("square"),
        layer_path=np.asarray("conv_init"),
        member_labels=np.asarray(["member_0"]),
        singular_values=np.asarray([[4.0, 3.0]], dtype=np.float32),
        normalized_singular_values=np.asarray([[9.0, 7.0]], dtype=np.float32),
        aspect_ratio=np.float32(2.0),
        mp_edge_normalized=np.float32(2.41421356),
        unfolded_shape=np.asarray([4, 2], dtype=np.int32),
    )

    loaded = _load_saved_step_spectra(
        spectra_dir=str(spectra_dir),
        width=64,
        step=10000,
        unfolding_mode="square",
        normalization_mode='empirical',
    )

    assert loaded is not None
    singular_values, normalized, aspect_ratio, mp_edge_normalized, unfolded_shape, *_ = loaded
    assert np.allclose(singular_values, [[4.0, 3.0]])
    assert np.allclose(normalized, [[9.0, 7.0]])
    assert np.isclose(aspect_ratio, 2.0)
    assert np.isclose(mp_edge_normalized, 2.41421356)
    assert unfolded_shape == (4, 2)


def test_load_saved_step_spectra_init_theory_recomputes_with_theoretical_scale(tmp_path: Path):
    spectra_dir = tmp_path / "spectra"
    width_dir = spectra_dir / "width_64"
    width_dir.mkdir(parents=True)
    np.savez_compressed(
        width_dir / "step_10000.npz",
        width=np.int32(64),
        images_seen=np.int64(10000),
        source_run_id=np.asarray("run_a"),
        unfolding_mode=np.asarray("square"),
        layer_path=np.asarray("conv_init"),
        member_labels=np.asarray(["member_0"]),
        singular_values=np.asarray([[4.0, 3.0]], dtype=np.float32),
        normalized_singular_values=np.asarray([[9.0, 7.0]], dtype=np.float32),
        aspect_ratio=np.float32(2.0),
        mp_edge_normalized=np.float32(2.41421356),
        unfolded_shape=np.asarray([4, 2], dtype=np.int32),
        theoretical_mp_scale_sq=np.float32(2.0),
    )

    loaded = _load_saved_step_spectra(
        spectra_dir=str(spectra_dir),
        width=64,
        step=10000,
        unfolding_mode="square",
        normalization_mode='init_theory',
    )

    assert loaded is not None
    singular_values, normalized, aspect_ratio, mp_edge_normalized, unfolded_shape, *_ = loaded
    assert np.allclose(singular_values, [[4.0, 3.0]])
    assert np.allclose(normalized, [[4.0 / np.sqrt(2.0), 3.0 / np.sqrt(2.0)]], atol=1e-6)
    assert np.isclose(aspect_ratio, 2.0)
    assert np.isclose(mp_edge_normalized, 2.41421356)
    assert unfolded_shape == (4, 2)


def test_legacy_saved_theoretical_mp_scale_sq_matches_old_saved_layouts():
    assert np.isclose(_legacy_saved_theoretical_mp_scale_sq((64, 147), ('conv_init',)), 2.0)
    assert np.isclose(
        _legacy_saved_theoretical_mp_scale_sq((192, 192), ('ResNetBlock_0', 'Conv_1')),
        2.0 / 3.0,
    )
