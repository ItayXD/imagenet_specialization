#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import re
from dataclasses import dataclass
from os.path import join

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:
    jax = None
    jnp = None


matplotlib.use('Agg')
import matplotlib.pyplot as plt

_RESNET_BLOCK_RE = re.compile(r'^ResNetBlock_(\d+)$')
_DIGIT_RE = re.compile(r'(\d+)')
@dataclass
class WidthSpectrumBundle:
    width: int
    source_run_id: str
    layer_path: tuple[str, ...]
    unfolded_shape: tuple[int, int]
    aspect_ratio: float
    mp_edge_normalized: float
    reference_edge_normalized: float
    common_steps: list[int]
    member_labels: list[str]
    normalized_by_step: list[np.ndarray]
    tail_by_step: list[np.ndarray]
    bulk_max: float
    tail_max: float


def _default_artifact_stem() -> str:
    return 'resnet_block1_conv1_spectra'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Extract and plot spectra of the second convolution in the first ResNet block '
            'from exchangeability ensemble checkpoints.'
        )
    )
    parser.add_argument(
        '--base-save-dir',
        required=True,
        help='Input root containing exchangeability_job* run directories.',
    )
    parser.add_argument(
        '--run-id',
        default='exchangeability',
        help='Run id prefix used to resolve width directories.',
    )
    parser.add_argument(
        '--run-id-resolution',
        choices=['exact', 'latest_prefix', 'auto'],
        default='latest_prefix',
        help='How to resolve --run-id when multiple matching runs exist.',
    )
    parser.add_argument(
        '--spectra-dir',
        default='',
        help=(
            'Directory where raw per-width/per-step spectra .npz files are written. '
            'Defaults to $IMAGENET_BASE_SAVE_DIR/resnet_block1_conv1_spectra.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        default='',
        help=(
            'Directory where figures are written. '
            'Defaults to $IMAGENET_BASE_SAVE_DIR/plots_<artifact_stem>.'
        ),
    )
    parser.add_argument(
        '--layer-selection',
        choices=['first_block_second_conv', 'conv_init'],
        default='first_block_second_conv',
        help='Which convolutional layer to analyze.',
    )
    parser.add_argument(
        '--artifact-stem',
        default=_default_artifact_stem(),
        help='Filename stem used for output directories and PDF names.',
    )
    parser.add_argument('--widths', type=int, nargs='*', default=None, help='Optional list of widths to analyze.')
    parser.add_argument('--bins', type=int, default=60, help='Histogram bins for the bulk panel.')
    parser.add_argument(
        '--normalization-mode',
        choices=['empirical', 'init_theory'],
        default='init_theory',
        help='How to normalize singular values on the x-axis.',
    )
    return parser.parse_args()


def _natural_key(text: str) -> list[object]:
    parts = _DIGIT_RE.split(str(text))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _default_spectra_dir(artifact_stem: str) -> str:
    base_dir = os.environ.get('IMAGENET_BASE_SAVE_DIR', '').strip() or os.environ.get('BASE_SAVE_DIR', '').strip()
    if not base_dir:
        return os.path.abspath(os.path.join('outputs', artifact_stem))
    return os.path.join(base_dir, artifact_stem)


def _default_output_dir(artifact_stem: str) -> str:
    base_dir = os.environ.get('IMAGENET_BASE_SAVE_DIR', '').strip() or os.environ.get('BASE_SAVE_DIR', '').strip()
    if not base_dir:
        return os.path.abspath(os.path.join('outputs', f'plots_{artifact_stem}'))
    return os.path.join(base_dir, f'plots_{artifact_stem}')


def _load_analysis_helpers():
    from scripts.analyze_exchangeability import (
        _collect_target_steps,
        _extract_train_state_fields,
        _flatten_members,
        _list_group_dirs,
        _progress,
        _resolve_width_dirs,
        _restore_state_checkpoint,
    )
    return (
        _collect_target_steps,
        _extract_train_state_fields,
        _flatten_members,
        _list_group_dirs,
        _progress,
        _resolve_width_dirs,
        _restore_state_checkpoint,
    )


def _iter_mapping_items(node) -> list[tuple[str, object]]:
    try:
        items = list(node.items())
    except Exception:
        return []
    return sorted(((str(key), value) for key, value in items), key=lambda item: _natural_key(item[0]))


def _path_get(node, path: tuple[str, ...]):
    current = node
    for key in path:
        current = current[key]
    return current


def _collect_conv_layer_paths(node, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    try:
        kernel = node['kernel']
    except Exception:
        kernel = None
    if kernel is not None:
        kernel_arr = np.asarray(kernel)
        if kernel_arr.ndim == 4:
            return [prefix]

    paths: list[tuple[str, ...]] = []
    for key, value in _iter_mapping_items(node):
        if key == 'kernel':
            continue
        paths.extend(_collect_conv_layer_paths(value, prefix + (key,)))
    return paths


def _select_first_block_second_conv_path(layer_paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    block_candidates: list[tuple[int, tuple[str, ...]]] = []
    for path in layer_paths:
        if len(path) < 2 or path[-1] != 'Conv_1':
            continue
        match = _RESNET_BLOCK_RE.match(path[-2])
        if match is None:
            continue
        block_candidates.append((int(match.group(1)), path))
    if not block_candidates:
        raise ValueError('Could not find a ResNetBlock_*/Conv_1 path in the restored parameters.')
    block_candidates.sort(key=lambda item: (item[0], tuple(_natural_key(piece) for piece in item[1])))
    return block_candidates[0][1]


def _select_conv_init_path(layer_paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    for path in layer_paths:
        if path == ('conv_init',):
            return path
    raise ValueError('Could not find conv_init in the restored parameters.')


def _resolve_layer_path(layer_paths: list[tuple[str, ...]], layer_selection: str) -> tuple[str, ...]:
    if layer_selection == 'first_block_second_conv':
        return _select_first_block_second_conv_path(layer_paths)
    if layer_selection == 'conv_init':
        return _select_conv_init_path(layer_paths)
    raise ValueError(f'Unsupported layer selection: {layer_selection}')


def _unfolding_mode_for_layer_selection(layer_selection: str) -> str:
    if layer_selection in {'conv_init', 'first_block_second_conv'}:
        return 'square'
    raise ValueError(f'Unsupported layer selection: {layer_selection}')


def _square_unfold_kernel(kernel) -> np.ndarray:
    kernel_arr = np.asarray(kernel, dtype=np.float32)
    if kernel_arr.ndim != 4:
        raise ValueError(f'Expected a 4D convolution kernel, got shape {kernel_arr.shape}.')
    k1, k2, cin, cout = map(int, kernel_arr.shape)
    return np.transpose(kernel_arr, (0, 3, 1, 2)).reshape((k1 * cout, k2 * cin))


def _unfold_kernel(kernel, unfolding_mode: str) -> np.ndarray:
    if unfolding_mode == 'square':
        return _square_unfold_kernel(kernel)
    raise ValueError(f'Unsupported unfolding mode: {unfolding_mode}')


def _normalize_singular_values(
    singular_values: np.ndarray,
    unfolded_shape: tuple[int, int],
    normalization_mode: str = 'empirical',
    theoretical_mp_scale_sq: float | None = None,
) -> tuple[np.ndarray, float, float]:
    num_rows, num_cols = map(int, unfolded_shape)
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError(f'Invalid unfolded shape for normalization: {unfolded_shape}.')
    aspect_ratio = float(num_rows) / float(num_cols)
    if normalization_mode == 'empirical':
        sum_sq_singular_values = float(np.sum(np.square(singular_values, dtype=np.float32)))
        mp_scale_sq = sum_sq_singular_values / float(num_rows)
    elif normalization_mode == 'init_theory':
        if theoretical_mp_scale_sq is None:
            raise ValueError('theoretical_mp_scale_sq is required for init_theory normalization.')
        mp_scale_sq = float(theoretical_mp_scale_sq)
    else:
        raise ValueError(f'Unsupported normalization mode: {normalization_mode}')
    if mp_scale_sq <= 0.0:
        raise ValueError('Encountered non-positive MP normalization scale while normalizing the spectrum.')
    normalized = singular_values / np.sqrt(mp_scale_sq)
    mp_edge_normalized = float(1.0 + np.sqrt(aspect_ratio))
    return normalized.astype(np.float32), aspect_ratio, mp_edge_normalized


def _reference_edge_normalized(unfolded_shape: tuple[int, int], aspect_ratio: float) -> float:
    del unfolded_shape
    return float(1.0 + np.sqrt(aspect_ratio))


def _theoretical_mp_scale_sq_from_kernel(kernel, unfolded_shape: tuple[int, int]) -> float:
    kernel_arr = np.asarray(kernel, dtype=np.float32)
    if kernel_arr.ndim != 4:
        raise ValueError(f'Expected a 4D convolution kernel, got shape {kernel_arr.shape}.')
    _, _, cin, _ = map(int, kernel_arr.shape)
    fan_in = int(np.prod(kernel_arr.shape[:-1], dtype=np.int64))
    init_variance = 2.0 / float(fan_in)
    num_cols = int(unfolded_shape[1])
    return float(num_cols) * init_variance


def _legacy_saved_theoretical_mp_scale_sq(
    unfolded_shape: tuple[int, int],
    layer_path: tuple[str, ...],
) -> float:
    del unfolded_shape
    if layer_path == ('conv_init',):
        # Legacy saved conv_init NPZs were flattened as (cout, fan_in), so with
        # He fan-in init the expected squared row norm is fan_in * (2 / fan_in).
        return 2.0
    # Legacy saved first-block conv spectra use 3x3 kernels and the square
    # unfolding from the original script. For a 3x3 kernel under He fan-in init,
    # the expected MP scale is (3 * cin) * (2 / (9 * cin)) = 2/3.
    return 2.0 / 3.0


if jax is not None:
    @jax.jit
    def _gram_matrix_left(unfolded: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(
            unfolded,
            jnp.swapaxes(unfolded, 0, 1),
            precision=jax.lax.Precision.HIGHEST,
        )


    @jax.jit
    def _gram_matrix_right(unfolded: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(
            jnp.swapaxes(unfolded, 0, 1),
            unfolded,
            precision=jax.lax.Precision.HIGHEST,
        )


    @jax.jit
    def _eigvalsh_nonnegative(matrix: jnp.ndarray) -> jnp.ndarray:
        eigenvalues = jnp.linalg.eigvalsh(matrix)
        return jnp.maximum(eigenvalues, 0.0)
else:
    def _gram_matrix_left(unfolded):
        raise RuntimeError('JAX is required to compute spectra from checkpoints.')


    def _gram_matrix_right(unfolded):
        raise RuntimeError('JAX is required to compute spectra from checkpoints.')


    def _eigvalsh_nonnegative(matrix):
        raise RuntimeError('JAX is required to compute spectra from checkpoints.')


def _member_spectrum_summary(
    kernel,
    unfolding_mode: str,
    normalization_mode: str = 'empirical',
) -> tuple[np.ndarray, np.ndarray, float, float, tuple[int, int]]:
    if jnp is None:
        raise RuntimeError('JAX is required to compute spectra from checkpoints.')
    unfolded = _unfold_kernel(kernel, unfolding_mode)
    num_rows, num_cols = map(int, unfolded.shape)
    unfolded_jax = jnp.asarray(unfolded, dtype=jnp.float32)
    if num_rows <= num_cols:
        gram = _gram_matrix_left(unfolded_jax)
    else:
        gram = _gram_matrix_right(unfolded_jax)
    eigenvalues = np.asarray(_eigvalsh_nonnegative(gram), dtype=np.float32)
    singular_values = np.sqrt(eigenvalues).astype(np.float32)
    theoretical_mp_scale_sq = None
    if normalization_mode == 'init_theory':
        theoretical_mp_scale_sq = _theoretical_mp_scale_sq_from_kernel(kernel, tuple(unfolded.shape))
    normalized, aspect_ratio, mp_edge_normalized = _normalize_singular_values(
        singular_values,
        tuple(unfolded.shape),
        normalization_mode=normalization_mode,
        theoretical_mp_scale_sq=theoretical_mp_scale_sq,
    )
    return singular_values.astype(np.float32), normalized.astype(np.float32), aspect_ratio, mp_edge_normalized, tuple(unfolded.shape)


def _mp_support(aspect_ratio: float) -> tuple[float, float]:
    sqrt_q = float(np.sqrt(aspect_ratio))
    return float(abs(1.0 - sqrt_q)), float(1.0 + sqrt_q)


def _mp_density(xs: np.ndarray, aspect_ratio: float) -> np.ndarray:
    s_minus, s_plus = _mp_support(aspect_ratio)
    density = np.zeros_like(xs, dtype=np.float64)
    if aspect_ratio <= 0.0:
        return density
    mask = (xs > s_minus) & (xs < s_plus)
    if not np.any(mask):
        return density
    numerator = np.sqrt((s_plus * s_plus - xs[mask] * xs[mask]) * (xs[mask] * xs[mask] - s_minus * s_minus))
    # For aspect_ratio > 1, the rectangular MP law has an atom at zero.
    # We plot only the nonzero singular values, so renormalize the continuous
    # part to unit mass while keeping the same support.
    denominator = np.pi * min(aspect_ratio, 1.0) * xs[mask]
    density[mask] = numerator / denominator
    return density


def _format_images_seen(step: int) -> str:
    value = int(step)
    if value >= 1_000_000:
        return f'{value / 1_000_000:g}M'
    if value >= 1_000:
        return f'{value / 1_000:g}k'
    return str(value)


def _collect_member_params(group_dirs: list[str], step: int, progress_label: str) -> tuple[list[dict], list[str]]:
    if jax is None:
        raise RuntimeError('JAX is required to restore checkpoints and compute spectra.')
    _, _extract_train_state_fields, _flatten_members, _, _progress, _, _restore_state_checkpoint = _load_analysis_helpers()
    members: list[dict] = []
    labels: list[str] = []
    group_iter = _progress(
        group_dirs,
        desc=f'{progress_label} restore-groups',
        total=len(group_dirs),
        leave=False,
    )
    for group_dir in group_iter:
        state_dir = join(group_dir, 'state_ckpts')
        state_obj = _restore_state_checkpoint(state_dir, step)
        params, _, _ = _extract_train_state_fields(state_obj)
        params = _flatten_members(params)
        leaves = jax.tree_util.tree_leaves(params)
        if not leaves:
            raise ValueError(f'No parameter leaves found in checkpoint {state_dir} at step {step}.')

        num_members = int(leaves[0].shape[0])
        group_name = os.path.basename(group_dir)
        for member_idx in range(num_members):
            members.append(jax.tree_util.tree_map(lambda z: z[member_idx], params))
            labels.append(f'{group_name}/member_{member_idx}')

        del state_obj
        del params
        gc.collect()
    return members, labels


def _save_step_spectra(
    spectra_dir: str,
    width: int,
    step: int,
    source_run_id: str,
    unfolding_mode: str,
    layer_path: tuple[str, ...],
    member_labels: list[str],
    singular_values: np.ndarray,
    normalized: np.ndarray,
    aspect_ratio: float,
    mp_edge_normalized: float,
    unfolded_shape: tuple[int, int],
    theoretical_mp_scale_sq: float | None = None,
) -> None:
    out_dir = os.path.join(spectra_dir, f'width_{width}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'step_{step}.npz')
    np.savez_compressed(
        out_path,
        width=np.int32(width),
        images_seen=np.int64(step),
        source_run_id=np.asarray(source_run_id),
        unfolding_mode=np.asarray(unfolding_mode),
        layer_path=np.asarray('/'.join(layer_path)),
        member_labels=np.asarray(member_labels),
        singular_values=singular_values.astype(np.float32),
        normalized_singular_values=normalized.astype(np.float32),
        aspect_ratio=np.float32(aspect_ratio),
        mp_edge_normalized=np.float32(mp_edge_normalized),
        unfolded_shape=np.asarray(unfolded_shape, dtype=np.int32),
        theoretical_mp_scale_sq=np.float32(theoretical_mp_scale_sq) if theoretical_mp_scale_sq is not None else np.float32(np.nan),
    )


def _load_saved_step_spectra(
    spectra_dir: str,
    width: int,
    step: int,
    unfolding_mode: str,
    normalization_mode: str,
) -> tuple[np.ndarray, np.ndarray, float, float, tuple[int, int], str, tuple[str, ...], list[str]] | None:
    npz_path = os.path.join(spectra_dir, f'width_{width}', f'step_{step}.npz')
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=False)
    if 'singular_values' not in data.files or 'unfolded_shape' not in data.files or 'unfolding_mode' not in data.files:
        return None
    saved_unfolding_mode = str(data['unfolding_mode'].tolist())
    if saved_unfolding_mode != unfolding_mode:
        return None
    singular_values = np.asarray(data['singular_values'], dtype=np.float32)
    unfolded_shape = tuple(int(x) for x in np.asarray(data['unfolded_shape']).tolist())
    source_run_id = str(data['source_run_id'].tolist())
    layer_path = tuple(str(data['layer_path'].tolist()).split('/'))
    if (
        normalization_mode == 'empirical'
        and 'normalized_singular_values' in data.files
        and 'aspect_ratio' in data.files
        and 'mp_edge_normalized' in data.files
    ):
        normalized = np.asarray(data['normalized_singular_values'], dtype=np.float32)
        aspect_ratio = float(np.asarray(data['aspect_ratio']).item())
        mp_edge_normalized = float(np.asarray(data['mp_edge_normalized']).item())
    else:
        theoretical_mp_scale_sq = None
        if normalization_mode == 'init_theory':
            if 'theoretical_mp_scale_sq' in data.files:
                raw_scale = float(np.asarray(data['theoretical_mp_scale_sq']).item())
                if np.isfinite(raw_scale) and raw_scale > 0.0:
                    theoretical_mp_scale_sq = raw_scale
            if theoretical_mp_scale_sq is None:
                theoretical_mp_scale_sq = _legacy_saved_theoretical_mp_scale_sq(unfolded_shape, layer_path)
        normalized, aspect_ratio, mp_edge_normalized = _normalize_singular_values(
            singular_values,
            unfolded_shape,
            normalization_mode=normalization_mode,
            theoretical_mp_scale_sq=theoretical_mp_scale_sq,
        )
    member_labels = [str(x) for x in data['member_labels'].tolist()]
    return (
        singular_values,
        normalized,
        aspect_ratio,
        mp_edge_normalized,
        unfolded_shape,
        source_run_id,
        layer_path,
        member_labels,
    )


def _clipped_density_histogram(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    flat_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat_values.size == 0:
        return np.zeros(bin_edges.size - 1, dtype=np.float64)
    counts, _ = np.histogram(flat_values, bins=bin_edges, density=False)
    widths = np.diff(bin_edges)
    return counts.astype(np.float64) / (flat_values.size * widths)


def _plot_histograms(
    ax: plt.Axes,
    normalized_by_step: list[np.ndarray],
    common_steps: list[int],
    bulk_max: float,
    bins: int,
    colors: np.ndarray,
) -> None:
    bin_edges = np.linspace(0.0, bulk_max, int(bins) + 1)
    for step_idx, values in enumerate(normalized_by_step):
        if values.size == 0:
            continue
        hist = _clipped_density_histogram(values, bin_edges)
        ax.stairs(
            hist,
            bin_edges,
            color=colors[step_idx],
            linewidth=1.6,
            alpha=0.95,
            label=f'P={_format_images_seen(common_steps[step_idx])}',
        )


def _plot_tail_panel(
    ax: plt.Axes,
    tail_by_step: list[np.ndarray],
    common_steps: list[int],
    colors: np.ndarray,
) -> None:
    for step_idx, tail_values in enumerate(tail_by_step):
        if tail_values.size == 0:
            continue
        if tail_values.size == 1:
            offsets = np.zeros(1, dtype=np.float64)
        else:
            offsets = np.linspace(-0.18, 0.18, tail_values.size, dtype=np.float64)
        y = np.full(tail_values.shape, float(step_idx), dtype=np.float64) + offsets
        ax.scatter(
            tail_values,
            y,
            s=18,
            color=colors[step_idx],
            alpha=0.8,
            edgecolors='none',
        )
    ax.set_yticks(np.arange(len(common_steps), dtype=np.float64))
    ax.set_yticklabels([_format_images_seen(step) for step in common_steps], fontsize=8)
    ax.set_ylabel('P', fontsize=9)


def _style_spectrum_axes(
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    width: int,
    unfolded_shape: tuple[int, int],
    aspect_ratio: float,
    mp_edge_normalized: float,
    reference_edge_normalized: float,
    bulk_max: float,
    tail_max: float,
) -> None:
    del unfolded_shape
    del reference_edge_normalized
    xs = np.linspace(1e-4, bulk_max, 600, dtype=np.float64)
    density = _mp_density(xs, aspect_ratio)
    ax_top.plot(xs, density, color='black', linewidth=2.0, label='MP bulk')
    ax_top.axvline(mp_edge_normalized, color='black', linestyle='--', linewidth=2.0, label='MP edge')
    ax_bottom.axvline(mp_edge_normalized, color='black', linestyle='--', linewidth=2.0)

    ax_top.set_xlim(0.0, bulk_max)
    ax_bottom.set_xlim(0.0, max(tail_max, mp_edge_normalized * 1.05))
    ax_top.set_ylabel('Density')
    ax_bottom.set_xlabel('MP-normalized singular value')
    ax_top.set_title(f'Width {width}')
    ax_top.grid(True, alpha=0.25)
    ax_bottom.grid(True, alpha=0.25)


def _build_width_bundle(
    width: int,
    width_dir: str,
    source_run_id: str,
    spectra_dir: str,
    layer_selection: str,
    normalization_mode: str,
) -> WidthSpectrumBundle:
    _collect_target_steps, _, _, _list_group_dirs, _, _, _ = _load_analysis_helpers()
    unfolding_mode = _unfolding_mode_for_layer_selection(layer_selection)
    group_dirs = _list_group_dirs(width_dir)
    common_steps = _collect_target_steps(group_dirs)
    if not common_steps:
        raise RuntimeError(f'Width {width} has no common checkpoints across groups.')

    print(
        f'Width {width}: using source_run_id={source_run_id}, '
        f'{len(group_dirs)} groups, {len(common_steps)} common checkpoints.'
    )

    layer_path: tuple[str, ...] | None = None
    member_labels: list[str] | None = None
    normalized_by_step: list[np.ndarray] = []
    tail_by_step: list[np.ndarray] = []
    bulk_max = 0.0
    tail_max = 0.0
    aspect_ratio: float | None = None
    mp_edge_normalized: float | None = None
    reference_edge_normalized: float | None = None
    unfolded_shape: tuple[int, int] | None = None

    for step in common_steps:
        progress_label = f'w{width} p{step}'
        cached = _load_saved_step_spectra(
            spectra_dir=spectra_dir,
            width=width,
            step=step,
            unfolding_mode=unfolding_mode,
            normalization_mode=normalization_mode,
        )
        if cached is not None:
            (
                spectra_matrix,
                normalized_matrix,
                current_aspect_ratio,
                current_mp_edge,
                current_unfolded_shape,
                cached_source_run_id,
                cached_layer_path,
                cached_member_labels,
            ) = cached
            print(f'Width {width}: reusing corrected spectra for step {step}')
            if not source_run_id:
                source_run_id = cached_source_run_id
            if layer_path is None:
                layer_path = cached_layer_path
            if member_labels is None:
                member_labels = cached_member_labels
        else:
            member_params, step_member_labels = _collect_member_params(group_dirs, step, progress_label)
            if not member_params:
                raise RuntimeError(f'No member parameters restored for width {width} at step {step}.')
            if member_labels is None:
                member_labels = step_member_labels
            elif member_labels != step_member_labels:
                raise ValueError(f'Member labels changed across checkpoints for width {width}.')

            if layer_path is None:
                layer_paths = _collect_conv_layer_paths(member_params[0])
                layer_path = _resolve_layer_path(layer_paths, layer_selection)
                print(f'Width {width}: selected layer {" / ".join(layer_path)}')

            spectra_rows: list[np.ndarray] = []
            normalized_rows: list[np.ndarray] = []
            current_aspect_ratio: float | None = None
            current_mp_edge: float | None = None
            current_unfolded_shape: tuple[int, int] | None = None
            for member_params_single in member_params:
                kernel = _path_get(member_params_single, layer_path + ('kernel',))
                singular_values, normalized, member_aspect_ratio, member_mp_edge, member_unfolded_shape = _member_spectrum_summary(
                    kernel,
                    unfolding_mode=unfolding_mode,
                    normalization_mode=normalization_mode,
                )
                spectra_rows.append(singular_values)
                normalized_rows.append(normalized)
                if current_aspect_ratio is None:
                    current_aspect_ratio = member_aspect_ratio
                    current_mp_edge = member_mp_edge
                    current_unfolded_shape = member_unfolded_shape
                else:
                    if not np.isclose(current_aspect_ratio, member_aspect_ratio):
                        raise ValueError(f'Width {width} has inconsistent aspect ratios across members.')
                    if not np.isclose(current_mp_edge, member_mp_edge):
                        raise ValueError(f'Width {width} has inconsistent MP edges across members.')
                    if current_unfolded_shape != member_unfolded_shape:
                        raise ValueError(f'Width {width} has inconsistent unfolded shapes across members.')

            spectra_matrix = np.stack(spectra_rows, axis=0).astype(np.float32)
            normalized_matrix = np.stack(normalized_rows, axis=0).astype(np.float32)
            _save_step_spectra(
                spectra_dir=spectra_dir,
                width=width,
                step=step,
                source_run_id=source_run_id,
                unfolding_mode=unfolding_mode,
                layer_path=layer_path,
                member_labels=step_member_labels,
                singular_values=spectra_matrix,
                normalized=normalized_matrix,
                aspect_ratio=float(current_aspect_ratio),
                mp_edge_normalized=float(current_mp_edge),
                unfolded_shape=current_unfolded_shape,
                theoretical_mp_scale_sq=_theoretical_mp_scale_sq_from_kernel(
                    _path_get(member_params[0], layer_path + ('kernel',)),
                    current_unfolded_shape,
                ) if normalization_mode == 'init_theory' else None,
            )

            del member_params
            gc.collect()

        normalized_by_step.append(normalized_matrix)
        aspect_ratio = float(current_aspect_ratio)
        mp_edge_normalized = float(current_mp_edge)
        unfolded_shape = current_unfolded_shape
        reference_edge_normalized = _reference_edge_normalized(unfolded_shape, aspect_ratio)
        tail_by_step.append(normalized_matrix[normalized_matrix > float(reference_edge_normalized)].astype(np.float32))
        finite_values = normalized_matrix[np.isfinite(normalized_matrix)]
        if finite_values.size:
            candidate_bulk_max = float(np.quantile(finite_values, 0.995))
            bulk_max = max(bulk_max, candidate_bulk_max)
            tail_max = max(tail_max, float(np.max(finite_values)))

    if layer_path is None or member_labels is None or aspect_ratio is None or mp_edge_normalized is None or reference_edge_normalized is None or unfolded_shape is None:
        raise RuntimeError(f'Width {width} did not produce a valid spectrum bundle.')

    bulk_max = max(bulk_max, mp_edge_normalized * 1.25)
    tail_max = max(tail_max, bulk_max)
    return WidthSpectrumBundle(
        width=int(width),
        source_run_id=str(source_run_id),
        layer_path=layer_path,
        unfolded_shape=unfolded_shape,
        aspect_ratio=float(aspect_ratio),
        mp_edge_normalized=float(mp_edge_normalized),
        reference_edge_normalized=float(reference_edge_normalized),
        common_steps=[int(step) for step in common_steps],
        member_labels=member_labels,
        normalized_by_step=normalized_by_step,
        tail_by_step=tail_by_step,
        bulk_max=float(bulk_max),
        tail_max=float(tail_max),
    )


def _plot_aggregate_figure(
    bundles: list[WidthSpectrumBundle],
    output_dir: str,
    bins: int,
    artifact_stem: str,
) -> str:
    if not bundles:
        raise ValueError('No width bundles were provided for aggregate plotting.')

    num_widths = len(bundles)
    ncols = min(3, num_widths)
    nrows = int(np.ceil(num_widths / ncols))
    fig = plt.figure(figsize=(6.2 * ncols, 4.6 * nrows), constrained_layout=True)
    outer = fig.add_gridspec(nrows * 2, ncols, height_ratios=[3.0, 1.0] * nrows)

    for idx, bundle in enumerate(bundles):
        row = idx // ncols
        col = idx % ncols
        ax_top = fig.add_subplot(outer[row * 2, col])
        ax_bottom = fig.add_subplot(outer[row * 2 + 1, col], sharex=ax_top)
        colors = plt.cm.magma(np.linspace(0.2, 0.82, len(bundle.common_steps), dtype=np.float64))

        _plot_histograms(
            ax=ax_top,
            normalized_by_step=bundle.normalized_by_step,
            common_steps=bundle.common_steps,
            bulk_max=bundle.bulk_max,
            bins=bins,
            colors=colors,
        )
        _plot_tail_panel(
            ax=ax_bottom,
            tail_by_step=bundle.tail_by_step,
            common_steps=bundle.common_steps,
            colors=colors,
        )
        _style_spectrum_axes(
            ax_top=ax_top,
            ax_bottom=ax_bottom,
            width=bundle.width,
            unfolded_shape=bundle.unfolded_shape,
            aspect_ratio=bundle.aspect_ratio,
            mp_edge_normalized=bundle.mp_edge_normalized,
            reference_edge_normalized=bundle.reference_edge_normalized,
            bulk_max=bundle.bulk_max,
            tail_max=bundle.tail_max,
        )

        if idx == 0:
            handles, labels = ax_top.get_legend_handles_labels()
            if handles:
                ax_top.legend(handles, labels, fontsize=8, loc='upper right')
        else:
            legend = ax_top.get_legend()
            if legend is not None:
                legend.remove()

    out_path = os.path.join(output_dir, f'{artifact_stem}_ensemble_aggregated.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _available_steps_and_indices(bundles: list[WidthSpectrumBundle]) -> list[tuple[int, list[int | None]]]:
    if not bundles:
        return []
    ordered_steps: list[int] = []
    seen_steps: set[int] = set()
    for bundle in bundles:
        for step in bundle.common_steps:
            step_int = int(step)
            if step_int in seen_steps:
                continue
            seen_steps.add(step_int)
            ordered_steps.append(step_int)
    ordered_steps.sort()

    step_indices: list[tuple[int, list[int | None]]] = []
    for step in ordered_steps:
        indices: list[int | None] = []
        for bundle in bundles:
            try:
                indices.append(bundle.common_steps.index(step))
            except ValueError:
                indices.append(None)
        step_indices.append((step, indices))
    return step_indices


def _plot_step_aggregate_tail_panel(
    ax: plt.Axes,
    bundles: list[WidthSpectrumBundle],
    step_indices: list[int | None],
    colors: np.ndarray,
) -> None:
    for width_idx, bundle in enumerate(bundles):
        step_idx = step_indices[width_idx]
        if step_idx is None:
            continue
        tail_values = bundle.tail_by_step[step_idx]
        if tail_values.size == 0:
            continue
        if tail_values.size == 1:
            offsets = np.zeros(1, dtype=np.float64)
        else:
            offsets = np.linspace(-0.18, 0.18, tail_values.size, dtype=np.float64)
        y = np.full(tail_values.shape, float(width_idx), dtype=np.float64) + offsets
        ax.scatter(
            tail_values,
            y,
            s=18,
            color=colors[width_idx],
            alpha=0.8,
            edgecolors='none',
        )
    ax.set_yticks(np.arange(len(bundles), dtype=np.float64))
    ax.set_yticklabels([str(bundle.width) for bundle in bundles], fontsize=8)
    ax.set_ylabel('Width', fontsize=9)


def _plot_step_aggregate_figure(
    bundles: list[WidthSpectrumBundle],
    output_dir: str,
    bins: int,
    artifact_stem: str,
) -> str:
    if not bundles:
        raise ValueError('No width bundles were provided for step-wise aggregate plotting.')

    shared_steps_and_indices = _available_steps_and_indices(bundles)
    if not shared_steps_and_indices:
        raise ValueError('No checkpoints are available for step-wise aggregate plotting.')

    num_steps = len(shared_steps_and_indices)
    ncols = min(4, num_steps)
    nrows = int(np.ceil(num_steps / ncols))
    fig = plt.figure(figsize=(5.8 * ncols, 4.6 * nrows), constrained_layout=True)
    outer = fig.add_gridspec(nrows * 2, ncols, height_ratios=[3.0, 1.0] * nrows)

    width_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(bundles), dtype=np.float64))
    width_legend_handles = [
        Line2D([0], [0], color=width_colors[idx], linewidth=1.6, label=f'W={bundle.width}')
        for idx, bundle in enumerate(bundles)
    ]
    bulk_max = max(bundle.bulk_max for bundle in bundles)
    tail_max = max(bundle.tail_max for bundle in bundles)
    same_mp_edge = all(np.isclose(bundle.mp_edge_normalized, bundles[0].mp_edge_normalized) for bundle in bundles[1:])
    same_aspect_ratio = all(np.isclose(bundle.aspect_ratio, bundles[0].aspect_ratio) for bundle in bundles[1:])

    for panel_idx, (step, step_indices) in enumerate(shared_steps_and_indices):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax_top = fig.add_subplot(outer[row * 2, col])
        ax_bottom = fig.add_subplot(outer[row * 2 + 1, col], sharex=ax_top)
        bin_edges = np.linspace(0.0, bulk_max, int(bins) + 1)

        for width_idx, bundle in enumerate(bundles):
            step_idx = step_indices[width_idx]
            if step_idx is None:
                continue
            values = bundle.normalized_by_step[step_idx]
            hist = _clipped_density_histogram(values, bin_edges)
            ax_top.stairs(
                hist,
                bin_edges,
                color=width_colors[width_idx],
                linewidth=1.6,
                alpha=0.95,
                label=f'W={bundle.width}',
            )

        _plot_step_aggregate_tail_panel(
            ax=ax_bottom,
            bundles=bundles,
            step_indices=step_indices,
            colors=width_colors,
        )

        if same_aspect_ratio:
            xs = np.linspace(1e-4, bulk_max, 600, dtype=np.float64)
            density = _mp_density(xs, bundles[0].aspect_ratio)
            ax_top.plot(xs, density, color='black', linewidth=2.0, label='MP bulk')
        if same_mp_edge:
            ax_top.axvline(bundles[0].mp_edge_normalized, color='black', linestyle='--', linewidth=2.0, label='MP edge')
            ax_bottom.axvline(bundles[0].mp_edge_normalized, color='black', linestyle='--', linewidth=2.0)
        else:
            for width_idx, bundle in enumerate(bundles):
                xs = np.linspace(1e-4, bulk_max, 600, dtype=np.float64)
                density = _mp_density(xs, bundle.aspect_ratio)
                ax_top.plot(xs, density, color=width_colors[width_idx], linewidth=1.2, alpha=0.7)
                ax_top.axvline(bundle.mp_edge_normalized, color=width_colors[width_idx], linestyle='--', linewidth=1.2)
                ax_bottom.axvline(bundle.mp_edge_normalized, color=width_colors[width_idx], linestyle='--', linewidth=1.2)

        ax_top.set_xlim(0.0, bulk_max)
        ax_bottom.set_xlim(0.0, max(tail_max, max(bundle.mp_edge_normalized for bundle in bundles) * 1.05))
        ax_top.set_title(f'P={_format_images_seen(step)}')
        ax_top.set_ylabel('Density')
        ax_bottom.set_xlabel('MP-normalized singular value')
        ax_top.grid(True, alpha=0.25)
        ax_bottom.grid(True, alpha=0.25)

        if panel_idx == 0:
            ax_top.legend(width_legend_handles, [handle.get_label() for handle in width_legend_handles], fontsize=8, loc='upper right')

    out_path = os.path.join(output_dir, f'{artifact_stem}_by_p_aggregated.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _plot_member_grid(bundle: WidthSpectrumBundle, output_dir: str, bins: int, artifact_stem: str) -> str:
    num_members = len(bundle.member_labels)
    ncols = max(1, min(6, int(np.ceil(np.sqrt(num_members)))))
    nrows = int(np.ceil(num_members / ncols))
    fig = plt.figure(figsize=(3.0 * ncols, 2.5 * nrows * 2), constrained_layout=True)
    outer = fig.add_gridspec(nrows * 2, ncols, height_ratios=[3.0, 1.0] * nrows)
    colors = plt.cm.magma(np.linspace(0.2, 0.82, len(bundle.common_steps), dtype=np.float64))

    for idx, member_label in enumerate(bundle.member_labels):
        row = idx // ncols
        col = idx % ncols
        ax_top = fig.add_subplot(outer[row * 2, col])
        ax_bottom = fig.add_subplot(outer[row * 2 + 1, col], sharex=ax_top)

        member_histories = [step_values[idx:idx + 1] for step_values in bundle.normalized_by_step]
        member_tails = [
            step_values[idx][step_values[idx] > bundle.mp_edge_normalized]
            for step_values in bundle.normalized_by_step
        ]

        _plot_histograms(
            ax=ax_top,
            normalized_by_step=member_histories,
            common_steps=bundle.common_steps,
            bulk_max=bundle.bulk_max,
            bins=bins,
            colors=colors,
        )
        _plot_tail_panel(
            ax=ax_bottom,
            tail_by_step=member_tails,
            common_steps=bundle.common_steps,
            colors=colors,
        )
        _style_spectrum_axes(
            ax_top=ax_top,
            ax_bottom=ax_bottom,
            width=bundle.width,
            unfolded_shape=bundle.unfolded_shape,
            aspect_ratio=bundle.aspect_ratio,
            mp_edge_normalized=bundle.mp_edge_normalized,
            reference_edge_normalized=bundle.reference_edge_normalized,
            bulk_max=bundle.bulk_max,
            tail_max=bundle.tail_max,
        )
        ax_top.set_title(member_label, fontsize=9)
        if col != 0:
            ax_top.set_ylabel('')
            ax_bottom.set_ylabel('')
            ax_bottom.set_yticklabels([])
        if row != nrows - 1:
            ax_bottom.set_xlabel('')

    for idx in range(num_members, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax_top = fig.add_subplot(outer[row * 2, col])
        ax_bottom = fig.add_subplot(outer[row * 2 + 1, col])
        ax_top.axis('off')
        ax_bottom.axis('off')

    fig.suptitle(
        f'Width {bundle.width} member-wise spectra: {" / ".join(bundle.layer_path)}',
        fontsize=14,
    )
    out_path = os.path.join(output_dir, f'{artifact_stem}_members_w{bundle.width}.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    _, _, _, _, _, _resolve_width_dirs, _ = _load_analysis_helpers()
    spectra_dir = os.path.abspath(args.spectra_dir) if args.spectra_dir else _default_spectra_dir(args.artifact_stem)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else _default_output_dir(args.artifact_stem)
    os.makedirs(spectra_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    width_dirs, width_sources = _resolve_width_dirs(
        base_save_dir=args.base_save_dir,
        run_id=args.run_id,
        resolution_mode=args.run_id_resolution,
        requested_widths=args.widths,
    )
    if not width_dirs:
        raise RuntimeError('No width directories were resolved for the requested run.')

    bundles: list[WidthSpectrumBundle] = []
    for width, width_dir in sorted(width_dirs.items()):
        source_run_id = width_sources.get(width, args.run_id)
        bundle = _build_width_bundle(
            width=int(width),
            width_dir=width_dir,
            source_run_id=source_run_id,
            spectra_dir=spectra_dir,
            layer_selection=args.layer_selection,
            normalization_mode=args.normalization_mode,
        )
        bundles.append(bundle)
        member_plot_path = _plot_member_grid(bundle, output_dir=output_dir, bins=args.bins, artifact_stem=args.artifact_stem)
        print(f'Wrote member-wise figure for width {width}: {member_plot_path}')

    aggregate_path = _plot_aggregate_figure(bundles, output_dir=output_dir, bins=args.bins, artifact_stem=args.artifact_stem)
    print(f'Wrote aggregate figure: {aggregate_path}')
    step_aggregate_path = _plot_step_aggregate_figure(bundles, output_dir=output_dir, bins=args.bins, artifact_stem=args.artifact_stem)
    print(f'Wrote step-wise aggregate figure: {step_aggregate_path}')
    print(f'Wrote raw spectra under: {spectra_dir}')


if __name__ == '__main__':
    main()
