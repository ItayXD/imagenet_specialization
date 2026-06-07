#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import replace

import numpy as np

from scripts.plot_resnet_block_spectra import (
    WidthSpectrumBundle,
    _default_artifact_stem,
    _default_output_dir,
    _legacy_saved_theoretical_mp_scale_sq,
    _normalize_singular_values,
    _plot_aggregate_figure,
    _plot_member_grid,
    _plot_outlier_stat_by_step,
    _plot_step_aggregate_figure,
    _reference_edge_normalized,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Rebuild spectrum figures from previously saved per-step NPZ files.'
    )
    parser.add_argument(
        '--spectra-dir',
        required=True,
        help='Directory containing width_*/step_*.npz outputs from plot_resnet_block_spectra.py',
    )
    parser.add_argument(
        '--output-dir',
        default='',
        help='Directory where refreshed figures should be written.',
    )
    parser.add_argument(
        '--artifact-stem',
        default=_default_artifact_stem(),
        help='Filename stem used for rebuilt PDF names.',
    )
    parser.add_argument('--bins', type=int, default=60, help='Histogram bins for the bulk panel.')
    return parser.parse_args()


def _load_bundle(width_dir: str) -> WidthSpectrumBundle:
    width_name = os.path.basename(width_dir)
    width = int(width_name.split('_')[-1])
    npz_paths = sorted(
        [
            os.path.join(width_dir, name)
            for name in os.listdir(width_dir)
            if name.endswith('.npz')
        ],
        key=lambda path: int(os.path.splitext(os.path.basename(path))[0].split('_')[-1]),
    )
    if not npz_paths:
        raise RuntimeError(f'No NPZ files found under {width_dir}')

    common_steps: list[int] = []
    normalized_by_step: list[np.ndarray] = []
    tail_by_step: list[np.ndarray] = []
    source_run_id = ''
    layer_path: tuple[str, ...] | None = None
    aspect_ratio: float | None = None
    mp_edge_normalized: float | None = None
    member_labels: list[str] | None = None
    bulk_max = 0.0
    tail_max = 0.0
    unfolded_shape: tuple[int, int] | None = None

    for npz_path in npz_paths:
        data = np.load(npz_path, allow_pickle=False)
        if 'singular_values' not in data.files or 'unfolded_shape' not in data.files:
            raise RuntimeError(
                f'{npz_path} does not contain the saved singular values needed for replotting.'
            )

        common_steps.append(int(data['images_seen']))
        singular_values = np.asarray(data['singular_values'], dtype=np.float32)
        current_unfolded_shape = tuple(int(x) for x in np.asarray(data['unfolded_shape']).tolist())
        current_layer = tuple(str(data['layer_path'].tolist()).split('/'))
        theoretical_mp_scale_sq = None
        if 'theoretical_mp_scale_sq' in data.files:
            raw_scale = float(np.asarray(data['theoretical_mp_scale_sq']).item())
            if np.isfinite(raw_scale) and raw_scale > 0.0:
                theoretical_mp_scale_sq = raw_scale
        if theoretical_mp_scale_sq is None:
            theoretical_mp_scale_sq = _legacy_saved_theoretical_mp_scale_sq(
                current_unfolded_shape,
                current_layer,
            )
        normalized, current_aspect_ratio, edge = _normalize_singular_values(
            singular_values,
            current_unfolded_shape,
            theoretical_mp_scale_sq=theoretical_mp_scale_sq,
        )
        reference_edge = _reference_edge_normalized(current_unfolded_shape, current_aspect_ratio)
        normalized_by_step.append(normalized)
        tail_by_step.append(normalized[normalized > reference_edge].astype(np.float32))

        finite_values = normalized[np.isfinite(normalized)]
        if finite_values.size:
            bulk_max = max(bulk_max, float(np.quantile(finite_values, 0.995)))
            tail_max = max(tail_max, float(np.max(finite_values)))

        if layer_path is None:
            layer_path = current_layer
            source_run_id = str(data['source_run_id'].tolist())
            aspect_ratio = current_aspect_ratio
            mp_edge_normalized = edge
            member_labels = [str(x) for x in data['member_labels'].tolist()]
            unfolded_shape = current_unfolded_shape
        else:
            if layer_path != current_layer:
                raise RuntimeError(f'Layer path changed inside {width_dir}')

    if layer_path is None or aspect_ratio is None or mp_edge_normalized is None or member_labels is None or unfolded_shape is None:
        raise RuntimeError(f'Failed to assemble width bundle from {width_dir}')

    bulk_max = max(bulk_max, mp_edge_normalized * 1.25)
    tail_max = max(tail_max, bulk_max)
    reference_edge_normalized = _reference_edge_normalized(unfolded_shape, aspect_ratio)
    return WidthSpectrumBundle(
        width=width,
        source_run_id=source_run_id,
        layer_path=layer_path,
        unfolded_shape=unfolded_shape,
        aspect_ratio=aspect_ratio,
        mp_edge_normalized=mp_edge_normalized,
        reference_edge_normalized=reference_edge_normalized,
        common_steps=common_steps,
        member_labels=member_labels,
        normalized_by_step=normalized_by_step,
        tail_by_step=tail_by_step,
        bulk_max=bulk_max,
        tail_max=tail_max,
    )


def main() -> None:
    args = parse_args()
    spectra_dir = os.path.abspath(args.spectra_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else _default_output_dir(args.artifact_stem)
    os.makedirs(output_dir, exist_ok=True)

    width_dirs = sorted(
        [
            os.path.join(spectra_dir, name)
            for name in os.listdir(spectra_dir)
            if name.startswith('width_') and os.path.isdir(os.path.join(spectra_dir, name))
        ],
        key=lambda path: int(os.path.basename(path).split('_')[-1]),
    )
    if not width_dirs:
        raise RuntimeError(f'No width_* directories found under {spectra_dir}')

    bundles = [_load_bundle(width_dir) for width_dir in width_dirs]
    for bundle in bundles:
        out_path = _plot_member_grid(bundle, output_dir=output_dir, bins=args.bins, artifact_stem=args.artifact_stem)
        print(f'Wrote member-wise figure for width {bundle.width}: {out_path}')

    aggregate_path = _plot_aggregate_figure(bundles, output_dir=output_dir, bins=args.bins, artifact_stem=args.artifact_stem)
    print(f'Wrote aggregate figure: {aggregate_path}')
    step_aggregate_path = _plot_step_aggregate_figure(bundles, output_dir=output_dir, bins=args.bins, artifact_stem=args.artifact_stem)
    print(f'Wrote step-wise aggregate figure: {step_aggregate_path}')
    outlier_count_path = _plot_outlier_stat_by_step(
        bundles,
        output_dir=output_dir,
        artifact_stem=args.artifact_stem,
        normalize_by_width=False,
    )
    print(f'Wrote outlier-count figure: {outlier_count_path}')
    outlier_fraction_path = _plot_outlier_stat_by_step(
        bundles,
        output_dir=output_dir,
        artifact_stem=args.artifact_stem,
        normalize_by_width=True,
    )
    print(f'Wrote outlier-fraction figure: {outlier_fraction_path}')


if __name__ == '__main__':
    main()
