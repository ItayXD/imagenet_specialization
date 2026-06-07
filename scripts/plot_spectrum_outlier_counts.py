#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


_WIDTH_RE = re.compile(r'^width_(\d+)$')
_STEP_RE = re.compile(r'^step_(\d+)\.npz$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot the number of MP-edge outliers vs P from cached spectrum NPZ files.'
    )
    parser.add_argument('--spectra-dir', required=True, help='Directory containing width_*/step_*.npz files.')
    parser.add_argument('--output-dir', required=True, help='Directory where the plot and CSV will be written.')
    parser.add_argument(
        '--artifact-stem',
        default='resnet_block1_conv1_spectra',
        help='Filename stem used for the output plot and CSV.',
    )
    return parser.parse_args()


def _discover_npz_paths(spectra_dir: str) -> dict[int, list[tuple[int, str]]]:
    by_width: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for name in sorted(os.listdir(spectra_dir)):
        width_match = _WIDTH_RE.match(name)
        if width_match is None:
            continue
        width = int(width_match.group(1))
        width_dir = os.path.join(spectra_dir, name)
        if not os.path.isdir(width_dir):
            continue
        for child in sorted(os.listdir(width_dir)):
            step_match = _STEP_RE.match(child)
            if step_match is None:
                continue
            step = int(step_match.group(1))
            by_width[width].append((step, os.path.join(width_dir, child)))
        by_width[width].sort(key=lambda item: item[0])
    return dict(sorted(by_width.items()))


def _legacy_saved_theoretical_mp_scale_sq(
    unfolded_shape: tuple[int, int],
    layer_path: tuple[str, ...],
) -> float:
    del unfolded_shape
    if layer_path == ('conv_init',):
        return 2.0
    return 2.0 / 3.0


def _load_theory_normalized_singular_values(npz_path: str) -> tuple[np.ndarray, float]:
    data = np.load(npz_path, allow_pickle=False)
    if 'singular_values' not in data.files or 'unfolded_shape' not in data.files:
        raise ValueError(f'Missing raw spectrum fields in {npz_path}.')
    singular_values = np.asarray(data['singular_values'], dtype=np.float32)
    unfolded_shape = tuple(int(x) for x in np.asarray(data['unfolded_shape']).tolist())
    layer_path = tuple(str(data['layer_path'].tolist()).split('/')) if 'layer_path' in data.files else ()
    theoretical_mp_scale_sq = None
    if 'theoretical_mp_scale_sq' in data.files:
        raw_scale = float(np.asarray(data['theoretical_mp_scale_sq']).item())
        if np.isfinite(raw_scale) and raw_scale > 0.0:
            theoretical_mp_scale_sq = raw_scale
    if theoretical_mp_scale_sq is None:
        theoretical_mp_scale_sq = _legacy_saved_theoretical_mp_scale_sq(unfolded_shape, layer_path)
    if theoretical_mp_scale_sq <= 0.0:
        raise ValueError(f'Encountered non-positive theoretical scale in {npz_path}.')
    aspect_ratio = float(unfolded_shape[0]) / float(unfolded_shape[1])
    mp_edge = float(1.0 + np.sqrt(aspect_ratio))
    normalized = singular_values / np.sqrt(float(theoretical_mp_scale_sq))
    return normalized.astype(np.float32), mp_edge


def _summarize_npz(width: int, step: int, npz_path: str) -> dict[str, float | int]:
    normalized, mp_edge = _load_theory_normalized_singular_values(npz_path)
    if normalized.ndim != 2 or normalized.shape[0] == 0:
        raise ValueError(f'Expected a 2D member-by-singular-value array in {npz_path}, got {normalized.shape}.')
    counts = np.sum(normalized > mp_edge, axis=1, dtype=np.int32)
    fractions = counts.astype(np.float64) / float(normalized.shape[1])
    excess = np.maximum(normalized - mp_edge, 0.0)
    excess_sums = np.sum(excess, axis=1, dtype=np.float64)
    return {
        'width': int(width),
        'images_seen': int(step),
        'mean_outlier_count': float(np.mean(counts, dtype=np.float64)),
        'std_outlier_count': float(np.std(counts, dtype=np.float64, ddof=0)),
        'mean_outlier_fraction': float(np.mean(fractions, dtype=np.float64)),
        'std_outlier_fraction': float(np.std(fractions, dtype=np.float64, ddof=0)),
        'mean_outlier_excess_sum': float(np.mean(excess_sums, dtype=np.float64)),
        'std_outlier_excess_sum': float(np.std(excess_sums, dtype=np.float64, ddof=0)),
        'num_members': int(counts.shape[0]),
    }


def _write_csv(path: str, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        raise RuntimeError('No outlier-count rows were produced.')
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float | int]], output_path: str) -> None:
    grouped: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for row in rows:
        grouped[int(row['width'])].append(row)
    for width in grouped:
        grouped[width].sort(key=lambda row: int(row['images_seen']))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    widths = sorted(grouped)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(widths), dtype=np.float64))

    for color, width in zip(colors, widths, strict=True):
        width_rows = grouped[width]
        xs = np.asarray([int(row['images_seen']) for row in width_rows], dtype=np.float64)
        ys = np.asarray([float(row['mean_outlier_count']) for row in width_rows], dtype=np.float64)
        yerr = np.asarray([float(row['std_outlier_count']) for row in width_rows], dtype=np.float64)
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            color=color,
            marker='o',
            markersize=4.5,
            linewidth=1.8,
            elinewidth=1.1,
            capsize=2.5,
            label=f'W={width}',
        )

    ax.set_xscale('log')
    ax.set_xlabel('P')
    ax.set_ylabel('Outlier count above MP edge')
    ax.set_title('Outlier count vs P by width')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    spectra_dir = os.path.abspath(args.spectra_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    by_width = _discover_npz_paths(spectra_dir)
    if not by_width:
        raise RuntimeError(f'No width spectra were found under {spectra_dir}.')

    rows: list[dict[str, float | int]] = []
    for width, step_paths in by_width.items():
        for step, npz_path in step_paths:
            rows.append(_summarize_npz(width=width, step=step, npz_path=npz_path))

    rows.sort(key=lambda row: (int(row['width']), int(row['images_seen'])))
    csv_path = os.path.join(output_dir, f'{args.artifact_stem}_outlier_count_by_p.csv')
    pdf_path = os.path.join(output_dir, f'{args.artifact_stem}_outlier_count_by_p.pdf')
    _write_csv(csv_path, rows)
    _plot(rows, pdf_path)
    print(f'Wrote CSV: {csv_path}')
    print(f'Wrote PDF: {pdf_path}')


if __name__ == '__main__':
    main()
