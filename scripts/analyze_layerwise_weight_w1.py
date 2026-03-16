#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import time
from os.path import join

import jax
import jax.numpy as jnp
import numpy as np

from scripts.analyze_exchangeability import (
    _aggregate_metrics,
    _collect_target_steps,
    _extract_train_state_fields,
    _flatten_members,
    _list_group_dirs,
    _progress,
    _resolve_width_dirs,
    _restore_state_checkpoint,
)
from src.experiment.exchangeability_utils import (
    w1_distance,
)


LAYERWISE_FIELDNAMES = [
    'width',
    'source_run_id',
    'images_seen',
    'layer_index',
    'layer_name',
    'member_width',
    'kernel_numel',
    'num_members',
    'w1_distance',
    'train_loss',
    'val_loss',
    'train_error',
    'val_error',
]

_OOM_SNIPPETS = (
    'RESOURCE_EXHAUSTED',
    'Out of memory',
    'out of memory',
)
_DIGIT_RE = re.compile(r'(\d+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute exact within-vs-across W1 for every convolutional weight layer and plot heatmaps.'
    )
    parser.add_argument('--base-save-dir', required=True, help='Root save directory (BASE_SAVE_DIR)')
    parser.add_argument('--run-id', default='exchangeability', help='Run id folder name under base-save-dir')
    parser.add_argument(
        '--run-id-resolution',
        choices=['exact', 'latest_prefix', 'auto'],
        default='auto',
        help='How to resolve --run-id when both exact and suffixed runs exist.',
    )
    parser.add_argument(
        '--output-csv',
        default='outputs/layerwise_weight_w1.csv',
        help='Output CSV path for exact per-layer observed W1 rows.',
    )
    parser.add_argument(
        '--gpu-block-rows',
        type=int,
        default=0,
        help='Optional row-block size for GPU similarity matmul. Set 0 to try the full matrix first.',
    )
    parser.add_argument('--widths', type=int, nargs='*', default=None, help='Optional list of widths to analyze')
    parser.add_argument('--resume', dest='resume', action='store_true', default=True, help='Resume from existing output CSV when possible')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Ignore existing output CSV and recompute everything')
    return parser.parse_args()


def _natural_key(text: str) -> list[object]:
    parts = _DIGIT_RE.split(text.lower())
    return [int(part) if part.isdigit() else part for part in parts]


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


def _flatten_kernel_filters(kernel) -> np.ndarray:
    kernel_arr = np.asarray(kernel, dtype=np.float32)
    out_channels = int(kernel_arr.shape[-1])
    return kernel_arr.reshape((-1, out_channels)).T


def _collect_member_params(group_dirs: list[str], step: int, progress_label: str = '') -> list[dict]:
    members: list[dict] = []
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
        for member_idx in range(num_members):
            members.append(jax.tree_util.tree_map(lambda z: z[member_idx], params))

        del state_obj
        del params
        gc.collect()

    return members


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc)
    return any(snippet in message for snippet in _OOM_SNIPPETS)


def _normalize_filter_rows(filters: np.ndarray) -> jnp.ndarray:
    x = jnp.asarray(filters, dtype=jnp.float32)
    norms = jnp.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = jnp.where(norms == 0.0, 1.0, norms)
    return x / safe_norms


def _similarity_matrix_from_filters(filters: np.ndarray, gpu_block_rows: int) -> np.ndarray:
    normalized = _normalize_filter_rows(filters)
    total_rows = int(normalized.shape[0])
    block_rows = int(gpu_block_rows)

    if block_rows <= 0:
        try:
            sim = jnp.matmul(
                normalized,
                jnp.swapaxes(normalized, 0, 1),
                precision=jax.lax.Precision.HIGHEST,
            )
            return np.asarray(sim, dtype=np.float32)
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            fallback_rows = min(total_rows, 8192)
            if fallback_rows <= 0 or fallback_rows >= total_rows:
                raise
            print(
                f'  full GPU similarity OOM for {total_rows} rows; '
                f'retrying with --gpu-block-rows={fallback_rows}'
            )
            return _similarity_matrix_from_filters(filters, fallback_rows)

    block_rows = min(block_rows, total_rows)
    sim = np.empty((total_rows, total_rows), dtype=np.float32)
    normalized_t = jnp.swapaxes(normalized, 0, 1)
    row_start = 0
    while row_start < total_rows:
        row_end = min(row_start + block_rows, total_rows)
        try:
            block = jnp.matmul(
                normalized[row_start:row_end],
                normalized_t,
                precision=jax.lax.Precision.HIGHEST,
            )
        except Exception as exc:
            if not _is_oom_error(exc) or block_rows <= 1:
                raise
            new_block_rows = max(1, block_rows // 2)
            if new_block_rows == block_rows:
                new_block_rows = block_rows - 1
            print(
                f'  GPU similarity OOM at block_rows={block_rows}; '
                f'retrying with --gpu-block-rows={new_block_rows}'
            )
            return _similarity_matrix_from_filters(filters, new_block_rows)
        sim[row_start:row_end] = np.asarray(block, dtype=np.float32)
        row_start = row_end
    return sim


def _stack_layer_filters(member_params: list[dict], layer_path: tuple[str, ...]) -> tuple[np.ndarray, int, int]:
    sample_flat = _flatten_kernel_filters(_path_get(member_params[0], layer_path + ('kernel',)))
    member_width, kernel_numel = map(int, sample_flat.shape)
    stacked = np.empty((len(member_params) * member_width, kernel_numel), dtype=np.float32)
    stacked[0:member_width] = sample_flat

    for member_idx in range(1, len(member_params)):
        flat = _flatten_kernel_filters(_path_get(member_params[member_idx], layer_path + ('kernel',)))
        if flat.shape != (member_width, kernel_numel):
            raise ValueError(
                f'Layer {"/".join(layer_path)} has inconsistent member shape: '
                f'expected {(member_width, kernel_numel)}, got {flat.shape}'
            )
        start = member_idx * member_width
        stacked[start:start + member_width] = flat

    return stacked, member_width, kernel_numel


def _extract_within_values_blocked(
    similarity_matrix: np.ndarray,
    num_members: int,
    member_width: int,
) -> np.ndarray:
    tri_i, tri_j = np.triu_indices(member_width, k=1)
    per_member = int(tri_i.size)
    within = np.empty(num_members * per_member, dtype=similarity_matrix.dtype)
    offset = 0

    for member_idx in range(num_members):
        start = member_idx * member_width
        end = start + member_width
        block = similarity_matrix[start:end, start:end]
        within[offset:offset + per_member] = block[tri_i, tri_j]
        offset += per_member

    return within


def _extract_across_values_blocked(
    similarity_matrix: np.ndarray,
    num_members: int,
    member_width: int,
) -> np.ndarray:
    per_pair = member_width * member_width
    total_pairs = num_members * (num_members - 1) // 2
    across = np.empty(total_pairs * per_pair, dtype=similarity_matrix.dtype)
    offset = 0

    for left_member in range(num_members):
        left_start = left_member * member_width
        left_end = left_start + member_width
        for right_member in range(left_member + 1, num_members):
            right_start = right_member * member_width
            right_end = right_start + member_width
            block = similarity_matrix[left_start:left_end, right_start:right_end]
            across[offset:offset + per_pair] = block.reshape(-1)
            offset += per_pair

    return across


def _row_identity(row: dict[str, str]) -> tuple[int, str, int, int]:
    return (
        int(row['width']),
        str(row.get('source_run_id', '')),
        int(row['images_seen']),
        int(row['layer_index']),
    )


def _read_existing_rows(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _append_rows(path: str, rows: list[dict], write_header: bool) -> None:
    if not rows:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mode = 'a'
    if write_header:
        mode = 'w'
    with open(path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LAYERWISE_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    normalized = np.full_like(matrix, np.nan, dtype=np.float64)
    for row_idx in range(matrix.shape[0]):
        row = matrix[row_idx]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        row_min = float(np.min(row[finite]))
        row_max = float(np.max(row[finite]))
        if row_max == row_min:
            normalized[row_idx, finite] = 0.5
        else:
            normalized[row_idx, finite] = (row[finite] - row_min) / (row_max - row_min)
    return normalized


def main() -> None:
    args = parse_args()
    output_csv = os.path.abspath(args.output_csv)

    if not args.resume and os.path.exists(output_csv):
        os.remove(output_csv)

    existing_rows = _read_existing_rows(output_csv) if args.resume else []
    completed = {_row_identity(row) for row in existing_rows}
    write_header = not bool(existing_rows)

    width_dirs, width_sources = _resolve_width_dirs(
        base_save_dir=args.base_save_dir,
        run_id=args.run_id,
        resolution_mode=args.run_id_resolution,
        requested_widths=args.widths,
    )
    if not width_dirs:
        raise RuntimeError('No width directories were resolved for the requested run.')

    for width, width_dir in sorted(width_dirs.items()):
        source_run_id = width_sources.get(width, args.run_id)
        group_dirs = _list_group_dirs(width_dir)
        common_steps = _collect_target_steps(group_dirs)
        metrics_by_step = _aggregate_metrics(group_dirs)

        print(
            f'Width {width}: {len(group_dirs)} groups, {len(common_steps)} common checkpoints, '
            f'backend={jax.default_backend()}'
        )

        layer_paths: list[tuple[str, ...]] | None = None
        for step in common_steps:
            step_label = f'w{width} p{step}'
            t0 = time.time()
            member_params = _collect_member_params(group_dirs, step, progress_label=step_label)
            if not member_params:
                raise ValueError(f'No member parameters restored for width={width} step={step}.')

            if layer_paths is None:
                layer_paths = _collect_conv_layer_paths(member_params[0])
                print(f'Width {width}: detected {len(layer_paths)} convolutional layers.')

            metric_payload = metrics_by_step.get(
                step,
                {
                    'train_loss': np.nan,
                    'val_loss': np.nan,
                    'train_error': np.nan,
                    'val_error': np.nan,
                },
            )

            new_rows: list[dict] = []
            for layer_index, layer_path in enumerate(layer_paths):
                row_key = (int(width), str(source_run_id), int(step), int(layer_index))
                if row_key in completed:
                    continue

                layer_name = '/'.join(layer_path)
                print(f'  Starting {step_label} layer={layer_index} name={layer_name}')
                layer_t0 = time.time()

                stacked_filters, member_width, kernel_numel = _stack_layer_filters(member_params, layer_path)
                similarity = _similarity_matrix_from_filters(stacked_filters, args.gpu_block_rows)
                across = _extract_across_values_blocked(similarity, len(member_params), member_width)
                within = _extract_within_values_blocked(similarity, len(member_params), member_width)
                observed_w1 = w1_distance(within, across)

                row = {
                    'width': int(width),
                    'source_run_id': str(source_run_id),
                    'images_seen': int(step),
                    'layer_index': int(layer_index),
                    'layer_name': layer_name,
                    'member_width': int(member_width),
                    'kernel_numel': int(kernel_numel),
                    'num_members': int(len(member_params)),
                    'w1_distance': float(observed_w1),
                    'train_loss': float(metric_payload['train_loss']),
                    'val_loss': float(metric_payload['val_loss']),
                    'train_error': float(metric_payload['train_error']),
                    'val_error': float(metric_payload['val_error']),
                }
                new_rows.append(row)
                completed.add(row_key)

                print(
                    f'  Completed {step_label} layer={layer_index} '
                    f'w1={observed_w1:.6f} elapsed={time.time() - layer_t0:.1f}s'
                )

                del stacked_filters
                del similarity
                del across
                del within
                gc.collect()

            if new_rows:
                _append_rows(output_csv, new_rows, write_header=write_header)
                write_header = False

            del member_params
            gc.collect()
            print(f'Completed {step_label} elapsed={time.time() - t0:.1f}s')

    print(f'Wrote layerwise weight W1 rows to {output_csv}')


if __name__ == '__main__':
    main()
