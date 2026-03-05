#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from os.path import join
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder, ImageNet
import torchvision.transforms as transforms

from src.experiment.exchangeability_utils import (
    abs_cosine_similarity_matrix,
    build_member_ids,
    extract_across_values,
    extract_within_values,
    ks_w1_stats,
    shuffled_similarity_values_batched,
    shuffled_similarity_values,
    two_sided_sigma_from_p,
)
from src.experiment.model.flax_mup.resnet import ResNet18
from src.run.constants import IMAGENET_FOLDER


STATE_PATTERN = re.compile(r'^state_(\d+)')


def _is_notebook_session() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    return ip.__class__.__name__ == 'ZMQInteractiveShell'


def _progress(iterable, **kwargs):
    kwargs.setdefault('dynamic_ncols', True)
    # In non-interactive log streams, tqdm prints one line per refresh.
    # Disable bars unless we're in a tty or notebook session.
    if 'disable' not in kwargs:
        kwargs['disable'] = (not sys.stdout.isatty()) and (not _is_notebook_session())
    return tqdm(iterable, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze exchangeability from grouped ensemble checkpoints.')
    parser.add_argument('--base-save-dir', required=True, help='Root save directory (BASE_SAVE_DIR)')
    parser.add_argument('--run-id', default='exchangeability', help='Run id folder name under base-save-dir')
    parser.add_argument(
        '--run-id-resolution',
        choices=['exact', 'latest_prefix', 'auto'],
        default='auto',
        help='How to resolve --run-id when both exact and suffixed runs exist.',
    )
    parser.add_argument('--output-csv', default='outputs/exchangeability_metrics.csv', help='Output CSV path')
    parser.add_argument('--shuffle-repeats', type=int, default=2000, help='Number of shuffle repeats')
    parser.add_argument(
        '--shuffle-batch-size',
        type=int,
        default=1,
        help='Number of shuffled permutations to generate in one vectorized batch',
    )
    parser.add_argument(
        '--shuffle-stats-workers',
        type=int,
        default=1,
        help='Thread workers for per-shuffle KS/W1 stats within each batch',
    )
    parser.add_argument(
        '--log-every-shuffles',
        type=int,
        default=100,
        help='Print shuffle progress every N iterations (<=0 disables periodic shuffle logs)',
    )
    parser.add_argument(
        '--write-every-shuffles',
        type=int,
        default=1,
        help='Flush CSV rows every N shuffles (<=0 disables periodic mid-representation writes)',
    )
    parser.add_argument('--probe-batch-size', type=int, default=1024, help='Total probe images for activation vectors')
    parser.add_argument(
        '--probe-loader-batch-size',
        type=int,
        default=128,
        help='Probe dataloader batch size used for streaming accumulation',
    )
    parser.add_argument('--probe-seed', type=int, default=1234, help='Seed for probe subset selection')
    parser.add_argument('--widths', type=int, nargs='*', default=None, help='Optional list of widths to analyze')
    parser.add_argument('--resume', dest='resume', action='store_true', default=True, help='Resume from existing output CSV when possible')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Ignore existing output CSV and recompute everything')
    return parser.parse_args()



def _get_nested_item(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    try:
        return obj[key]
    except Exception:
        return None



def _list_width_dirs(base_save_dir: str, run_id: str) -> dict[int, str]:
    root = join(base_save_dir, run_id)
    width_dirs = {}
    for d in sorted(glob(join(root, 'width_*'))):
        name = os.path.basename(d)
        width = int(name.split('_')[-1])
        width_dirs[width] = d
    return width_dirs



def _resolve_run_id(base_save_dir: str, run_id: str, resolution_mode: str) -> str:
    if not os.path.isdir(base_save_dir):
        raise FileNotFoundError(f'Base save dir does not exist: {base_save_dir}')

    exact_dir = join(base_save_dir, run_id)
    has_exact = os.path.isdir(exact_dir)

    prefix = f'{run_id}_'
    candidates: list[tuple[float, str]] = []
    for name in os.listdir(base_save_dir):
        path = join(base_save_dir, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith(prefix):
            continue
        candidates.append((os.path.getmtime(path), name))

    if resolution_mode == 'exact':
        if has_exact:
            return run_id
        raise FileNotFoundError(
            f'Run id "{run_id}" not found under {base_save_dir} (exact resolution mode).'
        )

    if resolution_mode == 'latest_prefix':
        if not candidates:
            if has_exact:
                print(
                    f'Run id resolution requested latest_prefix, but no "{prefix}*" dirs found. '
                    f'Falling back to exact run "{run_id}".'
                )
                return run_id
            raise FileNotFoundError(
                f'No "{prefix}*" runs found under {base_save_dir}, and exact run "{run_id}" is missing.'
            )
        candidates.sort(key=lambda x: x[0])
        resolved = candidates[-1][1]
        print(f'Run id "{run_id}" resolved via latest_prefix to "{resolved}".')
        return resolved

    # auto: choose newest between exact run and suffixed runs when both are present.
    if has_exact:
        candidates.append((os.path.getmtime(exact_dir), run_id))

    if not candidates:
        raise FileNotFoundError(
            f'Run id "{run_id}" not found under {base_save_dir}, and no "{prefix}*" runs were found.'
        )

    candidates.sort(key=lambda x: x[0])
    resolved = candidates[-1][1]
    if resolved == run_id:
        if len(candidates) > 1:
            print(
                f'Run id "{run_id}" resolved via auto to exact run "{resolved}" '
                f'(newer than available "{prefix}*" runs).'
            )
    else:
        print(f'Run id "{run_id}" resolved via auto to latest matching run "{resolved}".')
    return resolved



def _list_group_dirs(width_dir: str) -> list[str]:
    return sorted(glob(join(width_dir, 'group_*')))



def _list_state_steps(group_dir: str) -> set[int]:
    state_dir = join(group_dir, 'state_ckpts')
    if not os.path.isdir(state_dir):
        return set()

    steps = set()
    for name in os.listdir(state_dir):
        m = STATE_PATTERN.match(name)
        if m:
            steps.add(int(m.group(1)))
    return steps



def _load_group_metrics(group_dir: str) -> dict[int, dict]:
    metric_path = join(group_dir, 'metrics.jsonl')
    if not os.path.exists(metric_path):
        return {}

    out = {}
    with open(metric_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            out[int(row['images_seen'])] = row
    return out



def _collect_target_steps(group_dirs: list[str]) -> list[int]:
    if not group_dirs:
        return []
    common = None
    group_iter = _progress(
        group_dirs,
        desc='collect-target-steps',
        total=len(group_dirs),
        leave=False,
    )
    for d in group_iter:
        s = _list_state_steps(d)
        common = s if common is None else (common & s)
    return sorted(common) if common is not None else []



def _find_conv_init_kernel(params):
    conv_init = _get_nested_item(params, 'conv_init')
    if conv_init is not None:
        kernel = _get_nested_item(conv_init, 'kernel')
        if kernel is not None:
            return kernel

    if isinstance(params, dict):
        values = params.values()
    else:
        try:
            values = [v for _, v in params.items()]
        except Exception:
            values = []

    for value in values:
        kernel = _find_conv_init_kernel(value)
        if kernel is not None:
            return kernel

    return None



def _extract_weights_from_artifacts(group_dir: str, step: int) -> np.ndarray:
    path = join(group_dir, 'artifacts', f'first_layer_{step}.npz')
    data = np.load(path)
    raw = np.asarray(data['first_layer_weights'])
    if np.issubdtype(raw.dtype, np.floating):
        return raw.astype(np.float32, copy=False)

    # JAX bfloat16 can be materialized in NumPy archives as raw 2-byte records ("|V2").
    # Reconstruct float32 values from the bfloat16 bit pattern.
    if raw.dtype.kind == 'V' and raw.dtype.itemsize == 2:
        bits_u16 = np.frombuffer(raw.tobytes(), dtype=np.uint16).reshape(raw.shape)
        bits_u32 = bits_u16.astype(np.uint32) << 16
        return bits_u32.view(np.float32)

    try:
        return raw.astype(np.float32)
    except Exception as exc:
        raise ValueError(
            f'Unsupported first-layer weight dtype in artifact {path}: {raw.dtype}'
        ) from exc



def _missing_weight_artifacts(group_dirs: list[str], step: int) -> list[str]:
    missing = []
    filename = f'first_layer_{step}.npz'
    for group_dir in group_dirs:
        path = join(group_dir, 'artifacts', filename)
        if not os.path.exists(path):
            missing.append(path)
    return missing



def _extract_train_state_fields(state_obj):
    if isinstance(state_obj, dict):
        return state_obj['params'], state_obj['batch_stats'], state_obj['mup']
    return state_obj.params, state_obj.batch_stats, state_obj.mup



def _flatten_members(tree):
    return jax.tree_util.tree_map(lambda z: np.asarray(z).reshape((-1,) + np.asarray(z).shape[2:]), tree)



def _member_variables_from_state(state_obj):
    params, batch_stats, mup = _extract_train_state_fields(state_obj)
    params = _flatten_members(params)
    batch_stats = _flatten_members(batch_stats)
    mup = _flatten_members(mup)

    leaves = jax.tree_util.tree_leaves(params)
    n_members = leaves[0].shape[0]

    members = []
    for idx in range(n_members):
        members.append(
            {
                'params': jax.tree_util.tree_map(lambda z: z[idx], params),
                'batch_stats': jax.tree_util.tree_map(lambda z: z[idx], batch_stats),
                'mup': jax.tree_util.tree_map(lambda z: z[idx], mup),
            }
        )
    return members



def _build_probe_loader(probe_batch_size: int, probe_seed: int, probe_loader_batch_size: int):
    if IMAGENET_FOLDER is None:
        raise ValueError('IMAGENET_FOLDER must be set in environment/constants for activation analysis.')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    channels_last_transform = transforms.Lambda(lambda x: x.permute(1, 2, 0))
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            channels_last_transform,
        ]
    )

    val_split_dir = os.path.join(IMAGENET_FOLDER, 'val')
    if os.path.isdir(val_split_dir):
        dataset = ImageFolder(val_split_dir, transform=val_transform)
    else:
        dataset = ImageNet(IMAGENET_FOLDER, 'val', transform=val_transform)
    if probe_batch_size <= 0:
        raise ValueError('probe_batch_size must be positive.')
    if probe_loader_batch_size <= 0:
        raise ValueError('probe_loader_batch_size must be positive.')

    probe_loader_batch_size = min(probe_loader_batch_size, probe_batch_size)

    rng = np.random.default_rng(probe_seed)
    indices = rng.choice(len(dataset), size=probe_batch_size, replace=False)
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=probe_loader_batch_size, shuffle=False, num_workers=0, drop_last=False)



def _extract_conv_init_output(model, variables, batch_x):
    _, mutable = model.apply(
        variables,
        batch_x,
        train=False,
        capture_intermediates=lambda mdl, _: getattr(mdl, 'name', '') == 'conv_init',
        mutable=['intermediates'],
    )
    intermediates = mutable['intermediates']
    conv_init = intermediates['conv_init']
    out = conv_init['__call__'][0]
    return out



def _safe_cos_from_grams(gram: np.ndarray, norm_left: np.ndarray, norm_right: np.ndarray) -> np.ndarray:
    denom = np.outer(norm_left, norm_right)
    denom = np.where(denom == 0.0, 1.0, denom)
    return np.abs(gram / denom)



def _activation_similarity_matrix(
    member_variables: list[dict],
    width: int,
    probe_loader,
    progress_label: str = '',
) -> np.ndarray:
    num_members = len(member_variables)
    if num_members == 0:
        raise ValueError('No member variables found for activation analysis.')

    leaf = jax.tree_util.tree_leaves(member_variables[0]['params'])[0]
    param_dtype = jnp.asarray(leaf).dtype
    model = ResNet18(num_classes=1000, num_filters=width, param_dtype=param_dtype)

    self_grams = [jnp.zeros((width, width), dtype=jnp.float32) for _ in range(num_members)]
    cross_grams = {}
    for e in range(num_members):
        for f in range(e + 1, num_members):
            cross_grams[(e, f)] = jnp.zeros((width, width), dtype=jnp.float32)

    batch_iter = _progress(
        probe_loader,
        desc=f'{progress_label} probe-batches',
        leave=False,
    )
    for batch in batch_iter:
        batch_x, _ = batch
        x = jnp.asarray(batch_x)

        member_features = []
        member_iter = _progress(
            member_variables,
            desc=f'{progress_label} members',
            total=num_members,
            leave=False,
        )
        for vars_ in member_iter:
            conv_out = _extract_conv_init_output(model, vars_, x)
            feat = jnp.reshape(jnp.asarray(conv_out, dtype=jnp.float32), (-1, width))
            member_features.append(feat)

        for e in range(num_members):
            fe = member_features[e]
            self_grams[e] += fe.T @ fe

        for e in range(num_members):
            fe = member_features[e]
            for f in range(e + 1, num_members):
                ff = member_features[f]
                cross_grams[(e, f)] += fe.T @ ff

    self_grams_np = [np.asarray(g, dtype=np.float64) for g in self_grams]
    cross_grams_np = {k: np.asarray(v, dtype=np.float64) for k, v in cross_grams.items()}
    norms = [np.sqrt(np.maximum(np.diag(g), 1e-12)) for g in self_grams_np]

    total = num_members * width
    sim = np.zeros((total, total), dtype=np.float64)

    for e in range(num_members):
        e_start = e * width
        e_end = e_start + width
        block = _safe_cos_from_grams(self_grams_np[e], norms[e], norms[e])
        sim[e_start:e_end, e_start:e_end] = block

        for f in range(e + 1, num_members):
            f_start = f * width
            f_end = f_start + width
            cross = _safe_cos_from_grams(cross_grams_np[(e, f)], norms[e], norms[f])
            sim[e_start:e_end, f_start:f_end] = cross
            sim[f_start:f_end, e_start:e_end] = cross.T

    return sim



def _weight_similarity_matrix(weights: np.ndarray) -> np.ndarray:
    flat = weights.reshape((weights.shape[0] * weights.shape[1], -1))
    return abs_cosine_similarity_matrix(flat, flat)



def _collect_member_states(group_dirs: list[str], step: int, progress_label: str = '') -> list[dict]:
    members = []
    group_iter = _progress(
        group_dirs,
        desc=f'{progress_label} restore-groups',
        leave=False,
    )
    for group_dir in group_iter:
        state_dir = join(group_dir, 'state_ckpts')
        state_obj = checkpoints.restore_checkpoint(
            ckpt_dir=state_dir,
            target=None,
            step=step,
            prefix='state_',
        )
        members.extend(_member_variables_from_state(state_obj))
    return members



def _analysis_rows_for_similarity(
    similarity_matrix: np.ndarray,
    num_members: int,
    width: int,
    shuffle_repeats: int,
    shuffle_batch_size: int,
    shuffle_stats_workers: int,
    rng: np.random.Generator,
    metric_payload: dict,
    representation: str,
    log_every_shuffles: int,
    write_every_shuffles: int,
    row_callback: Callable[[list[dict]], None] | None = None,
):
    if shuffle_batch_size <= 0:
        raise ValueError('shuffle_batch_size must be positive.')
    if shuffle_stats_workers <= 0:
        raise ValueError('shuffle_stats_workers must be positive.')

    rows = []
    pending_rows: list[dict] = []
    member_ids = build_member_ids(num_members, width)
    across_real = extract_across_values(similarity_matrix, member_ids)
    within_real = extract_within_values(similarity_matrix, member_ids)

    # Analysis B observed: within_real vs across_real
    diag_stats = ks_w1_stats(within_real, across_real)
    diag_row = {
        **metric_payload,
        'representation': representation,
        'analysis_type': 'within_vs_across_real',
        'shuffle_id': -1,
        'ks_distance': diag_stats['ks_distance'],
        'ks_p_raw': diag_stats['ks_pvalue'],
        'ks_sigma_two_sided': two_sided_sigma_from_p(diag_stats['ks_pvalue']),
        'w1_distance': diag_stats['w1_distance'],
    }
    rows.append(diag_row)
    pending_rows.append(diag_row)
    if row_callback is not None and write_every_shuffles <= 0:
        row_callback(pending_rows)
        pending_rows = []

    shuffle_iter = _progress(
        range(shuffle_repeats),
        desc=f"w{metric_payload['width']} p{metric_payload['images_seen']} {representation} shuffles",
        total=shuffle_repeats,
        leave=False,
    )
    batch_across = None
    batch_within = None
    batch_stats = None
    batch_start = 0

    def _stats_for_offset(offset: int):
        assert batch_across is not None
        assert batch_within is not None
        across_shuf_local = batch_across[offset]
        within_shuf_local = batch_within[offset]
        return (
            ks_w1_stats(across_real, across_shuf_local),
            ks_w1_stats(within_shuf_local, across_real),
        )

    stats_executor = (
        ThreadPoolExecutor(max_workers=shuffle_stats_workers)
        if shuffle_stats_workers > 1
        else None
    )
    try:
        for shuffle_id in shuffle_iter:
            if batch_across is None or (shuffle_id - batch_start) >= batch_across.shape[0]:
                batch_start = shuffle_id
                current_batch = min(shuffle_batch_size, shuffle_repeats - shuffle_id)
                if current_batch == 1:
                    across_shuf_single, within_shuf_single = shuffled_similarity_values(
                        similarity_matrix,
                        num_members,
                        width,
                        rng,
                    )
                    batch_across = across_shuf_single.reshape((1, -1))
                    batch_within = within_shuf_single.reshape((1, -1))
                else:
                    batch_across, batch_within = shuffled_similarity_values_batched(
                        similarity_matrix=similarity_matrix,
                        num_members=num_members,
                        width=width,
                        rng=rng,
                        batch_size=current_batch,
                    )

                if stats_executor is None:
                    batch_stats = [_stats_for_offset(idx) for idx in range(current_batch)]
                else:
                    batch_stats = list(stats_executor.map(_stats_for_offset, range(current_batch)))

            offset = shuffle_id - batch_start
            assert batch_stats is not None
            baseline_stats, diag_shuffle_stats = batch_stats[offset]
            baseline_row = {
                **metric_payload,
                'representation': representation,
                'analysis_type': 'across_real_vs_across_shuffled',
                'shuffle_id': shuffle_id,
                'ks_distance': baseline_stats['ks_distance'],
                'ks_p_raw': baseline_stats['ks_pvalue'],
                'ks_sigma_two_sided': two_sided_sigma_from_p(baseline_stats['ks_pvalue']),
                'w1_distance': baseline_stats['w1_distance'],
            }
            rows.append(baseline_row)
            pending_rows.append(baseline_row)

            diag_shuffle_row = {
                **metric_payload,
                'representation': representation,
                'analysis_type': 'within_shuffled_vs_across_real',
                'shuffle_id': shuffle_id,
                'ks_distance': diag_shuffle_stats['ks_distance'],
                'ks_p_raw': diag_shuffle_stats['ks_pvalue'],
                'ks_sigma_two_sided': two_sided_sigma_from_p(diag_shuffle_stats['ks_pvalue']),
                'w1_distance': diag_shuffle_stats['w1_distance'],
            }
            rows.append(diag_shuffle_row)
            pending_rows.append(diag_shuffle_row)

            if (
                log_every_shuffles > 0
                and ((shuffle_id + 1) % log_every_shuffles == 0 or (shuffle_id + 1) == shuffle_repeats)
            ):
                print(
                    f"  width={metric_payload['width']} step={metric_payload['images_seen']} "
                    f"rep={representation}: shuffles {shuffle_id + 1}/{shuffle_repeats}"
                )

            if (
                row_callback is not None
                and write_every_shuffles > 0
                and ((shuffle_id + 1) % write_every_shuffles == 0 or (shuffle_id + 1) == shuffle_repeats)
            ):
                row_callback(pending_rows)
                pending_rows = []
    finally:
        if stats_executor is not None:
            stats_executor.shutdown(wait=True)

    if row_callback is not None and pending_rows:
        row_callback(pending_rows)

    return rows



def _aggregate_metrics(group_dirs: list[str]) -> dict[int, dict]:
    by_step = defaultdict(list)
    group_iter = _progress(
        group_dirs,
        desc='aggregate-metrics',
        total=len(group_dirs),
        leave=False,
    )
    for group_dir in group_iter:
        gm = _load_group_metrics(group_dir)
        for step, row in gm.items():
            by_step[step].append(row)

    aggregated = {}
    for step, rows in by_step.items():
        aggregated[step] = {
            'train_loss': float(np.mean([r['train_loss'] for r in rows])),
            'train_error': float(np.mean([r['train_error'] for r in rows])),
            'val_loss': float(np.mean([r['val_loss'] for r in rows])),
            'val_error': float(np.mean([r['val_error'] for r in rows])),
        }
    return aggregated



def write_rows(rows: list[dict], output_csv: str) -> None:
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        'width',
        'images_seen',
        'representation',
        'analysis_type',
        'shuffle_id',
        'ks_distance',
        'ks_p_raw',
        'ks_sigma_two_sided',
        'w1_distance',
        'train_loss',
        'val_loss',
        'train_error',
        'val_error',
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def _row_identity(row: dict) -> tuple[int, int, str, str, int]:
    return (
        int(row['width']),
        int(row['images_seen']),
        str(row['representation']),
        str(row['analysis_type']),
        int(row['shuffle_id']),
    )


def _rep_identity(row: dict) -> tuple[int, int, str]:
    return (
        int(row['width']),
        int(row['images_seen']),
        str(row['representation']),
    )


def _prepare_resume_state(
    output_csv: str,
    fieldnames: list[str],
    shuffle_repeats: int,
    resume: bool,
) -> tuple[set[tuple[int, int, str]], int, str]:
    if (not resume) or (not os.path.exists(output_csv)):
        return set(), 0, 'w'

    with open(output_csv, 'r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return set(), 0, 'w'

    dedup_rows = []
    seen_keys: set[tuple[int, int, str, str, int]] = set()
    for row in rows:
        key = _row_identity(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        dedup_rows.append(row)

    if len(dedup_rows) != len(rows):
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dedup_rows)
        print(f'Resume: removed {len(rows) - len(dedup_rows)} duplicate rows from existing output.')
        rows = dedup_rows

    expected_rep_rows = 1 + 2 * shuffle_repeats
    rep_counts: dict[tuple[int, int, str], int] = defaultdict(int)
    for row in rows:
        rep_counts[_rep_identity(row)] += 1

    incomplete_reps = {k for k, c in rep_counts.items() if c < expected_rep_rows}
    if incomplete_reps:
        kept_rows = [r for r in rows if _rep_identity(r) not in incomplete_reps]
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
        removed = len(rows) - len(kept_rows)
        print(
            f'Resume: dropped {removed} partial rows across {len(incomplete_reps)} incomplete representations.'
        )
        rows = kept_rows
        rep_counts = defaultdict(int)
        for row in rows:
            rep_counts[_rep_identity(row)] += 1

    completed_reps = {k for k, c in rep_counts.items() if c >= expected_rep_rows}
    if rows:
        print(
            f'Resume: keeping {len(rows)} existing rows, '
            f'skipping {len(completed_reps)} completed representations.'
        )
    return completed_reps, len(rows), ('a' if rows else 'w')


def main() -> None:
    args = parse_args()

    resolved_run_id = _resolve_run_id(args.base_save_dir, args.run_id, args.run_id_resolution)
    width_dirs = _list_width_dirs(args.base_save_dir, resolved_run_id)
    if not width_dirs:
        raise RuntimeError(
            f'No width directories found for run_id="{resolved_run_id}" under {args.base_save_dir}.'
        )
    if args.widths:
        width_dirs = {w: d for w, d in width_dirs.items() if w in set(args.widths)}
    if not width_dirs:
        raise RuntimeError(f'No matching widths found for requested filter: {args.widths}')

    probe_loader = _build_probe_loader(args.probe_batch_size, args.probe_seed, args.probe_loader_batch_size)
    rng = np.random.default_rng(args.probe_seed)

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        'width',
        'images_seen',
        'representation',
        'analysis_type',
        'shuffle_id',
        'ks_distance',
        'ks_p_raw',
        'ks_sigma_two_sided',
        'w1_distance',
        'train_loss',
        'val_loss',
        'train_error',
        'val_error',
    ]

    completed_reps, rows_written, file_mode = _prepare_resume_state(
        output_csv=args.output_csv,
        fieldnames=fieldnames,
        shuffle_repeats=args.shuffle_repeats,
        resume=args.resume,
    )

    with open(args.output_csv, file_mode, newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if file_mode == 'w':
            writer.writeheader()
            f_out.flush()

        width_iter = _progress(
            sorted(width_dirs.items()),
            desc='widths',
            total=len(width_dirs),
            leave=True,
        )
        for width, width_dir in width_iter:
            group_dirs = _list_group_dirs(width_dir)
            if not group_dirs:
                continue

            common_steps = _collect_target_steps(group_dirs)
            metrics_by_step = _aggregate_metrics(group_dirs)
            step_iter = _progress(
                common_steps,
                desc=f'width={width} steps',
                total=len(common_steps),
                leave=False,
            )
            for step in step_iter:
                metric_payload = {
                    'width': width,
                    'images_seen': step,
                    'train_loss': metrics_by_step.get(step, {}).get('train_loss', np.nan),
                    'train_error': metrics_by_step.get(step, {}).get('train_error', np.nan),
                    'val_loss': metrics_by_step.get(step, {}).get('val_loss', np.nan),
                    'val_error': metrics_by_step.get(step, {}).get('val_error', np.nan),
                }

                rows_step = 0

                # weights
                weights_rep_key = (width, step, 'weights')
                if weights_rep_key in completed_reps:
                    print(f'Skipping width={width} step={step} rep=weights (already complete in CSV)')
                else:
                    missing_artifacts = _missing_weight_artifacts(group_dirs, step)
                    if missing_artifacts:
                        preview = ', '.join(missing_artifacts[:2])
                        if len(missing_artifacts) > 2:
                            preview += f', ... (+{len(missing_artifacts) - 2} more)'
                        print(
                            f'Skipping width={width} step={step} rep=weights '
                            f'(missing artifact files: {preview})'
                        )
                    else:
                        t0 = time.time()
                        print(f'Starting width={width} step={step} rep=weights')
                        weight_chunks = [_extract_weights_from_artifacts(g, step) for g in group_dirs]
                        weights = np.concatenate(weight_chunks, axis=0)
                        num_members = weights.shape[0]
                        weight_sim = _weight_similarity_matrix(weights)
                        weight_rows = _analysis_rows_for_similarity(
                            similarity_matrix=weight_sim,
                            num_members=num_members,
                            width=width,
                            shuffle_repeats=args.shuffle_repeats,
                            shuffle_batch_size=args.shuffle_batch_size,
                            shuffle_stats_workers=args.shuffle_stats_workers,
                            rng=rng,
                            metric_payload=metric_payload,
                            representation='weights',
                            log_every_shuffles=args.log_every_shuffles,
                            write_every_shuffles=args.write_every_shuffles,
                            row_callback=lambda rows: (writer.writerows(rows), f_out.flush()),
                        )
                        rows_written += len(weight_rows)
                        rows_step += len(weight_rows)
                        completed_reps.add(weights_rep_key)
                        print(
                            f'Completed width={width} step={step} rep=weights '
                            f'rows_rep={len(weight_rows)} rows_written={rows_written} '
                            f'elapsed_s={(time.time() - t0):.2f}'
                        )

                # activations
                activations_rep_key = (width, step, 'activations')
                if activations_rep_key in completed_reps:
                    print(f'Skipping width={width} step={step} rep=activations (already complete in CSV)')
                else:
                    t1 = time.time()
                    print(f'Starting width={width} step={step} rep=activations')
                    member_states = _collect_member_states(group_dirs, step, progress_label=f'w{width} p{step}')
                    t2 = time.time()
                    act_sim = _activation_similarity_matrix(
                        member_states,
                        width,
                        probe_loader,
                        progress_label=f'w{width} p{step}',
                    )
                    activation_rows = _analysis_rows_for_similarity(
                        similarity_matrix=act_sim,
                        num_members=len(member_states),
                        width=width,
                        shuffle_repeats=args.shuffle_repeats,
                        shuffle_batch_size=args.shuffle_batch_size,
                        shuffle_stats_workers=args.shuffle_stats_workers,
                        rng=rng,
                        metric_payload=metric_payload,
                        representation='activations',
                        log_every_shuffles=args.log_every_shuffles,
                        write_every_shuffles=args.write_every_shuffles,
                        row_callback=lambda rows: (writer.writerows(rows), f_out.flush()),
                    )
                    rows_written += len(activation_rows)
                    rows_step += len(activation_rows)
                    completed_reps.add(activations_rep_key)
                    print(
                        f'Completed width={width} step={step} rep=activations '
                        f'rows_rep={len(activation_rows)} rows_written={rows_written} '
                        f'restore_s={(t2 - t1):.2f} activation_sim_s={(time.time() - t2):.2f}'
                    )

                print(
                    f'Analyzed width={width} step={step} '
                    f'rows_step={rows_step} rows_written={rows_written}'
                )

    print(f'Wrote {rows_written} rows to {args.output_csv}')


if __name__ == '__main__':
    main()
