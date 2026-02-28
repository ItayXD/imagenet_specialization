#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from glob import glob
from os.path import join

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms

from src.experiment.exchangeability_utils import (
    abs_cosine_similarity_matrix,
    build_member_ids,
    extract_across_values,
    extract_within_values,
    ks_w1_stats,
    shuffled_similarity_values,
    two_sided_sigma_from_p,
)
from src.experiment.model.flax_mup.resnet import ResNet18
from src.run.constants import IMAGENET_FOLDER


STATE_PATTERN = re.compile(r'^state_(\d+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze exchangeability from grouped ensemble checkpoints.')
    parser.add_argument('--base-save-dir', required=True, help='Root save directory (BASE_SAVE_DIR)')
    parser.add_argument('--run-id', default='exchangeability', help='Run id folder name under base-save-dir')
    parser.add_argument('--output-csv', default='outputs/exchangeability_metrics.csv', help='Output CSV path')
    parser.add_argument('--shuffle-repeats', type=int, default=2000, help='Number of shuffle repeats')
    parser.add_argument('--probe-batch-size', type=int, default=1024, help='Total probe images for activation vectors')
    parser.add_argument('--probe-loader-batch-size', type=int, default=1, help='Probe dataloader batch size used for streaming accumulation')
    parser.add_argument('--probe-seed', type=int, default=1234, help='Seed for probe subset selection')
    parser.add_argument('--widths', type=int, nargs='*', default=None, help='Optional list of widths to analyze')
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
    for d in group_dirs:
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
    return np.asarray(data['first_layer_weights'])



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

    dataset = ImageNet(IMAGENET_FOLDER, 'val', transform=val_transform)
    rng = np.random.default_rng(probe_seed)
    indices = rng.choice(len(dataset), size=probe_batch_size, replace=False)
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=probe_loader_batch_size, shuffle=False, num_workers=1, drop_last=False)



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



def _activation_similarity_matrix(member_variables: list[dict], width: int, probe_loader) -> np.ndarray:
    num_members = len(member_variables)
    if num_members == 0:
        raise ValueError('No member variables found for activation analysis.')

    leaf = jax.tree_util.tree_leaves(member_variables[0]['params'])[0]
    param_dtype = jnp.asarray(leaf).dtype
    model = ResNet18(num_classes=1000, num_filters=width, param_dtype=param_dtype)

    self_grams = [np.zeros((width, width), dtype=np.float64) for _ in range(num_members)]
    cross_grams = {}
    for e in range(num_members):
        for f in range(e + 1, num_members):
            cross_grams[(e, f)] = np.zeros((width, width), dtype=np.float64)

    for batch in probe_loader:
        batch_x, _ = batch
        x = jnp.array(batch_x)

        member_features = []
        for vars_ in member_variables:
            conv_out = _extract_conv_init_output(model, vars_, x)
            feat = np.asarray(conv_out, dtype=np.float32).reshape((-1, width))
            member_features.append(feat)

        for e in range(num_members):
            fe = member_features[e]
            self_grams[e] += fe.T @ fe

        for e in range(num_members):
            fe = member_features[e]
            for f in range(e + 1, num_members):
                ff = member_features[f]
                cross_grams[(e, f)] += fe.T @ ff

    norms = [np.sqrt(np.maximum(np.diag(g), 1e-12)) for g in self_grams]

    total = num_members * width
    sim = np.zeros((total, total), dtype=np.float64)

    for e in range(num_members):
        e_start = e * width
        e_end = e_start + width
        block = _safe_cos_from_grams(self_grams[e], norms[e], norms[e])
        sim[e_start:e_end, e_start:e_end] = block

        for f in range(e + 1, num_members):
            f_start = f * width
            f_end = f_start + width
            cross = _safe_cos_from_grams(cross_grams[(e, f)], norms[e], norms[f])
            sim[e_start:e_end, f_start:f_end] = cross
            sim[f_start:f_end, e_start:e_end] = cross.T

    return sim



def _weight_similarity_matrix(weights: np.ndarray) -> np.ndarray:
    flat = weights.reshape((weights.shape[0] * weights.shape[1], -1))
    return abs_cosine_similarity_matrix(flat, flat)



def _collect_member_states(group_dirs: list[str], step: int) -> list[dict]:
    members = []
    for group_dir in group_dirs:
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
    rng: np.random.Generator,
    metric_payload: dict,
    representation: str,
):
    rows = []
    member_ids = build_member_ids(num_members, width)
    across_real = extract_across_values(similarity_matrix, member_ids)
    within_real = extract_within_values(similarity_matrix, member_ids)

    # Analysis B observed: within_real vs across_real
    diag_stats = ks_w1_stats(within_real, across_real)
    rows.append(
        {
            **metric_payload,
            'representation': representation,
            'analysis_type': 'within_vs_across_real',
            'shuffle_id': -1,
            'ks_distance': diag_stats['ks_distance'],
            'ks_p_raw': diag_stats['ks_pvalue'],
            'ks_sigma_two_sided': two_sided_sigma_from_p(diag_stats['ks_pvalue']),
            'w1_distance': diag_stats['w1_distance'],
        }
    )

    for shuffle_id in range(shuffle_repeats):
        across_shuf, within_shuf = shuffled_similarity_values(similarity_matrix, num_members, width, rng)

        baseline_stats = ks_w1_stats(across_real, across_shuf)
        rows.append(
            {
                **metric_payload,
                'representation': representation,
                'analysis_type': 'across_real_vs_across_shuffled',
                'shuffle_id': shuffle_id,
                'ks_distance': baseline_stats['ks_distance'],
                'ks_p_raw': baseline_stats['ks_pvalue'],
                'ks_sigma_two_sided': two_sided_sigma_from_p(baseline_stats['ks_pvalue']),
                'w1_distance': baseline_stats['w1_distance'],
            }
        )

        diag_shuffle_stats = ks_w1_stats(within_shuf, across_real)
        rows.append(
            {
                **metric_payload,
                'representation': representation,
                'analysis_type': 'within_shuffled_vs_across_real',
                'shuffle_id': shuffle_id,
                'ks_distance': diag_shuffle_stats['ks_distance'],
                'ks_p_raw': diag_shuffle_stats['ks_pvalue'],
                'ks_sigma_two_sided': two_sided_sigma_from_p(diag_shuffle_stats['ks_pvalue']),
                'w1_distance': diag_shuffle_stats['w1_distance'],
            }
        )

    return rows



def _aggregate_metrics(group_dirs: list[str]) -> dict[int, dict]:
    by_step = defaultdict(list)
    for group_dir in group_dirs:
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
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
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



def main() -> None:
    args = parse_args()

    width_dirs = _list_width_dirs(args.base_save_dir, args.run_id)
    if args.widths:
        width_dirs = {w: d for w, d in width_dirs.items() if w in set(args.widths)}

    probe_loader = _build_probe_loader(args.probe_batch_size, args.probe_seed, args.probe_loader_batch_size)
    rng = np.random.default_rng(args.probe_seed)

    output_rows = []

    for width, width_dir in sorted(width_dirs.items()):
        group_dirs = _list_group_dirs(width_dir)
        if not group_dirs:
            continue

        common_steps = _collect_target_steps(group_dirs)
        metrics_by_step = _aggregate_metrics(group_dirs)

        for step in common_steps:
            metric_payload = {
                'width': width,
                'images_seen': step,
                'train_loss': metrics_by_step.get(step, {}).get('train_loss', np.nan),
                'train_error': metrics_by_step.get(step, {}).get('train_error', np.nan),
                'val_loss': metrics_by_step.get(step, {}).get('val_loss', np.nan),
                'val_error': metrics_by_step.get(step, {}).get('val_error', np.nan),
            }

            # weights
            weight_chunks = [_extract_weights_from_artifacts(g, step) for g in group_dirs]
            weights = np.concatenate(weight_chunks, axis=0)
            num_members = weights.shape[0]
            weight_sim = _weight_similarity_matrix(weights)
            output_rows.extend(
                _analysis_rows_for_similarity(
                    similarity_matrix=weight_sim,
                    num_members=num_members,
                    width=width,
                    shuffle_repeats=args.shuffle_repeats,
                    rng=rng,
                    metric_payload=metric_payload,
                    representation='weights',
                )
            )

            # activations
            member_states = _collect_member_states(group_dirs, step)
            act_sim = _activation_similarity_matrix(member_states, width, probe_loader)
            output_rows.extend(
                _analysis_rows_for_similarity(
                    similarity_matrix=act_sim,
                    num_members=len(member_states),
                    width=width,
                    shuffle_repeats=args.shuffle_repeats,
                    rng=rng,
                    metric_payload=metric_payload,
                    representation='activations',
                )
            )

            print(f'Analyzed width={width} step={step}')

    write_rows(output_rows, args.output_csv)
    print(f'Wrote {len(output_rows)} rows to {args.output_csv}')


if __name__ == '__main__':
    main()
