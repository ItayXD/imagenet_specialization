#!/usr/bin/env python3
"""Linearized (tangent / NTK) random-feature readout of a ResNet18 at initialization.

The GAP/last-layer analysis (analyze_init_regression_powerlaw.py) trains ONLY the readout
on the last hidden layer's features -> the conjugate/NNGP kernel, a weak predictor on hard
tasks. This script instead fits the *linearized network* (the lazy / NTK regime, e.g. the
"lin-CNN-GAP" of Lee et al. 2020): the model is

    f_lin(x) = f0(x) + <grad_theta f0(x), Delta_theta>,

trained by least squares over Delta_theta. That is linear regression in the TANGENT feature
map grad_theta f0(x) (all layers), NOT a kernel. Its raw dimension is the parameter count,
so we use the standard explicit random-feature realization: m random parameter-space
directions s_k and forward-mode JVPs

    phi_k(x) = grad_theta f0(x) . s_k     (a Jacobian-vector product),

whose Gram is an unbiased sketch of the NTK (validated: rel-err ~ 1/sqrt(m)). We restrict
Delta_theta = sum_k a_k s_k (a in R^m, shared across the C outputs) and solve the ridge

    min_a  sum_{x,c} ( f0(x)_c + phi(x)_c . a - y_{x,c} )^2 + lambda ||a||^2 ,

then evaluate f_lin on a held-out val split. Compares the NTK/tangent readout accuracy to
the last-layer (conjugate) readout. Runs on the cluster (GPU); JVP cost ~ m forward passes
per batch, so use small images (CIFAR-5M) and moderate m first.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from scripts.analyze_init_regression_powerlaw import (
    _build_split_loader,
    _calibrate_batchnorm,
    _init_variables,
    _to_dtype_tree,
    activate_residual_scales,
)


def _tangent_keys(feat_seed: int, m: int):
    """m fixed per-direction PRNG keys (identical across data batches and train/val)."""
    import jax

    return jax.random.split(jax.random.PRNGKey(int(feat_seed)), m)


def _make_forward(model, base_variables):
    """f(params, x) -> logits (C-dim), with batch_stats/mup frozen at their calibrated values."""
    import jax.numpy as jnp  # noqa: F401

    bs = base_variables['batch_stats']
    mup = base_variables['mup']

    def f(params, x):
        return model.apply({'params': params, 'batch_stats': bs, 'mup': mup}, x, train=False)

    return f


def _batch_tangent_features(f, params, x, keys_chunk):
    """Phi_chunk[k] = JVP of f along random direction s_k, for a chunk of directions.

    Returns array (chunk, B, C): the output perturbation d f(x) along each s_k.
    """
    import jax
    import jax.numpy as jnp

    leaves, treedef = jax.tree_util.tree_flatten(params)

    def one_direction(key):
        subkeys = jax.random.split(key, len(leaves))
        tangent = jax.tree_util.tree_unflatten(
            treedef, [jax.random.normal(k, leaf.shape, leaf.dtype)
                      for k, leaf in zip(subkeys, leaves)])
        _, out = jax.jvp(lambda p: f(p, x), (params,), (tangent,))  # (B, C)
        return out

    return jax.vmap(one_direction)(keys_chunk)  # (chunk, B, C)


def accumulate_gram(f, params, loader, keys, chunk: int, num_classes: int):
    """One streaming pass: return (G (m,m), rhs (m,), f0/labels for accuracy), NTK-sketch fit.

    G = sum_{x,c} phi(x)_c phi(x)_c^T ; rhs = sum_{x,c} phi(x)_c (y_{x,c} - f0(x)_c).
    Also returns stacked f0 (N,C) and labels (N,) so the caller can score train accuracy.
    """
    import jax
    import jax.numpy as jnp

    m = keys.shape[0]
    forward = jax.jit(lambda p, x: f(p, x))
    feat_fn = jax.jit(lambda p, x, kc: _batch_tangent_features(f, p, x, kc))

    gram = np.zeros((m, m), dtype=np.float64)
    rhs = np.zeros(m, dtype=np.float64)
    f0_all, labels_all = [], []
    for bx, by in loader:
        x = jnp.asarray(np.asarray(bx))
        y = np.asarray(by).reshape(-1).astype(np.int64)
        f0 = np.asarray(forward(params, x), dtype=np.float64)  # (B, C)
        onehot = np.eye(num_classes, dtype=np.float64)[y]  # (B, C)
        resid = (onehot - f0)  # (B, C)
        b = f0.shape[0]
        phi = np.empty((m, b, f0.shape[1]), dtype=np.float64)  # (m, B, C)
        for c0 in range(0, m, chunk):
            kc = keys[c0:c0 + chunk]
            phi[c0:c0 + chunk] = np.asarray(feat_fn(params, x, kc), dtype=np.float64)
        phi_flat = phi.transpose(1, 2, 0).reshape(b * f0.shape[1], m)  # (B*C, m)
        gram += phi_flat.T @ phi_flat
        rhs += phi_flat.T @ resid.reshape(-1)
        f0_all.append(f0)
        labels_all.append(y)
    return gram, rhs, np.concatenate(f0_all), np.concatenate(labels_all)


def predict_pass(f, params, loader, keys, chunk: int, num_classes: int, a_by_lambda: dict):
    """Second pass: for each lambda's coefficient vector a, predict f_lin on the loader.

    Returns {rel_lambda: (accuracy, cross_entropy)} plus the count of scored samples.
    """
    import jax
    import jax.numpy as jnp

    m = keys.shape[0]
    forward = jax.jit(lambda p, x: f(p, x))
    feat_fn = jax.jit(lambda p, x, kc: _batch_tangent_features(f, p, x, kc))

    correct = {k: 0 for k in a_by_lambda}
    ce_sum = {k: 0.0 for k in a_by_lambda}
    n = 0
    for bx, by in loader:
        x = jnp.asarray(np.asarray(bx))
        y = np.asarray(by).reshape(-1).astype(np.int64)
        f0 = np.asarray(forward(params, x), dtype=np.float64)  # (B, C)
        b = f0.shape[0]
        phi = np.empty((m, b, f0.shape[1]), dtype=np.float64)
        for c0 in range(0, m, chunk):
            phi[c0:c0 + chunk] = np.asarray(feat_fn(params, x, keys[c0:c0 + chunk]), dtype=np.float64)
        phi_bcm = phi.transpose(1, 2, 0)  # (B, C, m)
        for k, a in a_by_lambda.items():
            logits = f0 + phi_bcm @ a  # (B, C)
            correct[k] += int(np.sum(logits.argmax(1) == y))
            shifted = logits - logits.max(1, keepdims=True)
            logp = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
            ce_sum[k] += float(-np.sum(logp[np.arange(b), y]))
        n += b
    return {k: (correct[k] / n, ce_sum[k] / n) for k in a_by_lambda}, n


def solve_ridge_paths(gram: np.ndarray, rhs: np.ndarray, rel_lambdas, min_norm_tol=1e-10):
    """Solve a = (G + lambda I)^-1 rhs for each rel lambda (relative to mean eigenvalue of G)."""
    mu, q = np.linalg.eigh(gram)
    mu = np.clip(mu, 0.0, None)
    scale = float(np.mean(mu)) if mu.size else 0.0
    max_eig = float(mu.max()) if mu.size else 0.0
    qtr = q.T @ rhs
    out = {}
    for rel in rel_lambdas:
        lam = float(rel) * scale
        if lam > 0:
            inv = 1.0 / (mu + lam)
        else:
            tol = max_eig * min_norm_tol
            inv = np.where(mu > tol, 1.0 / np.where(mu > tol, mu, 1.0), 0.0)
        out[float(rel)] = q @ (inv * qtr)
    return out, mu, scale


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['imagenet', 'cifar5m'], default='cifar5m')
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--num-features', type=int, default=4000, help='m: random tangent directions.')
    p.add_argument('--num-train-images', type=int, default=20000)
    p.add_argument('--num-val-images', type=int, default=10000)
    p.add_argument('--num-calib-batches', type=int, default=50)
    p.add_argument('--residual-scale-init', choices=['ones', 'random', 'zeros'], default='ones')
    p.add_argument('--residual-scale-std', type=float, default=0.5)
    p.add_argument('--ridge-rel-lambdas', type=float, nargs='*',
                   default=[0.0, 1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument('--init-seed', type=int, default=0)
    p.add_argument('--feat-seed', type=int, default=7, help='Seed for the random tangent directions.')
    p.add_argument('--train-seed', type=int, default=1234)
    p.add_argument('--seed', type=int, default=2423)
    p.add_argument('--eval-batch-size', type=int, default=200)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--jvp-chunk', type=int, default=16, help='Directions per vmapped JVP call.')
    p.add_argument('--compute-dtype', choices=['float32'], default='float32')
    p.add_argument('--output-dir', default='')
    p.add_argument('--base-save-dir', default='')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import jax.numpy as jnp

    from scripts.analyze_classifier_powerlaw import _default_base_save_dir
    from src.experiment.dataset_specs import get_dataset_spec
    from src.experiment.model.flax_mup.resnet import ResNet18

    dataset = args.dataset
    spec = get_dataset_spec(dataset)
    num_classes = spec.num_classes
    width = int(args.width)
    m = int(args.num_features)
    compute_dtype = jnp.float32
    base_save_dir = args.base_save_dir or _default_base_save_dir(dataset)
    print(f'dataset={dataset} width={width} m={m} num_classes={num_classes} '
          f'train={args.num_train_images} val={args.num_val_images}')

    model = ResNet18(num_classes=num_classes, num_filters=width,
                     param_dtype=compute_dtype, stem_type=spec.stem_type)
    variables = _init_variables(model, spec.input_shape, compute_dtype, args.init_seed + width)
    rng = np.random.default_rng(args.init_seed + width)
    params_np, n_act = activate_residual_scales(
        variables['params'], args.residual_scale_init, rng=rng, std=args.residual_scale_std)
    variables['params'] = _to_dtype_tree(params_np, compute_dtype)
    print(f'residual_scale_init={args.residual_scale_init}: activated {n_act} scales')

    train_loader = _build_split_loader(dataset, 'train', int(args.num_train_images),
                                       int(args.train_seed), int(args.eval_batch_size),
                                       int(args.num_workers))
    val_loader = _build_split_loader(dataset, 'val', int(args.num_val_images), int(args.seed),
                                     int(args.eval_batch_size), int(args.num_workers))

    if args.residual_scale_init != 'zeros' and args.num_calib_batches > 0:
        variables = _calibrate_batchnorm(model, variables, train_loader, args.num_calib_batches)

    params = variables['params']
    f = _make_forward(model, variables)
    keys = _tangent_keys(args.feat_seed, m)

    print(f'accumulating NTK-sketch Gram over train (m={m}, chunk={args.jvp_chunk})...')
    gram, rhs, f0_tr, y_tr = accumulate_gram(
        f, params, train_loader, keys, args.jvp_chunk, num_classes)
    a_by_lambda, mu, scale = solve_ridge_paths(gram, rhs, args.ridge_rel_lambdas)
    ntrain = y_tr.size
    print(f'gram done (ntrain={ntrain}); NTK eig: max={mu.max():.3e} '
          f'mean={scale:.3e} eff_rank={(mu.sum()**2/(mu**2).sum()):.1f}')

    # Train accuracy (in-sample) reusing the stored f0/labels + a fresh feature pass is costly;
    # instead score train via a second pass (same cost as val) for the gap.
    train_scores, _ = predict_pass(f, params, train_loader, keys, args.jvp_chunk,
                                   num_classes, a_by_lambda)
    val_scores, nval = predict_pass(f, params, val_loader, keys, args.jvp_chunk,
                                    num_classes, a_by_lambda)

    print(f'{"rel_lambda":>10} {"train_acc":>10} {"val_acc":>10} {"val_ce":>8}')
    rows = []
    for rel in args.ridge_rel_lambdas:
        tr = train_scores[float(rel)]
        va = val_scores[float(rel)]
        print(f'{rel:>10g} {tr[0]:>10.4f} {va[0]:>10.4f} {va[1]:>8.3f}')
        rows.append({'rel_lambda': float(rel), 'train_acc': tr[0],
                     'val_acc': va[0], 'val_ce': va[1]})

    out_dir = args.output_dir or os.path.join(
        base_save_dir, 'init_ntk_regression', dataset, f'ntk_w{width}_m{m}')
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        'dataset': dataset, 'width': width, 'num_features_m': m,
        'num_classes': num_classes, 'num_train': int(ntrain), 'num_val': int(nval),
        'residual_scale_init': args.residual_scale_init,
        'num_calib_batches': int(args.num_calib_batches),
        'ntk_eig_max': float(mu.max()), 'ntk_eig_mean': float(scale),
        'ntk_eff_rank': float(mu.sum() ** 2 / (mu ** 2).sum()),
        'paths': rows,
    }
    with open(os.path.join(out_dir, 'ntk_summary.json'), 'w') as h:
        json.dump(summary, h, indent=2)
    np.savez_compressed(os.path.join(out_dir, 'ntk_arrays.npz'),
                        ntk_eigenvalues=mu.astype(np.float64))
    print(f'wrote {out_dir}/ntk_summary.json')


if __name__ == '__main__':
    main()
