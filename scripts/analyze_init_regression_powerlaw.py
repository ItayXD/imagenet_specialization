#!/usr/bin/env python3
"""Source/capacity/SVD power-law structure of a ResNet18 *at initialization*, with the
last layer learned by ridge linear regression on the frozen at-init features.

This is the untrained counterpart to scripts/analyze_classifier_powerlaw.py. Instead of a
trained readout, we:

  1. Build ResNet18 at random init and ACTIVATE the residual branches. The repo zero-inits
     the last BatchNorm scale in every ResNetBlock (a training-stability trick), which at
     init collapses each block to relu(shortcut) and makes the deep 3x3 residual convs
     inert. `--residual-scale-init ones` (default) sets those gammas back to the conventional
     value 1 so the full ResNet18 depth is exercised at init.
  2. CALIBRATE BatchNorm running statistics with a few train-mode forward passes over data,
     so the at-init feature map is well-scaled (rather than normalizing by the mean-0/var-1
     init defaults).
  3. Extract penultimate features h(x) (mean-pooled last ResNetBlock output) on a TRAIN split
     and fit a ridge classifier W: min_W ||h W^T + b - Y||^2 + lambda ||W||^2 to one-hot
     targets. A small grid of ridge lambdas is swept (relative to the feature-covariance
     scale) so you can see how regularization reshapes the capacity/source/SVD structure.
  4. Run the SAME downstream analysis on a held-out VAL split (feature PCA -> capacity
     exponent b, per-class source exponents a_i, PCA-truncation recovery), writing the same
     powerlaw_arrays.npz schema so scripts/analyze_svd_diagonal.py (and the other
     analyze_classifier_* consumers) work unchanged.

Runs on the cluster (JAX + torch dataloaders + GPU). One output dir per (width, lambda):
  <output-root>/<dataset>/init_reg_w<width>_lam<rel>/
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from scripts.analyze_classifier_powerlaw import (
    _chw_to_hwc,
    _default_base_save_dir,
    _extract_features,
    _make_loader,
    run_powerlaw_analysis,
)


# --------------------------------------------------------------------------------------
# Pure helpers (no JAX / no data; unit-tested in test_init_regression_powerlaw)
# --------------------------------------------------------------------------------------
def activate_residual_scales(
    params: Mapping,
    mode: str = 'ones',
    *,
    rng: np.random.Generator | None = None,
    std: float = 0.5,
    atol: float = 1e-8,
) -> tuple[dict, int]:
    """Replace the zero-init residual-branch BatchNorm gammas with active values.

    In this repo every ResNetBlock's *last* BatchNorm uses scale_init=zeros, so at init its
    'scale' leaf is all-zero while every other BatchNorm 'scale' is 1. We target exactly the
    all-zero 'scale' leaves and set them to 1 (mode='ones', the conventional BN init) or to
    positive random draws N(1, std) (mode='random'). mode='zeros' is a no-op (faithful t=0).

    Returns (new_params_dict, num_activated). Operates on / returns plain nested dicts of
    numpy arrays, so it is agnostic to FrozenDict vs dict inputs.
    """
    if mode not in ('ones', 'random', 'zeros'):
        raise ValueError(f"mode must be 'ones', 'random', or 'zeros'; got {mode!r}.")
    count = 0

    def _recurse(node):
        nonlocal count
        out: dict[str, Any] = {}
        for key, value in node.items():
            if isinstance(value, Mapping):
                out[key] = _recurse(value)
                continue
            arr = np.asarray(value)
            is_zero_scale = (
                key == 'scale' and arr.size > 0 and bool(np.all(np.abs(arr) <= atol))
            )
            if mode != 'zeros' and is_zero_scale:
                count += 1
                if mode == 'ones':
                    out[key] = np.ones_like(arr)
                else:
                    draw = rng.normal(1.0, std, size=arr.shape) if rng is not None \
                        else np.ones(arr.shape)
                    out[key] = np.clip(draw, 1e-3, None).astype(arr.dtype)
            else:
                out[key] = arr
        return out

    return _recurse(params), count


def fit_ridge_paths(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    rel_lambdas: list[float],
    *,
    scale_mode: str = 'mean_eig',
    min_norm_tol: float = 1e-10,
) -> list[dict[str, Any]]:
    """Ridge regression of one-hot class targets on features, over a grid of lambdas.

    Fits, for each lambda, W (C, D) and bias b (C,) minimizing
        sum_n || h_n W^T + b - onehot(y_n) ||^2 + lambda ||W||^2
    with an unregularized intercept (features and targets are centered; b restores the mean).
    lambda = rel * scale, where scale is the mean (or max) eigenvalue of the centered feature
    Gram H_c^T H_c, so rel is a dimensionless regularization strength. rel <= 0 gives the
    min-norm ordinary-least-squares solution (zero eigenvalues pseudo-inverted to 0).

    The Gram is eigendecomposed once and reused across lambdas. One-hot targets are handled
    without materializing the (N, C) matrix (H_c^T Y is a class-wise sum of centered rows).
    Returns a list (aligned with rel_lambdas) of dicts with keys rel_lambda, abs_lambda,
    scale, dof (effective degrees of freedom sum mu/(mu+lambda)), W, b.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    num_samples, dim = features.shape
    if labels.shape[0] != num_samples:
        raise ValueError('features and labels disagree on the sample count.')

    feature_mean = features.mean(axis=0)  # (D,)
    centered = features - feature_mean  # (N, D)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)  # (C,)
    target_mean = counts / float(num_samples)  # (C,) class frequencies

    # R = centered^T @ Y  (D, C): per-class sum of centered feature rows (Y one-hot). The
    # target-mean term drops out because centered columns sum to zero.
    class_sum = np.zeros((num_classes, dim), dtype=np.float64)
    np.add.at(class_sum, labels, centered)
    rhs = class_sum.T  # (D, C)

    gram = centered.T @ centered  # (D, D)
    eigvals, eigvecs = np.linalg.eigh(gram)  # ascending, symmetric
    eigvals = np.clip(eigvals, 0.0, None)
    max_eig = float(eigvals.max()) if eigvals.size else 0.0
    mean_eig = float(np.mean(eigvals)) if eigvals.size else 0.0
    scale = mean_eig if scale_mode == 'mean_eig' else max_eig
    proj_rhs = eigvecs.T @ rhs  # (D, C)
    zero_tol = max_eig * min_norm_tol

    results: list[dict[str, Any]] = []
    for rel in rel_lambdas:
        lam = float(rel) * scale
        if lam > 0.0:
            inv = 1.0 / (eigvals + lam)
            dof = float(np.sum(eigvals / (eigvals + lam)))
        else:
            inv = np.where(eigvals > zero_tol, 1.0 / np.where(eigvals > zero_tol, eigvals, 1.0), 0.0)
            dof = float(np.sum(eigvals > zero_tol))
        weight_centered = eigvecs @ (inv[:, None] * proj_rhs)  # (D, C) = (G + lam I)^-1 R
        weight = weight_centered.T  # (C, D)
        bias = target_mean - feature_mean @ weight_centered  # (C,)
        results.append({
            'rel_lambda': float(rel), 'abs_lambda': lam, 'scale': scale, 'dof': dof,
            'W': weight, 'b': bias,
        })
    return results


def _lambda_tag(rel: float) -> str:
    """Filesystem-safe tag for a relative ridge lambda (e.g. 0 -> 'lam0', 0.001 -> 'lam0.001')."""
    return f'lam{rel:g}'


# Short filesystem tags for the non-default feature-preprocessing modes (used in run dirs).
_PREPROC_TAG = {'per_sample_center': 'psc', 'per_sample_standardize': 'pss', 'pre_relu': 'prerelu'}


def apply_feature_preproc(features: np.ndarray, mode: str) -> np.ndarray:
    """Per-sample preprocessing to tame the ReLU+global-average-pool DC common mode.

    The dominant feature eigenmode of GLOBAL-AVERAGE-POOLED post-ReLU features at init is the
    all-positive DC/common mode that ReLU rectification injects and pooling preserves (see the
    'pre_relu' feature_mode in _extract_features for the source-level alternative). These
    per-sample transforms remove it downstream of extraction:

      'none'/'pre_relu'      -> identity (pre_relu is handled at extraction, not post hoc).
      'per_sample_center'    -> subtract each sample's mean over feature dims (removes the
                                ~all-ones common component that is the dominant eigenmode).
      'per_sample_standardize' -> additionally divide by each sample's std over feature dims
                                (Coates & Ng 2011 per-sample brightness/contrast normalization).
    """
    f = np.asarray(features, dtype=np.float64)
    if mode in ('none', 'pre_relu', 'post_relu'):
        return f
    centered = f - f.mean(axis=1, keepdims=True)
    if mode == 'per_sample_center':
        return centered
    if mode == 'per_sample_standardize':
        std = f.std(axis=1, keepdims=True)
        return centered / np.where(std > 0, std, 1.0)
    raise ValueError(f'unknown feature preproc {mode!r}')


# --------------------------------------------------------------------------------------
# Model-at-init / feature extraction plumbing (JAX + torch; runs on the cluster)
# --------------------------------------------------------------------------------------
def _build_split_loader(dataset: str, split: str, num_images: int, seed: int,
                        batch_size: int, num_workers: int):
    """DataLoader over a dataset split with the deterministic eval transform (channels-last)."""
    from torch.utils.data import Subset

    dataset = str(dataset).strip().lower()
    if dataset == 'cifar5m':
        from scripts.analyze_exchangeability import _load_cifar5m_probe_subset_builder
        from src.run.constants import CIFAR5M_FOLDER

        if CIFAR5M_FOLDER is None:
            raise ValueError('CIFAR5M_FOLDER must be set for cifar5m analysis.')
        size = num_images if num_images > 0 else 50000
        # Distinct seed per split so train/val draws are (near-)disjoint from the 5M pool.
        subset = _load_cifar5m_probe_subset_builder()(CIFAR5M_FOLDER, size, seed)
        # The cifar5m probe transform holds an unpicklable local lambda, so spawn workers
        # cannot serialize the dataset; load in-process (num_workers=0). Images are 32x32.
        return _make_loader(subset, batch_size, 0)

    from scripts.analyze_exchangeability import _load_imagenet_torchvision
    from src.run.constants import IMAGENET_FOLDER

    if IMAGENET_FOLDER is None:
        raise ValueError('IMAGENET_FOLDER must be set for imagenet analysis.')
    ImageFolder, ImageNet, transforms = _load_imagenet_torchvision()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    channels_last = transforms.Lambda(_chw_to_hwc)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        channels_last,
    ])
    split_dir = os.path.join(IMAGENET_FOLDER, split)
    if os.path.isdir(split_dir):
        base_dataset = ImageFolder(split_dir, transform=transform)
    else:
        base_dataset = ImageNet(IMAGENET_FOLDER, split, transform=transform)

    total = len(base_dataset)
    if num_images <= 0 or num_images >= total:
        dataset_obj = base_dataset
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=num_images, replace=False)
        dataset_obj = Subset(base_dataset, indices.tolist())
    return _make_loader(dataset_obj, batch_size, num_workers)


def _init_variables(model, input_shape, compute_dtype, init_seed: int) -> dict:
    """model.init at train=True to create params + batch_stats + mup collections."""
    import jax
    import jax.numpy as jnp
    from flax.core import unfreeze

    dummy = jnp.zeros((1,) + tuple(input_shape), dtype=compute_dtype)
    variables = model.init(jax.random.PRNGKey(int(init_seed)), dummy, train=True)
    return unfreeze(variables)


def _calibrate_batchnorm(model, variables: dict, loader, max_batches: int) -> dict:
    """Populate BatchNorm running stats with train-mode forward passes (EMA over batches)."""
    import jax.numpy as jnp

    seen = 0
    for batch_x, _ in loader:
        x = jnp.asarray(np.asarray(batch_x))
        _, updated = model.apply(variables, x, train=True, mutable=['batch_stats'])
        variables = {**variables, 'batch_stats': updated['batch_stats']}
        seen += 1
        if seen >= max_batches:
            break
    print(f'calibrated BatchNorm over {seen} batches')
    return variables


def _analyze_width(*, width: int, args, dataset: str, num_classes: int, input_shape,
                   compute_dtype, spec, base_save_dir: str, train_loader, val_loader) -> None:
    """Init one width, activate residuals, calibrate BN, fit ridge paths, analyze on val."""
    from src.experiment.model.flax_mup.resnet import ResNet18

    width = int(width)
    print(f'\n=== dataset={dataset} width={width} (ResNet18 at init) ===')

    model = ResNet18(num_classes=num_classes, num_filters=width,
                     param_dtype=compute_dtype, stem_type=spec.stem_type)
    variables = _init_variables(model, input_shape, compute_dtype, args.init_seed + width)

    # Activate the inert residual branches (zero-init gammas -> ones/random).
    rng = np.random.default_rng(args.init_seed + width)
    params, n_activated = activate_residual_scales(
        variables['params'], args.residual_scale_init, rng=rng, std=args.residual_scale_std)
    variables['params'] = _to_dtype_tree(params, compute_dtype)
    print(f'residual_scale_init={args.residual_scale_init}: activated {n_activated} '
          f'zero-init BatchNorm scales')

    # BatchNorm calibration over train images.
    if args.residual_scale_init != 'zeros' and args.num_calib_batches > 0:
        variables = _calibrate_batchnorm(model, variables, train_loader, args.num_calib_batches)

    # Features for fitting the ridge classifier (train) and for analysis (val). feature_mode
    # 'pre_relu' pools the block pre-activation (avoids the DC common mode at source); the
    # per-sample preprocs are applied post hoc below.
    feature_mode = 'pre_relu' if args.feature_preproc == 'pre_relu' else 'post_relu'
    print(f'extracting TRAIN features (for ridge fit; feature_mode={feature_mode})...')
    feats_train, _logits_train, labels_train = _extract_features(
        model, variables, train_loader, feature_mode=feature_mode)
    print(f'train features={feats_train.shape} labels={labels_train.shape}')
    print('extracting VAL features (for analysis)...')
    feats_val, _logits_val, labels_val = _extract_features(
        model, variables, val_loader, feature_mode=feature_mode)
    print(f'val features={feats_val.shape} labels={labels_val.shape}')

    # Per-sample feature preprocessing (identity for 'none'/'pre_relu').
    feats_train = apply_feature_preproc(feats_train, args.feature_preproc)
    feats_val = apply_feature_preproc(feats_val, args.feature_preproc)
    lam1_share = float('nan')
    try:
        ev = np.linalg.eigvalsh(np.cov(feats_val.astype(np.float64), rowvar=False))
        ev = np.clip(ev, 0.0, None)
        lam1_share = float(ev.max() / ev.sum()) if ev.sum() > 0 else float('nan')
        eff_rank = float(ev.sum() ** 2 / (ev ** 2).sum()) if (ev ** 2).sum() > 0 else float('nan')
        print(f'feature_preproc={args.feature_preproc}: val lambda1_share={lam1_share:.3f} '
              f'eff_rank={eff_rank:.1f}')
    except Exception:
        pass
    num_features = int(feats_val.shape[1])

    # Ridge classifier paths over the lambda grid (fit on train features).
    paths = fit_ridge_paths(feats_train, labels_train, num_classes, args.ridge_rel_lambdas,
                            scale_mode=args.ridge_lambda_scale)

    root = os.path.abspath(args.output_root) if args.output_root \
        else os.path.join(base_save_dir, 'init_regression_powerlaw')
    pp_tag = _PREPROC_TAG.get(args.feature_preproc, '')
    dir_suffix = f'_{pp_tag}' if pp_tag else ''
    for path in paths:
        rel = path['rel_lambda']
        tag = _lambda_tag(rel)
        run_label = f'init_reg_w{width}_{tag}{dir_suffix}'
        output_dir = args.output_dir if (args.output_dir and not args.widths and len(paths) == 1) \
            else os.path.join(root, dataset, f'init_reg_w{width}_{tag}{dir_suffix}')
        os.makedirs(output_dir, exist_ok=True)
        # In-sample train accuracy of this ridge fit (diagnostic, not the analyzed metric).
        train_logits = feats_train @ path['W'].T + path['b']
        train_acc = float(np.mean(train_logits.argmax(1) == labels_train))
        print(f'\n--- {run_label}: rel_lambda={rel:g} abs_lambda={path["abs_lambda"]:.4g} '
              f'dof={path["dof"]:.1f}/{num_features} train_acc={train_acc:.4f} -> {output_dir} ---')
        run_powerlaw_analysis(
            features=feats_val.astype(np.float64),
            labels=labels_val,
            weight_eff=path['W'],
            bias=path['b'],
            num_classes=num_classes,
            run_label=run_label,
            output_dir=output_dir,
            num_bins=args.num_bins,
            fit_seeds=args.fit_seeds,
            fit_rmse_tol=args.fit_rmse_tol,
            seed=int(args.seed),
            trunc_list=args.trunc_list,
            no_plots=args.no_plots,
            model_logits=None,  # regression predictions: feats_val @ W^T + b
            source_arrays={
                'dataset': dataset,
                'width': np.int64(width),
                'classifier': 'init_ridge_regression',
                'ridge_rel_lambda': np.float64(rel),
                'ridge_abs_lambda': np.float64(path['abs_lambda']),
                'ridge_dof': np.float64(path['dof']),
                'residual_scale_init': args.residual_scale_init,
                'feature_preproc': args.feature_preproc,
                'feature_lambda1_share': np.float64(lam1_share),
                'num_train_images': np.int64(labels_train.size),
                'init_seed': np.int64(args.init_seed + width),
                'train_accuracy': np.float64(train_acc),
            },
            source_summary={
                'dataset': dataset,
                'width': width,
                'classifier': 'init_ridge_regression',
                'ridge_rel_lambda': float(rel),
                'ridge_abs_lambda': float(path['abs_lambda']),
                'ridge_lambda_scale_mode': args.ridge_lambda_scale,
                'ridge_effective_dof': float(path['dof']),
                'residual_scale_init': args.residual_scale_init,
                'feature_preproc': args.feature_preproc,
                'feature_lambda1_share': float(lam1_share),
                'bn_calibration_batches': int(args.num_calib_batches),
                'num_train_images': int(labels_train.size),
                'num_val_images': int(labels_val.size),
                'init_seed': int(args.init_seed + width),
                'train_accuracy': train_acc,
            },
        )


def _to_dtype_tree(tree, dtype):
    """Cast floating-point leaves of a nested dict to a jax dtype; leave integers alone."""
    import jax
    import jax.numpy as jnp

    def _cast(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(dtype) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf

    return jax.tree_util.tree_map(_cast, tree)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['imagenet', 'cifar5m'], default='imagenet')
    parser.add_argument('--width', type=int, default=64, help='Single width (if --widths unset).')
    parser.add_argument('--widths', type=int, nargs='*', default=None,
                        help='Multiple widths in one process (data loaders built once).')
    parser.add_argument('--base-save-dir', default='', help='Override dataset save root.')
    parser.add_argument('--init-seed', type=int, default=0,
                        help='Base PRNG seed for model init (per-width offset added).')
    parser.add_argument('--residual-scale-init', choices=['ones', 'random', 'zeros'],
                        default='ones',
                        help="How to set the zero-init residual-branch BatchNorm gammas. "
                             "'ones' (default) activates them at the conventional value 1; "
                             "'random' draws N(1, std); 'zeros' keeps the inert faithful-t=0 map.")
    parser.add_argument('--residual-scale-std', type=float, default=0.5,
                        help="Std for --residual-scale-init random.")
    parser.add_argument('--feature-preproc',
                        choices=['none', 'per_sample_center', 'per_sample_standardize', 'pre_relu'],
                        default='none',
                        help="Tame the ReLU+global-average-pool DC common mode of at-init "
                             "features. none: raw post-ReLU GAP features. per_sample_center / "
                             "per_sample_standardize: subtract each sample's mean (and divide by "
                             "std) over feature dims (Coates & Ng 2011). pre_relu: pool the block "
                             "pre-activation (before ReLU) instead, avoiding the DC mode at source.")
    parser.add_argument('--num-train-images', type=int, default=150000,
                        help='Train images for the ridge fit (<=0 uses the full split).')
    parser.add_argument('--num-val-images', type=int, default=50000,
                        help='Val images for the PCA/capacity/source/truncation analysis.')
    parser.add_argument('--num-calib-batches', type=int, default=100,
                        help='Train-mode forward passes to calibrate BatchNorm running stats.')
    parser.add_argument('--ridge-rel-lambdas', type=float, nargs='*',
                        default=[0.0, 1e-3, 1e-2, 1e-1, 1.0],
                        help='Ridge strengths relative to the feature-covariance scale.')
    parser.add_argument('--ridge-lambda-scale', choices=['mean_eig', 'max_eig'],
                        default='mean_eig',
                        help='Feature-Gram eigenvalue that rel-lambda multiplies.')
    parser.add_argument('--train-seed', type=int, default=1234,
                        help='Seed for the train-subset selection.')
    parser.add_argument('--seed', type=int, default=2423,
                        help='Seed for the val-subset selection and the capacity seed-grow fit.')
    parser.add_argument('--eval-batch-size', type=int, default=250)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-bins', type=int, default=24)
    parser.add_argument('--fit-seeds', type=int, default=64)
    parser.add_argument('--fit-rmse-tol', type=float, default=0.10)
    parser.add_argument('--compute-dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--trunc-list', type=int, nargs='*', default=None)
    parser.add_argument('--output-dir', default='',
                        help='Output dir for a single width + single lambda. Ignored otherwise.')
    parser.add_argument('--output-root', default='',
                        help='Parent dir; defaults to <base-save-dir>/init_regression_powerlaw.')
    parser.add_argument('--no-plots', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import jax.numpy as jnp

    from src.experiment.dataset_specs import get_dataset_spec

    dataset = args.dataset
    widths = [int(w) for w in args.widths] if args.widths else [int(args.width)]
    base_save_dir = args.base_save_dir or _default_base_save_dir(dataset)
    spec = get_dataset_spec(dataset)
    num_classes = spec.num_classes
    input_shape = spec.input_shape
    compute_dtype = jnp.float32 if args.compute_dtype == 'float32' else jnp.bfloat16
    print(f'dataset={dataset} widths={widths} num_classes={num_classes} '
          f'compute_dtype={args.compute_dtype}')
    print(f'base_save_dir={base_save_dir} residual_scale_init={args.residual_scale_init} '
          f'ridge_rel_lambdas={args.ridge_rel_lambdas}')

    # Build the train/val loaders once; re-iterated per width (features differ per init).
    print(f'building loaders (train={args.num_train_images}, val={args.num_val_images})...')
    train_loader = _build_split_loader(dataset, 'train', int(args.num_train_images),
                                       int(args.train_seed), int(args.eval_batch_size),
                                       int(args.num_workers))
    val_loader = _build_split_loader(dataset, 'val', int(args.num_val_images), int(args.seed),
                                     int(args.eval_batch_size), int(args.num_workers))

    failures: list[tuple[int, str]] = []
    for width in widths:
        try:
            _analyze_width(
                width=width, args=args, dataset=dataset, num_classes=num_classes,
                input_shape=input_shape, compute_dtype=compute_dtype, spec=spec,
                base_save_dir=base_save_dir, train_loader=train_loader, val_loader=val_loader,
            )
        except Exception as exc:  # keep going so one bad width doesn't lose the rest
            import traceback

            traceback.print_exc()
            failures.append((int(width), repr(exc)))

    if failures:
        print(f'FAILURES ({len(failures)}): {failures}')
        raise SystemExit(1)
    print(f'all done for widths {widths}')


if __name__ == '__main__':
    main()
