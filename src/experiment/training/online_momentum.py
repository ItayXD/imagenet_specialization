import json
import os
import time
from functools import partial
from logging import info
from typing import Any

import chex
import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
from flax.training import checkpoints, train_state
from jax import ShapeDtypeStruct, jit, tree_map, value_and_grad, vmap
from jax.lax import scan
from jax.random import split

from src.experiment.model.flax_mup.mup import Mup
from src.experiment.model.flax_mup.resnet import ResNet18
from src.experiment.exchangeability_utils import make_target_points
from src.run.constants import BASE_SAVE_DIR

NUM_CLASSES = 1_000
IMAGENET_SHAPE = (224, 224, 3)

# Backward-compatible alias for existing callers/tests that used the old location.
_make_target_points = make_target_points


def _get_nested_item(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    try:
        return obj[key]
    except Exception:
        return None


def _find_conv_init_kernel(params):
    conv_init = _get_nested_item(params, 'conv_init')
    if conv_init is not None:
        kernel = _get_nested_item(conv_init, 'kernel')
        if kernel is not None:
            return kernel

    if isinstance(params, dict):
        for value in params.values():
            kernel = _find_conv_init_kernel(value)
            if kernel is not None:
                return kernel
    else:
        try:
            for _, value in params.items():
                kernel = _find_conv_init_kernel(value)
                if kernel is not None:
                    return kernel
        except Exception:
            return None
    return None


def _extract_first_layer_weights(params_tree) -> np.ndarray:
    kernel = _find_conv_init_kernel(params_tree)
    if kernel is None:
        raise ValueError('Could not find conv_init kernel in parameter tree.')
    kernel = np.asarray(kernel)
    if kernel.ndim < 6:
        raise ValueError('Expected ensemble-shaped kernel with dimensions [subset, member, kh, kw, in_ch, out_ch].')

    subset, member = kernel.shape[:2]
    out_ch = kernel.shape[-1]
    flattened = kernel.reshape(subset * member, -1, out_ch)
    flattened = np.transpose(flattened, (0, 2, 1))
    return flattened


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _compute_loss_error(logits: jnp.ndarray, labels: jnp.ndarray) -> tuple[float, float]:
    labels = labels.reshape((-1,))
    ensemble_logits = jnp.mean(logits.reshape((-1,) + logits.shape[2:]), axis=0)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(ensemble_logits, labels))
    preds = jnp.argmax(ensemble_logits, axis=1)
    acc = jnp.mean(preds == labels)
    return float(loss), float(1.0 - acc)


def _write_jsonl(path: str, row: dict) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')


def _resolve_run_dirs(training_params: dict, N: int, n_ensemble: int) -> dict[str, str]:
    run_id = str(training_params.get('run_id', 'exchangeability'))
    width = int(training_params.get('width', N))
    group_id = int(training_params.get('group_id', 0))

    base_dir = BASE_SAVE_DIR or os.path.join(os.getcwd(), 'outputs')
    run_dir = os.path.join(base_dir, run_id, f'width_{width}', f'group_{group_id}')

    state_ckpt_dir = os.path.join(run_dir, 'state_ckpts')
    artifact_dir = os.path.join(run_dir, 'artifacts')
    metric_file = os.path.join(run_dir, 'metrics.jsonl')
    metadata_file = os.path.join(run_dir, 'metadata.json')

    _ensure_dir(run_dir)
    _ensure_dir(state_ckpt_dir)
    _ensure_dir(artifact_dir)

    metadata = {
        'run_id': run_id,
        'width': width,
        'group_id': group_id,
        'ensemble_size': n_ensemble,
        'created_at': time.time(),
        'target_images_seen': int(training_params.get('target_images_seen', 10_000_000)),
        'p_targets_images_seen': [int(v) for v in training_params.get('p_targets_images_seen', [])],
    }
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return {
        'run_dir': run_dir,
        'state_ckpt_dir': state_ckpt_dir,
        'artifact_dir': artifact_dir,
        'metric_file': metric_file,
    }


def _init_wandb(training_params: dict, model_params: dict, n_ensemble: int):
    if not bool(training_params.get('wandb_enabled', False)):
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        info(f'W&B unavailable: {e}')
        return None

    project = str(training_params.get('wandb_project', 'imagenet-exchangeability'))
    entity = str(training_params.get('wandb_entity', '')).strip() or None
    mode = str(training_params.get('wandb_mode', 'online'))
    run_id = str(training_params.get('run_id', 'exchangeability'))
    width = int(training_params.get('width', model_params.get('N', 0)))
    group_id = int(training_params.get('group_id', 0))

    common_kwargs = {
        'project': project,
        'entity': entity,
        'name': f'{run_id}-w{width}-g{group_id}',
        'group': run_id,
        'config': {
            'width': width,
            'group_id': group_id,
            'ensemble_size': n_ensemble,
            'training_params': training_params,
            'model_params': model_params,
        },
    }

    try:
        return wandb.init(mode=mode, **common_kwargs)
    except Exception as e:
        info(f'W&B init failed in mode={mode}: {e}; falling back to offline.')
        try:
            return wandb.init(mode='offline', **common_kwargs)
        except Exception as ee:
            info(f'W&B offline init failed: {ee}')
            return None


def initialize(keys, N: int, num_ensemble_subsets: int, mup, param_dtype):
    model = ResNet18(num_classes=NUM_CLASSES, num_filters=N, param_dtype=param_dtype)
    dummy_input = jnp.zeros((1,) + IMAGENET_SHAPE, dtype=param_dtype)

    width_mults = mup._width_mults
    readout_zero_init = mup.readout_zero_init

    def train_init(key, inp):
        vars_ = model.init(key, inp, train=True)
        mup_vars = dict(
            mup.rescale_parameters({'params': vars_['params']}, width_mults, readout_zero_init),
            **{'batch_stats': vars_['batch_stats']},
        )
        mup_vars.update({'mup': vars_['mup']})
        return mup_vars

    within_subset_size = keys.shape[0] // num_ensemble_subsets
    sub_keys = keys.reshape((num_ensemble_subsets, within_subset_size, 2))

    ensemble_get_params = vmap(
        vmap(train_init, in_axes=(0, None), axis_name='within_subset'),
        in_axes=(0, None),
        axis_name='over_subsets',
    )
    fn = jit(ensemble_get_params).lower(
        ShapeDtypeStruct(sub_keys.shape, jnp.uint32),
        ShapeDtypeStruct(dummy_input.shape, jnp.float32),
    ).compile()
    return fn(sub_keys, dummy_input)


def train(
    vars_0: chex.ArrayTree,
    N: int,
    optimizer: optax.GradientTransformation,
    train_loader,
    val_data,
    batch_size: int,
    n_ensemble: int,
    ensemble_subsets: int,
    data_dtype: Any,
    target_images_seen: int,
    p_targets_images_seen: list[int],
    training_params: dict,
    model_params: dict,
    learning_rate_fn,
):
    tranche_size = train_loader.batch_size
    num_batches = tranche_size // batch_size

    if num_batches <= 0:
        raise ValueError('microbatch_size must be <= minibatch_size.')

    class TrainState(train_state.TrainState):
        batch_stats: chex.ArrayTree
        mup: chex.ArrayTree

    @partial(vmap, in_axes=(0, None, None), axis_name='ensemble')
    def _subset_update(state: TrainState, Xtr_sb: chex.ArrayDevice, ytr_sb: chex.ArrayDevice) -> TrainState:
        def apply_fn(vars_, Xin):
            y_hat, dict_updated_bs = state.apply_fn(vars_, Xin, train=True, mutable=['batch_stats'])
            return y_hat, dict_updated_bs['batch_stats']

        def loss_fn(params, batch_stats, mup_col, Xin, yin):
            vars_ = {'params': params, 'batch_stats': batch_stats, 'mup': mup_col}
            y_hat, update_bs = apply_fn(vars_, Xin)
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(y_hat, yin))
            return loss, update_bs

        loss_grad_fn = value_and_grad(loss_fn, has_aux=True)

        def step(step_state: TrainState, data: tuple) -> TrainState:
            batch, labels = data
            (_, update_bs), grads = loss_grad_fn(step_state.params, step_state.batch_stats, step_state.mup, batch, labels)
            return step_state.apply_gradients(grads=grads, batch_stats=update_bs), None

        updated_state, _ = scan(step, state, (Xtr_sb, ytr_sb))
        return updated_state

    @jit
    def update(state: TrainState, Xtr: chex.ArrayDevice, ytr: chex.ArrayDevice):
        Xtr_sb = Xtr.reshape((num_batches, batch_size, *Xtr.shape[1:]))
        ytr_sb = ytr.reshape((num_batches, batch_size, *ytr.shape[1:]))

        def partial_subset_update(state_stacked):
            return _subset_update(state_stacked, Xtr_sb, ytr_sb)

        return jax.lax.map(partial_subset_update, state)

    model = ResNet18(num_classes=NUM_CLASSES, num_filters=N, param_dtype=data_dtype)

    def create_train_state(params, tx, bs, mup):
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=bs, mup=mup)

    init_params = vars_0['params']
    init_bs = vars_0['batch_stats']
    mup_col = vars_0['mup']
    state = vmap(
        vmap(create_train_state, axis_name='within_subset', in_axes=(0, None, 0, 0)),
        axis_name='over_subsets',
        in_axes=(0, None, 0, 0),
    )(init_params, optimizer, init_bs, mup_col)

    @jit
    def predict_logits(params, batch_stats, mup_vars, x):
        def one_member(p, bs, mu):
            vars_ = {'params': p, 'batch_stats': bs, 'mup': mu}
            return model.apply(vars_, x, train=False)

        pred_within = vmap(one_member, in_axes=(0, 0, 0))
        pred_all = vmap(pred_within, in_axes=(0, 0, 0))
        return pred_all(params, batch_stats, mup_vars)

    def loader_to_jax(batch):
        x_ch, y_list = batch
        x_jnp = jnp.array(x_ch, dtype=data_dtype)
        y_jnp = jnp.array(y_list)
        return x_jnp, y_jnp

    val_x, val_y = loader_to_jax(val_data)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    run_paths = _resolve_run_dirs(training_params, N, n_ensemble)
    metric_file = run_paths['metric_file']

    target_points = make_target_points(target_images_seen, p_targets_images_seen)
    target_index = 0

    images_seen = 0
    tranches_seen = 0
    max_tranches = int(training_params.get('max_tranches', 0))
    log_every_tranches = max(1, int(training_params.get('log_every_tranches', 10)))

    wandb_run = _init_wandb(training_params, model_params, n_ensemble)

    info('Entering training loop...')
    start = time.time()

    while images_seen < target_images_seen:
        for tranche in train_loader:
            x, y = loader_to_jax(tranche)
            state = update(state, x, y)
            tranches_seen += 1
            images_seen += tranche_size

            step_value = int(np.asarray(state.step)[0, 0])
            lr_value = float(learning_rate_fn(step_value))

            if tranches_seen % log_every_tranches == 0 and wandb_run is not None:
                wandb_run.log(
                    {
                        'images_seen': images_seen,
                        'step': step_value,
                        'lr': lr_value,
                    },
                    step=images_seen,
                )

            while target_index < len(target_points) and images_seen >= target_points[target_index]:
                target_p = target_points[target_index]

                train_logits = predict_logits(state.params, state.batch_stats, state.mup, x)
                val_logits = predict_logits(state.params, state.batch_stats, state.mup, val_x)
                train_loss, train_error = _compute_loss_error(train_logits, y)
                val_loss, val_error = _compute_loss_error(val_logits, val_y)

                checkpoints.save_checkpoint(
                    ckpt_dir=run_paths['state_ckpt_dir'],
                    target=state,
                    step=int(target_p),
                    overwrite=False,
                    keep=1_000_000,
                    prefix='state_',
                    orbax_checkpointer=checkpointer,
                )

                first_layer_weights = _extract_first_layer_weights(state.params)
                np.savez_compressed(
                    os.path.join(run_paths['artifact_dir'], f'first_layer_{target_p}.npz'),
                    first_layer_weights=first_layer_weights,
                    images_seen=np.array([target_p], dtype=np.int64),
                    train_loss=np.array([train_loss], dtype=np.float64),
                    train_error=np.array([train_error], dtype=np.float64),
                    val_loss=np.array([val_loss], dtype=np.float64),
                    val_error=np.array([val_error], dtype=np.float64),
                )

                row = {
                    'images_seen': int(target_p),
                    'step': step_value,
                    'train_loss': train_loss,
                    'train_error': train_error,
                    'val_loss': val_loss,
                    'val_error': val_error,
                    'lr': lr_value,
                }
                _write_jsonl(metric_file, row)

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            'images_seen': int(target_p),
                            'step': step_value,
                            'lr': lr_value,
                            'train_loss': train_loss,
                            'train_error': train_error,
                            'val_loss': val_loss,
                            'val_error': val_error,
                        },
                        step=int(target_p),
                    )

                info(
                    f'checkpoint target={target_p} images_seen={images_seen} '
                    f'train_loss={train_loss:.4f} val_loss={val_loss:.4f}'
                )
                target_index += 1

            if max_tranches > 0 and tranches_seen >= max_tranches:
                info(f'Stopping early due to max_tranches={max_tranches}.')
                images_seen = target_images_seen
                break

            if images_seen >= target_images_seen:
                break

        if images_seen >= target_images_seen:
            break

    if wandb_run is not None:
        wandb_run.finish()

    info(f'...exiting loop: elapsed time {time.time() - start:.1f}s')
    return None, None


def apply(key, train_loader, val_data, devices, model_params, training_params):
    n_ensemble = int(model_params['ensemble_size'])
    if n_ensemble <= 0:
        raise ValueError('ensemble_size must be > 0')

    ensemble_subsets = int(training_params['ensemble_subsets'])
    if ensemble_subsets <= 0 or n_ensemble % ensemble_subsets != 0:
        raise ValueError('ensemble_subsets must be > 0 and divide ensemble_size.')

    BASE_N = int(model_params['BASE_N'])
    N = int(model_params['N'])
    if N <= 0:
        raise ValueError('N must be > 0')

    try:
        dtype = jnp.dtype(model_params['dtype'])
    except TypeError as e:
        raise ValueError('model_params.dtype must be a valid jax dtype') from e

    mup = Mup()

    init_input = jnp.zeros((1,) + IMAGENET_SHAPE, dtype=dtype)
    base_model = ResNet18(num_classes=NUM_CLASSES, num_filters=BASE_N, param_dtype=dtype)
    vars_ = base_model.init(jax.random.PRNGKey(0), init_input)
    mup.set_base_shapes({'params': vars_['params']})

    target_model = ResNet18(num_classes=NUM_CLASSES, num_filters=N, param_dtype=dtype)
    vars_target = target_model.init(jax.random.PRNGKey(0), init_input)
    mup.set_target_shapes({'params': vars_target['params']})

    init_keys = split(key, num=n_ensemble)
    vars_0 = initialize(init_keys, N, ensemble_subsets, mup, dtype)
    info('Initialized parameters.')

    eta_0 = float(training_params['eta_0'])
    use_warmup_cosine_decay = bool(training_params.get('use_warmup_cosine_decay', True))

    target_images_seen = int(training_params.get('target_images_seen', 10_000_000))
    microbatch_size = int(training_params['microbatch_size'])
    total_micro_steps = int(np.ceil(target_images_seen / microbatch_size))

    lr_schedule = None
    if use_warmup_cosine_decay:
        wcd_params = training_params['wcd_params']
        init_lr = float(wcd_params['init_lr'])
        min_lr = float(wcd_params['min_lr'])

        warmup_steps = max(1, int(total_micro_steps * 0.01))
        decay_steps = max(warmup_steps + 1, total_micro_steps)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=init_lr,
            peak_value=eta_0,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=min_lr,
        )
        base_optimizer = optax.adam(learning_rate=lr_schedule)
    else:
        base_optimizer = optax.adam(eta_0)
        lr_schedule = lambda _: eta_0

    optimizer = mup.wrap_optimizer(base_optimizer, adam=True)

    info('Entering train function.')
    _ = train(
        vars_0=vars_0,
        N=N,
        optimizer=optimizer,
        train_loader=train_loader,
        val_data=val_data,
        batch_size=int(training_params['microbatch_size']),
        n_ensemble=n_ensemble,
        ensemble_subsets=ensemble_subsets,
        data_dtype=dtype,
        target_images_seen=target_images_seen,
        p_targets_images_seen=training_params.get('p_targets_images_seen', []),
        training_params=training_params,
        model_params=model_params,
        learning_rate_fn=lr_schedule,
    )

    return None
