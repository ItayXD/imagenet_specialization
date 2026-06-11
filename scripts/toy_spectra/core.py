"""Toy models for width-consistency of weight spectra: muP SGD vs idealized Muon.

Three online-regression models (input dim d, output dim M, width N, batch B,
fresh Gaussian samples each step, MSE loss, no label noise, constant lr):

  lin3:    f = W2 (W1 (W0 x / sqrt(d)) / sqrt(N)) / (gamma0 N)
           target y = A x, A = U diag(linspace(1, 4, M)) V^T of rank M.
  nonlin2: f = W1 relu(W0 x / sqrt(d)) / (gamma0 N)
           target y = V relu(B x / sqrt(d)), committee teacher with k hidden units.
  nonlin3: f = W2 relu(W1 relu(W0 x / sqrt(d)) / sqrt(N)) / (gamma0 N)
           same committee-teacher target.

All weights init N(0, 1) (mean-field convention, cf. muon_spectra_toy/).

Optimizers:
  sgd:  muP -- one global lr = eta_sgd * gamma0^2 * N on every layer.
  muon: idealized Muon on hidden layers, momentum-free with exact
        orthogonalization: dW = -eta_muon * (sqrt(fan_in) + sqrt(fan_out)) *
        msign(grad); readout always via muP SGD. This deliberately differs
        from optax.contrib.scale_by_muon (momentum + Newton-Schulz): the
        idealization is part of the experiment design. In MP-normalized units
        the update spectral norm is eta_muon * (1 + sqrt(aspect)), i.e.
        width-independent by construction.

Spectra conventions follow scripts/plot_resnet_block_spectra.py and
muon_spectra_toy/width_spectra_toy.py: normalized sigma = sigma /
sqrt(cols * init_var) with init_var = 1, aspect = rows / cols, MP edge =
1 + sqrt(aspect).
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

GAMMA0_DEFAULT = 3.0
D_DEFAULT = 512
M_DEFAULT = 8
BATCH_SIZE_DEFAULT = 512
TARGET_SEED_DEFAULT = 4242
TEACHER_K = 5

MODELS = ("lin3", "nonlin2", "nonlin3")
OPTIMIZERS = ("sgd", "muon")
TRACKED_LAYERS = {
    "lin3": ("W0", "W1"),
    "nonlin2": ("W0", "W1"),
    "nonlin3": ("W0", "W1"),
}


def make_p_targets(
    p_min: int = 10_000,
    p_max: int = 2_048_000,
    num: int = 15,
    batch_size: int = BATCH_SIZE_DEFAULT,
    include_zero: bool = True,
) -> list[int]:
    """Log-spaced sample-count checkpoints, each rounded up to a batch multiple."""
    raw = np.logspace(math.log10(p_min), math.log10(p_max), num)
    targets = sorted({int(math.ceil(p / batch_size)) * batch_size for p in raw})
    if include_zero:
        targets = [0] + targets
    return targets


# --------------------------------------------------------------------- msign
def msign_thin(L: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """Polar factor of G = L @ R.T from thin (tall) factors.

    Requires L (n, r), R (m, r) with n >= r and m >= r; cost O((n + m) r^2).
    """
    QL, SL = jnp.linalg.qr(L)
    QR_, SR = jnp.linalg.qr(R)
    U, _, Vt = jnp.linalg.svd(SL @ SR.T, full_matrices=False)
    return (QL @ U) @ (Vt @ QR_.T)


def msign_full(G: jnp.ndarray) -> jnp.ndarray:
    """Polar factor of G via thin SVD (used when no tall factorization exists)."""
    U, _, Vt = jnp.linalg.svd(G, full_matrices=False)
    return U @ Vt


def msign_from_factors(L: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """Polar factor of L @ R.T, picking the cheap thin path when both factors
    are tall (rank-limited gradients); otherwise SVD of the full product."""
    if L.shape[0] >= L.shape[1] and R.shape[0] >= R.shape[1]:
        return msign_thin(L, R)
    return msign_full(L @ R.T)


def muon_scale(rows: int, cols: int) -> float:
    return math.sqrt(rows) + math.sqrt(cols)


# ------------------------------------------------------------------- targets
def make_target(model: str, target_seed: int, d: int, M: int) -> dict[str, jnp.ndarray]:
    """Frozen target, fixed by target_seed only (shared across N/seed/optimizer)."""
    key = jax.random.PRNGKey(target_seed)
    k_u, k_v, k_b, k_c = jax.random.split(key, 4)
    if model == "lin3":
        s = jnp.linspace(1.0, 4.0, M)
        U = jnp.linalg.qr(jax.random.normal(k_u, (M, M)))[0]
        V = jnp.linalg.qr(jax.random.normal(k_v, (d, M)))[0]
        return {"A": (U * s) @ V.T}  # (M, d)
    if model in ("nonlin2", "nonlin3"):
        B = jax.random.normal(k_b, (TEACHER_K, d))
        B = B / jnp.linalg.norm(B, axis=1, keepdims=True) * math.sqrt(d)
        V = jax.random.normal(k_c, (M, TEACHER_K)) / math.sqrt(TEACHER_K)
        return {"B": B, "V": V}
    raise ValueError(f"unknown model {model!r}")


def target_fn(model: str, target: dict[str, jnp.ndarray], X: jnp.ndarray) -> jnp.ndarray:
    """X: (d, B) -> Y: (M, B)."""
    if model == "lin3":
        return target["A"] @ X
    d = X.shape[0]
    return target["V"] @ jax.nn.relu(target["B"] @ X / math.sqrt(d))


# -------------------------------------------------------------- params/model
def init_params(model: str, key, N: int, d: int, M: int) -> dict[str, jnp.ndarray]:
    k0, k1, k2 = jax.random.split(key, 3)
    if model == "lin3" or model == "nonlin3":
        return {
            "W0": jax.random.normal(k0, (N, d)),
            "W1": jax.random.normal(k1, (N, N)),
            "W2": jax.random.normal(k2, (M, N)),
        }
    if model == "nonlin2":
        return {
            "W0": jax.random.normal(k0, (N, d)),
            "W1": jax.random.normal(k1, (M, N)),
        }
    raise ValueError(f"unknown model {model!r}")


def forward(model: str, params: dict, X: jnp.ndarray, gamma0: float) -> jnp.ndarray:
    """X: (d, B) -> f: (M, B)."""
    d = X.shape[0]
    if model == "lin3":
        N = params["W1"].shape[0]
        H1 = params["W0"] @ X / math.sqrt(d)
        H2 = params["W1"] @ H1 / math.sqrt(N)
        return params["W2"] @ H2 / (gamma0 * N)
    if model == "nonlin2":
        N = params["W0"].shape[0]
        H = jax.nn.relu(params["W0"] @ X / math.sqrt(d))
        return params["W1"] @ H / (gamma0 * N)
    if model == "nonlin3":
        N = params["W1"].shape[0]
        H0 = jax.nn.relu(params["W0"] @ X / math.sqrt(d))
        H1 = jax.nn.relu(params["W1"] @ H0 / math.sqrt(N))
        return params["W2"] @ H1 / (gamma0 * N)
    raise ValueError(f"unknown model {model!r}")


def batch_loss(model: str, params: dict, target: dict, X: jnp.ndarray,
               gamma0: float) -> jnp.ndarray:
    E = forward(model, params, X, gamma0) - target_fn(model, target, X)
    return 0.5 * jnp.mean(jnp.sum(E**2, axis=0))


# --------------------------------------------------------------- single step
def _step(model: str, opt: str, params: dict, target: dict, X: jnp.ndarray,
          gamma0: float, lr_sgd: float, eta_muon: float):
    """One online step on a fresh batch X (d, B). Returns (params, loss).

    Explicit backward pass: Muon needs the low-rank factor structure of the
    hidden-layer gradients (rank <= M for lin3, rank <= B for nonlin).
    """
    d, B = X.shape
    Y = target_fn(model, target, X)

    if model == "lin3":
        N = params["W1"].shape[0]
        W0, W1, W2 = params["W0"], params["W1"], params["W2"]
        H1 = W0 @ X / math.sqrt(d)                     # (N, B)
        H2 = W1 @ H1 / math.sqrt(N)                    # (N, B)
        E = W2 @ H2 / (gamma0 * N) - Y                 # (M, B)
        loss = 0.5 * jnp.mean(jnp.sum(E**2, axis=0))
        G2 = E @ H2.T / (B * gamma0 * N)               # (M, N)
        if opt == "sgd":
            dH2 = W2.T @ E / (gamma0 * N)              # (N, B)
            G1 = dH2 @ H1.T / (B * math.sqrt(N))       # (N, N), rank <= M
            dH1 = W1.T @ dH2 / math.sqrt(N)            # (N, B)
            G0 = dH1 @ X.T / (B * math.sqrt(d))        # (N, d), rank <= M
            new = {"W0": W0 - lr_sgd * G0, "W1": W1 - lr_sgd * G1,
                   "W2": W2 - lr_sgd * G2}
        else:
            # G1 = L1 @ R1.T, L1 = W2.T (N, M), R1 = H1 @ E.T (N, M)
            # G0 = L0 @ R0.T, L0 = (W2 @ W1).T (N, M), R0 = X @ E.T (d, M)
            M1 = msign_from_factors(W2.T, H1 @ E.T)
            M0 = msign_from_factors((W2 @ W1).T, X @ E.T)
            new = {
                "W1": W1 - eta_muon * muon_scale(N, N) * M1,
                "W0": W0 - eta_muon * muon_scale(N, d) * M0,
                "W2": W2 - lr_sgd * G2,
            }
        return new, loss

    if model == "nonlin2":
        N = params["W0"].shape[0]
        W0, W1 = params["W0"], params["W1"]
        Z = W0 @ X / math.sqrt(d)                      # (N, B)
        H = jax.nn.relu(Z)
        E = W1 @ H / (gamma0 * N) - Y                  # (M, B)
        loss = 0.5 * jnp.mean(jnp.sum(E**2, axis=0))
        G1 = E @ H.T / (B * gamma0 * N)                # (M, N)
        delta = (W1.T @ E) * (Z > 0) / (gamma0 * N)    # (N, B)
        if opt == "sgd":
            G0 = delta @ X.T / (B * math.sqrt(d))      # (N, d), rank <= B
            new = {"W0": W0 - lr_sgd * G0, "W1": W1 - lr_sgd * G1}
        else:
            M0 = msign_from_factors(delta, X)
            new = {"W0": W0 - eta_muon * muon_scale(N, d) * M0,
                   "W1": W1 - lr_sgd * G1}
        return new, loss

    if model == "nonlin3":
        N = params["W1"].shape[0]
        W0, W1, W2 = params["W0"], params["W1"], params["W2"]
        Z0 = W0 @ X / math.sqrt(d)
        H0 = jax.nn.relu(Z0)                           # (N, B)
        Z1 = W1 @ H0 / math.sqrt(N)
        H1 = jax.nn.relu(Z1)                           # (N, B)
        E = W2 @ H1 / (gamma0 * N) - Y                 # (M, B)
        loss = 0.5 * jnp.mean(jnp.sum(E**2, axis=0))
        G2 = E @ H1.T / (B * gamma0 * N)               # (M, N)
        d1 = (W2.T @ E) * (Z1 > 0) / (gamma0 * N)      # (N, B)
        d0 = (W1.T @ d1 / math.sqrt(N)) * (Z0 > 0)     # (N, B)
        if opt == "sgd":
            G1 = d1 @ H0.T / (B * math.sqrt(N))        # (N, N), rank <= B
            G0 = d0 @ X.T / (B * math.sqrt(d))         # (N, d), rank <= B
            new = {"W0": W0 - lr_sgd * G0, "W1": W1 - lr_sgd * G1,
                   "W2": W2 - lr_sgd * G2}
        else:
            M1 = msign_from_factors(d1, H0)
            M0 = msign_from_factors(d0, X)
            new = {
                "W1": W1 - eta_muon * muon_scale(N, N) * M1,
                "W0": W0 - eta_muon * muon_scale(N, d) * M0,
                "W2": W2 - lr_sgd * G2,
            }
        return new, loss

    raise ValueError(f"unknown model {model!r}")


@partial(jax.jit, static_argnames=("model", "opt", "n_steps", "batch_size",
                                   "gamma0", "lr_sgd", "eta_muon"))
def train_chunk(model: str, opt: str, params: dict, target: dict,
                train_key: jnp.ndarray, step_offset: jnp.ndarray,
                n_steps: int, batch_size: int, gamma0: float,
                lr_sgd: float, eta_muon: float):
    """Run n_steps online steps; fresh X per step keyed by global step index.

    Returns (params, losses) with losses of shape (n_steps,).
    """
    d = (target["A"] if model == "lin3" else target["B"]).shape[1]

    def body(carry, i):
        prm, = carry
        key_t = jax.random.fold_in(train_key, step_offset + i)
        X = jax.random.normal(key_t, (d, batch_size))
        prm, loss = _step(model, opt, prm, target, X, gamma0, lr_sgd, eta_muon)
        return (prm,), loss

    (params,), losses = jax.lax.scan(body, (params,), jnp.arange(n_steps))
    return params, losses


def eval_loss(model: str, params: dict, target: dict, eval_key: jnp.ndarray,
              checkpoint_index: int, batch_size: int, gamma0: float) -> float:
    """Loss on a fresh eval batch; eval stream shared across all runs."""
    d = (target["A"] if model == "lin3" else target["B"]).shape[1]
    key = jax.random.fold_in(eval_key, checkpoint_index)
    X = jax.random.normal(key, (d, batch_size))
    return float(batch_loss(model, params, target, X, gamma0))


# ------------------------------------------------------------------- spectra
def normalized_svals(W, init_var: float = 1.0):
    """Returns (sv sorted desc, sv_norm, aspect, mp_edge).

    sv_norm = sv / sqrt(cols * init_var); aspect = rows / cols;
    mp_edge = 1 + sqrt(aspect).
    """
    W = np.asarray(W)
    rows, cols = W.shape
    sv = np.linalg.svd(W, compute_uv=False)
    aspect = rows / cols
    return sv, sv / math.sqrt(cols * init_var), aspect, 1.0 + math.sqrt(aspect)
