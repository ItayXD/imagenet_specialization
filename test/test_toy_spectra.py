import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.toy_spectra import core
from scripts.toy_spectra import build_toy_spectra_manifest as manifest_builder
from scripts.toy_spectra import run_toy_spectra


def test_msign_thin_matches_full_polar():
    rng = np.random.default_rng(0)
    L = jnp.asarray(rng.standard_normal((24, 3)), jnp.float32)
    R = jnp.asarray(rng.standard_normal((16, 3)), jnp.float32)
    M = np.asarray(core.msign_thin(L, R))
    G = np.asarray(L) @ np.asarray(R).T
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    # G has rank 3; compare against the polar factor restricted to its range.
    np.testing.assert_allclose(M, U[:, :3] @ Vt[:3, :], atol=1e-4)
    sv = np.linalg.svd(M, compute_uv=False)
    np.testing.assert_allclose(sv[:3], 1.0, atol=1e-4)
    np.testing.assert_allclose(sv[3:], 0.0, atol=1e-4)


def test_msign_from_factors_wide_fallback():
    rng = np.random.default_rng(1)
    # L wide (rows < cols): falls back to SVD of the product.
    L = jnp.asarray(rng.standard_normal((8, 32)), jnp.float32)
    R = jnp.asarray(rng.standard_normal((16, 32)), jnp.float32)
    M = np.asarray(core.msign_from_factors(L, R))
    sv = np.linalg.svd(M, compute_uv=False)
    np.testing.assert_allclose(sv, 1.0, atol=1e-4)


def test_muon_update_spectral_norm_width_independent():
    rng = np.random.default_rng(2)
    eta = 0.01
    for rows, cols in [(16, 8), (64, 32), (128, 64)]:
        G = jnp.asarray(rng.standard_normal((rows, cols)), jnp.float32)
        update = eta * core.muon_scale(rows, cols) * np.asarray(core.msign_full(G))
        spec_norm = np.linalg.svd(update, compute_uv=False)[0]
        normalized = spec_norm / np.sqrt(cols)
        expected = eta * np.sqrt(max(1.0, rows / cols))
        np.testing.assert_allclose(normalized, expected, rtol=1e-4)


@pytest.mark.parametrize("model", core.MODELS)
def test_mup_sgd_function_update_width_consistent(model):
    d, M, B, gamma0, eta = 32, 4, 64, 3.0, 0.01
    target = core.make_target(model, 123, d, M)
    X = jax.random.normal(jax.random.PRNGKey(7), (d, B))
    deltas = {}
    for N in (64, 256):
        params = core.init_params(model, jax.random.PRNGKey(0), N, d, M)
        f0 = core.forward(model, params, X, gamma0)
        new, _ = core.train_chunk(model, "sgd", params, target,
                                  jax.random.PRNGKey(9), 0, 1, B, gamma0,
                                  eta * gamma0**2 * N, 0.0)
        f1 = core.forward(model, new, X, gamma0)
        deltas[N] = float(jnp.sqrt(jnp.mean((f1 - f0) ** 2)))
    ratio = deltas[256] / deltas[64]
    assert 0.25 < ratio < 4.0, f"Delta-f ratio {ratio} not width-consistent"


def test_make_p_targets():
    targets = core.make_p_targets(10_000, 2_048_000, 15, 512)
    assert targets[0] == 0
    assert targets[-1] == 2_048_000
    assert all(t % 512 == 0 for t in targets)
    assert targets == sorted(set(targets))
    assert len(targets) <= 16
    assert targets[1] >= 10_000


def test_aspect_and_edge_bookkeeping():
    rng = np.random.default_rng(3)
    sv, sv_norm, aspect, edge = core.normalized_svals(
        rng.standard_normal((64, 16)))
    assert aspect == 4.0
    assert edge == 3.0
    assert sv_norm.shape == (16,)
    _, sv_norm, aspect, edge = core.normalized_svals(
        rng.standard_normal((8, 64)))
    assert aspect == 0.125
    # A pure-noise matrix should not exceed the MP edge by much.
    _, sv_norm, _, edge = core.normalized_svals(
        rng.standard_normal((256, 128)))
    assert sv_norm.max() < edge * 1.3


def test_target_is_seed_stable_across_widths_and_seeds():
    t1 = core.make_target("nonlin2", 4242, 32, 4)
    t2 = core.make_target("nonlin2", 4242, 32, 4)
    for k in t1:
        np.testing.assert_array_equal(np.asarray(t1[k]), np.asarray(t2[k]))
    p1 = core.init_params("nonlin2", jax.random.PRNGKey(0), 16, 32, 4)
    p2 = core.init_params("nonlin2", jax.random.PRNGKey(1), 16, 32, 4)
    assert not np.allclose(np.asarray(p1["W0"]), np.asarray(p2["W0"]))


@pytest.mark.parametrize("model", core.MODELS)
@pytest.mark.parametrize("optimizer", core.OPTIMIZERS)
def test_end_to_end_smoke(model, optimizer, tmp_path):
    argv = [
        "--model", model, "--optimizer", optimizer, "--N", "32",
        "--seed", "0", "--lr", "0.01", "--eta-muon", "0.01",
        "--d", "16", "--M", "4", "--batch-size", "8",
        "--p-targets", "16,48", "--eval-batch-size", "64",
        "--log-every", "1", "--output-dir", str(tmp_path),
    ]
    run_toy_spectra.main(argv)
    run_dir = tmp_path / f"toy_{model}_{optimizer}_N32_s0"
    assert run_dir.is_dir()
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["p_targets"] == [0, 16, 48]
    assert meta["tracked_layers"] == list(core.TRACKED_LAYERS[model])
    rows = [json.loads(line) for line in
            (run_dir / "metrics.jsonl").read_text().splitlines()]
    ckpt_rows = [r for r in rows if r.get("checkpoint")]
    assert len(ckpt_rows) == 3
    assert all(np.isfinite(r["eval_loss"]) for r in ckpt_rows)
    for p in (0, 16, 48):
        with np.load(run_dir / f"spectra_{p}.npz") as z:
            for layer in core.TRACKED_LAYERS[model]:
                sv_norm = z[f"sv_norm_{layer}"]
                rows_, cols_ = z[f"shape_{layer}"]
                assert sv_norm.shape == (min(rows_, cols_),)
                assert np.all(np.isfinite(sv_norm))
                assert float(z[f"mp_edge_{layer}"]) == pytest.approx(
                    1.0 + np.sqrt(rows_ / cols_))
            assert np.isfinite(float(z["eval_loss"]))


def test_manifest_builder_row_count_and_schema(tmp_path):
    out = tmp_path / "manifest.csv"
    manifest_builder.main([
        "--output", str(out), "--models", "lin3", "--optimizers",
        "sgd", "muon", "--widths", "64", "128", "--seeds", "0",
    ])
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert list(rows[0].keys()) == manifest_builder.FIELDNAMES
    targets = [int(v) for v in rows[0]["p_targets"].split(",")]
    assert targets == core.make_p_targets()
    sgd_rows = [r for r in rows if r["optimizer"] == "sgd"]
    assert all(r["eta_muon"] == "" for r in sgd_rows)


def test_manifest_mode_resolves_row(tmp_path):
    out = tmp_path / "manifest.csv"
    manifest_builder.main([
        "--output", str(out), "--models", "nonlin2", "--optimizers", "muon",
        "--widths", "16", "--seeds", "3", "--d", "8", "--M", "2",
        "--batch-size", "4", "--p-min", "8", "--p-max", "16", "--p-num", "2",
        "--eval-batch-size", "16",
    ])
    args = run_toy_spectra.resolve(run_toy_spectra.parse_args([
        "--manifest", str(out), "--index", "0",
        "--output-dir", str(tmp_path),
    ]))
    assert (args.model, args.optimizer, args.N, args.seed) == \
        ("nonlin2", "muon", 16, 3)
    assert args.run_id == "toy_nonlin2_muon_N16_s3"
    assert args.p_target_list[0] == 0
    assert all(t % 4 == 0 for t in args.p_target_list)
