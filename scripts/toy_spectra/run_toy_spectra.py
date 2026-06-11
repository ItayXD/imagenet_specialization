"""Train one toy-spectra run and save per-checkpoint weight spectra.

Online training with fresh Gaussian samples, constant lr, MSE loss; saves
metadata.json, metrics.jsonl, and spectra_{P}.npz at log-spaced sample counts
(see scripts/toy_spectra/core.py for the models and conventions).

Usage (direct):
  uv run python scripts/toy_spectra/run_toy_spectra.py \
    --model lin3 --optimizer muon --N 256 --seed 0 \
    --lr 0.004 --eta-muon 0.01 --output-dir outputs/toy_spectra_smoke

Usage (manifest row, as in the SLURM array wrapper):
  uv run python scripts/toy_spectra/run_toy_spectra.py \
    --manifest conf/toy_spectra_manifest.csv --index 3 \
    --output-dir "$TOY_SPECTRA_BASE_SAVE_DIR" --run-id-suffix job123
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.toy_spectra import core


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=str, default="",
                   help="Optional manifest CSV; row values override flags.")
    p.add_argument("--index", type=int, default=-1,
                   help="Row index into --manifest (0-based, excludes header).")
    p.add_argument("--model", choices=core.MODELS, default=None)
    p.add_argument("--optimizer", choices=core.OPTIMIZERS, default=None)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="eta_sgd; effective SGD lr is eta_sgd * gamma0^2 * N.")
    p.add_argument("--eta-muon", type=float, default=float("nan"),
                   help="Muon step size (required iff optimizer=muon).")
    p.add_argument("--d", type=int, default=core.D_DEFAULT)
    p.add_argument("--M", type=int, default=core.M_DEFAULT)
    p.add_argument("--M-ratio", type=float, default=0.0,
                   help="If > 0, extensive output dim: M = int(M_ratio * N).")
    p.add_argument("--gamma0", type=float, default=core.GAMMA0_DEFAULT)
    p.add_argument("--batch-size", type=int, default=core.BATCH_SIZE_DEFAULT)
    p.add_argument("--target-seed", type=int, default=core.TARGET_SEED_DEFAULT)
    p.add_argument("--p-min", type=int, default=10_000)
    p.add_argument("--p-max", type=int, default=2_048_000)
    p.add_argument("--p-num", type=int, default=15)
    p.add_argument("--p-targets", type=str, default="",
                   help="Optional explicit comma-separated checkpoint sample "
                        "counts (overrides --p-min/--p-max/--p-num).")
    p.add_argument("--eval-batch-size", type=int, default=8192)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--run-id", type=str, default="",
                   help="Default: toy_{model}_{optimizer}_N{N}_s{seed}")
    p.add_argument("--run-id-suffix", type=str, default="")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args(argv)


_MANIFEST_FIELDS = {
    "model": str, "optimizer": str, "N": int, "seed": int, "lr": float,
    "eta_muon": float, "d": int, "M": int, "M_ratio": float, "gamma0": float,
    "batch_size": int, "target_seed": int, "p_min": int, "p_max": int,
    "p_num": int, "p_targets": str, "eval_batch_size": int, "run_id": str,
}


def apply_manifest_row(args: argparse.Namespace) -> None:
    rows = list(csv.DictReader(open(args.manifest, newline="")))
    if not (0 <= args.index < len(rows)):
        raise SystemExit(f"--index {args.index} out of range (manifest has "
                         f"{len(rows)} rows)")
    row = rows[args.index]
    for field, cast in _MANIFEST_FIELDS.items():
        if field in row and row[field] != "":
            setattr(args, field.replace("-", "_"), cast(row[field]))


def resolve(args: argparse.Namespace) -> argparse.Namespace:
    if args.manifest:
        apply_manifest_row(args)
    for field in ("model", "optimizer", "N", "seed", "lr"):
        if getattr(args, field) is None:
            raise SystemExit(f"--{field} is required (flag or manifest)")
    if args.optimizer == "muon" and math.isnan(args.eta_muon):
        raise SystemExit("--eta-muon is required when optimizer=muon")
    if args.M_ratio > 0:
        args.M = max(1, int(round(args.M_ratio * args.N)))
    if args.p_targets:
        cleaned = str(args.p_targets).strip().strip("[]")
        targets = sorted({int(v) for v in cleaned.split(",") if v.strip()})
        if targets[0] != 0:
            targets = [0] + targets
        bad = [t for t in targets if t % args.batch_size]
        if bad:
            raise SystemExit(f"p_targets must be multiples of batch_size: {bad}")
        args.p_target_list = targets
    else:
        args.p_target_list = core.make_p_targets(
            args.p_min, args.p_max, args.p_num, args.batch_size)
    if not args.run_id:
        args.run_id = f"toy_{args.model}_{args.optimizer}_N{args.N}_s{args.seed}"
    return args


def save_spectra(run_dir: Path, args, params, samples_seen: int, step: int,
                 train_loss_recent: float, train_loss_ema: float,
                 eval_loss_value: float) -> None:
    payload = {
        "run_id": args.run_id, "model": args.model, "optimizer": args.optimizer,
        "N": np.int32(args.N), "d": np.int32(args.d), "M": np.int32(args.M),
        "seed": np.int32(args.seed), "target_seed": np.int32(args.target_seed),
        "gamma0": np.float32(args.gamma0), "init_var": np.float32(1.0),
        "samples_seen": np.int64(samples_seen), "step": np.int64(step),
        "train_loss_recent": np.float32(train_loss_recent),
        "train_loss_ema": np.float32(train_loss_ema),
        "eval_loss": np.float32(eval_loss_value),
        "eval_batch_size": np.int32(args.eval_batch_size),
        "lr_sgd": np.float32(args.lr * args.gamma0**2 * args.N),
        "eta_muon": np.float32(args.eta_muon),
    }
    for layer in core.TRACKED_LAYERS[args.model]:
        sv, sv_norm, aspect, mp_edge = core.normalized_svals(params[layer])
        payload[f"sv_{layer}"] = sv.astype(np.float32)
        payload[f"sv_norm_{layer}"] = sv_norm.astype(np.float32)
        payload[f"shape_{layer}"] = np.asarray(params[layer].shape, np.int32)
        payload[f"aspect_{layer}"] = np.float32(aspect)
        payload[f"mp_edge_{layer}"] = np.float32(mp_edge)
    np.savez_compressed(run_dir / f"spectra_{samples_seen}.npz", **payload)


def main(argv=None) -> None:
    args = resolve(parse_args(argv))
    t_start = time.time()

    run_name = args.run_id + (f"_{args.run_id_suffix}" if args.run_id_suffix else "")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    lr_sgd = args.lr * args.gamma0**2 * args.N
    metadata = {
        **{k: v for k, v in vars(args).items() if k not in ("manifest", "index")},
        "lr_sgd_effective": lr_sgd,
        "tracked_layers": list(core.TRACKED_LAYERS[args.model]),
        "p_targets": args.p_target_list,
        "run_name": run_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    target = core.make_target(args.model, args.target_seed, args.d, args.M)
    key_master = jax.random.PRNGKey(args.seed)
    key_train = jax.random.fold_in(key_master, 0)
    key_init = jax.random.fold_in(jax.random.fold_in(key_master, 1), args.N)
    key_eval = jax.random.fold_in(jax.random.PRNGKey(args.target_seed), 999)
    params = core.init_params(args.model, key_init, args.N, args.d, args.M)

    metrics_path = run_dir / "metrics.jsonl"
    metrics_file = open(metrics_path, "a")

    def write_row(row: dict) -> None:
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()

    ema = None
    ema_decay = 0.99

    def checkpoint(ckpt_index: int, samples_seen: int, step: int,
                   recent_loss: float) -> None:
        ev = core.eval_loss(args.model, params, target, key_eval, ckpt_index,
                            args.eval_batch_size, args.gamma0)
        save_spectra(run_dir, args, params, samples_seen, step,
                     recent_loss, ema if ema is not None else recent_loss, ev)
        write_row({"samples_seen": samples_seen, "step": step,
                   "train_loss": recent_loss,
                   "train_loss_ema": ema if ema is not None else recent_loss,
                   "eval_loss": ev, "lr": lr_sgd, "checkpoint": True})
        print(f"[{args.run_id}] P={samples_seen} step={step} "
              f"train={recent_loss:.5f} eval={ev:.5f}", flush=True)

    # Checkpoint at init (P=0): bulk should sit on MP.
    init_loss = float(core.batch_loss(
        args.model, params, target,
        jax.random.normal(jax.random.fold_in(key_eval, 10**6),
                          (args.d, args.batch_size)),
        args.gamma0))
    checkpoint(0, 0, 0, init_loss)

    targets = args.p_target_list
    step = 0
    for ckpt_index in range(1, len(targets)):
        n_steps = (targets[ckpt_index] - targets[ckpt_index - 1]) // args.batch_size
        params, losses = core.train_chunk(
            args.model, args.optimizer, params, target, key_train,
            int(step), n_steps, args.batch_size, args.gamma0,
            lr_sgd, float(args.eta_muon))
        losses = np.asarray(losses, dtype=np.float64)
        if not np.all(np.isfinite(losses)):
            first_bad = int(np.argmax(~np.isfinite(losses)))
            raise SystemExit(
                f"[{args.run_id}] non-finite loss at step {step + first_bad}")
        for i, loss in enumerate(losses):
            ema = loss if ema is None else ema_decay * ema + (1 - ema_decay) * loss
            global_step = step + i + 1
            if global_step % args.log_every == 0:
                write_row({"samples_seen": global_step * args.batch_size,
                           "step": global_step, "train_loss": float(loss),
                           "train_loss_ema": float(ema), "lr": lr_sgd})
        step += n_steps
        recent = float(losses[-min(50, len(losses)):].mean())
        checkpoint(ckpt_index, targets[ckpt_index], step, recent)

    metrics_file.close()
    print(f"[{args.run_id}] done in {time.time() - t_start:.1f}s "
          f"({step} steps, {targets[-1]} samples) -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
