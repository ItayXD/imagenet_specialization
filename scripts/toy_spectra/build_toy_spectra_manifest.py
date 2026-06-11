"""Build the toy-spectra sweep manifest (one CSV row per training run).

Default sweep: models x {sgd, muon} x widths x seeds, fixed d/M/target.
With --lr-grid, instead builds an LR-tuning pilot at the smallest width,
seed 0 only, sweeping both eta_sgd and eta_muon over the grid.

Usage:
  uv run python scripts/toy_spectra/build_toy_spectra_manifest.py \
    --output conf/toy_spectra_manifest.csv
  uv run python scripts/toy_spectra/build_toy_spectra_manifest.py \
    --output conf/toy_spectra_lr_pilot_manifest.csv \
    --lr-grid 0.003,0.01,0.03,0.1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.toy_spectra import core

FIELDNAMES = [
    "job_id", "run_id", "model", "optimizer", "N", "seed", "lr", "eta_muon",
    "d", "M", "gamma0", "batch_size", "target_seed", "p_min", "p_max",
    "p_num", "p_targets", "eval_batch_size",
]

# Tuned by the N=128 LR pilot (job 21586270): best final eval loss, ties
# broken toward the smallest LR on the plateau.
DEFAULT_LR_SGD = {"lin3": 0.01, "nonlin2": 0.1, "nonlin3": 0.1}
DEFAULT_ETA_MUON = {"lin3": 0.003, "nonlin2": 0.003, "nonlin3": 0.003}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=str, default="conf/toy_spectra_manifest.csv")
    p.add_argument("--models", nargs="+", default=list(core.MODELS),
                   choices=core.MODELS)
    p.add_argument("--optimizers", nargs="+", default=list(core.OPTIMIZERS),
                   choices=core.OPTIMIZERS)
    p.add_argument("--widths", nargs="+", type=int,
                   default=[128, 256, 512, 1024])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    for model in core.MODELS:
        p.add_argument(f"--lr-sgd-{model}", type=float,
                       default=DEFAULT_LR_SGD[model])
        p.add_argument(f"--eta-muon-{model}", type=float,
                       default=DEFAULT_ETA_MUON[model])
    p.add_argument("--lr-grid", type=str, default="",
                   help="Comma-separated grid; if set, build an LR pilot "
                        "(smallest width, seed 0) instead of the sweep.")
    p.add_argument("--d", type=int, default=core.D_DEFAULT)
    p.add_argument("--M", type=int, default=core.M_DEFAULT)
    p.add_argument("--gamma0", type=float, default=core.GAMMA0_DEFAULT)
    p.add_argument("--batch-size", type=int, default=core.BATCH_SIZE_DEFAULT)
    p.add_argument("--target-seed", type=int, default=core.TARGET_SEED_DEFAULT)
    p.add_argument("--p-min", type=int, default=10_000)
    p.add_argument("--p-max", type=int, default=2_048_000)
    p.add_argument("--p-num", type=int, default=15)
    p.add_argument("--eval-batch-size", type=int, default=8192)
    return p.parse_args(argv)


def build_rows(args: argparse.Namespace) -> list[dict]:
    p_targets = core.make_p_targets(args.p_min, args.p_max, args.p_num,
                                    args.batch_size)
    common = {
        "d": args.d, "M": args.M, "gamma0": args.gamma0,
        "batch_size": args.batch_size, "target_seed": args.target_seed,
        "p_min": args.p_min, "p_max": args.p_max, "p_num": args.p_num,
        "p_targets": ",".join(str(t) for t in p_targets),
        "eval_batch_size": args.eval_batch_size,
    }

    rows: list[dict] = []

    def add(model, optimizer, N, seed, lr, eta_muon, run_id=None):
        rows.append({
            "job_id": len(rows),
            "run_id": run_id or f"toy_{model}_{optimizer}_N{N}_s{seed}",
            "model": model, "optimizer": optimizer, "N": N, "seed": seed,
            "lr": lr, "eta_muon": eta_muon if optimizer == "muon" else "",
            **common,
        })

    if args.lr_grid:
        grid = [float(v) for v in args.lr_grid.split(",")]
        N, seed = min(args.widths), 0
        for model in args.models:
            for lr in grid:
                if "sgd" in args.optimizers:
                    add(model, "sgd", N, seed, lr,
                        "", run_id=f"toy_{model}_sgd_N{N}_s{seed}_lr{lr:g}")
                if "muon" in args.optimizers:
                    add(model, "muon", N, seed,
                        getattr(args, f"lr_sgd_{model}"), lr,
                        run_id=f"toy_{model}_muon_N{N}_s{seed}_eta{lr:g}")
    else:
        for model in args.models:
            for optimizer in args.optimizers:
                for N in args.widths:
                    for seed in args.seeds:
                        add(model, optimizer, N, seed,
                            getattr(args, f"lr_sgd_{model}"),
                            getattr(args, f"eta_muon_{model}"))
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    rows = build_rows(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
