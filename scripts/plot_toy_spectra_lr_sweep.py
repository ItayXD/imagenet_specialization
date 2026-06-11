"""LR-tuning plots for toy_spectra runs: final eval loss vs learning rate,
one curve per width, one panel per optimizer — the muP-transfer diagnostic
(optimum should sit at the same lr across widths).

Reads run dirs produced by run_toy_spectra.py from an LR-grid manifest
(run_ids like toy_lin3_sgd_N128_s0_lr0.01 / toy_lin3_muon_N128_s0_eta0.01).
The swept variable is read from metadata.json: `lr` for sgd, `eta_muon` for
muon. Diverged runs (metrics.jsonl ends with a diverged row) are drawn as x
markers at the top of the axis.

Usage:
  uv run python scripts/plot_toy_spectra_lr_sweep.py \
    --results-root artifacts/toy_spectra_lr_sweep --output-dir artifacts/...
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--run-glob", type=str, default="toy_*")
    p.add_argument("--job-suffix", type=str, default="")
    p.add_argument("--format", type=str, default="png", choices=["pdf", "png"])
    return p.parse_args(argv)


def collect(args):
    """{model: {optimizer: {N: [(lr, final_eval_loss or nan-if-diverged)]}}}"""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run_dir in sorted(Path(args.results_root).glob(args.run_glob)):
        meta_path = run_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        if args.job_suffix and not run_dir.name.endswith(args.job_suffix):
            continue
        meta = json.loads(meta_path.read_text())
        rows = [json.loads(line) for line in
                (run_dir / "metrics.jsonl").read_text().splitlines()
                if line.strip()]
        diverged = any(r.get("diverged") for r in rows)
        ckpts = [r for r in rows if r.get("checkpoint")]
        final = float("nan")
        if not diverged and ckpts:
            final = ckpts[-1]["eval_loss"]
        lr = meta["eta_muon"] if meta["optimizer"] == "muon" else meta["lr"]
        data[meta["model"]][meta["optimizer"]][meta["N"]].append(
            (float(lr), final))
    return data


def main(argv=None) -> None:
    args = parse_args(argv)
    data = collect(args)
    if not data:
        raise SystemExit(f"No runs found under {args.results_root}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    for model, by_opt in sorted(data.items()):
        opts = sorted(by_opt)
        fig, axs = plt.subplots(1, len(opts), figsize=(5.5 * len(opts), 4.2),
                                squeeze=False)
        for ax, opt in zip(axs[0], opts):
            widths = sorted(by_opt[opt])
            colors = {n: cmap(i / max(1, len(widths) - 1))
                      for i, n in enumerate(widths)}
            finite_vals = [v for n in widths for _, v in by_opt[opt][n]
                           if np.isfinite(v)]
            top = max(finite_vals) * 2 if finite_vals else 1.0
            for n in widths:
                pts = sorted(by_opt[opt][n])
                lrs = np.array([p[0] for p in pts])
                vals = np.array([p[1] for p in pts])
                ok = np.isfinite(vals)
                ax.plot(lrs[ok], vals[ok], marker="o", ms=4, lw=1.3,
                        color=colors[n], label=f"W={n}")
                if (~ok).any():
                    ax.plot(lrs[~ok], np.full((~ok).sum(), top), "x", ms=7,
                            color=colors[n])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("eta_muon" if opt == "muon" else "lr (eta_sgd)")
            ax.set_title(opt.upper())
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axs[0][0].set_ylabel("Final eval loss")
        fig.suptitle(f"{model}: LR tuning across widths "
                     f"(x = diverged)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / f"{model}_lr_sweep.{args.format}", dpi=200)
        plt.close(fig)
    print(f"LR-sweep plots written to {out_dir}")


if __name__ == "__main__":
    main()
