"""Plot toy-spectra results: MP-normalized spectra by P, outlier-fraction-vs-P
width-collapse curves (the key SGD-vs-Muon comparison), and loss curves.

Reads run directories produced by scripts/toy_spectra/run_toy_spectra.py
(metadata.json + spectra_{P}.npz + metrics.jsonl). Conventions follow
scripts/plot_resnet_block_spectra.py: normalized sigma = sigma/sqrt(cols),
MP edge = 1 + sqrt(rows/cols), outlier fraction = #{sigma_norm > edge} / N.

Usage:
  uv run python scripts/plot_toy_spectra.py \
    --results-root /path/to/toy_spectra_results \
    --output-dir artifacts/toy_spectra_plots [--job-suffix job1234567]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

STRIP_EDGE_FACTOR = 1.02  # display-only margin for the outlier strip


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--run-glob", type=str, default="toy_*")
    p.add_argument("--job-suffix", type=str, default="",
                   help="e.g. job1234567; restrict to run dirs ending with it.")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--optimizers", nargs="*", default=None)
    p.add_argument("--widths", nargs="*", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--loss-vs-width", action="store_true",
                   help="Also emit eval-loss-vs-width-by-P figures.")
    p.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    return p.parse_args(argv)


# ----------------------------------------------------------------- discovery
def discover_runs(args) -> dict:
    """Returns {(model, optimizer): [run dicts]} keyed from metadata.json."""
    root = Path(args.results_root)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run_dir in sorted(root.glob(args.run_glob)):
        meta_path = run_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        if args.job_suffix and not run_dir.name.endswith(args.job_suffix):
            continue
        meta = json.loads(meta_path.read_text())
        if args.models and meta["model"] not in args.models:
            continue
        if args.optimizers and meta["optimizer"] not in args.optimizers:
            continue
        if args.widths and meta["N"] not in args.widths:
            continue
        if args.seeds and meta["seed"] not in args.seeds:
            continue
        groups[(meta["model"], meta["optimizer"])].append(
            {"dir": run_dir, "meta": meta, "N": meta["N"],
             "seed": meta["seed"], "p_targets": meta["p_targets"],
             "layers": meta["tracked_layers"]})
    return groups


def load_spectra(run: dict) -> dict:
    """{P: npz dict} for one run (lazy-loaded once, kept as numpy arrays)."""
    if "spectra" not in run:
        spectra = {}
        for p in run["p_targets"]:
            path = run["dir"] / f"spectra_{p}.npz"
            if path.is_file():
                with np.load(path, allow_pickle=False) as z:
                    spectra[p] = {k: np.array(z[k]) for k in z.files
                                  if k.startswith(("sv_norm_", "mp_edge_",
                                                   "aspect_", "eval_loss"))}
        run["spectra"] = spectra
    return run["spectra"]


def load_metrics(run: dict) -> list[dict]:
    if "metrics" not in run:
        path = run["dir"] / "metrics.jsonl"
        rows = []
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        run["metrics"] = rows
    return run["metrics"]


# -------------------------------------------------------------------- pieces
def mp_density(xs: np.ndarray, aspect: float) -> np.ndarray:
    """Density of normalized singular values of a variance-1 iid matrix."""
    a = max(aspect, 1.0 / aspect) if aspect < 1 else aspect
    lo, hi = (math.sqrt(a) - 1) ** 2, (math.sqrt(a) + 1) ** 2
    x2 = xs**2
    mask = (x2 > lo) & (x2 < hi) & (xs > 0)
    rho = np.zeros_like(xs)
    rho[mask] = np.sqrt((hi - x2[mask]) * (x2[mask] - lo)) / (np.pi * xs[mask])
    if aspect < 1:
        # rows < cols: only `rows` nonzero singular values; same density in
        # the normalized variable up to the aspect flip.
        rho = rho / aspect
        rho = rho * aspect  # keep unit mass over the nonzero spectrum
    return rho


def format_p(p: int) -> str:
    if p >= 1_000_000:
        return f"{p / 1e6:g}M"
    if p >= 1_000:
        return f"{p / 1e3:g}k"
    return str(p)


def width_colors(widths: list[int]) -> dict[int, tuple]:
    cmap = plt.get_cmap("viridis")
    return {n: cmap(i / max(1, len(widths) - 1))
            for i, n in enumerate(sorted(widths))}


def spectra_panel(ax_top, ax_bot, data_by_width: dict[int, np.ndarray],
                  edge_by_width: dict[int, float],
                  aspect_by_width: dict[int, float], title: str, bins: int):
    """One checkpoint panel: histogram + MP overlay on top, outlier strip below."""
    widths = sorted(data_by_width)
    colors = width_colors(widths)
    allv = np.concatenate([data_by_width[n] for n in widths])
    bulk_max = max(max(edge_by_width.values()) * 1.2,
                   float(np.quantile(allv, 0.995)) * 1.05)
    xmax = min(float(allv.max()) * 1.05 + 1e-9, bulk_max * 4)
    xmax = max(xmax, bulk_max)
    xs = np.linspace(1e-3, xmax, 400)
    distinct_aspects = len({round(a, 3) for a in aspect_by_width.values()}) > 1
    for n in widths:
        ax_top.hist(data_by_width[n], bins=np.linspace(0, xmax, bins),
                    density=True, histtype="step", color=colors[n],
                    label=f"W={n}")
        ax_top.axvline(edge_by_width[n], color=colors[n], ls="--", lw=0.8)
        ax_bot.axvline(edge_by_width[n], color=colors[n], ls="--", lw=0.8)
        if distinct_aspects:
            ax_top.plot(xs, mp_density(xs, aspect_by_width[n]),
                        color=colors[n], lw=0.8, alpha=0.6)
    if not distinct_aspects:
        ax_top.plot(xs, mp_density(xs, aspect_by_width[widths[-1]]),
                    "k-", lw=1.2, label="MP")
    for i, n in enumerate(widths):
        sv = data_by_width[n]
        tail = sv[sv > edge_by_width[n] * STRIP_EDGE_FACTOR]
        ax_bot.plot(tail, np.full_like(tail, i), "o", ms=2.0, color=colors[n],
                    alpha=0.7)
    ax_bot.set_yticks(range(len(widths)))
    ax_bot.set_yticklabels([str(n) for n in widths], fontsize=6)
    ax_bot.set_ylim(-0.7, len(widths) - 0.3)
    ax_bot.set_xlim(0, xmax)
    ax_top.set_xlim(0, xmax)
    ax_top.set_title(title, fontsize=9)


# -------------------------------------------------------------------- plots
def plot_spectra_by_p(runs: list[dict], model: str, opt: str, layer: str,
                      out_dir: Path, bins: int, fmt: str):
    p_targets = sorted({p for run in runs for p in load_spectra(run)})
    if not p_targets:
        return
    ncols = 4
    nrows = math.ceil(len(p_targets) / ncols)
    fig, axs = plt.subplots(
        2 * nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows),
        gridspec_kw={"height_ratios": [3, 1] * nrows}, squeeze=False)
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax_top, ax_bot = axs[2 * r, c], axs[2 * r + 1, c]
        if idx >= len(p_targets):
            ax_top.axis("off")
            ax_bot.axis("off")
            continue
        p = p_targets[idx]
        data, edges, aspects = {}, {}, {}
        for run in runs:
            spec = load_spectra(run).get(p)
            if spec is None or f"sv_norm_{layer}" not in spec:
                continue
            n = run["N"]
            data.setdefault(n, []).append(spec[f"sv_norm_{layer}"])
            edges[n] = float(spec[f"mp_edge_{layer}"])
            aspects[n] = float(spec[f"aspect_{layer}"])
        if not data:
            ax_top.axis("off")
            ax_bot.axis("off")
            continue
        pooled = {n: np.concatenate(vals) for n, vals in data.items()}
        spectra_panel(ax_top, ax_bot, pooled, edges, aspects,
                      f"P={format_p(p)}", bins)
        if idx == 0:
            ax_top.legend(fontsize=6)
            ax_top.set_ylabel("Density")
        if c == 0:
            ax_bot.set_ylabel("width", fontsize=7)
        if r == nrows - 1:
            ax_bot.set_xlabel("MP-normalized singular value", fontsize=8)
    fig.suptitle(f"{model} {opt.upper()} layer {layer}: MP-normalized spectra "
                 f"by samples seen", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"{model}_{opt}_{layer}_spectra_by_p.{fmt}", dpi=200)
    plt.close(fig)


def outlier_stats(runs: list[dict], layer: str):
    """{N: (p_array, mean_frac, sem_frac, mean_count, sem_count)}."""
    per_width: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for p, spec in load_spectra(run).items():
            if f"sv_norm_{layer}" not in spec:
                continue
            count = int(np.sum(spec[f"sv_norm_{layer}"]
                               > float(spec[f"mp_edge_{layer}"])))
            per_width[run["N"]][p].append(count)
    stats = {}
    for n, by_p in per_width.items():
        ps = np.array(sorted(by_p))
        counts = [np.asarray(by_p[p], float) for p in ps]
        mean_c = np.array([c.mean() for c in counts])
        sem_c = np.array([c.std(ddof=1) / math.sqrt(len(c)) if len(c) > 1
                          else 0.0 for c in counts])
        stats[n] = (ps, mean_c / n, sem_c / n, mean_c, sem_c)
    return stats


def _plot_outlier_axis(ax, stats: dict, normalized: bool):
    colors = width_colors(sorted(stats))
    for n in sorted(stats):
        ps, frac, frac_sem, count, count_sem = stats[n]
        y, yerr = (frac, frac_sem) if normalized else (count, count_sem)
        pos = ps > 0
        ax.errorbar(np.maximum(ps, 1)[pos], y[pos], yerr=yerr[pos],
                    marker="o", ms=3, lw=1.2, capsize=2, color=colors[n],
                    label=f"W={n}")
    ax.set_xscale("log")
    ax.set_xlabel("P (samples seen)")
    ax.grid(alpha=0.3)


def plot_outlier_fraction(runs: list[dict], model: str, opt: str, layer: str,
                          out_dir: Path, fmt: str):
    stats = outlier_stats(runs, layer)
    if not stats:
        return
    for normalized, stem, ylabel in [
            (True, "outlier_fraction", "Outlier fraction above MP edge"),
            (False, "outlier_count", "Outlier count above MP edge")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        _plot_outlier_axis(ax, stats, normalized)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model} {opt.upper()} layer {layer}: "
                     f"{stem.replace('_', ' ')} vs P by width")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_{opt}_{layer}_{stem}_by_p.{fmt}",
                    dpi=200)
        plt.close(fig)


def plot_outlier_fraction_combined(groups: dict, model: str, layer: str,
                                   out_dir: Path, fmt: str):
    stats_by_opt = {}
    for opt in ("sgd", "muon"):
        runs = groups.get((model, opt))
        if runs:
            stats = outlier_stats(runs, layer)
            if stats:
                stats_by_opt[opt] = stats
    if len(stats_by_opt) < 2:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (opt, stats) in zip(axs, stats_by_opt.items()):
        _plot_outlier_axis(ax, stats, normalized=True)
        ax.set_title(opt.upper())
        ax.legend(fontsize=7)
    axs[0].set_ylabel("Outlier fraction above MP edge")
    fig.suptitle(f"{model} layer {layer}: outlier fraction vs P by width",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"{model}_{layer}_outlier_fraction_sgd_vs_muon.{fmt}",
                dpi=200)
    plt.close(fig)


def plot_losses(runs: list[dict], model: str, opt: str, out_dir: Path,
                fmt: str):
    colors = width_colors(sorted({r["N"] for r in runs}))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for run in runs:
        rows = load_metrics(run)
        n = run["N"]
        train = [(r["samples_seen"], r["train_loss_ema"]) for r in rows
                 if not r.get("checkpoint") and r["samples_seen"] > 0]
        evals = [(r["samples_seen"], r["eval_loss"]) for r in rows
                 if r.get("checkpoint") and r["samples_seen"] > 0]
        if train:
            xs, ys = zip(*train)
            axs[0].plot(xs, ys, lw=0.8, alpha=0.7, color=colors[n])
        if evals:
            xs, ys = zip(*evals)
            axs[1].plot(xs, ys, lw=1.0, marker="o", ms=2.5, alpha=0.8,
                        color=colors[n])
    for ax, title in zip(axs, ["train loss (EMA)", "eval loss"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("P (samples seen)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=c, label=f"W={n}")
               for n, c in sorted(colors.items())]
    axs[0].legend(handles=handles, fontsize=7)
    fig.suptitle(f"{model} {opt.upper()}: loss vs samples seen", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"{model}_{opt}_loss_by_width.{fmt}", dpi=200)
    plt.close(fig)


def plot_loss_vs_width(runs: list[dict], model: str, opt: str, out_dir: Path,
                       fmt: str):
    by_p: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for p, spec in load_spectra(run).items():
            if p > 0 and "eval_loss" in spec:
                by_p[p][run["N"]].append(float(spec["eval_loss"]))
    if not by_p:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cmap = plt.get_cmap("viridis")
    ps = sorted(by_p)
    for i, p in enumerate(ps):
        widths = sorted(by_p[p])
        means = [float(np.mean(by_p[p][n])) for n in widths]
        ax.plot(widths, means, marker="o", ms=3, lw=1.1,
                color=cmap(i / max(1, len(ps) - 1)), label=f"P={format_p(p)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Width N")
    ax.set_ylabel("eval loss")
    ax.set_title(f"{model} {opt.upper()}: eval loss vs width by P")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model}_{opt}_loss_vs_width_by_p.{fmt}", dpi=200)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    groups = discover_runs(args)
    if not groups:
        raise SystemExit(f"No runs found under {args.results_root} "
                         f"(glob {args.run_glob!r}, suffix {args.job_suffix!r})")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({m for m, _ in groups})
    for (model, opt), runs in sorted(groups.items()):
        layers = runs[0]["layers"]
        print(f"{model}/{opt}: {len(runs)} runs, layers {layers}")
        for layer in layers:
            plot_spectra_by_p(runs, model, opt, layer, out_dir, args.bins,
                              args.format)
            plot_outlier_fraction(runs, model, opt, layer, out_dir, args.format)
        plot_losses(runs, model, opt, out_dir, args.format)
        if args.loss_vs_width:
            plot_loss_vs_width(runs, model, opt, out_dir, args.format)
    for model in models:
        runs = groups.get((model, "sgd")) or groups.get((model, "muon"))
        for layer in runs[0]["layers"]:
            plot_outlier_fraction_combined(groups, model, layer, out_dir,
                                           args.format)
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
