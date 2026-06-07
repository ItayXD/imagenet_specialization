#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


REMOTE_FILE_RE = re.compile(
    r"(?P<run_id>[^/]+)/width_(?P<width>\d+)/group_(?P<group>\d+)/(?P<name>metrics\.jsonl|metadata\.json)$"
)


@dataclass
class RunCandidate:
    run_id: str
    width: int
    metrics_paths: dict[int, str] = field(default_factory=dict)
    metadata_paths: dict[int, str] = field(default_factory=dict)

    @property
    def job_id(self) -> int:
        match = re.search(r"(\d+)$", self.run_id)
        if match is None:
            return -1
        return int(match.group(1))

    @property
    def available_groups(self) -> list[int]:
        return sorted(set(self.metrics_paths) | set(self.metadata_paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate loss-vs-width exchangeability plots by reading remote metrics.jsonl "
            "files over ssh."
        )
    )
    parser.add_argument(
        "--ssh-host",
        default="ilavie@cannon",
        help="SSH host used to access the remote results directory.",
    )
    parser.add_argument(
        "--remote-root",
        default="/n/pehlevan_lab/Users/ilavie/imagenet_specialization_results",
        help="Remote directory containing exchangeability_job* run folders.",
    )
    parser.add_argument(
        "--remote-run-glob",
        default="exchangeability_job*",
        help="Remote run glob relative to --remote-root.",
    )
    parser.add_argument(
        "--metric-key",
        choices=["train_loss", "val_loss"],
        required=True,
        help="Which saved loss metric to plot.",
    )
    parser.add_argument(
        "--p-mode",
        choices=["common", "available"],
        default="common",
        help=(
            "Use only P values shared by every selected width, or allow each P to use the "
            "widths where that checkpoint exists."
        ),
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="*",
        default=None,
        help="Optional width filter.",
    )
    parser.add_argument(
        "--manifest-path",
        default="conf/exchangeability_manifest.csv",
        help="Local manifest used to infer the expected number of groups per width.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/exchangeability_loss_vs_width",
        help="Directory where plots and CSV summaries are written.",
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        default=["pdf"],
        help="Plot formats to write, for example png pdf.",
    )
    return parser.parse_args()


def _run_remote(host: str, command: str) -> str:
    if host in {"", "local", "localhost"}:
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-lc", command],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    quoted = shlex.quote(command)
    result = subprocess.run(
        ["ssh", host, f"bash --noprofile --norc -lc {quoted}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _cat_remote_file(host: str, remote_path: str) -> str:
    return _run_remote(host, f"cat {shlex.quote(remote_path)}")


def _load_expected_group_counts(manifest_path: str) -> dict[int, int]:
    path = Path(manifest_path)
    if not path.is_file():
        return {}
    groups_by_width: dict[int, set[int]] = defaultdict(set)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            groups_by_width[int(row["width"])].add(int(row["group_id"]))
    return {width: len(groups) for width, groups in groups_by_width.items()}


def _discover_remote_candidates(
    host: str,
    remote_root: str,
    remote_run_glob: str,
    requested_widths: set[int] | None,
) -> dict[int, list[RunCandidate]]:
    find_command = (
        f"cd {shlex.quote(remote_root)} && "
        f"find {remote_run_glob} \\( -name metrics.jsonl -o -name metadata.json \\) -print | sort"
    )
    output = _run_remote(host, find_command)
    by_run_width: dict[tuple[str, int], RunCandidate] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = REMOTE_FILE_RE.search(line)
        if match is None:
            continue
        run_id = match.group("run_id")
        width = int(match.group("width"))
        group_id = int(match.group("group"))
        name = match.group("name")
        if requested_widths is not None and width not in requested_widths:
            continue
        key = (run_id, width)
        candidate = by_run_width.setdefault(key, RunCandidate(run_id=run_id, width=width))
        remote_path = f"{remote_root.rstrip('/')}/{line}"
        if name == "metrics.jsonl":
            candidate.metrics_paths[group_id] = remote_path
        else:
            candidate.metadata_paths[group_id] = remote_path

    candidates_by_width: dict[int, list[RunCandidate]] = defaultdict(list)
    for candidate in by_run_width.values():
        candidates_by_width[candidate.width].append(candidate)
    for width in candidates_by_width:
        candidates_by_width[width].sort(key=lambda item: item.job_id)
    return candidates_by_width


def _select_candidate(
    candidates: list[RunCandidate],
    expected_groups: int | None,
) -> tuple[RunCandidate, list[str]]:
    warnings: list[str] = []
    if not candidates:
        raise ValueError("No run candidates were provided.")

    def candidate_score(candidate: RunCandidate) -> tuple[int, int, int, int]:
        metrics_count = len(candidate.metrics_paths)
        metadata_count = len(candidate.metadata_paths)
        if expected_groups is not None and expected_groups > 0:
            fully_expected = int(metrics_count == expected_groups)
            completeness = -abs(expected_groups - metrics_count)
        else:
            fully_expected = int(metrics_count > 0 and metrics_count == metadata_count)
            completeness = min(metrics_count, metadata_count)
        return (fully_expected, completeness, metrics_count, candidate.job_id)

    selected = max(candidates, key=candidate_score)
    metrics_count = len(selected.metrics_paths)
    metadata_count = len(selected.metadata_paths)
    if expected_groups is not None and metrics_count != expected_groups:
        warnings.append(
            f"Width {selected.width}: selected {selected.run_id} with {metrics_count}/{expected_groups} metric files."
        )
    elif expected_groups is None and metrics_count != metadata_count:
        warnings.append(
            f"Width {selected.width}: selected {selected.run_id} with {metrics_count} metrics and {metadata_count} metadata files."
        )
    return selected, warnings


def _load_group_metrics(host: str, remote_path: str) -> list[dict[str, float | int]]:
    content = _cat_remote_file(host, remote_path)
    rows: list[dict[str, float | int]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _format_p(images_seen: int) -> str:
    if images_seen >= 1_000_000:
        value = images_seen / 1_000_000.0
        suffix = "M"
    elif images_seen >= 1_000:
        value = images_seen / 1_000.0
        suffix = "k"
    else:
        value = float(images_seen)
        suffix = ""
    if suffix and abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}{suffix}"
    if suffix:
        return f"{value:.2f}{suffix}"
    return str(images_seen)


def _build_aggregate_rows(
    host: str,
    selected_by_width: dict[int, RunCandidate],
    metric_key: str,
) -> list[dict[str, float | int | str]]:
    aggregated: list[dict[str, float | int | str]] = []
    for width in sorted(selected_by_width):
        candidate = selected_by_width[width]
        values_by_p: dict[int, list[float]] = defaultdict(list)
        steps_by_p: dict[int, int] = {}
        for group_id, remote_path in sorted(candidate.metrics_paths.items()):
            group_rows = _load_group_metrics(host, remote_path)
            for row in group_rows:
                images_seen = int(row["images_seen"])
                steps_by_p.setdefault(images_seen, int(row["step"]))
                values_by_p[images_seen].append(float(row[metric_key]))
        for images_seen, losses in sorted(values_by_p.items()):
            loss_array = np.asarray(losses, dtype=np.float64)
            aggregated.append(
                {
                    "run_id": candidate.run_id,
                    "width": width,
                    "images_seen": images_seen,
                    "step": steps_by_p[images_seen],
                    "group_count": int(loss_array.size),
                    "mean_loss": float(loss_array.mean()),
                    "std_loss": float(loss_array.std(ddof=0)),
                    "min_loss": float(loss_array.min()),
                    "max_loss": float(loss_array.max()),
                }
            )
    return aggregated


def _filter_plot_rows(
    aggregate_rows: list[dict[str, float | int | str]],
    p_mode: str,
) -> list[dict[str, float | int | str]]:
    widths = sorted({int(row["width"]) for row in aggregate_rows})
    p_by_width: dict[int, set[int]] = defaultdict(set)
    for row in aggregate_rows:
        p_by_width[int(row["width"])].add(int(row["images_seen"]))
    if p_mode == "common":
        allowed_p = set.intersection(*(p_by_width[width] for width in widths))
    else:
        allowed_p = set.union(*(p_by_width[width] for width in widths))
    return [row for row in aggregate_rows if int(row["images_seen"]) in allowed_p]


def _group_rows_by_p(
    aggregate_rows: list[dict[str, float | int | str]],
) -> dict[int, list[dict[str, float | int | str]]]:
    grouped: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in aggregate_rows:
        grouped[int(row["images_seen"])].append(row)
    for images_seen in grouped:
        grouped[images_seen].sort(key=lambda row: int(row["width"]))
    return grouped


def _power_law_with_offset(
    widths: np.ndarray,
    amplitude: float,
    exponent: float,
    offset: float,
) -> np.ndarray:
    return amplitude * np.power(widths, exponent) + offset


def _build_delta_rows_from_widest(
    grouped_rows: dict[int, list[dict[str, float | int | str]]],
) -> tuple[list[dict[str, float | int | str]], list[str]]:
    delta_rows: list[dict[str, float | int | str]] = []
    warnings: list[str] = []
    for images_seen, rows in sorted(grouped_rows.items()):
        widest_row = max(rows, key=lambda row: int(row["width"]))
        widest_width = int(widest_row["width"])
        widest_loss = float(widest_row["mean_loss"])
        for row in rows:
            width = int(row["width"])
            if width == widest_width:
                continue
            delta_loss = float(row["mean_loss"]) - widest_loss
            if delta_loss <= 0.0:
                warnings.append(
                    f"P={images_seen}: width {width} has non-positive widest-subtracted loss {delta_loss:.6g}; skipped in delta plots."
                )
                continue
            delta_rows.append(
                {
                    "run_id": row["run_id"],
                    "width": width,
                    "images_seen": images_seen,
                    "step": row["step"],
                    "group_count": row["group_count"],
                    "widest_width": widest_width,
                    "widest_loss": widest_loss,
                    "mean_loss": row["mean_loss"],
                    "delta_loss": delta_loss,
                }
            )
    return delta_rows, warnings


def _save_figure(fig: plt.Figure, output_stem: Path, output_formats: list[str]) -> None:
    for fmt in output_formats:
        fmt_clean = fmt.strip().lstrip(".").lower()
        if not fmt_clean:
            continue
        fig.savefig(output_stem.with_suffix(f".{fmt_clean}"), dpi=200, bbox_inches="tight")


def _plot_loss_curves(
    grouped_rows: dict[int, list[dict[str, float | int | str]]],
    output_stem: Path,
    output_formats: list[str],
    title: str,
    ylabel: str,
    normalize_each_p: bool,
    value_key: str = "mean_loss",
) -> None:
    ps = sorted(grouped_rows)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(ps), dtype=np.float64))
    fig, ax = plt.subplots(figsize=(9, 6))
    for color, images_seen in zip(colors, ps, strict=True):
        rows = grouped_rows[images_seen]
        widths = np.asarray([int(row["width"]) for row in rows], dtype=np.float64)
        losses = np.asarray([float(row[value_key]) for row in rows], dtype=np.float64)
        valid = (widths > 0.0) & (losses > 0.0)
        if valid.sum() == 0:
            continue
        widths = widths[valid]
        losses = losses[valid]
        if normalize_each_p:
            losses = losses / losses.max()
        ax.loglog(
            widths,
            losses,
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            color=color,
            label=f"P={_format_p(images_seen)}",
        )
    ax.set_xlabel("Width")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_stem, output_formats)
    plt.close(fig)


def _fit_power_law_slopes(
    grouped_rows: dict[int, list[dict[str, float | int | str]]],
) -> list[dict[str, float | int]]:
    fitted_rows: list[dict[str, float | int]] = []
    for images_seen, rows in sorted(grouped_rows.items()):
        widths = np.asarray([int(row["width"]) for row in rows], dtype=np.float64)
        losses = np.asarray([float(row["mean_loss"]) for row in rows], dtype=np.float64)
        valid = (widths > 0.0) & (losses > 0.0)
        if valid.sum() < 2:
            continue
        widths_valid = widths[valid]
        losses_valid = losses[valid]
        offset0 = max(0.0, float(losses_valid.min()) * 0.95)
        shifted = np.maximum(losses_valid - offset0, 1e-12)
        exponent0, log_amplitude0 = np.polyfit(np.log(widths_valid), np.log(shifted), deg=1)
        amplitude0 = float(np.exp(log_amplitude0))
        max_abs_loss = max(float(np.max(np.abs(losses_valid))), 1.0)
        params, _ = curve_fit(
            _power_law_with_offset,
            widths_valid,
            losses_valid,
            p0=np.asarray([amplitude0, exponent0, offset0], dtype=np.float64),
            bounds=(
                np.asarray([0.0, -10.0, -10.0 * max_abs_loss], dtype=np.float64),
                np.asarray([np.inf, 10.0, 10.0 * max_abs_loss], dtype=np.float64),
            ),
            maxfev=20000,
        )
        amplitude, exponent, offset = [float(value) for value in params]
        fitted = _power_law_with_offset(widths_valid, amplitude, exponent, offset)
        residual = float(np.sqrt(np.mean(np.square(losses_valid - fitted))))
        fitted_rows.append(
            {
                "images_seen": images_seen,
                "num_widths": int(valid.sum()),
                "amplitude": amplitude,
                "exponent": exponent,
                "offset": offset,
                "rmse": residual,
            }
        )
    return fitted_rows


def _fit_delta_loglog_slopes(
    grouped_rows: dict[int, list[dict[str, float | int | str]]],
) -> list[dict[str, float | int]]:
    fitted_rows: list[dict[str, float | int]] = []
    for images_seen, rows in sorted(grouped_rows.items()):
        widths = np.asarray([int(row["width"]) for row in rows], dtype=np.float64)
        delta_losses = np.asarray([float(row["delta_loss"]) for row in rows], dtype=np.float64)
        valid = (widths > 0.0) & (delta_losses > 0.0)
        if valid.sum() < 2:
            continue
        widths_valid = widths[valid]
        delta_valid = delta_losses[valid]
        slope, intercept = np.polyfit(np.log(widths_valid), np.log(delta_valid), deg=1)
        fitted = np.exp(intercept) * np.power(widths_valid, slope)
        residual = float(np.sqrt(np.mean(np.square(np.log(delta_valid) - np.log(fitted)))))
        fitted_rows.append(
            {
                "images_seen": images_seen,
                "num_widths": int(valid.sum()),
                "amplitude": float(np.exp(intercept)),
                "exponent": float(slope),
                "log_rmse": residual,
            }
        )
    return fitted_rows


def _plot_slopes(
    slope_rows: list[dict[str, float | int]],
    output_stem: Path,
    output_formats: list[str],
    title: str,
    ylabel: str,
) -> None:
    if not slope_rows:
        raise RuntimeError("No valid power-law slopes could be fitted.")
    ps = np.asarray([int(row["images_seen"]) for row in slope_rows], dtype=np.float64)
    slopes = np.asarray([float(row["exponent"]) for row in slope_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(ps, slopes, marker="o", linewidth=1.6, markersize=4.5)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("P")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_stem, output_formats)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path}.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_widths = None if args.widths is None else set(args.widths)
    expected_group_counts = _load_expected_group_counts(args.manifest_path)
    candidates_by_width = _discover_remote_candidates(
        host=args.ssh_host,
        remote_root=args.remote_root,
        remote_run_glob=args.remote_run_glob,
        requested_widths=requested_widths,
    )
    if not candidates_by_width:
        raise RuntimeError("No remote run candidates were discovered.")

    selected_by_width: dict[int, RunCandidate] = {}
    selection_rows: list[dict[str, int | str]] = []
    selection_warnings: list[str] = []
    for width in sorted(candidates_by_width):
        expected = expected_group_counts.get(width)
        selected, warnings = _select_candidate(candidates_by_width[width], expected)
        selected_by_width[width] = selected
        selection_warnings.extend(warnings)
        selection_rows.append(
            {
                "width": width,
                "run_id": selected.run_id,
                "job_id": selected.job_id,
                "metrics_count": len(selected.metrics_paths),
                "metadata_count": len(selected.metadata_paths),
                "expected_groups": expected if expected is not None else "",
            }
        )

    aggregate_rows = _build_aggregate_rows(
        host=args.ssh_host,
        selected_by_width=selected_by_width,
        metric_key=args.metric_key,
    )
    filtered_rows = _filter_plot_rows(aggregate_rows, args.p_mode)
    grouped_rows = _group_rows_by_p(filtered_rows)
    delta_rows, delta_warnings = _build_delta_rows_from_widest(grouped_rows)
    delta_grouped_rows = _group_rows_by_p(delta_rows)
    slope_rows = _fit_power_law_slopes(grouped_rows)
    delta_slope_rows = _fit_delta_loglog_slopes(delta_grouped_rows)

    stem = f"{args.metric_key}_{args.p_mode}"
    _write_csv(output_dir / f"{stem}_selected_runs.csv", selection_rows)
    _write_csv(output_dir / f"{stem}_mean_losses.csv", filtered_rows)
    _write_csv(output_dir / f"{stem}_mean_losses_minus_widest.csv", delta_rows)
    _write_csv(output_dir / f"{stem}_powerlaw_slopes.csv", slope_rows)
    _write_csv(output_dir / f"{stem}_loss_minus_widest_powerlaw_slopes.csv", delta_slope_rows)

    selection_warnings.extend(delta_warnings)
    if selection_warnings:
        warnings_path = output_dir / f"{stem}_warnings.txt"
        warnings_path.write_text("\n".join(selection_warnings) + "\n")
        for warning in selection_warnings:
            print(f"WARNING: {warning}")

    _plot_loss_curves(
        grouped_rows=grouped_rows,
        output_stem=output_dir / f"{stem}_loss_vs_width",
        output_formats=args.output_formats,
        title=f"{args.metric_key} vs width by P ({args.p_mode} P selection)",
        ylabel=args.metric_key,
        normalize_each_p=False,
    )
    _plot_loss_curves(
        grouped_rows=grouped_rows,
        output_stem=output_dir / f"{stem}_loss_vs_width_rescaled",
        output_formats=args.output_formats,
        title=f"{args.metric_key} / max_width_loss vs width by P ({args.p_mode} P selection)",
        ylabel=f"{args.metric_key} / max over widths",
        normalize_each_p=True,
    )
    _plot_loss_curves(
        grouped_rows=delta_grouped_rows,
        output_stem=output_dir / f"{stem}_loss_minus_widest_vs_width",
        output_formats=args.output_formats,
        title=f"{args.metric_key} - widest-network loss vs width by P ({args.p_mode} P selection)",
        ylabel=f"{args.metric_key} - widest-network loss",
        normalize_each_p=False,
        value_key="delta_loss",
    )
    _plot_loss_curves(
        grouped_rows=delta_grouped_rows,
        output_stem=output_dir / f"{stem}_loss_minus_widest_vs_width_rescaled",
        output_formats=args.output_formats,
        title=f"({args.metric_key} - widest-network loss) / max vs width by P ({args.p_mode} P selection)",
        ylabel=f"({args.metric_key} - widest-network loss) / max over widths",
        normalize_each_p=True,
        value_key="delta_loss",
    )
    _plot_slopes(
        slope_rows=slope_rows,
        output_stem=output_dir / f"{stem}_powerlaw_slope_vs_p",
        output_formats=args.output_formats,
        title=f"Exponent a in {args.metric_key}(width) = c * width^a + b",
        ylabel="Exponent a in loss = c * width^a + b",
    )
    _plot_slopes(
        slope_rows=delta_slope_rows,
        output_stem=output_dir / f"{stem}_loss_minus_widest_powerlaw_slope_vs_p",
        output_formats=args.output_formats,
        title=f"Log-log slope of ({args.metric_key} - widest-network loss)(width) vs P",
        ylabel="Exponent a in (loss - widest loss) = c * width^a",
    )

    print("Selected runs:")
    for row in selection_rows:
        print(
            f"  width={row['width']}: run_id={row['run_id']} "
            f"metrics={row['metrics_count']} expected={row['expected_groups']}"
        )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
