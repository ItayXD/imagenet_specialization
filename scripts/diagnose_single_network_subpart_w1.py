#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.analyze_exchangeability import (
    _extract_weights_from_artifacts,
    _list_group_dirs,
    _load_group_metadata,
    _resolve_width_dirs,
    _weight_similarity_matrix,
)
from src.experiment.exchangeability_utils import ks_w1_stats


GROUP_PATTERN = re.compile(r"group_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute single-network subpart-vs-full W1 diagnostics from saved "
            "first-layer weight artifacts."
        )
    )
    parser.add_argument(
        "--base-save-dir",
        required=True,
        help=(
            "Either a dataset/run root containing width_* directories directly, "
            "or a parent directory containing run subdirectories."
        ),
    )
    parser.add_argument(
        "--run-id",
        default="",
        help=(
            "Run id to resolve under --base-save-dir when the base dir does not "
            "contain width_* directories directly."
        ),
    )
    parser.add_argument(
        "--resolution-mode",
        choices=["exact", "latest_prefix", "auto"],
        default="auto",
        help="Run id resolution mode when --base-save-dir is a parent directory.",
    )
    parser.add_argument(
        "--widths",
        nargs="*",
        type=int,
        default=None,
        help="Optional width filter.",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        type=int,
        default=None,
        help="Optional images_seen filter.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="Subpart fractions to evaluate.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=32,
        help="Number of random subparts per width/step/member/fraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260311,
        help="Base RNG seed. The notebook-compatible per-width/step/fraction offset is preserved.",
    )
    parser.add_argument(
        "--member-indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit member indices. Default: all members in each artifact.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Per-group/member diagnostic CSV output path.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional summary CSV path. Defaults to <output_csv stem>_summary.csv.",
    )
    return parser.parse_args()


def _list_direct_width_dirs(base_dir: str, requested_widths: list[int] | None) -> tuple[dict[int, str], dict[int, str]]:
    base_path = Path(base_dir)
    width_dirs: dict[int, str] = {}
    width_sources: dict[int, str] = {}
    requested = None if requested_widths is None else {int(width) for width in requested_widths}

    for child in sorted(base_path.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("width_"):
            continue
        try:
            width = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        if requested is not None and width not in requested:
            continue
        width_dirs[width] = str(child)
        width_sources[width] = base_path.name
    return width_dirs, width_sources


def resolve_width_dirs(
    base_save_dir: str,
    run_id: str,
    resolution_mode: str,
    widths: list[int] | None,
) -> tuple[dict[int, str], dict[int, str]]:
    direct_width_dirs, direct_sources = _list_direct_width_dirs(base_save_dir, widths)
    if direct_width_dirs:
        return direct_width_dirs, direct_sources
    return _resolve_width_dirs(
        base_save_dir=str(base_save_dir),
        run_id=str(run_id),
        resolution_mode=str(resolution_mode),
        requested_widths=widths,
    )


def _artifact_steps(group_dir: str) -> list[int]:
    artifacts_dir = Path(group_dir) / "artifacts"
    steps = []
    if not artifacts_dir.is_dir():
        return steps
    for artifact_path in sorted(artifacts_dir.glob("first_layer_*.npz")):
        try:
            step = int(artifact_path.stem.split("_")[-1])
        except ValueError:
            continue
        steps.append(step)
    return sorted(set(steps))


def _group_id_from_dir(group_dir: str) -> int:
    match = GROUP_PATTERN.search(Path(group_dir).name)
    if not match:
        return -1
    return int(match.group(1))


def _stats_dict(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def _row_seed(base_seed: int, width: int, step: int, frac_index: int) -> int:
    return int(base_seed) + int(width) * 10_000_000 + int(step) * 1_000 + int(frac_index) * 1_000_000


def _member_rows_for_weights(
    *,
    weights: np.ndarray,
    dataset: str,
    width: int,
    source_run_id: str,
    group_dir: str,
    group_rank: int,
    step: int,
    fractions: list[float],
    repeats: int,
    base_seed: int,
    member_indices: list[int] | None,
) -> list[dict[str, object]]:
    if weights.ndim < 2:
        return []

    num_members = int(weights.shape[0])
    width_channels = int(weights.shape[1])
    chosen_members = list(range(num_members)) if member_indices is None else [int(idx) for idx in member_indices]

    rows: list[dict[str, object]] = []
    for member_index in chosen_members:
        if member_index < 0 or member_index >= num_members:
            continue

        member_weights = np.asarray(weights[member_index:member_index + 1], dtype=np.float32)
        sim_full = _weight_similarity_matrix(member_weights)
        tri_full = np.triu_indices(sim_full.shape[0], k=1)
        full_values = np.asarray(sim_full[tri_full], dtype=np.float64)
        if full_values.size == 0:
            continue

        full_stats = _stats_dict(full_values, "full_similarity")

        for frac_index, frac in enumerate(fractions):
            subpart_width = int(round(float(frac) * width_channels))
            subpart_width = max(2, min(width_channels, subpart_width))
            rng = np.random.default_rng(_row_seed(base_seed, width, step, frac_index))

            w1_samples = []
            pooled_subparts = []
            subset_stats_accumulator = []
            for _ in range(int(repeats)):
                subset_idx = rng.choice(width_channels, size=subpart_width, replace=False)
                sub_sim = sim_full[np.ix_(subset_idx, subset_idx)]
                tri_sub = np.triu_indices(subpart_width, k=1)
                sub_values = np.asarray(sub_sim[tri_sub], dtype=np.float64)
                if sub_values.size == 0:
                    continue
                pooled_subparts.append(sub_values)
                subset_stats_accumulator.append(
                    {
                        "mean": float(np.mean(sub_values)),
                        "std": float(np.std(sub_values)),
                        "p10": float(np.percentile(sub_values, 10)),
                        "p50": float(np.percentile(sub_values, 50)),
                        "p90": float(np.percentile(sub_values, 90)),
                    }
                )
                w1_samples.append(float(ks_w1_stats(sub_values, full_values)["w1_distance"]))

            if not w1_samples:
                continue

            w1_array = np.asarray(w1_samples, dtype=np.float64)
            pooled_values = np.concatenate(pooled_subparts, axis=0)
            pooled_subpart_w1 = float(ks_w1_stats(pooled_values, full_values)["w1_distance"])
            subset_means = np.asarray([entry["mean"] for entry in subset_stats_accumulator], dtype=np.float64)
            subset_stds = np.asarray([entry["std"] for entry in subset_stats_accumulator], dtype=np.float64)
            subset_p10s = np.asarray([entry["p10"] for entry in subset_stats_accumulator], dtype=np.float64)
            subset_p50s = np.asarray([entry["p50"] for entry in subset_stats_accumulator], dtype=np.float64)
            subset_p90s = np.asarray([entry["p90"] for entry in subset_stats_accumulator], dtype=np.float64)

            row: dict[str, object] = {
                "dataset": dataset,
                "width": int(width),
                "source_run_id": str(source_run_id),
                "group_name": Path(group_dir).name,
                "group_id": _group_id_from_dir(group_dir),
                "group_rank": int(group_rank),
                "images_seen": int(step),
                "member_index": int(member_index),
                "is_notebook_group": int(group_rank == 0),
                "is_notebook_member": int(member_index == 0),
                "is_notebook_row": int(group_rank == 0 and member_index == 0),
                "fraction": float(frac),
                "subpart_width": int(subpart_width),
                "width_channels": int(width_channels),
                "repeats": int(repeats),
                "full_pairs": int(full_values.size),
                "subpart_pairs": int(subpart_width * (subpart_width - 1) // 2),
                "pooled_subpart_w1": pooled_subpart_w1,
                "width_times_pooled_subpart_w1": float(width * pooled_subpart_w1),
            }
            row.update(full_stats)
            row.update(_stats_dict(w1_array, "w1"))
            row["width_times_w1_mean"] = float(width * float(row["w1_mean"]))
            row["width_times_w1_p10"] = float(width * float(row["w1_p10"]))
            row["width_times_w1_p90"] = float(width * float(row["w1_p90"]))
            row["subset_similarity_mean_mean"] = float(np.mean(subset_means))
            row["subset_similarity_std_mean"] = float(np.mean(subset_stds))
            row["subset_similarity_p10_mean"] = float(np.mean(subset_p10s))
            row["subset_similarity_p50_mean"] = float(np.mean(subset_p50s))
            row["subset_similarity_p90_mean"] = float(np.mean(subset_p90s))
            rows.append(row)

    return rows


def collect_diagnostic_rows(
    *,
    base_save_dir: str,
    run_id: str,
    resolution_mode: str,
    widths: list[int] | None,
    steps: list[int] | None,
    fractions: list[float],
    repeats: int,
    seed: int,
    member_indices: list[int] | None,
) -> list[dict[str, object]]:
    width_dirs, width_sources = resolve_width_dirs(base_save_dir, run_id, resolution_mode, widths)
    rows: list[dict[str, object]] = []

    for width in sorted(width_dirs):
        width_dir = width_dirs[width]
        source_run_id = width_sources.get(width, "")
        group_dirs = _list_group_dirs(width_dir)
        if not group_dirs:
            continue

        for group_rank, group_dir in enumerate(group_dirs):
            metadata = _load_group_metadata(group_dir)
            dataset = str(metadata.get("dataset", "")).strip() or "unknown"
            available_steps = _artifact_steps(group_dir)
            target_steps = available_steps if steps is None else [int(step) for step in steps if int(step) in set(available_steps)]
            for step in target_steps:
                artifact_path = Path(group_dir) / "artifacts" / f"first_layer_{int(step)}.npz"
                if not artifact_path.exists():
                    continue
                weights = _extract_weights_from_artifacts(group_dir, int(step))
                rows.extend(
                    _member_rows_for_weights(
                        weights=weights,
                        dataset=dataset,
                        width=int(width),
                        source_run_id=source_run_id,
                        group_dir=group_dir,
                        group_rank=group_rank,
                        step=int(step),
                        fractions=fractions,
                        repeats=int(repeats),
                        base_seed=int(seed),
                        member_indices=member_indices,
                    )
                )
    return rows


def summarize_diagnostic_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped_rows: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["source_run_id"],
            row["width"],
            row["images_seen"],
            row["fraction"],
        )
        grouped_rows[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped_rows):
        members = grouped_rows[key]
        dataset, source_run_id, width, images_seen, fraction = key
        w1_means = np.asarray([float(row["w1_mean"]) for row in members], dtype=np.float64)
        pooled_w1s = np.asarray([float(row["pooled_subpart_w1"]) for row in members], dtype=np.float64)
        full_means = np.asarray([float(row["full_similarity_mean"]) for row in members], dtype=np.float64)

        group_ids = sorted({int(row["group_id"]) for row in members})
        member_keys = {(int(row["group_id"]), int(row["member_index"])) for row in members}
        notebook_rows = [row for row in members if int(row["is_notebook_row"]) == 1]

        per_group_w1: dict[int, list[float]] = defaultdict(list)
        for row in members:
            per_group_w1[int(row["group_id"])].append(float(row["w1_mean"]))
        group_means = np.asarray([float(np.mean(vals)) for _, vals in sorted(per_group_w1.items())], dtype=np.float64)

        summary_row: dict[str, object] = {
            "dataset": dataset,
            "source_run_id": source_run_id,
            "width": int(width),
            "images_seen": int(images_seen),
            "fraction": float(fraction),
            "row_count": int(len(members)),
            "group_count": int(len(group_ids)),
            "group_member_count": int(len(member_keys)),
            "all_rows_w1_mean_mean": float(np.mean(w1_means)),
            "all_rows_w1_mean_std": float(np.std(w1_means)),
            "all_rows_w1_mean_p10": float(np.percentile(w1_means, 10)),
            "all_rows_w1_mean_p50": float(np.percentile(w1_means, 50)),
            "all_rows_w1_mean_p90": float(np.percentile(w1_means, 90)),
            "all_rows_width_times_w1_mean_mean": float(int(width) * np.mean(w1_means)),
            "all_rows_pooled_subpart_w1_mean": float(np.mean(pooled_w1s)),
            "all_rows_width_times_pooled_subpart_w1_mean": float(int(width) * np.mean(pooled_w1s)),
            "all_rows_full_similarity_mean_mean": float(np.mean(full_means)),
            "group_mean_w1_mean": float(np.mean(group_means)),
            "group_mean_width_times_w1_mean": float(int(width) * np.mean(group_means)),
            "notebook_row_present": int(bool(notebook_rows)),
        }
        if notebook_rows:
            notebook_row = notebook_rows[0]
            notebook_w1 = float(notebook_row["w1_mean"])
            summary_row["notebook_group_id"] = int(notebook_row["group_id"])
            summary_row["notebook_member_index"] = int(notebook_row["member_index"])
            summary_row["notebook_w1_mean"] = notebook_w1
            summary_row["notebook_width_times_w1_mean"] = float(int(width) * notebook_w1)
            summary_row["notebook_pooled_subpart_w1"] = float(notebook_row["pooled_subpart_w1"])
            summary_row["notebook_minus_all_rows_w1_mean"] = float(notebook_w1 - np.mean(w1_means))
            summary_row["notebook_minus_all_rows_width_times_w1_mean"] = float(int(width) * (notebook_w1 - np.mean(w1_means)))
        else:
            summary_row["notebook_group_id"] = -1
            summary_row["notebook_member_index"] = -1
            summary_row["notebook_w1_mean"] = float("nan")
            summary_row["notebook_width_times_w1_mean"] = float("nan")
            summary_row["notebook_pooled_subpart_w1"] = float("nan")
            summary_row["notebook_minus_all_rows_w1_mean"] = float("nan")
            summary_row["notebook_minus_all_rows_width_times_w1_mean"] = float("nan")
        summary_rows.append(summary_row)

    return summary_rows


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows were produced for {path_obj}.")

    fieldnames = sorted({key for row in rows for key in row})
    with path_obj.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    fractions = list(dict.fromkeys(float(value) for value in args.fractions))
    if not fractions:
        raise ValueError("--fractions must be non-empty.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    rows = collect_diagnostic_rows(
        base_save_dir=args.base_save_dir,
        run_id=args.run_id,
        resolution_mode=args.resolution_mode,
        widths=None if args.widths is None else [int(width) for width in args.widths],
        steps=None if args.steps is None else [int(step) for step in args.steps],
        fractions=fractions,
        repeats=int(args.repeats),
        seed=int(args.seed),
        member_indices=None if args.member_indices is None else [int(idx) for idx in args.member_indices],
    )
    summary_rows = summarize_diagnostic_rows(rows)

    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_csv.with_name(f"{output_csv.stem}_summary.csv")
    _write_csv(str(output_csv), rows)
    _write_csv(str(summary_csv), summary_rows)

    print(
        f"Wrote {len(rows)} diagnostic rows to {output_csv.resolve()} "
        f"and {len(summary_rows)} summary rows to {summary_csv.resolve()}."
    )


if __name__ == "__main__":
    main()
