from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiment.exchangeability_utils import ks_w1_stats


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _ordered_representations(values: list[str], preferred_order: list[str]) -> list[str]:
    present = set(str(v) for v in values)
    ordered = [rep for rep in preferred_order if rep in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _ecdf_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(values, dtype=np.float64))
    ys = np.arange(1, xs.size + 1, dtype=np.float64) / float(xs.size)
    return xs, ys


def export_plot_vectors(context: Mapping[str, Any], out_path: str | Path | None = None) -> dict[str, Any]:
    required_keys = [
        "CSV_PATH",
        "OUT_DIR",
        "df",
        "curves",
        "KS_W1_ANALYSIS_TYPES",
        "KS_W1_ANALYSIS_LABELS",
        "KS_W1_REPRESENTATION_ORDER",
        "ECDF_WIDTH",
        "ECDF_REPRESENTATION",
        "ecdf_data",
        "AVAILABLE_WIDTHS",
        "PAIRWISE_ACROSS_REPRESENTATIONS",
        "PAIRWISE_ACROSS_RNG_SEED",
        "PAIRWISE_ACROSS_MAX_POINTS",
        "ECDF_STEPS",
        "_prepare_across_real_by_step",
        "_subsample",
        "results_by_fraction",
        "SINGLE_NET_REPRESENTATION",
        "SINGLE_NET_MEMBER_INDEX",
        "similarity_stats_df",
    ]
    missing = [key for key in required_keys if key not in context]
    if missing:
        raise KeyError(f"Missing notebook state for plot export: {missing}")

    csv_path = Path(context["CSV_PATH"])
    out_dir = Path(context["OUT_DIR"])
    df = context["df"]
    curves = context["curves"]
    ks_w1_analysis_types = list(context["KS_W1_ANALYSIS_TYPES"])
    ks_w1_analysis_labels = dict(context["KS_W1_ANALYSIS_LABELS"])
    ks_w1_representation_order = [str(v) for v in context["KS_W1_REPRESENTATION_ORDER"]]
    ecdf_width = int(context["ECDF_WIDTH"])
    ecdf_representation = str(context["ECDF_REPRESENTATION"])
    ecdf_data = dict(context["ecdf_data"])
    available_widths = [int(v) for v in context["AVAILABLE_WIDTHS"]]
    pairwise_across_representations = [str(v) for v in context["PAIRWISE_ACROSS_REPRESENTATIONS"]]
    pairwise_across_rng_seed = int(context["PAIRWISE_ACROSS_RNG_SEED"])
    pairwise_across_max_points = int(context["PAIRWISE_ACROSS_MAX_POINTS"])
    ecdf_steps = context["ECDF_STEPS"]
    prepare_across_real_by_step = context["_prepare_across_real_by_step"]
    subsample = context["_subsample"]
    results_by_fraction = context["results_by_fraction"]
    single_net_representation = str(context["SINGLE_NET_REPRESENTATION"])
    single_net_member_index = int(context["SINGLE_NET_MEMBER_INDEX"])
    similarity_stats_df = context["similarity_stats_df"]
    layerwise_df = context.get("layerwise_df")

    array_store: dict[str, np.ndarray] = {}
    array_lookup: dict[tuple[int, str, tuple[int, ...], str], str] = {}
    manifest = {
        "version": 1,
        "source_csv": str(csv_path),
        "output_dir": str(out_dir),
        "transforms": {
            "multiply": "plotted_values = factor * source_values",
            "difference_then_multiply": "plotted_values = factor * (lhs_values - rhs_values)",
            "row_normalize_per_row": "apply min-max normalization independently within each matrix row",
            "scatter_from_mask": "overlay points are the base x/y values wherever mask is true",
        },
        "plots": [],
    }

    def register_array(values: Any, prefix: str) -> str:
        arr = np.ascontiguousarray(np.asarray(values))
        if arr.ndim not in (1, 2):
            raise ValueError(f"Only 1D/2D arrays are supported, got shape={arr.shape}.")
        signature = (arr.ndim, str(arr.dtype), tuple(arr.shape), hashlib.sha1(arr.tobytes()).hexdigest())
        existing = array_lookup.get(signature)
        if existing is not None:
            return existing
        key = f"{prefix}_{len(array_store):04d}"
        array_lookup[signature] = key
        array_store[key] = arr
        return key

    def register_vector(values: Any, prefix: str = "vector") -> str:
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError(f"Expected a vector, got shape={arr.shape}.")
        return register_array(arr, prefix)

    def register_matrix(values: Any, prefix: str = "matrix") -> str:
        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ValueError(f"Expected a matrix, got shape={arr.shape}.")
        return register_array(arr, prefix)

    def record_plot(**entry: Any) -> None:
        manifest["plots"].append(_json_ready(entry))

    curve_specs = [
        {
            "plot_id": "ks_distance_vs_images_seen",
            "title": "KS Distance vs Images Seen",
            "ylabel": "KS distance",
            "metric": "ks_distance",
            "lo": "ks_distance_p10",
            "hi": "ks_distance_p90",
            "y_transform": None,
            "band_transform": None,
        },
        {
            "plot_id": "w1_distance_vs_images_seen",
            "title": "W1 Distance vs Images Seen",
            "ylabel": "W1 distance",
            "metric": "w1_distance",
            "lo": "w1_distance_p10",
            "hi": "w1_distance_p90",
            "y_transform": None,
            "band_transform": None,
        },
        {
            "plot_id": "w1_distance_times_sqrt_width_vs_images_seen",
            "title": "N * W1 Distance vs Images Seen",
            "ylabel": "N * W1 distance",
            "metric": "w1_distance",
            "lo": "w1_distance_p10",
            "hi": "w1_distance_p90",
            "y_transform": {"kind": "multiply"},
            "band_transform": {"kind": "multiply"},
        },
    ]

    plot_df = curves[curves["analysis_type"].isin(ks_w1_analysis_types)].copy()
    ordered_reps = _ordered_representations(
        plot_df["representation"].astype(str).unique().tolist(),
        ks_w1_representation_order,
    )
    group_rows = plot_df[["analysis_type", "width", "representation"]].drop_duplicates()
    ordered_groups = sorted(
        group_rows.itertuples(index=False, name=None),
        key=lambda row: (
            str(row[0]),
            int(row[1]),
            ordered_reps.index(str(row[2])) if str(row[2]) in ordered_reps else 10**9,
        ),
    )
    for spec in curve_specs:
        for analysis_type, width, representation in ordered_groups:
            sub = plot_df[
                (plot_df["analysis_type"] == analysis_type)
                & (plot_df["width"] == width)
                & (plot_df["representation"] == representation)
            ].sort_values("images_seen")
            if sub.empty:
                continue
            width_int = int(width)
            display_analysis = ks_w1_analysis_labels.get(str(analysis_type), str(analysis_type))
            entry = {
                "plot_id": spec["plot_id"],
                "plot_kind": "line_with_band",
                "series_label": f"{display_analysis}/{representation} N={width_int}",
                "x_key": register_vector(sub["images_seen"].to_numpy(dtype=np.int64)),
                "y_key": register_vector(sub[spec["metric"]].to_numpy(dtype=np.float64)),
                "band": {
                    "lo_key": register_vector(sub[spec["lo"]].to_numpy(dtype=np.float64)),
                    "hi_key": register_vector(sub[spec["hi"]].to_numpy(dtype=np.float64)),
                },
                "metadata": {
                    "analysis_type": str(analysis_type),
                    "representation": str(representation),
                    "width": width_int,
                    "title": spec["title"],
                    "xlabel": "Images seen (P)",
                    "ylabel": spec["ylabel"],
                },
            }
            if spec["y_transform"] is not None:
                entry["y_transform"] = {"kind": spec["y_transform"]["kind"], "factor": width_int}
                entry["band_transform"] = {"kind": spec["band_transform"]["kind"], "factor": width_int}
            record_plot(**entry)

    train_val_dedup = (
        df[["width", "images_seen", "train_loss", "val_loss", "train_error", "val_error"]]
        .drop_duplicates()
        .sort_values(["width", "images_seen"])
    )
    for metric in ["train_loss", "val_loss", "train_error", "val_error"]:
        for width, sub in train_val_dedup.groupby("width"):
            record_plot(
                plot_id=f"{metric}_vs_images_seen",
                plot_kind="line",
                series_label=f"N={int(width)}",
                x_key=register_vector(sub["images_seen"].to_numpy(dtype=np.int64)),
                y_key=register_vector(sub[metric].to_numpy(dtype=np.float64)),
                metadata={
                    "metric": metric,
                    "width": int(width),
                    "title": f"{metric.replace('_', ' ').title()} vs Images Seen",
                    "xlabel": "Images seen (P)",
                    "ylabel": metric.replace("_", " ").title(),
                },
            )

    significance_subset = df[
        (df["analysis_type"] == "within_vs_across_real")
        & (df["shuffle_id"] == -1)
    ].copy()
    if not significance_subset.empty:
        null_counts = (
            df[
                (df["analysis_type"] == "within_shuffled_vs_across_real")
                & (df["shuffle_id"] >= 0)
            ]
            .groupby(["width", "images_seen", "representation"], as_index=False)
            .size()
            .rename(columns={"size": "empirical_null_count"})
        )
        significance_subset = significance_subset.merge(
            null_counts,
            on=["width", "images_seen", "representation"],
            how="left",
        )
        significance_specs = [
            (
                "ks_sigma_two_sided",
                "KS Raw Sigma (Two-Sided) vs Images Seen",
                "ks_sigma_two_sided_vs_images_seen",
                "KS Raw Sigma (Two-Sided)",
                None,
            ),
            (
                "ks_sigma_empirical_two_sided",
                "KS Empirical Sigma vs Images Seen",
                "ks_sigma_empirical_two_sided_vs_images_seen",
                "KS Empirical Sigma",
                "ks_p_empirical",
            ),
            (
                "w1_sigma_empirical_two_sided",
                "W1 Empirical Sigma vs Images Seen",
                "w1_sigma_empirical_two_sided_vs_images_seen",
                "W1 Empirical Sigma",
                "w1_p_empirical",
            ),
        ]
        for metric, title, plot_id, ylabel, empirical_p_col in significance_specs:
            plot_metric_df = significance_subset[np.isfinite(significance_subset[metric])].copy()
            if plot_metric_df.empty:
                continue
            ordered_reps = _ordered_representations(
                plot_metric_df["representation"].astype(str).unique().tolist(),
                ks_w1_representation_order,
            )
            ordered_groups = sorted(
                plot_metric_df[["representation", "width"]].drop_duplicates().itertuples(index=False, name=None),
                key=lambda row: (
                    ordered_reps.index(str(row[0])) if str(row[0]) in ordered_reps else 10**9,
                    int(row[1]),
                ),
            )
            for representation, width in ordered_groups:
                sub = plot_metric_df[
                    (plot_metric_df["representation"] == representation)
                    & (plot_metric_df["width"] == width)
                ].sort_values("images_seen")
                if sub.empty:
                    continue
                entry = {
                    "plot_id": plot_id,
                    "plot_kind": "line",
                    "series_label": f"{representation} N={int(width)}",
                    "x_key": register_vector(sub["images_seen"].to_numpy(dtype=np.int64)),
                    "y_key": register_vector(sub[metric].to_numpy(dtype=np.float64)),
                    "metadata": {
                        "metric": metric,
                        "representation": str(representation),
                        "width": int(width),
                        "title": title,
                        "xlabel": "Images seen (P)",
                        "ylabel": ylabel,
                    },
                }
                if empirical_p_col is not None and empirical_p_col in sub.columns and "empirical_null_count" in sub.columns:
                    floor_p = 1.0 / (sub["empirical_null_count"] + 1.0)
                    floor_mask = (
                        np.isfinite(sub[empirical_p_col].to_numpy(dtype=np.float64))
                        & np.isfinite(floor_p.to_numpy(dtype=np.float64))
                        & np.isclose(
                            sub[empirical_p_col].to_numpy(dtype=np.float64),
                            floor_p.to_numpy(dtype=np.float64),
                            rtol=1e-6,
                            atol=1e-12,
                        )
                    )
                    if np.any(floor_mask):
                        entry["overlay"] = {
                            "kind": "scatter_from_mask",
                            "mask_key": register_vector(floor_mask.astype(bool)),
                            "marker": "x",
                        }
                record_plot(**entry)

    if ecdf_data:
        plot_id = f"ecdf_similarity_real_by_p_w{ecdf_width}_{ecdf_representation}"
        for step in sorted(ecdf_data):
            for distribution in ["within_real", "across_real"]:
                xs, ys = _ecdf_xy(np.asarray(ecdf_data[step][distribution]))
                record_plot(
                    plot_id=plot_id,
                    plot_kind="line",
                    series_label=f"P={int(step)} {distribution}",
                    x_key=register_vector(xs.astype(np.float64)),
                    y_key=register_vector(ys.astype(np.float64)),
                    metadata={
                        "distribution": distribution,
                        "images_seen": int(step),
                        "width": ecdf_width,
                        "representation": ecdf_representation,
                        "title": (
                            f"ECDF by P (width={ecdf_width}, "
                            f"representation={ecdf_representation}, real only)"
                        ),
                        "xlabel": "Cosine similarity",
                        "ylabel": "ECDF",
                    },
                )

    if len(available_widths) >= 2:
        width_pairs = list(combinations(available_widths, 2))
        selected_steps = set(int(v) for v in ecdf_steps) if ecdf_steps is not None else None
        for representation in pairwise_across_representations:
            across_by_width: dict[int, dict[int, np.ndarray]] = {}
            for width in available_widths:
                prepared, _meta = prepare_across_real_by_step(int(width), representation)
                across_by_width[int(width)] = prepared
            for width_a, width_b in width_pairs:
                steps_common = sorted(set(across_by_width[width_a]) & set(across_by_width[width_b]))
                if selected_steps is not None:
                    steps_common = [step for step in steps_common if step in selected_steps]
                if not steps_common:
                    continue
                w1_values = []
                for step in steps_common:
                    rng = np.random.default_rng(
                        pairwise_across_rng_seed + int(step) + 1000 * int(width_a) + int(width_b)
                    )
                    sample_a = subsample(
                        np.asarray(across_by_width[width_a][step]),
                        pairwise_across_max_points,
                        rng,
                    )
                    sample_b = subsample(
                        np.asarray(across_by_width[width_b][step]),
                        pairwise_across_max_points,
                        rng,
                    )
                    w1_values.append(float(ks_w1_stats(sample_a, sample_b)["w1_distance"]))
                if not w1_values:
                    continue
                record_plot(
                    plot_id="w1_across_width_pairs_vs_images_seen",
                    plot_kind="line",
                    panel=str(representation),
                    series_label=f"N={int(width_a)} vs N={int(width_b)}",
                    x_key=register_vector(np.asarray(steps_common, dtype=np.int64)),
                    y_key=register_vector(np.asarray(w1_values, dtype=np.float64)),
                    metadata={
                        "representation": str(representation),
                        "width_a": int(width_a),
                        "width_b": int(width_b),
                        "title": f"Across-width W1 vs P ({representation})",
                        "xlabel": "Images seen (P)",
                        "ylabel": "W1 distance between across_real distributions",
                    },
                )

    for frac, width_results in sorted(results_by_fraction.items(), key=lambda item: float(item[0])):
        frac_label = f"{float(frac):.3f}".rstrip("0").rstrip(".").replace(".", "p")
        plot_id = f"w1_single_network_subparts_vs_full_r{frac_label}_{single_net_representation}"
        for width in available_widths:
            if width not in width_results:
                continue
            row = width_results[width]
            order = np.argsort(np.asarray(row["steps"], dtype=np.int64))
            xs = np.asarray(row["steps"], dtype=np.int64)[order]
            record_plot(
                plot_id=plot_id,
                plot_kind="line_with_band",
                series_label=f"N={width} ({int(row['subpart_width'])} channels)",
                x_key=register_vector(xs),
                y_key=register_vector(np.asarray(row["w1_mean"], dtype=np.float64)[order]),
                y_transform={"kind": "multiply", "factor": int(width)},
                band={
                    "lo_key": register_vector(np.asarray(row["w1_p10"], dtype=np.float64)[order]),
                    "hi_key": register_vector(np.asarray(row["w1_p90"], dtype=np.float64)[order]),
                },
                band_transform={"kind": "multiply", "factor": int(width)},
                metadata={
                    "fraction": float(frac),
                    "width": int(width),
                    "subpart_width": int(row["subpart_width"]),
                    "representation": single_net_representation,
                    "member_index": single_net_member_index,
                    "title": (
                        "Single-network subpart-vs-full W1 "
                        f"(r={float(frac):g}, rep={single_net_representation}, "
                        f"member={single_net_member_index})"
                    ),
                    "xlabel": "Images seen (P)",
                    "ylabel": "N*W1 distance (subpart vs full similarity distributions)",
                },
            )

    ordered_reps = _ordered_representations(
        similarity_stats_df["representation"].astype(str).unique().tolist(),
        ks_w1_representation_order,
    )
    for metric, ylabel, title in [
        ("mean", "Mean similarity", "Mean similarity vs Images Seen"),
        ("variance", "Similarity variance", "Similarity variance vs Images Seen"),
    ]:
        for distribution in ["within", "across"]:
            for width in available_widths:
                for representation in ordered_reps:
                    sub = similarity_stats_df[
                        (similarity_stats_df["width"] == width)
                        & (similarity_stats_df["representation"] == representation)
                        & (similarity_stats_df["distribution"] == distribution)
                    ].sort_values("images_seen")
                    if sub.empty:
                        continue
                    x_key = register_vector(sub["images_seen"].to_numpy(dtype=np.int64))
                    y_key = register_vector(sub[metric].to_numpy(dtype=np.float64))
                    base_metadata = {
                        "metric": metric,
                        "distribution": distribution,
                        "representation": str(representation),
                        "width": int(width),
                        "xlabel": "Images seen (P)",
                    }
                    record_plot(
                        plot_id="similarity_mean_variance_vs_images_seen",
                        plot_kind="line",
                        panel=metric,
                        series_label=f"{distribution}/{representation} N={width}",
                        x_key=x_key,
                        y_key=y_key,
                        metadata={**base_metadata, "title": title, "ylabel": ylabel},
                    )
                    record_plot(
                        plot_id="similarity_N_scaled_mean_variance_vs_images_seen",
                        plot_kind="line",
                        panel=metric,
                        series_label=f"{distribution}/{representation} N={width}",
                        x_key=x_key,
                        y_key=y_key,
                        y_transform={"kind": "multiply", "factor": int(width)},
                        metadata={
                            **base_metadata,
                            "title": f"N*{ylabel} vs Images Seen",
                            "ylabel": f"N*{ylabel}",
                        },
                    )

    for metric, ylabel, title in [
        (
            "mean",
            "N*Mean similarity difference (across - within)",
            "N*Mean similarity difference (across - within) vs Images Seen",
        ),
        (
            "variance",
            "Similarity N*variance difference (across - within)",
            "Similarity N*variance difference (across - within) vs Images Seen",
        ),
    ]:
        for width in available_widths:
            for representation in ordered_reps:
                within_sub = similarity_stats_df[
                    (similarity_stats_df["width"] == width)
                    & (similarity_stats_df["representation"] == representation)
                    & (similarity_stats_df["distribution"] == "within")
                ][["images_seen", metric]].rename(columns={metric: "within_value"})
                across_sub = similarity_stats_df[
                    (similarity_stats_df["width"] == width)
                    & (similarity_stats_df["representation"] == representation)
                    & (similarity_stats_df["distribution"] == "across")
                ][["images_seen", metric]].rename(columns={metric: "across_value"})
                sub = within_sub.merge(across_sub, on="images_seen", how="inner").sort_values("images_seen")
                if sub.empty:
                    continue
                record_plot(
                    plot_id="similarity_mean_variance_difference_vs_images_seen",
                    plot_kind="derived_line",
                    panel=metric,
                    series_label=f"across-within/{representation} N={width}",
                    x_key=register_vector(sub["images_seen"].to_numpy(dtype=np.int64)),
                    y_sources={
                        "lhs_key": register_vector(sub["across_value"].to_numpy(dtype=np.float64)),
                        "rhs_key": register_vector(sub["within_value"].to_numpy(dtype=np.float64)),
                    },
                    y_transform={"kind": "difference_then_multiply", "factor": int(width)},
                    metadata={
                        "metric": metric,
                        "representation": str(representation),
                        "width": int(width),
                        "title": title,
                        "xlabel": "Images seen (P)",
                        "ylabel": ylabel,
                        "reference_hline": 0.0,
                    },
                )

    if isinstance(layerwise_df, pd.DataFrame) and not layerwise_df.empty:
        for width, sub in sorted(layerwise_df.groupby("width"), key=lambda item: int(item[0])):
            steps = sorted({int(v) for v in sub["images_seen"].dropna().tolist()})
            layer_indices = sorted({int(v) for v in sub["layer_index"].dropna().tolist()})
            if not steps or not layer_indices:
                continue
            step_to_col = {step: idx for idx, step in enumerate(steps)}
            layer_to_row = {layer_index: idx for idx, layer_index in enumerate(layer_indices)}
            matrix = np.full((len(layer_indices), len(steps)), np.nan, dtype=np.float64)
            for row in sub.itertuples(index=False):
                matrix[layer_to_row[int(row.layer_index)], step_to_col[int(row.images_seen)]] = float(row.w1_distance)
            x_key = register_vector(np.asarray(steps, dtype=np.int64))
            y_key = register_vector(np.asarray(layer_indices, dtype=np.int64))
            z_key = register_matrix(matrix)
            record_plot(
                plot_id=f"layerwise_weight_w1_width_{int(width)}_global",
                plot_kind="heatmap",
                x_key=x_key,
                y_key=y_key,
                z_key=z_key,
                metadata={
                    "width": int(width),
                    "mode": "global",
                    "title": f"Layerwise weight W1 heatmap (width={int(width)}, mode=global)",
                    "xlabel": "Images seen (P)",
                    "ylabel": "Layer depth",
                    "zlabel": "W1 distance",
                },
            )
            record_plot(
                plot_id=f"layerwise_weight_w1_width_{int(width)}_per_row",
                plot_kind="heatmap",
                x_key=x_key,
                y_key=y_key,
                z_key=z_key,
                z_transform={"kind": "row_normalize_per_row"},
                metadata={
                    "width": int(width),
                    "mode": "per_row",
                    "title": f"Layerwise weight W1 heatmap (width={int(width)}, mode=per_row)",
                    "xlabel": "Images seen (P)",
                    "ylabel": "Layer depth",
                    "zlabel": "Row-normalized W1",
                },
            )

    export_path = Path(out_path) if out_path is not None else out_dir / "plot_vectors.npz"
    np.savez_compressed(
        export_path,
        manifest_json=np.asarray(json.dumps(manifest, indent=2)),
        **array_store,
    )
    return {
        "path": str(export_path),
        "manifest": manifest,
        "unique_array_count": len(array_store),
        "plot_count": len(manifest["plots"]),
    }
