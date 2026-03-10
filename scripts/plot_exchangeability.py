#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot exchangeability analysis outputs.')
    parser.add_argument('--input-csv', default='outputs/exchangeability_metrics.csv', help='Input analysis CSV')
    parser.add_argument('--output-dir', default='outputs/plots_exchangeability', help='Directory for plots')
    return parser.parse_args()



def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        'width',
        'images_seen',
        'representation',
        'analysis_type',
    ]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(
            f'Input CSV is missing required columns: {missing_required}. '
            f'Expected raw analysis output from analyze_exchangeability.py (not summary CSVs).'
        )

    numeric_cols = [
        'width',
        'images_seen',
        'shuffle_id',
        'ks_distance',
        'ks_p_raw',
        'ks_sigma_two_sided',
        'ks_p_empirical',
        'ks_sigma_empirical_two_sided',
        'w1_distance',
        'w1_p_empirical',
        'w1_sigma_empirical_two_sided',
        'train_loss',
        'val_loss',
        'train_error',
        'val_error',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    return df



def _aggregate_for_curves(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = ['width', 'images_seen', 'representation', 'analysis_type']
    return (
        df.groupby(grp_cols, as_index=False)
        .agg(
            ks_distance=('ks_distance', 'mean'),
            ks_distance_p10=('ks_distance', lambda s: np.nanpercentile(s, 10)),
            ks_distance_p90=('ks_distance', lambda s: np.nanpercentile(s, 90)),
            w1_distance=('w1_distance', 'mean'),
            w1_distance_p10=('w1_distance', lambda s: np.nanpercentile(s, 10)),
            w1_distance_p90=('w1_distance', lambda s: np.nanpercentile(s, 90)),
        )
    )



def _plot_metric(
    curves: pd.DataFrame,
    metric: str,
    lo: str,
    hi: str,
    out_path: str,
    title: str,
    close: bool = True,
    analysis_types: Sequence[str] | None = None,
    analysis_labels: Mapping[str, str] | None = None,
    analysis_colors: Mapping[str, str] | None = None,
    width_colors: Mapping[int | str, str] | None = None,
    analysis_color_adjust: Mapping[str, float] | None = None,
    representation_linestyles: Mapping[str, str] | None = None,
    representation_order: Sequence[str] | None = None,
):
    plot_df = curves.copy()
    if analysis_types is not None:
        allowed = set(analysis_types)
        plot_df = plot_df[plot_df['analysis_type'].isin(allowed)].copy()

    if plot_df.empty:
        raise ValueError('No rows available for metric plotting after applying analysis-type filters.')

    if representation_order is None:
        ordered_representations = sorted(plot_df['representation'].astype(str).unique().tolist())
    else:
        present = set(plot_df['representation'].astype(str).unique().tolist())
        ordered_representations = [rep for rep in representation_order if rep in present]
        ordered_representations.extend(sorted(present - set(ordered_representations)))

    fig = plt.figure(figsize=(10, 6))
    group_cols = ['analysis_type', 'width', 'representation']
    for analysis_type, width, representation in sorted(
        plot_df[group_cols].drop_duplicates().itertuples(index=False, name=None),
        key=lambda t: (str(t[0]), float(t[1]), ordered_representations.index(str(t[2])) if str(t[2]) in ordered_representations else 10**9),
    ):
        sub = plot_df[
            (plot_df['analysis_type'] == analysis_type)
            & (plot_df['width'] == width)
            & (plot_df['representation'] == representation)
        ].sort_values('images_seen')
        if sub.empty:
            continue

        display_analysis = (
            analysis_labels.get(str(analysis_type), str(analysis_type))
            if analysis_labels is not None
            else str(analysis_type)
        )
        color = None
        if width_colors is not None:
            width_int = int(width)
            width_key = width_int if width_int in width_colors else str(width_int)
            base_color = width_colors.get(width_key)
            if base_color is not None and analysis_color_adjust is not None:
                adjust = float(analysis_color_adjust.get(str(analysis_type), 0.0))
                if adjust > 0.0:
                    base_rgba = np.asarray(mcolors.to_rgba(base_color), dtype=np.float64)
                    white = np.array([1.0, 1.0, 1.0, base_rgba[3]], dtype=np.float64)
                    base_color = tuple(((1.0 - adjust) * base_rgba + adjust * white).tolist())
            color = base_color
        elif analysis_colors is not None:
            color = analysis_colors.get(str(analysis_type))
        linestyle = (
            representation_linestyles.get(str(representation), '-')
            if representation_linestyles is not None
            else '-'
        )
        label = f'{display_analysis}/{representation} N={int(width)}'
        plt.plot(
            sub['images_seen'],
            sub[metric],
            label=label,
            color=color,
            linestyle=linestyle,
        )
        plt.fill_between(
            sub['images_seen'],
            sub[lo],
            sub[hi],
            alpha=0.12,
            color=color,
        )

    plt.xscale('log')
    plt.xlabel('Images seen (P)')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    if close:
        plt.close(fig)
    return fig



def _plot_train_val(
    metrics_df: pd.DataFrame,
    out_dir: str,
    close: bool = True,
):
    dedup = (
        metrics_df[['width', 'images_seen', 'train_loss', 'val_loss', 'train_error', 'val_error']]
        .drop_duplicates()
        .sort_values(['width', 'images_seen'])
    )

    figures = []
    for metric in ['train_loss', 'val_loss', 'train_error', 'val_error']:
        fig = plt.figure(figsize=(8, 5))
        for width, sub in dedup.groupby('width'):
            plt.plot(sub['images_seen'], sub[metric], marker='o', label=f'N={int(width)}')
        plt.xscale('log')
        plt.xlabel('Images seen (P)')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} vs Images Seen')
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{metric}_vs_images_seen.pdf'), bbox_inches='tight')
        figures.append(fig)
        if close:
            plt.close(fig)
    return figures


def _plot_within_observed_significance(
    metrics_df: pd.DataFrame,
    out_dir: str,
    close: bool = True,
    width_colors: Mapping[int | str, str] | None = None,
    representation_linestyles: Mapping[str, str] | None = None,
    representation_order: Sequence[str] | None = None,
):
    subset = metrics_df[
        (metrics_df['analysis_type'] == 'within_vs_across_real')
        & (metrics_df['shuffle_id'] == -1)
    ].copy()
    if subset.empty:
        return []

    null_counts = (
        metrics_df[
            (metrics_df['analysis_type'] == 'within_shuffled_vs_across_real')
            & (metrics_df['shuffle_id'] >= 0)
        ]
        .groupby(['width', 'images_seen', 'representation'], as_index=False)
        .size()
        .rename(columns={'size': 'empirical_null_count'})
    )
    subset = subset.merge(
        null_counts,
        on=['width', 'images_seen', 'representation'],
        how='left',
    )

    metric_specs = [
        (
            'ks_sigma_two_sided',
            'KS Raw Sigma (Two-Sided) vs Images Seen',
            'ks_sigma_two_sided_vs_images_seen.pdf',
            None,
        ),
        (
            'ks_sigma_empirical_two_sided',
            'KS Empirical Sigma vs Images Seen',
            'ks_sigma_empirical_two_sided_vs_images_seen.pdf',
            'ks_p_empirical',
        ),
        (
            'w1_sigma_empirical_two_sided',
            'W1 Empirical Sigma vs Images Seen',
            'w1_sigma_empirical_two_sided_vs_images_seen.pdf',
            'w1_p_empirical',
        ),
    ]

    figures = []
    for metric, title, filename, empirical_p_col in metric_specs:
        if metric not in subset.columns:
            continue
        plot_df = subset[np.isfinite(subset[metric])].copy()
        if plot_df.empty:
            continue

        fig = plt.figure(figsize=(10, 6))
        if representation_order is None:
            ordered_representations = sorted(plot_df['representation'].astype(str).unique().tolist())
        else:
            present = set(plot_df['representation'].astype(str).unique().tolist())
            ordered_representations = [rep for rep in representation_order if rep in present]
            ordered_representations.extend(sorted(present - set(ordered_representations)))

        group_rows = plot_df[['representation', 'width']].drop_duplicates()
        ordered_groups = sorted(
            group_rows.itertuples(index=False, name=None),
            key=lambda t: (
                ordered_representations.index(str(t[0])) if str(t[0]) in ordered_representations else 10**9,
                float(t[1]),
            ),
        )

        for representation, width in ordered_groups:
            sub = plot_df[
                (plot_df['representation'] == representation)
                & (plot_df['width'] == width)
            ]
            sub = sub.sort_values('images_seen')
            label = f'{representation} N={int(width)}'
            color = None
            if width_colors is not None:
                width_int = int(width)
                width_key = width_int if width_int in width_colors else str(width_int)
                color = width_colors.get(width_key)
            linestyle = (
                representation_linestyles.get(str(representation), '-')
                if representation_linestyles is not None
                else '-'
            )

            plt.plot(
                sub['images_seen'],
                sub[metric],
                marker='o',
                label=label,
                color=color,
                linestyle=linestyle,
            )

            # Mark finite-shuffle floor saturation points for empirical statistics.
            if empirical_p_col and empirical_p_col in sub.columns and 'empirical_null_count' in sub.columns:
                floor_p = 1.0 / (sub['empirical_null_count'] + 1.0)
                floor_mask = (
                    np.isfinite(sub[empirical_p_col].to_numpy())
                    & np.isfinite(floor_p.to_numpy())
                    & np.isclose(
                        sub[empirical_p_col].to_numpy(),
                        floor_p.to_numpy(),
                        rtol=1e-6,
                        atol=1e-12,
                    )
                )
                if np.any(floor_mask):
                    floor_sub = sub.loc[floor_mask]
                    plt.scatter(
                        floor_sub['images_seen'],
                        floor_sub[metric],
                        marker='x',
                        s=40,
                        linewidths=1.4,
                        color=color if color is not None else 'black',
                        alpha=0.8,
                        label='_nolegend_',
                    )

        plt.xscale('log')
        plt.xlabel('Images seen (P)')
        ylabel = metric.replace('_', ' ').title()
        if empirical_p_col is not None:
            ylabel = ylabel.replace(' Two Sided', '')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
        figures.append(fig)
        if close:
            plt.close(fig)

    return figures



def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = _prepare(df)

    curves = _aggregate_for_curves(df)

    _plot_metric(
        curves,
        metric='ks_distance',
        lo='ks_distance_p10',
        hi='ks_distance_p90',
        out_path=os.path.join(args.output_dir, 'ks_distance_vs_images_seen.pdf'),
        title='KS Distance vs Images Seen',
    )
    _plot_metric(
        curves,
        metric='w1_distance',
        lo='w1_distance_p10',
        hi='w1_distance_p90',
        out_path=os.path.join(args.output_dir, 'w1_distance_vs_images_seen.pdf'),
        title='W1 Distance vs Images Seen',
    )

    _plot_train_val(df, args.output_dir)
    _plot_within_observed_significance(df, args.output_dir)

    print(f'Wrote plots to {args.output_dir}')


if __name__ == '__main__':
    main()
