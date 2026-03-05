#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
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
):
    fig = plt.figure(figsize=(10, 6))
    for (representation, analysis_type), sub in curves.groupby(['representation', 'analysis_type']):
        for width, wsub in sub.groupby('width'):
            wsub = wsub.sort_values('images_seen')
            label = f'{representation}/{analysis_type} N={int(width)}'
            plt.plot(wsub['images_seen'], wsub[metric], label=label)
            plt.fill_between(wsub['images_seen'], wsub[lo], wsub[hi], alpha=0.12)

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
            'KS Empirical Sigma (Two-Sided) vs Images Seen',
            'ks_sigma_empirical_two_sided_vs_images_seen.pdf',
            'ks_p_empirical',
        ),
        (
            'w1_sigma_empirical_two_sided',
            'W1 Empirical Sigma (Two-Sided) vs Images Seen',
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
        for (representation, width), sub in plot_df.groupby(['representation', 'width']):
            sub = sub.sort_values('images_seen')
            label = f'{representation} N={int(width)}'
            plt.plot(sub['images_seen'], sub[metric], marker='o', label=label)

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
                        color='black',
                        alpha=0.8,
                        label=f'{label} (floor)',
                    )

        plt.xscale('log')
        plt.xlabel('Images seen (P)')
        plt.ylabel(metric.replace('_', ' ').title())
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
