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
    numeric_cols = [
        'width',
        'images_seen',
        'shuffle_id',
        'ks_distance',
        'ks_p_raw',
        'ks_sigma_two_sided',
        'w1_distance',
        'train_loss',
        'val_loss',
        'train_error',
        'val_error',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
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

    print(f'Wrote plots to {args.output_dir}')


if __name__ == '__main__':
    main()
