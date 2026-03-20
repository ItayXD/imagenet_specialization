#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare ImageNet and CIFAR-5M exchangeability CSVs on common checkpoints.')
    parser.add_argument('--imagenet-csv', required=True, help='ImageNet analysis CSV')
    parser.add_argument('--cifar5m-csv', required=True, help='CIFAR-5M analysis CSV')
    parser.add_argument('--output-dir', default='outputs/compare/imagenet_vs_cifar5m', help='Comparison output directory')
    return parser.parse_args()


def _load_dataset_csv(path: str, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'dataset' not in df.columns:
        df['dataset'] = dataset
    else:
        df['dataset'] = df['dataset'].fillna(dataset).replace('', dataset)
    numeric_cols = [
        'width',
        'images_seen',
        'shuffle_id',
        'ks_distance',
        'w1_distance',
        'train_loss',
        'val_loss',
        'train_error',
        'val_error',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _common_images_seen(dfs: list[pd.DataFrame]) -> list[int]:
    step_sets = []
    for df in dfs:
        observed = df[
            (df['analysis_type'] == 'within_vs_across_real')
            & (df['shuffle_id'] == -1)
        ]
        step_sets.append(set(observed['images_seen'].dropna().astype(int).tolist()))
    common = set.intersection(*step_sets) if step_sets else set()
    return sorted(common)


def _observed_rows(df: pd.DataFrame, common_steps: list[int]) -> pd.DataFrame:
    return df[
        (df['analysis_type'] == 'within_vs_across_real')
        & (df['shuffle_id'] == -1)
        & (df['images_seen'].isin(common_steps))
    ].copy()


def _plot_metric(observed: pd.DataFrame, metric: str, out_path: str, title: str) -> None:
    fig = plt.figure(figsize=(10, 6))
    for (dataset, width, representation), sub in observed.groupby(['dataset', 'width', 'representation']):
        sub = sub.sort_values('images_seen')
        linestyle = '-' if str(dataset) == 'imagenet' else '--'
        label = f'{dataset}/{representation} N={int(width)}'
        plt.plot(sub['images_seen'], sub[metric], linestyle=linestyle, label=label)
    plt.xscale('log')
    plt.xlabel('Images seen (common checkpoints)')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _plot_train_metrics(observed: pd.DataFrame, out_dir: str) -> None:
    metric_rows = (
        observed[['dataset', 'width', 'images_seen', 'train_loss', 'val_loss', 'train_error', 'val_error']]
        .drop_duplicates()
        .sort_values(['dataset', 'width', 'images_seen'])
    )
    for metric in ['train_loss', 'val_loss', 'train_error', 'val_error']:
        fig = plt.figure(figsize=(9, 5))
        for (dataset, width), sub in metric_rows.groupby(['dataset', 'width']):
            linestyle = '-' if str(dataset) == 'imagenet' else '--'
            label = f'{dataset} N={int(width)}'
            plt.plot(sub['images_seen'], sub[metric], linestyle=linestyle, marker='o', label=label)
        plt.xscale('log')
        plt.xlabel('Images seen (common checkpoints)')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{metric}_comparison.pdf'), bbox_inches='tight')
        plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    imagenet_df = _load_dataset_csv(args.imagenet_csv, 'imagenet')
    cifar_df = _load_dataset_csv(args.cifar5m_csv, 'cifar5m')
    common_steps = _common_images_seen([imagenet_df, cifar_df])
    if not common_steps:
        raise RuntimeError('No common observed checkpoints were found between ImageNet and CIFAR-5M.')

    observed = pd.concat(
        [
            _observed_rows(imagenet_df, common_steps),
            _observed_rows(cifar_df, common_steps),
        ],
        ignore_index=True,
    )

    observed.to_csv(os.path.join(args.output_dir, 'common_checkpoint_observed_rows.csv'), index=False)
    latest_step = max(common_steps)
    latest = observed[observed['images_seen'] == latest_step].copy()
    latest.to_csv(os.path.join(args.output_dir, 'latest_common_checkpoint_summary.csv'), index=False)

    _plot_metric(
        observed,
        metric='ks_distance',
        out_path=os.path.join(args.output_dir, 'ks_distance_common_checkpoints.pdf'),
        title='KS Distance on Common Checkpoints',
    )
    _plot_metric(
        observed,
        metric='w1_distance',
        out_path=os.path.join(args.output_dir, 'w1_distance_common_checkpoints.pdf'),
        title='W1 Distance on Common Checkpoints',
    )
    _plot_train_metrics(observed, args.output_dir)

    print(
        f'Wrote comparison outputs to {os.path.abspath(args.output_dir)} '
        f'using {len(common_steps)} common checkpoints; latest_common_step={latest_step}.'
    )


if __name__ == '__main__':
    main()
