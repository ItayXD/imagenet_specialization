#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot final CIFAR-5M loss as a function of learning rate.')
    parser.add_argument(
        '--manifest-path',
        default='conf/manifests/cifar5m/exchangeability_manifest_sgd_lr_tuning_w64.csv',
        help='Manifest CSV path',
    )
    parser.add_argument(
        '--metric-key',
        choices=('train_loss', 'val_loss'),
        default='val_loss',
        help='Final loss metric to plot',
    )
    parser.add_argument(
        '--output-dir',
        default='artifacts/cifar5m_sgd_lr_tuning_w64',
        help='Directory where the summary CSV and plot are written',
    )
    parser.add_argument(
        '--title',
        default='CIFAR-5M width-64 LR tuning',
        help='Plot title',
    )
    return parser.parse_args()


def _load_final_row(metrics_path: Path) -> dict[str, float | int]:
    final_row = None
    with metrics_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                final_row = json.loads(stripped)
    if final_row is None:
        raise ValueError(f'No metrics found in {metrics_path}')
    return final_row


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | str]] = []
    with manifest_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metrics_path = Path(row['save_dir']) / 'metrics.jsonl'
            final_row = _load_final_row(metrics_path)
            expected_images_seen = int(row['target_images_seen'])
            final_images_seen = int(final_row['images_seen'])
            if final_images_seen != expected_images_seen:
                raise ValueError(
                    f'Incomplete run for {row["run_id"]}: expected images_seen={expected_images_seen}, '
                    f'found {final_images_seen}'
                )

            eta_0 = float(row.get('eta_0') or final_row['lr'])
            summary_rows.append(
                {
                    'run_id': row['run_id'],
                    'optimizer': row.get('optimizer', 'sgd'),
                    'eta_0': eta_0,
                    'final_train_loss': float(final_row['train_loss']),
                    'final_val_loss': float(final_row['val_loss']),
                    'final_images_seen': final_images_seen,
                }
            )

    summary_rows.sort(key=lambda item: float(item['eta_0']))

    summary_path = output_dir / 'final_loss_vs_lr.csv'
    with summary_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'run_id',
                'optimizer',
                'eta_0',
                'final_train_loss',
                'final_val_loss',
                'final_images_seen',
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    x_values = [float(row['eta_0']) for row in summary_rows]
    y_values = [float(row[f'final_{args.metric_key}']) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x_values, y_values, marker='o')
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel(f'Final {args.metric_key.replace("_", " ")}')
    ax.set_title(args.title)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / f'final_{args.metric_key}_vs_lr.png'
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f'Wrote summary to {summary_path}')
    print(f'Wrote plot to {plot_path}')


if __name__ == '__main__':
    main()
