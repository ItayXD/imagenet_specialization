#!/usr/bin/env python3
import argparse
import os
from os.path import dirname, isabs, join

from scripts.build_cifar5m_sweep import (
    DEFAULT_CLUSTER_ROOT,
    build_configs,
    clear_cifar5m_exchangeability_configs,
    members_per_group_for_width,
)
from scripts.build_exchangeability_manifest import _parse_row, write_manifest
from src.run.constants import CIFAR5M_BASE_SAVE_DIR

PROJECT_ROOT = dirname(dirname(__file__))

DEFAULT_WIDTH = 64
DEFAULT_LEARNING_RATES = [
    0.0002,
    0.0004,
    0.0008,
    0.0016,
    0.0032,
    0.0064,
    0.0128,
    0.0256,
    0.0512,
    0.1024,
]
DEFAULT_CONFIG_PREFIX = 'cifar5m_exchangeability_muon_lr_tuning_w64'
DEFAULT_RUN_PREFIX = 'exchangeability_cifar5m_muon_lr_tuning_w64'
DEFAULT_BASE_DIR_ROOT = DEFAULT_CLUSTER_ROOT + '/exchangeability_runs/cifar5m_muon_lr_tuning/w64'
DEFAULT_MANIFEST_PATH = 'conf/manifests/cifar5m/exchangeability_manifest_muon_lr_tuning_w64.csv'
DEFAULT_BASE_SAVE_DIR = (
    CIFAR5M_BASE_SAVE_DIR or '/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_cifar5m'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build width-64 CIFAR-5M Muon learning-rate tuning configs.')
    parser.add_argument('--config-dir', default='conf/experiment', help='Output directory for generated configs')
    parser.add_argument('--manifest-output', default=DEFAULT_MANIFEST_PATH, help='Output manifest CSV path')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH, help='Model width to tune')
    parser.add_argument(
        '--learning-rates',
        type=float,
        nargs='+',
        default=DEFAULT_LEARNING_RATES,
        help='Learning rates to sweep',
    )
    parser.add_argument('--seed-base', type=int, default=20260319, help='Base RNG seed for task seeds')
    parser.add_argument('--data-seed', type=int, default=2423, help='Dataset seed')
    parser.add_argument(
        '--config-prefix',
        default=DEFAULT_CONFIG_PREFIX,
        help='Filename prefix for generated configs',
    )
    parser.add_argument(
        '--run-prefix',
        default=DEFAULT_RUN_PREFIX,
        help='Run-id prefix for generated configs',
    )
    parser.add_argument(
        '--base-dir-root',
        default=DEFAULT_BASE_DIR_ROOT,
        help='Base directory root used for generated configs',
    )
    parser.add_argument(
        '--base-save-dir',
        default=DEFAULT_BASE_SAVE_DIR,
        help='Base save directory written into the manifest',
    )
    return parser.parse_args()


def _resolve_path(path: str) -> str:
    if isabs(path):
        return path
    return join(PROJECT_ROOT, path)


def main() -> None:
    args = parse_args()
    config_dir = _resolve_path(args.config_dir)
    manifest_output = _resolve_path(args.manifest_output)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(dirname(manifest_output), exist_ok=True)

    clear_cifar5m_exchangeability_configs(config_dir, args.config_prefix)

    config_paths: list[str] = []
    target_members_per_width = members_per_group_for_width(args.width)

    for lr_idx, lr in enumerate(args.learning_rates):
        run_id = f'{args.run_prefix}_r{lr_idx:02d}'
        config_name_template = f'{args.config_prefix}_r{lr_idx:02d}_g{{group_id}}.yaml'
        base_dir_template = f'{args.base_dir_root}/r{lr_idx:02d}/g{{group_id}}'
        outputs = build_configs(
            seed_base=args.seed_base,
            data_seed=args.data_seed,
            eta_0=lr,
            optimizer='muon',
            widths=(args.width,),
            target_members_per_width=target_members_per_width,
            run_id=run_id,
            config_name_template=config_name_template,
            base_dir_template=base_dir_template,
            use_warmup_cosine_decay=False,
        )

        for name, content in outputs:
            config_path = join(config_dir, name)
            with open(config_path, 'w', encoding='utf-8') as handle:
                handle.write(content)
            config_paths.append(config_path)

    rows = [
        _parse_row(job_id=job_id, cfg_path=cfg_path, base_save_dir=args.base_save_dir)
        for job_id, cfg_path in enumerate(sorted(config_paths))
    ]
    write_manifest(rows, manifest_output)

    print(f'Wrote {len(config_paths)} configs to {config_dir}')
    print(f'Wrote manifest to {manifest_output}')


if __name__ == '__main__':
    main()
