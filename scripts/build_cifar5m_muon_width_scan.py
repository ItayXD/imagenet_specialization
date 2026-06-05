#!/usr/bin/env python3
import argparse
import os
from os.path import dirname, isabs, join

from scripts.build_cifar5m_sweep import (
    DEFAULT_CLUSTER_ROOT,
    WIDTHS,
    build_configs,
    clear_cifar5m_exchangeability_configs,
)
from scripts.build_exchangeability_manifest import _parse_row, write_manifest
from src.run.constants import CIFAR5M_BASE_SAVE_DIR

PROJECT_ROOT = dirname(dirname(__file__))

DEFAULT_CONFIG_PREFIX = 'cifar5m_exchangeability_muon_width_scan'
DEFAULT_RUN_ID = 'exchangeability_cifar5m_muon'
DEFAULT_BASE_DIR_ROOT = DEFAULT_CLUSTER_ROOT + '/exchangeability_runs/cifar5m_muon'
DEFAULT_MANIFEST_PATH = 'conf/manifests/cifar5m/exchangeability_manifest_muon.csv'
DEFAULT_BASE_SAVE_DIR = (
    CIFAR5M_BASE_SAVE_DIR or '/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_cifar5m'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build CIFAR-5M Muon width-scan configs with one member per width.')
    parser.add_argument('--config-dir', default='conf/experiment', help='Output directory for generated configs')
    parser.add_argument('--manifest-output', default=DEFAULT_MANIFEST_PATH, help='Output manifest CSV path')
    parser.add_argument('--seed-base', type=int, default=20260319, help='Base RNG seed for task seeds')
    parser.add_argument('--data-seed', type=int, default=2423, help='Dataset seed')
    parser.add_argument('--eta-0', type=float, default=6e-3, help='Muon learning rate')
    parser.add_argument(
        '--config-prefix',
        default=DEFAULT_CONFIG_PREFIX,
        help='Filename prefix for generated configs',
    )
    parser.add_argument(
        '--run-id',
        default=DEFAULT_RUN_ID,
        help='Run-id written into generated configs',
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

    clear_cifar5m_exchangeability_configs(config_dir, f'{args.config_prefix}_')

    outputs = build_configs(
        seed_base=args.seed_base,
        data_seed=args.data_seed,
        eta_0=args.eta_0,
        optimizer='muon',
        widths=WIDTHS,
        target_members_per_width=1,
        members_per_group_override=1,
        run_id=args.run_id,
        config_name_template=f'{args.config_prefix}_w{{width}}_g{{group_id}}.yaml',
        base_dir_template=f'{args.base_dir_root}/w{{width}}/g{{group_id}}',
        use_warmup_cosine_decay=True,
    )

    config_paths: list[str] = []
    for name, content in outputs:
        config_path = join(config_dir, name)
        with open(config_path, 'w', encoding='utf-8') as handle:
            handle.write(content)
        config_paths.append(config_path)

    rows = [
        _parse_row(job_id=job_id, cfg_path=cfg_path, base_save_dir=args.base_save_dir)
        for job_id, cfg_path in enumerate(config_paths)
    ]
    write_manifest(rows, manifest_output)

    print(f'Wrote {len(config_paths)} configs to {config_dir}')
    print(f'Wrote manifest to {manifest_output}')


if __name__ == '__main__':
    main()
