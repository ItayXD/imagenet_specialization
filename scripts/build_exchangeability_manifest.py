#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from os.path import basename, dirname, join

from omegaconf import OmegaConf

from src.run.constants import BASE_SAVE_DIR


@dataclass
class ManifestRow:
    job_id: int
    dataset: str
    run_id: str
    width: int
    group_id: int
    member_seed_list: list[int]
    data_seed: int
    target_images_seen: int
    p_targets_images_seen: list[int]
    base_dir: str
    save_dir: str
    wandb_group: str
    experiment_name: str



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build manifest for exchangeability SLURM array runs.')
    parser.add_argument('--config-dir', default='conf/experiment', help='Directory containing exchangeability_*.yaml configs')
    parser.add_argument('--output', default='conf/exchangeability_manifest.csv', help='Manifest CSV path')
    parser.add_argument('--base-save-dir', default=BASE_SAVE_DIR or '/tmp/exchangeability_outputs', help='Base save directory used by training runs')
    parser.add_argument('--dataset', default='', help='Optional dataset filter (for example imagenet or cifar5m)')
    return parser.parse_args()



def _derive_member_seeds(task_seed: int, member_group_size: int) -> list[int]:
    rng = random.Random(task_seed)
    return [rng.randrange(0, 10**9) for _ in range(member_group_size)]



def _iter_config_paths(config_dir: str) -> list[str]:
    names = sorted(
        n for n in os.listdir(config_dir)
        if ('exchangeability' in n) and n.endswith('.yaml')
    )
    return [join(config_dir, n) for n in names]



def _parse_row(job_id: int, cfg_path: str, base_save_dir: str) -> ManifestRow:
    cfg = OmegaConf.load(cfg_path)
    dataset = str(cfg.setting.dataset)
    tp = cfg.hyperparams.task_list[0].training_params
    mp = cfg.hyperparams.task_list[0].model_params
    task_seed = int(cfg.hyperparams.task_list[0].seed)

    width = int(mp.N)
    group_id = int(tp.group_id)
    run_id = str(tp.run_id)

    member_group_size = int(tp.member_group_size)
    member_seed_list = _derive_member_seeds(task_seed, member_group_size)

    experiment_name = basename(cfg_path).replace('.yaml', '')
    save_dir = join(base_save_dir, run_id, f'width_{width}', f'group_{group_id}')

    return ManifestRow(
        job_id=job_id,
        dataset=dataset,
        run_id=run_id,
        width=width,
        group_id=group_id,
        member_seed_list=member_seed_list,
        data_seed=int(cfg.hyperparams.data_params.data_seed),
        target_images_seen=int(tp.target_images_seen),
        p_targets_images_seen=[int(v) for v in tp.p_targets_images_seen],
        base_dir=str(cfg.base_dir),
        save_dir=save_dir,
        wandb_group=run_id,
        experiment_name=experiment_name,
    )



def write_manifest(rows: list[ManifestRow], output_path: str) -> None:
    os.makedirs(dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'job_id',
                'dataset',
                'run_id',
                'width',
                'group_id',
                'member_seed_list',
                'data_seed',
                'target_images_seen',
                'p_targets_images_seen',
                'base_dir',
                'save_dir',
                'wandb_group',
                'experiment_name',
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    'job_id': row.job_id,
                    'dataset': row.dataset,
                    'run_id': row.run_id,
                    'width': row.width,
                    'group_id': row.group_id,
                    'member_seed_list': json.dumps(row.member_seed_list),
                    'data_seed': row.data_seed,
                    'target_images_seen': row.target_images_seen,
                    'p_targets_images_seen': json.dumps(row.p_targets_images_seen),
                    'base_dir': row.base_dir,
                    'save_dir': row.save_dir,
                    'wandb_group': row.wandb_group,
                    'experiment_name': row.experiment_name,
                }
            )



def main() -> None:
    args = parse_args()
    config_paths = _iter_config_paths(args.config_dir)
    rows = []
    for path in config_paths:
        row = _parse_row(job_id=len(rows), cfg_path=path, base_save_dir=args.base_save_dir)
        if args.dataset and row.dataset != args.dataset:
            continue
        rows.append(row)
    write_manifest(rows, args.output)
    print(f'Wrote {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
