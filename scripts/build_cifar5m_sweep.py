import random
import argparse
import os
from os.path import dirname, join
import sys

from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config_structs import (
    Config,
    DataParams,
    ModelParams,
    Setting,
    TaskConfig,
    TaskListConfig,
    TrainingParams,
)

CONFIG_DIR = '../conf/experiment'
DEFAULT_CONFIG_NAME_TEMPLATE = 'cifar5m_exchangeability_w{width}_g{group_id}.yaml'

WIDTHS = (32, 64, 128, 256, 512)
TARGET_MEMBERS_PER_WIDTH = 16
DEFAULT_MEMBERS_PER_GROUP = 4
TARGET_IMAGES_SEEN = 5_000_000
FULL_CIFAR5M_TRAIN_SIZE = 5_000_000

DEFAULT_CLUSTER_ROOT = os.environ.get(
    'EXCHANGEABILITY_ROOT',
    '/n/netscratch/kempner_pehlevan_lab/Lab/ilavie',
)
DEFAULT_BASE_DIR_TEMPLATE = DEFAULT_CLUSTER_ROOT + '/exchangeability_runs/cifar5m/w{width}/g{group_id}'
DEFAULT_RUN_ID = 'exchangeability_cifar5m'
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'imagenet_specialization')
WANDB_ENTITY = os.environ.get('WANDB_ENTITY', '')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build CIFAR-5M exchangeability experiment configs.')
    parser.add_argument('--config-dir', default=CONFIG_DIR, help='Output directory for generated configs')
    parser.add_argument('--seed-base', type=int, default=20260319, help='Base RNG seed for task seeds')
    parser.add_argument('--data-seed', type=int, default=2423, help='Dataset seed')
    parser.add_argument('--eta-0', type=float, default=6e-3, help='Learning rate')
    parser.add_argument('--optimizer', default='adam', choices=('adam', 'sgd', 'muon'), help='Optimizer name')
    parser.add_argument(
        '--widths',
        type=int,
        nargs='+',
        default=list(WIDTHS),
        help='Widths to generate configs for',
    )
    parser.add_argument(
        '--target-members-per-width',
        type=int,
        default=TARGET_MEMBERS_PER_WIDTH,
        help='Total ensemble members to allocate per width',
    )
    parser.add_argument(
        '--members-per-group-override',
        type=int,
        default=0,
        help='Optional fixed member_group_size/ensemble_size to use for every width; 0 keeps width-based defaults',
    )
    parser.add_argument(
        '--run-id',
        default=DEFAULT_RUN_ID,
        help='training_params.run_id and output folder name under BASE_SAVE_DIR',
    )
    parser.add_argument(
        '--config-name-template',
        default=DEFAULT_CONFIG_NAME_TEMPLATE,
        help='Filename template for generated experiment configs',
    )
    parser.add_argument(
        '--base-dir-template',
        default=DEFAULT_BASE_DIR_TEMPLATE,
        help='base_dir template for generated configs',
    )
    parser.add_argument(
        '--clear-prefix',
        default='',
        help='Optional filename prefix to delete before regenerating configs',
    )
    parser.set_defaults(use_warmup_cosine_decay=True)
    parser.add_argument(
        '--use-warmup-cosine-decay',
        dest='use_warmup_cosine_decay',
        action='store_true',
        help='Enable warmup cosine decay scheduling',
    )
    parser.add_argument(
        '--no-warmup-cosine-decay',
        dest='use_warmup_cosine_decay',
        action='store_false',
        help='Use a constant learning rate',
    )
    return parser.parse_args()


def members_per_group_for_width(width: int, override: int | None = None) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError('members_per_group_override must be > 0 when provided')
        return override
    if width >= 512:
        return 1
    if width == 256:
        return 2
    return DEFAULT_MEMBERS_PER_GROUP


def num_groups_for_width(
    width: int,
    target_members_per_width: int = TARGET_MEMBERS_PER_WIDTH,
    members_per_group_override: int | None = None,
) -> int:
    members_per_group = members_per_group_for_width(width, override=members_per_group_override)
    if target_members_per_width <= 0:
        raise ValueError('target_members_per_width must be > 0')
    if target_members_per_width % members_per_group != 0:
        raise ValueError(
            'target_members_per_width must be divisible by members_per_group; '
            f'got target_members_per_width={target_members_per_width}, members_per_group={members_per_group}'
        )
    return target_members_per_width // members_per_group


def minibatch_size_for_width(width: int) -> int:
    del width
    return 1024


def microbatch_size_for_width(width: int) -> int:
    del width
    return 128


def num_workers_for_width(width: int) -> int:
    del width
    return 8


def build_p_targets() -> list[int]:
    return [
        10000,
        21544,
        46415,
        100000,
        138949,
        193069,
        268269,
        372759,
        517947,
        719686,
        1000000,
        1389495,
        1930698,
        2682695,
        3727593,
        5000000,
    ]


def clear_cifar5m_exchangeability_configs(folder: str, prefix: str) -> None:
    for name in os.listdir(folder):
        if name.startswith(prefix) and name.endswith('.yaml'):
            os.unlink(join(folder, name))


def config_name_prefix(config_name_template: str) -> str:
    prefix = config_name_template.split('{width}', 1)[0]
    if not prefix:
        raise ValueError(f'Could not derive filename prefix from template: {config_name_template}')
    return prefix


def build_configs(
    *,
    seed_base: int = 20260319,
    data_seed: int = 2423,
    eta_0: float = 6e-3,
    optimizer: str = 'adam',
    widths: tuple[int, ...] = WIDTHS,
    target_members_per_width: int = TARGET_MEMBERS_PER_WIDTH,
    members_per_group_override: int | None = None,
    run_id: str = DEFAULT_RUN_ID,
    config_name_template: str = DEFAULT_CONFIG_NAME_TEMPLATE,
    base_dir_template: str = DEFAULT_BASE_DIR_TEMPLATE,
    use_warmup_cosine_decay: bool = True,
) -> list[tuple[str, str]]:
    rng = random.Random(seed_base)
    p_targets = build_p_targets()

    outputs: list[tuple[str, str]] = []

    for width in widths:
        members_per_group = members_per_group_for_width(width, override=members_per_group_override)
        num_groups = num_groups_for_width(
            width,
            target_members_per_width=target_members_per_width,
            members_per_group_override=members_per_group_override,
        )

        for group_id in range(num_groups):
            task_seed = rng.randrange(0, 10**9)

            tp = TrainingParams(
                eta_0=eta_0,
                optimizer=optimizer,
                minibatch_size=minibatch_size_for_width(width),
                microbatch_size=microbatch_size_for_width(width),
                num_workers=num_workers_for_width(width),
                epochs=50,
                ensemble_subsets=1,
                use_warmup_cosine_decay=use_warmup_cosine_decay,
                target_images_seen=TARGET_IMAGES_SEEN,
                p_targets_images_seen=p_targets,
                wandb_enabled=True,
                wandb_project=WANDB_PROJECT,
                wandb_entity=WANDB_ENTITY,
                wandb_mode='online',
                run_id=run_id,
                width=width,
                group_id=group_id,
                member_group_size=members_per_group,
                probe_batch_size=1024,
                log_every_tranches=10,
                max_tranches=0,
            )

            mp = ModelParams(
                BASE_N=64,
                N=width,
                ensemble_size=members_per_group,
                dtype='bfloat16',
            )

            task = TaskConfig(training_params=tp, model_params=mp, seed=task_seed)
            data_params = DataParams(
                P=FULL_CIFAR5M_TRAIN_SIZE,
                data_seed=data_seed,
                root_dir='data-dir',
                val_P=1024,
            )
            tlc = TaskListConfig(task_list=[task], data_params=data_params)
            cfg = Config(
                setting=Setting(dataset='cifar5m', model='resnet18'),
                hyperparams=tlc,
                base_dir=base_dir_template.format(width=width, group_id=group_id),
            )

            cfg_name = config_name_template.format(width=width, group_id=group_id)
            cfg_text = '# @package _global_\n' + OmegaConf.to_yaml(cfg)
            outputs.append((cfg_name, cfg_text))

    return outputs


if __name__ == '__main__':
    args = parse_args()
    curr_dir = dirname(__file__)
    config_save_folder = join(curr_dir, args.config_dir)
    clear_prefix = args.clear_prefix or config_name_prefix(args.config_name_template)
    clear_cifar5m_exchangeability_configs(config_save_folder, clear_prefix)

    for name, content in build_configs(
        seed_base=args.seed_base,
        data_seed=args.data_seed,
        eta_0=args.eta_0,
        optimizer=args.optimizer,
        widths=tuple(args.widths),
        target_members_per_width=args.target_members_per_width,
        members_per_group_override=(args.members_per_group_override or None),
        run_id=args.run_id,
        config_name_template=args.config_name_template,
        base_dir_template=args.base_dir_template,
        use_warmup_cosine_decay=args.use_warmup_cosine_decay,
    ):
        with open(join(config_save_folder, name), 'w', encoding='utf-8') as f:
            f.write(content)
