import os
from os.path import dirname, join
import random
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
CONFIG_NAME = 'exchangeability_w{width}_g{group_id}.yaml'

WIDTHS = (32, 64, 128, 256, 512)
NUM_GROUPS = 4
MEMBERS_PER_GROUP = 4

DEFAULT_CLUSTER_ROOT = os.environ.get(
    'EXCHANGEABILITY_ROOT',
    '/n/netscratch/kempner_pehlevan_lab/Lab/ilavie',
)
BASE_DIR = DEFAULT_CLUSTER_ROOT + '/exchangeability_runs/w{width}/g{group_id}'
RUN_ID = 'exchangeability'
FULL_IMAGENET_TRAIN_SIZE = 1_281_167
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'imagenet_specialization')
WANDB_ENTITY = os.environ.get('WANDB_ENTITY', '')


def ensemble_subsets_for_width(width: int) -> int:
    # Keep full vectorization for smaller widths, split width-512 for memory safety.
    return MEMBERS_PER_GROUP if width >= 512 else 1


def minibatch_size_for_width(width: int) -> int:
    return 512 if width >= 512 else 1024


def microbatch_size_for_width(width: int) -> int:
    return 64 if width >= 512 else 128



def build_p_targets() -> list[int]:
    return [
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
        5179474,
        7196856,
        10000000,
    ]



def clear_exchangeability_configs(folder: str) -> None:
    for name in os.listdir(folder):
        if name.startswith('exchangeability_') and name.endswith('.yaml'):
            os.unlink(join(folder, name))



def build_configs(seed_base: int = 20260228, data_seed: int = 2423) -> list[tuple[str, str]]:
    rng = random.Random(seed_base)
    p_targets = build_p_targets()

    outputs: list[tuple[str, str]] = []

    for width in WIDTHS:
        for group_id in range(NUM_GROUPS):
            task_seed = rng.randrange(0, 10**9)

            tp = TrainingParams(
                eta_0=8e-3,
                minibatch_size=minibatch_size_for_width(width),
                microbatch_size=microbatch_size_for_width(width),
                num_workers=24,
                epochs=50,
                ensemble_subsets=ensemble_subsets_for_width(width),
                use_warmup_cosine_decay=True,
                target_images_seen=10_000_000,
                p_targets_images_seen=p_targets,
                wandb_enabled=True,
                wandb_project=WANDB_PROJECT,
                wandb_entity=WANDB_ENTITY,
                wandb_mode='online',
                run_id=RUN_ID,
                width=width,
                group_id=group_id,
                member_group_size=MEMBERS_PER_GROUP,
                probe_batch_size=1024,
                log_every_tranches=10,
                max_tranches=0,
            )

            mp = ModelParams(
                BASE_N=64,
                N=width,
                ensemble_size=MEMBERS_PER_GROUP,
                dtype='bfloat16',
            )

            task = TaskConfig(training_params=tp, model_params=mp, seed=task_seed)
            data_params = DataParams(
                P=FULL_IMAGENET_TRAIN_SIZE,
                data_seed=data_seed,
                root_dir='data-dir',
                val_P=1024,
            )
            tlc = TaskListConfig(task_list=[task], data_params=data_params)
            cfg = Config(setting=Setting(dataset='imagenet', model='resnet18'), hyperparams=tlc, base_dir=BASE_DIR.format(width=width, group_id=group_id))

            cfg_name = CONFIG_NAME.format(width=width, group_id=group_id)
            cfg_text = '# @package _global_\n' + OmegaConf.to_yaml(cfg)
            outputs.append((cfg_name, cfg_text))

    return outputs



if __name__ == '__main__':
    curr_dir = dirname(__file__)
    config_save_folder = join(curr_dir, CONFIG_DIR)
    clear_exchangeability_configs(config_save_folder)

    for name, content in build_configs():
        with open(join(config_save_folder, name), 'w', encoding='utf-8') as f:
            f.write(content)
