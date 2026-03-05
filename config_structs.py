from dataclasses import dataclass, field
from omegaconf import MISSING
# from hydra.core.config_store import ConfigStore


def _default_p_targets() -> list[int]:
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
        5179474,
        7196856,
        10000000,
    ]

@dataclass
class WarmupCosineDecayParameters:
    warmup_epochs: float = 0.5
    init_lr: float = 8e-5
    min_lr: float = 8e-5


@dataclass
class TrainingParams:
    eta_0: float = 8e-3
    # momentum: float = 0.9
    # weight_decay: float = 1e-5 # TODO: not implemented; replace with batch_size
    minibatch_size: int = 1024 # changed
    microbatch_size: int = 64
    num_workers: int = 24
    epochs: int = 50
    full_batch_gradient: bool = False
    ensemble_subsets: int = 1 # number of subsets of the ensemble to be run synchronously (increase from 1 if out-of-memory during training); divides ensemble_size
    use_checkpoint: bool = False
    ckpt_dir: str = ''
    model_ckpt_dir: str = ''
    use_warmup_cosine_decay: bool = True
    wcd_params: WarmupCosineDecayParameters = field(default_factory=WarmupCosineDecayParameters)
    target_images_seen: int = 10_000_000
    p_targets_images_seen: list[int] = field(default_factory=_default_p_targets)
    wandb_enabled: bool = False
    wandb_project: str = 'imagenet_specialization'
    wandb_entity: str = ''
    wandb_mode: str = 'online'
    run_id: str = 'exchangeability'
    width: int = 128
    group_id: int = 0
    member_group_size: int = 4
    probe_batch_size: int = 1024
    log_every_tranches: int = 10
    max_tranches: int = 0


@dataclass
class DataParams:
    P: int = 2 ** 20
    # k: int = 0 # no target fn
    # on_device: bool = True
    # random_subset: bool = True
    data_seed: int = MISSING
    root_dir: str = 'data-dir'
    val_P: int = 1024


@dataclass
class ModelParams:
    BASE_N: int = 64
    N: int = 128
    ensemble_size: int = 1
    dtype: str = 'bfloat16'


@dataclass
class TaskConfig:
    training_params: TrainingParams = field(default_factory=TrainingParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    seed: int = MISSING


@dataclass
class TaskListConfig:
    task_list: list[TaskConfig] = field(default_factory=list)
    data_params: DataParams = field(default_factory=DataParams)

@dataclass
class Setting:
  dataset: str = 'imagenet'
  model: str = 'resnet18'

@dataclass
class Config:
    setting: Setting
    hyperparams: TaskListConfig
    base_dir: str = ''
