from enum import Enum
from typing import Mapping

from jax.random import PRNGKey
from omegaconf import OmegaConf

from src.experiment.dataset.cifar5m import load_cifar5m_data
from src.experiment.training.online_momentum import apply
from src.run.OnlinePreprocessDevice import OnlinePreprocessDevice as OPD
import src.run.constants as constants
from src.tasks.read_tasks import TaskReader as TR
from src.tasks.task import Task


class TaskType(Enum):
    TRAIN_NN = 0
    TRAIN_NTK = 1


class Callbacks(Enum):
    APPLY = apply


class PreprocessDevice(OPD):
    def load_data(self, data_params):
        return load_cifar5m_data(constants.CIFAR5M_FOLDER, data_params)


class TaskReader(TR):
    task_type = TaskType.TRAIN_NN

    def validate_task(self, task: Task):
        super().validate_task(task)

    def _read_task(self, config: Mapping):
        try:
            key = PRNGKey(config['seed'])
            task = Task(
                model='resnet18',
                dataset='cifar5m',
                model_params=OmegaConf.to_container(config['model_params']),
                training_params=OmegaConf.to_container(config['training_params']),
                type_=self.task_type,
                seed=key,
                apply_callback=Callbacks.APPLY,
            )
        except KeyError as exc:
            raise ValueError('Task not properly configured.') from exc
        return task
