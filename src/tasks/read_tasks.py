from abc import abstractmethod, ABC
from typing import Mapping

from src.tasks.task import Task
from src.tasks.build_task_graph import order_tasks

from jax import device_count


class TaskReader(ABC):
    def __init__(self, config_list: list[Mapping]):
        self.tasks = config_list
        self._num_devices = device_count()

    @property
    def tasks(self) -> tuple[Task]:
        return self._tasks
    
    @tasks.setter
    def tasks(self, config_list: list[Mapping]):
        self._tasks = self._read_tasks(config_list)
    
    @tasks.deleter
    def tasks(self):
        del self._tasks

    def _read_tasks(self, config_list: list[Mapping]) -> tuple:
        task_tuple = tuple(self._read_task(hp) for hp in config_list)
        for task in task_tuple:
            self.validate_task(task)
        order_tasks(dict(map(lambda t: (t._id, t.dependencies), task_tuple)))
        return task_tuple

    @abstractmethod
    def _read_task(self, hyperparams: dict) -> Task:
        pass

    @abstractmethod
    def validate_task(self, task: Task) -> None:
        if task.parallelize and task.repeat > 1 and task.repeat % self._num_devices != 0:
            raise ValueError('Number of repeats is not a multiple of the number of devices.')
        minibatch_size = int(task.training_params['minibatch_size'])
        microbatch_size = int(task.training_params['microbatch_size'])
        if minibatch_size % microbatch_size != 0:
            raise ValueError("'microbatch_size' must divide 'minibatch_size'.")
        
