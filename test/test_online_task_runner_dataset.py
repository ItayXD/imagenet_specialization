import jax

from src.run.OnlineTaskRunner import OnlineTaskRunner
from src.tasks.task import Task


class _DummyPreprocessDevice:
    def __init__(self, save_dir, train_dataset, val_data):
        self.save_dir = str(save_dir)
        self.train_dataset = train_dataset
        self.val_data = val_data
        self.devices = ['cpu']


def test_online_task_runner_injects_dataset_into_training_params(tmp_path):
    observed = {}

    def _apply(key, train_loader, val_data, devices, model_params, training_params):
        del key, train_loader, val_data, devices, model_params
        observed['dataset'] = training_params['dataset']

    save_dir = tmp_path / 'results'
    save_dir.mkdir(parents=True, exist_ok=True)
    preprocess = _DummyPreprocessDevice(
        save_dir=save_dir,
        train_dataset=[(jax.numpy.zeros((32, 32, 3)), 0) for _ in range(4)],
        val_data=(jax.numpy.zeros((2, 32, 32, 3)), jax.numpy.zeros((2,), dtype=jax.numpy.int32)),
    )
    runner = OnlineTaskRunner(preprocess)
    task = Task(
        model='resnet18',
        dataset='cifar5m',
        model_params={'N': 32, 'ensemble_size': 1},
        training_params={'minibatch_size': 2, 'microbatch_size': 1, 'num_workers': 0},
        type_=0,
        seed=jax.random.PRNGKey(0),
        apply_callback=_apply,
    )

    runner.run_repeat_task(task)

    assert observed['dataset'] == 'cifar5m'


def test_online_task_runner_builds_loader_from_training_config(tmp_path, monkeypatch):
    observed = {}

    def _fake_make_dataloader(dataset, *args, **kwargs):
        observed['dataset'] = dataset
        observed['args'] = args
        observed['kwargs'] = kwargs
        return 'train-loader'

    def _apply(key, train_loader, val_data, devices, model_params, training_params):
        del key, val_data, devices, model_params, training_params
        observed['train_loader'] = train_loader

    monkeypatch.setattr('src.run.OnlineTaskRunner.make_dataloader', _fake_make_dataloader)

    save_dir = tmp_path / 'results'
    save_dir.mkdir(parents=True, exist_ok=True)
    preprocess = _DummyPreprocessDevice(
        save_dir=save_dir,
        train_dataset=[(jax.numpy.zeros((32, 32, 3)), 0) for _ in range(4)],
        val_data=(jax.numpy.zeros((2, 32, 32, 3)), jax.numpy.zeros((2,), dtype=jax.numpy.int32)),
    )
    runner = OnlineTaskRunner(preprocess)
    task = Task(
        model='resnet18',
        dataset='cifar5m',
        model_params={'N': 32, 'ensemble_size': 1},
        training_params={'minibatch_size': 2, 'microbatch_size': 1, 'num_workers': 3},
        type_=0,
        seed=jax.random.PRNGKey(0),
        apply_callback=_apply,
    )

    runner.run_repeat_task(task)

    assert observed['dataset'] == preprocess.train_dataset
    assert observed['args'] == (2,)
    assert observed['kwargs']['num_workers'] == 3
    assert observed['kwargs']['persistent_workers'] is True
    assert observed['kwargs']['shuffle'] is True
    assert observed['train_loader'] == 'train-loader'
