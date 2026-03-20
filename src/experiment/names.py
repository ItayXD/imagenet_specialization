import importlib

_MODULE_PATHS = {
    ('cifar10', 'resnet18'): 'src.experiment.cifar10_resnet',
    ('cifar5m', 'resnet18'): 'src.experiment.cifar5m_resnet',
    ('imagenet', 'resnet18'): 'src.experiment.imagenet_resnet',
}


def get_experiment_module(dataset: str, model: str):
    path = _MODULE_PATHS[dataset, model]
    return importlib.import_module(path)
