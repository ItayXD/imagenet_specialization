import os

DEFAULT_CLUSTER_ROOT = os.environ.get(
    'EXCHANGEABILITY_ROOT',
    '/n/netscratch/kempner_pehlevan_lab/Lab/ilavie',
)

CIFAR_FOLDER = os.environ.get('CIFAR_FOLDER')
CIFAR5M_FOLDER = os.environ.get(
    'CIFAR5M_FOLDER',
    os.path.join(DEFAULT_CLUSTER_ROOT, 'cifar5m'),
)
REMOTE_RESULTS_FOLDER = os.environ.get(
    'REMOTE_RESULTS_FOLDER',
    DEFAULT_CLUSTER_ROOT,
)
IMAGENET_FOLDER = os.environ.get(
    'IMAGENET_FOLDER',
    os.path.join(DEFAULT_CLUSTER_ROOT, 'imagenet'),
)
IMAGENET_BASE_SAVE_DIR = os.environ.get(
    'IMAGENET_BASE_SAVE_DIR',
    os.path.join(DEFAULT_CLUSTER_ROOT, 'exchangeability_imagenet'),
)
CIFAR5M_BASE_SAVE_DIR = os.environ.get(
    'CIFAR5M_BASE_SAVE_DIR',
    os.path.join(DEFAULT_CLUSTER_ROOT, 'exchangeability_cifar5m'),
)
BASE_SAVE_DIR = os.environ.get(
    'BASE_SAVE_DIR',
    IMAGENET_BASE_SAVE_DIR,
)

LOCAL_RESULTS_FOLDER = "results"
