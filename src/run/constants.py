import os

DEFAULT_CLUSTER_ROOT = os.environ.get(
    'EXCHANGEABILITY_ROOT',
    '/n/netscratch/kempner_pehlevan_lab/Lab/ilavie',
)

CIFAR_FOLDER = os.environ.get('CIFAR_FOLDER')
REMOTE_RESULTS_FOLDER = os.environ.get(
    'REMOTE_RESULTS_FOLDER',
    DEFAULT_CLUSTER_ROOT,
)
IMAGENET_FOLDER = os.environ.get(
    'IMAGENET_FOLDER',
    os.path.join(DEFAULT_CLUSTER_ROOT, 'imagenet'),
)
BASE_SAVE_DIR = os.environ.get(
    'BASE_SAVE_DIR',
    os.path.join(DEFAULT_CLUSTER_ROOT, 'exchangeability_outputs'),
)

LOCAL_RESULTS_FOLDER = "results"
