import os

CIFAR_FOLDER = os.environ.get('CIFAR_FOLDER')  # location of CIFAR data
REMOTE_RESULTS_FOLDER = os.environ.get('REMOTE_RESULTS_FOLDER')  # location to save results
IMAGENET_FOLDER = os.environ.get('IMAGENET_FOLDER')  # location of ImageNet data
BASE_SAVE_DIR = os.environ.get('BASE_SAVE_DIR')  # location to save logged logits and model states

LOCAL_RESULTS_FOLDER = "results"
