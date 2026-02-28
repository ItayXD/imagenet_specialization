from numpy import int32, asarray
from omegaconf import OmegaConf
from os.path import join, dirname
import os, shutil

from template import SBATCH_TEMPLATE
from config_structs import Config, DataParams, ModelParams, TrainingParams, Setting, TaskConfig, TaskListConfig

CONFIG_DIR = '../conf/experiment'
SBATCH_DIR = '../sbatch_files'

BASE_DIR = '/tmp/{id}'

CONFIG_NAME = 'sweep_alpha_{id}.yaml'
SBATCH_NAME = 'sweep_alpha_{id}.bat'


def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == '__main__':
    # clear_folder(CONFIG_DIR)
    # clear_folder(SBATCH_DIR)
    widths = [2 ** i for i in range(2, 10)]
    width_es_map =  {512: 1, 256: 2, 128: 6, 64: 12, 32: 36, 16: 64, 8: 64, 4: 128, 2: 128}
    
    data_seed = 2423
    seed = 3442
    dp = DataParams(data_seed=data_seed)

    tlcs = [TaskListConfig(task_list=[TaskConfig(training_params=TrainingParams(), model_params=ModelParams(N=16, alpha=a, ensemble_size=4), seed=seed)], data_params=dp) for a in (1/5.65, 1, 5.65, 32)]
    configs = [Config(setting=Setting(), hyperparams=tlc_inst, base_dir=BASE_DIR.format(id=id)) for id, tlc_inst in enumerate(tlcs)]
    str_configs = ['# @package _global_\n' + OmegaConf.to_yaml(conf) for conf in configs]

    curr_dir = dirname(__file__)
    config_save_folder = join(curr_dir, CONFIG_DIR)
    sbatch_save_folder = join(curr_dir, SBATCH_DIR)

    for id, strc in enumerate(str_configs):
        config_fname = CONFIG_NAME.format(id=id)
        config_rel_loc = join(config_save_folder, config_fname)

        sbatch_str = SBATCH_TEMPLATE.format(id=id)

        sbatch_fname = SBATCH_NAME.format(id=id)
        sbatch_rel_loc = join(sbatch_save_folder, sbatch_fname)

        with open(config_rel_loc, mode='x') as fi:
            fi.write(strc) # TODO: add file exists exception handler + clean up
        with open(sbatch_rel_loc, mode='x') as fi:
            fi.write(sbatch_str) # TODO: add file exists exception handler + clean up
