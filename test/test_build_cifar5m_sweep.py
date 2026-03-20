from omegaconf import OmegaConf

from scripts.build_cifar5m_sweep import build_configs, build_p_targets


def test_cifar5m_p_targets_include_terminal_checkpoint():
    assert build_p_targets()[-1] == 5000000


def test_cifar5m_build_configs_use_expected_dataset_and_run_id():
    outputs = build_configs(seed_base=123, data_seed=456)
    assert len(outputs) == 36

    name, text = outputs[0]
    assert name.startswith('cifar5m_exchangeability_w')

    cfg = OmegaConf.create(text.split('\n', 1)[1])
    assert cfg.setting.dataset == 'cifar5m'
    assert cfg.hyperparams.task_list[0].training_params.run_id == 'exchangeability_cifar5m'
    assert cfg.hyperparams.task_list[0].training_params.target_images_seen == 5000000
    assert cfg.hyperparams.task_list[0].training_params.p_targets_images_seen[-1] == 5000000
