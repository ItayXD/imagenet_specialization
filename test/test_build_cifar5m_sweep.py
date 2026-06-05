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


def test_cifar5m_build_configs_support_sgd_constant_lr_variant():
    outputs = build_configs(
        seed_base=123,
        data_seed=456,
        eta_0=0.01,
        optimizer='sgd',
        run_id='exchangeability_cifar5m_sgd',
        config_name_template='cifar5m_exchangeability_sgd_w{width}_g{group_id}.yaml',
        base_dir_template='/tmp/exchangeability_runs/cifar5m_sgd/w{width}/g{group_id}',
        use_warmup_cosine_decay=False,
    )

    name, text = outputs[0]
    assert name.startswith('cifar5m_exchangeability_sgd_w')

    cfg = OmegaConf.create(text.split('\n', 1)[1])
    tp = cfg.hyperparams.task_list[0].training_params
    assert tp.optimizer == 'sgd'
    assert tp.eta_0 == 0.01
    assert tp.use_warmup_cosine_decay is False
    assert tp.run_id == 'exchangeability_cifar5m_sgd'
    assert cfg.base_dir.startswith('/tmp/exchangeability_runs/cifar5m_sgd/')


def test_cifar5m_build_configs_support_muon_variant():
    outputs = build_configs(
        seed_base=123,
        data_seed=456,
        eta_0=0.006,
        optimizer='muon',
        run_id='exchangeability_cifar5m_muon',
        config_name_template='cifar5m_exchangeability_muon_w{width}_g{group_id}.yaml',
        base_dir_template='/tmp/exchangeability_runs/cifar5m_muon/w{width}/g{group_id}',
        use_warmup_cosine_decay=True,
    )

    name, text = outputs[0]
    assert name.startswith('cifar5m_exchangeability_muon_w')

    cfg = OmegaConf.create(text.split('\n', 1)[1])
    tp = cfg.hyperparams.task_list[0].training_params
    assert tp.optimizer == 'muon'
    assert tp.eta_0 == 0.006
    assert tp.use_warmup_cosine_decay is True
    assert tp.run_id == 'exchangeability_cifar5m_muon'
    assert cfg.base_dir.startswith('/tmp/exchangeability_runs/cifar5m_muon/')


def test_cifar5m_build_configs_can_limit_widths_and_groups():
    outputs = build_configs(
        seed_base=123,
        data_seed=456,
        widths=(64,),
        target_members_per_width=4,
    )

    assert len(outputs) == 1

    _, text = outputs[0]
    cfg = OmegaConf.create(text.split('\n', 1)[1])
    tp = cfg.hyperparams.task_list[0].training_params
    mp = cfg.hyperparams.task_list[0].model_params

    assert int(mp.N) == 64
    assert int(tp.group_id) == 0
    assert int(tp.member_group_size) == 4


def test_cifar5m_build_configs_support_single_member_width_scan_override():
    outputs = build_configs(
        seed_base=123,
        data_seed=456,
        optimizer='muon',
        widths=(32, 256, 512),
        target_members_per_width=1,
        members_per_group_override=1,
        run_id='exchangeability_cifar5m_muon',
        config_name_template='cifar5m_exchangeability_muon_w{width}_g{group_id}.yaml',
        base_dir_template='/tmp/exchangeability_runs/cifar5m_muon/w{width}/g{group_id}',
    )

    assert len(outputs) == 3

    for _, text in outputs:
        cfg = OmegaConf.create(text.split('\n', 1)[1])
        tp = cfg.hyperparams.task_list[0].training_params
        mp = cfg.hyperparams.task_list[0].model_params
        assert int(tp.group_id) == 0
        assert int(tp.member_group_size) == 1
        assert int(mp.ensemble_size) == 1
