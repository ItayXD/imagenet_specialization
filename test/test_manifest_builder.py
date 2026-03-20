from scripts.build_exchangeability_manifest import _derive_member_seeds, _parse_row



def test_member_seed_derivation_is_deterministic():
    seeds_a = _derive_member_seeds(12345, 4)
    seeds_b = _derive_member_seeds(12345, 4)
    assert seeds_a == seeds_b
    assert len(seeds_a) == 4


def test_parse_row_includes_dataset_and_run_id(tmp_path):
    cfg_path = tmp_path / 'cifar5m_exchangeability_w32_g0.yaml'
    cfg_path.write_text(
        '\n'.join(
            [
                '# @package _global_',
                'setting:',
                '  dataset: cifar5m',
                '  model: resnet18',
                'hyperparams:',
                '  data_params:',
                '    P: 5000000',
                '    data_seed: 2423',
                '    root_dir: data-dir',
                '    val_P: 1024',
                '  task_list:',
                '    - training_params:',
                '        run_id: exchangeability_cifar5m',
                '        group_id: 0',
                '        target_images_seen: 5000000',
                '        p_targets_images_seen: [10000, 5000000]',
                '        member_group_size: 4',
                '      model_params:',
                '        N: 32',
                '      seed: 123',
                'base_dir: /tmp/demo',
            ]
        ),
        encoding='utf-8',
    )

    row = _parse_row(job_id=0, cfg_path=str(cfg_path), base_save_dir='/tmp/outputs')
    assert row.dataset == 'cifar5m'
    assert row.run_id == 'exchangeability_cifar5m'
    assert row.save_dir == '/tmp/outputs/exchangeability_cifar5m/width_32/group_0'
