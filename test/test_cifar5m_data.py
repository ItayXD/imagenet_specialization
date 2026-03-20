import json
import os

import numpy as np

from src.experiment.dataset.cifar5m import (
    Cifar5mShardDataset,
    build_split_manifest,
    load_split_manifest,
)


def _write_shard(path, start_label, count):
    images = np.zeros((count, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(start_label, start_label + count, dtype=np.int64) % 10
    for idx in range(count):
        images[idx, :, :, :] = (start_label + idx) % 255
    np.savez(path, images=images, labels=labels)


def test_build_split_manifest_and_cross_shard_reads(tmp_path):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    lengths = [2, 2, 2, 2, 2, 1]
    start_label = 0
    for shard_idx, count in enumerate(lengths):
        _write_shard(raw_dir / f'part{shard_idx}.npz', start_label, count)
        start_label += count

    manifest = build_split_manifest(str(tmp_path), train_size=6)
    assert manifest['train_size'] == 6
    assert manifest['heldout_size'] == 5
    assert manifest['total_size'] == 11
    assert len(manifest['shards']) == 6

    split_manifest_path = tmp_path / 'splits' / 'split_manifest.json'
    assert split_manifest_path.exists()
    with open(split_manifest_path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert loaded['train_size'] == 6

    dataset = Cifar5mShardDataset(loaded, start=0, end=6, transform=None, explicit_indices=[1, 2, 5])
    image0, label0 = dataset[0]
    image1, label1 = dataset[1]
    image2, label2 = dataset[2]

    assert image0.shape == (32, 32, 3)
    assert label0 == 1
    assert label1 == 2
    assert label2 == 5


def test_load_split_manifest_builds_if_missing(tmp_path):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    for shard_idx in range(6):
        _write_shard(raw_dir / f'part{shard_idx}.npz', shard_idx, 2)

    manifest = load_split_manifest(str(tmp_path), train_size=6)
    assert manifest['train_size'] == 6
    assert os.path.exists(tmp_path / 'splits' / 'split_manifest.json')
