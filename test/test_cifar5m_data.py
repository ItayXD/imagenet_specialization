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


def test_cifar5m_dataset_caches_shard_arrays(monkeypatch):
    class _FakeNpz:
        def __init__(self):
            self.calls = {'images': 0, 'labels': 0}
            self.arrays = {
                'images': np.zeros((3, 32, 32, 3), dtype=np.uint8),
                'labels': np.array([3, 4, 5], dtype=np.int64),
            }

        def __getitem__(self, key):
            self.calls[key] += 1
            return self.arrays[key]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    fake_npz = _FakeNpz()
    observed = {'loads': 0}

    def _fake_np_load(path, mmap_mode='r'):
        del path, mmap_mode
        observed['loads'] += 1
        return fake_npz

    monkeypatch.setattr('src.experiment.dataset.cifar5m.np.load', _fake_np_load)

    dataset = Cifar5mShardDataset(
        {
            'shards': [
                {
                    'path': '/tmp/part0.npz',
                    'length': 3,
                    'global_start': 0,
                    'global_end': 3,
                    'image_key': 'images',
                    'label_key': 'labels',
                }
            ]
        },
        start=0,
        end=3,
        transform=None,
    )

    _, label0 = dataset[0]
    _, label1 = dataset[1]

    assert label0 == 3
    assert label1 == 4
    assert observed['loads'] == 1
    assert fake_npz.calls == {'images': 1, 'labels': 1}
