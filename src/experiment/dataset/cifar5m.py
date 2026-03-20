from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass
from typing import Mapping

import jax.random as jr
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms


TRAIN_SIZE = 5_000_000
SHARD_COUNT = 6
SPLIT_MANIFEST_RELATIVE_PATH = os.path.join('splits', 'split_manifest.json')
IMAGE_KEY_CANDIDATES = ('images', 'X', 'x', 'data')
LABEL_KEY_CANDIDATES = ('labels', 'Y', 'y', 'targets')

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class ShardInfo:
    path: str
    length: int
    global_start: int
    global_end: int
    image_key: str
    label_key: str


def _default_raw_paths(root: str) -> list[str]:
    raw_dir = os.path.join(root, 'raw')
    return [os.path.join(raw_dir, f'part{i}.npz') for i in range(SHARD_COUNT)]


def _find_array_keys(npz: np.lib.npyio.NpzFile) -> tuple[str, str]:
    image_key = next((key for key in IMAGE_KEY_CANDIDATES if key in npz.files), '')
    label_key = next((key for key in LABEL_KEY_CANDIDATES if key in npz.files), '')
    if not image_key or not label_key:
        raise ValueError(
            f'Could not find image/label arrays in shard; files={sorted(npz.files)} '
            f'expected image keys {IMAGE_KEY_CANDIDATES} and label keys {LABEL_KEY_CANDIDATES}.'
        )
    return image_key, label_key


def _scan_shard(path: str) -> ShardInfo:
    if not os.path.exists(path):
        raise FileNotFoundError(f'CIFAR-5M shard not found: {path}')
    with np.load(path, mmap_mode='r') as npz:
        image_key, label_key = _find_array_keys(npz)
        images = npz[image_key]
        labels = npz[label_key]
        if images.ndim != 4:
            raise ValueError(f'Expected image array with 4 dims in {path}, got shape {images.shape}.')
        if labels.shape[0] != images.shape[0]:
            raise ValueError(
                f'Label count {labels.shape[0]} does not match image count {images.shape[0]} in {path}.'
            )
        return ShardInfo(
            path=os.path.abspath(path),
            length=int(images.shape[0]),
            global_start=0,
            global_end=0,
            image_key=image_key,
            label_key=label_key,
        )


def build_split_manifest(root: str, *, train_size: int = TRAIN_SIZE) -> dict:
    shard_infos = []
    global_start = 0
    for shard_path in _default_raw_paths(root):
        shard = _scan_shard(shard_path)
        shard_infos.append(
            {
                'path': shard.path,
                'length': shard.length,
                'global_start': global_start,
                'global_end': global_start + shard.length,
                'image_key': shard.image_key,
                'label_key': shard.label_key,
            }
        )
        global_start += shard.length

    total_size = global_start
    if train_size <= 0 or train_size >= total_size:
        raise ValueError(f'train_size must be in (0, total_size); got train_size={train_size}, total_size={total_size}.')

    manifest = {
        'version': 1,
        'train_size': int(train_size),
        'total_size': int(total_size),
        'heldout_size': int(total_size - train_size),
        'shards': shard_infos,
    }
    split_path = os.path.join(root, SPLIT_MANIFEST_RELATIVE_PATH)
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_split_manifest(root: str, *, train_size: int = TRAIN_SIZE) -> dict:
    split_path = os.path.join(root, SPLIT_MANIFEST_RELATIVE_PATH)
    if not os.path.exists(split_path):
        return build_split_manifest(root, train_size=train_size)
    with open(split_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _channels_last_transform() -> transforms.Lambda:
    return transforms.Lambda(lambda x: x.permute(1, 2, 0))


def _train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD),
            _channels_last_transform(),
        ]
    )


def _eval_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD),
            _channels_last_transform(),
        ]
    )


class Cifar5mShardDataset(Dataset):
    def __init__(
        self,
        manifest: Mapping,
        *,
        start: int,
        end: int,
        transform=None,
        explicit_indices: list[int] | None = None,
    ) -> None:
        if start < 0 or end <= start:
            raise ValueError(f'Invalid dataset range start={start}, end={end}.')
        self.start = int(start)
        self.end = int(end)
        self.transform = transform
        self.explicit_indices = explicit_indices
        self.shards = [ShardInfo(**row) for row in manifest['shards']]
        self._cumulative_ends = [int(shard.global_end) for shard in self.shards]
        self._npz_cache: dict[int, np.lib.npyio.NpzFile] = {}

    def __len__(self) -> int:
        if self.explicit_indices is not None:
            return len(self.explicit_indices)
        return self.end - self.start

    def _resolve_global_index(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.explicit_indices is not None:
            return int(self.explicit_indices[index])
        return self.start + index

    def _open_shard(self, shard_idx: int) -> np.lib.npyio.NpzFile:
        npz = self._npz_cache.get(shard_idx)
        if npz is None:
            npz = np.load(self.shards[shard_idx].path, mmap_mode='r')
            self._npz_cache[shard_idx] = npz
        return npz

    def __getitem__(self, index: int):
        global_index = self._resolve_global_index(index)
        shard_idx = bisect.bisect_right(self._cumulative_ends, global_index)
        shard = self.shards[shard_idx]
        local_index = global_index - int(shard.global_start)
        npz = self._open_shard(shard_idx)
        image = np.asarray(npz[shard.image_key][local_index], dtype=np.uint8)
        label = int(np.asarray(npz[shard.label_key][local_index]).reshape(()))
        image_obj = Image.fromarray(image)
        if self.transform is not None:
            image_out = self.transform(image_obj)
        else:
            image_out = image
        return image_out, label


def make_heldout_indices(manifest: Mapping, val_P: int, data_seed: int) -> list[int]:
    heldout_start = int(manifest['train_size'])
    heldout_size = int(manifest['heldout_size'])
    if val_P <= 0:
        raise ValueError('val_P must be positive.')
    if val_P > heldout_size:
        raise ValueError(f'val_P={val_P} exceeds held-out size {heldout_size}.')
    key = jr.PRNGKey(int(data_seed))
    offsets = jr.choice(key, heldout_size, shape=(val_P,), replace=False)
    heldout_indices = (np.asarray(offsets, dtype=np.int64) + heldout_start).tolist()
    return heldout_indices


def load_cifar5m_data(root: str, data_params: Mapping) -> tuple[Dataset, Dataset]:
    manifest = load_split_manifest(root)
    val_P = int(data_params['val_P'])
    data_seed = int(data_params['data_seed'])
    train_dataset = Cifar5mShardDataset(
        manifest,
        start=0,
        end=int(manifest['train_size']),
        transform=_train_transform(),
    )
    heldout_indices = make_heldout_indices(manifest, val_P=val_P, data_seed=data_seed)
    heldout_dataset = Cifar5mShardDataset(
        manifest,
        start=int(manifest['train_size']),
        end=int(manifest['total_size']),
        transform=_eval_transform(),
        explicit_indices=heldout_indices,
    )
    return train_dataset, heldout_dataset


def build_probe_subset(root: str, probe_batch_size: int, probe_seed: int) -> Dataset:
    manifest = load_split_manifest(root)
    probe_indices = make_heldout_indices(manifest, val_P=probe_batch_size, data_seed=probe_seed)
    heldout_dataset = Cifar5mShardDataset(
        manifest,
        start=int(manifest['train_size']),
        end=int(manifest['total_size']),
        transform=_eval_transform(),
        explicit_indices=probe_indices,
    )
    return heldout_dataset
