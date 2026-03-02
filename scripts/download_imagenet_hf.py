#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable

from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download ImageNet from Hugging Face and materialize torchvision-style train/val folders.')
    parser.add_argument('--repo-id', default=os.environ.get('HF_IMAGENET_REPO_ID', 'ILSVRC/imagenet-1k'), help='HF dataset repo id')
    parser.add_argument('--root', default=os.environ.get('IMAGENET_FOLDER', ''), help='Output root (contains train/ and val/)')
    parser.add_argument('--train-split', default='train', help='HF split name for training data')
    parser.add_argument('--val-split', default='validation', help='HF split name for validation data')
    parser.add_argument(
        '--cache-dir',
        default=os.environ.get('HF_DATASETS_CACHE', os.environ.get('HF_HOME', '')),
        help='HF datasets cache dir',
    )
    parser.add_argument('--max-train', type=int, default=0, help='Optional cap for train samples (0 = full split)')
    parser.add_argument('--max-val', type=int, default=0, help='Optional cap for val samples (0 = full split)')
    parser.add_argument('--force', action='store_true', help='Re-export split even if done marker exists')
    return parser.parse_args()


def _sanitize_label_name(name: str) -> str:
    out = name.lower()
    out = out.replace('/', '_')
    out = out.replace(' ', '_')
    out = out.replace(',', '_')
    out = out.replace('-', '_')
    out = out.replace('(', '')
    out = out.replace(')', '')
    return ''.join(ch for ch in out if ch.isalnum() or ch == '_')


def _get_label_names(dataset) -> list[str]:
    label_feature = dataset.features['label']
    names = getattr(label_feature, 'names', None)
    if names is None:
        label_count = int(label_feature.num_classes)
        return [f'class_{idx:04d}' for idx in range(label_count)]
    return [str(name) for name in names]


def _build_folder_names(label_names: Iterable[str]) -> list[str]:
    folder_names: list[str] = []
    for idx, label_name in enumerate(label_names):
        sanitized = _sanitize_label_name(label_name)
        if not sanitized:
            sanitized = f'class_{idx:04d}'
        folder_names.append(f'class_{idx:04d}__{sanitized}')
    return folder_names


def _write_label_map(root: str, label_names: list[str], folder_names: list[str]) -> None:
    mapping = []
    for idx, (label_name, folder_name) in enumerate(zip(label_names, folder_names, strict=True)):
        mapping.append(
            {
                'label_idx': idx,
                'label_name': label_name,
                'folder_name': folder_name,
            }
        )
    out_path = os.path.join(root, 'hf_label_map.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)


def _ensure_class_dirs(split_root: str, folder_names: list[str]) -> None:
    for folder_name in folder_names:
        os.makedirs(os.path.join(split_root, folder_name), exist_ok=True)


def _export_split(
    *,
    dataset,
    split_name: str,
    split_root: str,
    folder_names: list[str],
    max_samples: int,
) -> int:
    total = len(dataset)
    if max_samples > 0:
        total = min(total, max_samples)

    written = 0
    pbar = tqdm(total=total, desc=f'export {split_name}', unit='img')
    for idx in range(total):
        sample = dataset[idx]
        label = int(sample['label'])
        image = sample['image']

        out_dir = os.path.join(split_root, folder_names[label])
        out_path = os.path.join(out_dir, f'{split_name}_{idx:08d}.jpg')
        if os.path.exists(out_path):
            pbar.update(1)
            continue

        image.convert('RGB').save(out_path, format='JPEG')
        written += 1
        pbar.update(1)

    pbar.close()
    return written


def _done_marker(split_root: str) -> str:
    return os.path.join(split_root, '.hf_export_done')


def main() -> None:
    args = parse_args()
    if not args.root:
        raise ValueError('Missing output root. Set --root or IMAGENET_FOLDER.')

    token = os.environ.get('HF_TOKEN', '').strip()
    if not token:
        raise ValueError('HF_TOKEN is not set. Put it in ~/.secrets or export it in your shell.')

    os.makedirs(args.root, exist_ok=True)

    load_kwargs = {
        'path': args.repo_id,
        'token': token,
    }
    if args.cache_dir:
        load_kwargs['cache_dir'] = args.cache_dir

    print(f'Loading train split `{args.train_split}` from {args.repo_id}')
    train_ds = load_dataset(split=args.train_split, **load_kwargs)
    label_names = _get_label_names(train_ds)
    folder_names = _build_folder_names(label_names)
    _write_label_map(args.root, label_names, folder_names)

    train_root = os.path.join(args.root, 'train')
    val_root = os.path.join(args.root, 'val')
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)
    _ensure_class_dirs(train_root, folder_names)
    _ensure_class_dirs(val_root, folder_names)

    train_done = _done_marker(train_root)
    if not args.force and os.path.exists(train_done):
        print(f'Skipping train export; marker exists: {train_done}')
    else:
        train_written = _export_split(
            dataset=train_ds,
            split_name='train',
            split_root=train_root,
            folder_names=folder_names,
            max_samples=args.max_train,
        )
        with open(train_done, 'w', encoding='utf-8') as f:
            f.write('ok\n')
        print(f'Train export done. New files written: {train_written}')

    print(f'Loading val split `{args.val_split}` from {args.repo_id}')
    val_ds = load_dataset(split=args.val_split, **load_kwargs)
    val_done = _done_marker(val_root)
    if not args.force and os.path.exists(val_done):
        print(f'Skipping val export; marker exists: {val_done}')
    else:
        val_written = _export_split(
            dataset=val_ds,
            split_name='val',
            split_root=val_root,
            folder_names=folder_names,
            max_samples=args.max_val,
        )
        with open(val_done, 'w', encoding='utf-8') as f:
            f.write('ok\n')
        print(f'Val export done. New files written: {val_written}')

    print(f'Finished HF ImageNet export into: {args.root}')


if __name__ == '__main__':
    main()
