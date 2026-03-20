#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from urllib.request import urlretrieve

from src.experiment.dataset.cifar5m import SHARD_COUNT, build_split_manifest


DEFAULT_URL_TEMPLATE = 'https://storage.googleapis.com/gresearch/cifar5m/part{index}.npz'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download CIFAR-5M public shards and build a local split manifest.')
    parser.add_argument('--root', default=os.environ.get('CIFAR5M_FOLDER', ''), help='CIFAR-5M root directory')
    parser.add_argument(
        '--url-template',
        default=os.environ.get('CIFAR5M_URL_TEMPLATE', DEFAULT_URL_TEMPLATE),
        help='Shard URL template with {index} placeholder',
    )
    parser.add_argument('--force', action='store_true', help='Re-download shards even if they already exist')
    return parser.parse_args()


def _download_shard(url_template: str, root: str, index: int, force: bool) -> str:
    raw_dir = os.path.join(root, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, f'part{index}.npz')
    if os.path.exists(out_path) and not force:
        return out_path
    url = url_template.format(index=index)
    print(f'Downloading {url} -> {out_path}')
    urlretrieve(url, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    if not args.root:
        raise ValueError('Missing CIFAR-5M root. Set --root or CIFAR5M_FOLDER.')
    os.makedirs(args.root, exist_ok=True)

    for index in range(SHARD_COUNT):
        _download_shard(args.url_template, args.root, index, args.force)

    manifest = build_split_manifest(args.root)
    print(
        f'Finished CIFAR-5M download into {args.root}; '
        f'train_size={manifest["train_size"]}, heldout_size={manifest["heldout_size"]}'
    )


if __name__ == '__main__':
    main()
