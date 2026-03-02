#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare torchvision ImageNet train/val folders from downloaded archives.')
    parser.add_argument('--imagenet-root', default=os.environ.get('IMAGENET_FOLDER', ''), help='ImageNet root directory')
    parser.add_argument('--verify-only', action='store_true', help='Only verify availability, do not trigger parsing')
    return parser.parse_args()


def _required_files(root: str) -> list[str]:
    return [
        os.path.join(root, 'ILSVRC2012_img_train.tar'),
        os.path.join(root, 'ILSVRC2012_img_val.tar'),
        os.path.join(root, 'ILSVRC2012_devkit_t12.tar.gz'),
    ]


def main() -> None:
    args = parse_args()
    root = args.imagenet_root
    if not root:
        raise ValueError('No ImageNet root provided. Set --imagenet-root or IMAGENET_FOLDER.')

    missing = [p for p in _required_files(root) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('Missing required ImageNet archives:\n' + '\n'.join(missing))

    if args.verify_only:
        print(f'ImageNet archives present in {root}')
        return

    from torchvision.datasets import ImageNet

    print(f'Preparing ImageNet train split from archives in: {root}')
    _ = ImageNet(root, split='train')
    print(f'Preparing ImageNet val split from archives in: {root}')
    _ = ImageNet(root, split='val')
    print('ImageNet archive parsing finished.')


if __name__ == '__main__':
    main()
