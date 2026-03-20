import csv

from scripts.compare_exchangeability_datasets import _common_images_seen, _load_dataset_csv


def _write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_common_images_seen_uses_observed_rows_only(tmp_path):
    fieldnames = [
        'width',
        'images_seen',
        'representation',
        'analysis_type',
        'shuffle_id',
        'ks_distance',
        'w1_distance',
        'train_loss',
        'val_loss',
        'train_error',
        'val_error',
    ]
    imagenet_csv = tmp_path / 'imagenet.csv'
    cifar_csv = tmp_path / 'cifar.csv'
    _write_csv(
        imagenet_csv,
        fieldnames,
        [
            {'width': '32', 'images_seen': '100000', 'representation': 'weights', 'analysis_type': 'within_vs_across_real', 'shuffle_id': '-1', 'ks_distance': '0.1', 'w1_distance': '0.2', 'train_loss': '1.0', 'val_loss': '1.1', 'train_error': '0.3', 'val_error': '0.31'},
            {'width': '32', 'images_seen': '3727593', 'representation': 'weights', 'analysis_type': 'within_vs_across_real', 'shuffle_id': '-1', 'ks_distance': '0.2', 'w1_distance': '0.3', 'train_loss': '0.9', 'val_loss': '1.0', 'train_error': '0.2', 'val_error': '0.21'},
        ],
    )
    _write_csv(
        cifar_csv,
        fieldnames,
        [
            {'width': '32', 'images_seen': '100000', 'representation': 'weights', 'analysis_type': 'within_vs_across_real', 'shuffle_id': '-1', 'ks_distance': '0.3', 'w1_distance': '0.4', 'train_loss': '0.8', 'val_loss': '0.9', 'train_error': '0.1', 'val_error': '0.11'},
            {'width': '32', 'images_seen': '5000000', 'representation': 'weights', 'analysis_type': 'within_vs_across_real', 'shuffle_id': '-1', 'ks_distance': '0.4', 'w1_distance': '0.5', 'train_loss': '0.7', 'val_loss': '0.8', 'train_error': '0.05', 'val_error': '0.06'},
        ],
    )

    imagenet_df = _load_dataset_csv(str(imagenet_csv), 'imagenet')
    cifar_df = _load_dataset_csv(str(cifar_csv), 'cifar5m')

    assert _common_images_seen([imagenet_df, cifar_df]) == [100000]
