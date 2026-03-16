import csv

from scripts.merge_layerwise_weight_w1_csvs import LAYERWISE_FIELDNAMES, _merge_rows


def _write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_row():
    return {
        'width': '256',
        'source_run_id': 'exchangeability_job123',
        'images_seen': '7196856',
        'layer_index': '7',
        'layer_name': 'ResNetBlock_3/Conv_1',
        'member_width': '1024',
        'kernel_numel': '9216',
        'num_members': '16',
        'w1_distance': '0.2',
        'train_loss': '1.0',
        'val_loss': '1.1',
        'train_error': '0.3',
        'val_error': '0.31',
    }


def test_merge_rows_deduplicates_identical_layer_records(tmp_path):
    row = _base_row()
    left_csv = tmp_path / 'layerwise_weight_w1.csv'
    right_csv = tmp_path / 'layerwise_weight_w1_w256.csv'

    _write_csv(left_csv, LAYERWISE_FIELDNAMES, [row])
    _write_csv(right_csv, LAYERWISE_FIELDNAMES, [row])

    merged = _merge_rows([str(left_csv), str(right_csv)])

    assert len(merged) == 1
    assert merged[0]['layer_index'] == '7'
