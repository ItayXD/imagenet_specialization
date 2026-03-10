import csv

from scripts.merge_exchangeability_analysis_csvs import ANALYSIS_FIELDNAMES, _merge_rows


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
        'representation': 'weights',
        'analysis_type': 'within_vs_across_real',
        'shuffle_id': '-1',
        'ks_distance': '0.1',
        'ks_p_raw': '0.9',
        'ks_sigma_two_sided': '0.1',
        'ks_p_empirical': '0.9',
        'ks_sigma_empirical_two_sided': '0.1',
        'w1_distance': '0.2',
        'w1_p_empirical': '0.8',
        'w1_sigma_empirical_two_sided': '0.2',
        'train_loss': '1.0',
        'val_loss': '1.1',
        'train_error': '0.3',
        'val_error': '0.31',
    }


def test_merge_rows_handles_legacy_blank_shuffle_id(tmp_path):
    legacy_row = _base_row()
    legacy_row.pop('shuffle_id')

    current_row = _base_row()

    legacy_csv = tmp_path / 'exchangeability_metrics.csv'
    current_csv = tmp_path / 'exchangeability_metrics_w256.csv'

    _write_csv(
        legacy_csv,
        [field for field in ANALYSIS_FIELDNAMES if field != 'shuffle_id'],
        [legacy_row],
    )
    _write_csv(current_csv, ANALYSIS_FIELDNAMES, [current_row])

    merged = _merge_rows([str(legacy_csv), str(current_csv)])

    assert len(merged) == 1
    assert merged[0]['shuffle_id'] == '-1'
