import csv

from scripts.merge_single_network_subpart_w1_csvs import _merge_rows


def _write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _raw_row():
    return {
        'dataset': 'imagenet',
        'width': '128',
        'source_run_id': 'exchangeability_20260407',
        'images_seen': '100000',
        'fraction': '0.5',
        'group_id': '0',
        'member_index': '0',
        'w1_mean': '0.01',
    }


def _summary_row():
    return {
        'dataset': 'imagenet',
        'width': '128',
        'source_run_id': 'exchangeability_20260407',
        'images_seen': '100000',
        'fraction': '0.5',
        'all_rows_w1_mean_mean': '0.02',
    }


def test_merge_rows_deduplicates_raw_records(tmp_path):
    row = _raw_row()
    left_csv = tmp_path / 'single_network_subpart_w1.csv'
    right_csv = tmp_path / 'single_network_subpart_w1_w128.csv'

    _write_csv(left_csv, list(row), [row])
    _write_csv(right_csv, list(row), [row])

    fieldnames, merged = _merge_rows([str(left_csv), str(right_csv)], mode='raw')

    assert len(merged) == 1
    assert 'member_index' in fieldnames
    assert merged[0]['group_id'] == '0'


def test_merge_rows_deduplicates_summary_records(tmp_path):
    row = _summary_row()
    left_csv = tmp_path / 'single_network_subpart_w1_summary.csv'
    right_csv = tmp_path / 'single_network_subpart_w1_summary_w128.csv'

    _write_csv(left_csv, list(row), [row])
    _write_csv(right_csv, list(row), [row])

    fieldnames, merged = _merge_rows([str(left_csv), str(right_csv)], mode='summary')

    assert len(merged) == 1
    assert 'all_rows_w1_mean_mean' in fieldnames
    assert merged[0]['fraction'] == '0.5'
