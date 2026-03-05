import csv
import math

import numpy as np

from scripts.analyze_exchangeability import (
    ANALYSIS_FIELDNAMES,
    _annotate_within_observed_empirical_p,
    _empirical_upper_tail_p,
    _prepare_resume_state,
)


def _base_row(**overrides):
    row = {
        'width': 32,
        'images_seen': 100000,
        'representation': 'weights',
        'analysis_type': 'within_vs_across_real',
        'shuffle_id': -1,
        'ks_distance': 0.5,
        'ks_p_raw': 0.01,
        'ks_sigma_two_sided': 2.57,
        'w1_distance': 0.4,
        'train_loss': 1.0,
        'val_loss': 1.1,
        'train_error': 0.3,
        'val_error': 0.31,
    }
    row.update(overrides)
    return row


def _read_csv_rows(path):
    with open(path, 'r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_empirical_plus_one_formula():
    observed = 0.5
    null = np.asarray([0.1, 0.5, 0.7], dtype=np.float64)
    p = _empirical_upper_tail_p(observed=observed, null_values=null)
    assert p == (1 + 2) / (3 + 1)


def test_empirical_columns_only_on_within_observed_row():
    rows = [
        _base_row(analysis_type='within_vs_across_real', shuffle_id=-1, ks_distance=0.6, w1_distance=0.6),
        _base_row(analysis_type='across_real_vs_across_shuffled', shuffle_id=0, ks_distance=0.2, w1_distance=0.2),
        _base_row(analysis_type='within_shuffled_vs_across_real', shuffle_id=0, ks_distance=0.3, w1_distance=0.3),
        _base_row(analysis_type='within_shuffled_vs_across_real', shuffle_id=1, ks_distance=0.7, w1_distance=0.7),
    ]

    changed = _annotate_within_observed_empirical_p(rows)
    assert changed

    observed = rows[0]
    assert observed['ks_p_empirical'] == (1 + 1) / (2 + 1)
    assert observed['w1_p_empirical'] == (1 + 1) / (2 + 1)
    assert np.isfinite(observed['ks_sigma_empirical_two_sided'])
    assert np.isfinite(observed['w1_sigma_empirical_two_sided'])

    for row in rows[1:]:
        assert math.isnan(float(row['ks_p_empirical']))
        assert math.isnan(float(row['ks_sigma_empirical_two_sided']))
        assert math.isnan(float(row['w1_p_empirical']))
        assert math.isnan(float(row['w1_sigma_empirical_two_sided']))


def test_empirical_handles_nan_and_empty_null():
    rows = [
        _base_row(analysis_type='within_vs_across_real', shuffle_id=-1, ks_distance=0.4, w1_distance=0.2),
        _base_row(analysis_type='within_shuffled_vs_across_real', shuffle_id=0, ks_distance=np.nan, w1_distance=np.nan),
        _base_row(analysis_type='within_shuffled_vs_across_real', shuffle_id=1, ks_distance=0.3, w1_distance=np.nan),
    ]
    _annotate_within_observed_empirical_p(rows)
    observed = rows[0]
    assert observed['ks_p_empirical'] == (1 + 0) / (1 + 1)
    assert math.isnan(float(observed['w1_p_empirical']))
    assert math.isnan(float(observed['w1_sigma_empirical_two_sided']))


def test_resume_backfill_populates_new_columns_without_recompute(tmp_path):
    csv_path = tmp_path / 'legacy.csv'
    rows = [
        _base_row(analysis_type='within_vs_across_real', shuffle_id=-1, ks_distance=0.55, w1_distance=0.45),
        _base_row(analysis_type='across_real_vs_across_shuffled', shuffle_id=0, ks_distance=0.2, w1_distance=0.2),
        _base_row(analysis_type='within_shuffled_vs_across_real', shuffle_id=0, ks_distance=0.3, w1_distance=0.25),
        _base_row(analysis_type='across_real_vs_across_shuffled', shuffle_id=1, ks_distance=0.1, w1_distance=0.1),
        _base_row(analysis_type='within_shuffled_vs_across_real', shuffle_id=1, ks_distance=0.8, w1_distance=0.7),
    ]
    legacy_fieldnames = [
        'width',
        'images_seen',
        'representation',
        'analysis_type',
        'shuffle_id',
        'ks_distance',
        'ks_p_raw',
        'ks_sigma_two_sided',
        'w1_distance',
        'train_loss',
        'val_loss',
        'train_error',
        'val_error',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=legacy_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    completed_reps, rows_written, file_mode = _prepare_resume_state(
        output_csv=str(csv_path),
        fieldnames=ANALYSIS_FIELDNAMES,
        shuffle_repeats=2,
        resume=True,
    )

    assert completed_reps == {(32, '', 100000, 'weights')}
    assert rows_written == 5
    assert file_mode == 'a'

    updated_rows = _read_csv_rows(csv_path)
    observed = next(r for r in updated_rows if r['analysis_type'] == 'within_vs_across_real')
    assert float(observed['ks_p_empirical']) == (1 + 1) / (2 + 1)
    assert float(observed['w1_p_empirical']) == (1 + 1) / (2 + 1)


def test_existing_asymptotic_columns_unchanged():
    rows = [
        _base_row(analysis_type='within_vs_across_real', shuffle_id=-1, ks_distance=0.6, ks_p_raw=0.123, ks_sigma_two_sided=1.98),
        _base_row(analysis_type='within_shuffled_vs_across_real', shuffle_id=0, ks_distance=0.7),
        _base_row(analysis_type='within_shuffled_vs_across_real', shuffle_id=1, ks_distance=0.2),
    ]
    _annotate_within_observed_empirical_p(rows)
    observed = rows[0]
    assert observed['ks_p_raw'] == 0.123
    assert observed['ks_sigma_two_sided'] == 1.98


def test_resume_recomputes_when_source_job_changes(tmp_path):
    csv_path = tmp_path / 'source_bound.csv'
    rows = [
        _base_row(
            source_run_id='exchangeability_job_old',
            analysis_type='within_vs_across_real',
            shuffle_id=-1,
            ks_distance=0.55,
            w1_distance=0.45,
        ),
        _base_row(
            source_run_id='exchangeability_job_old',
            analysis_type='across_real_vs_across_shuffled',
            shuffle_id=0,
            ks_distance=0.2,
            w1_distance=0.2,
        ),
        _base_row(
            source_run_id='exchangeability_job_old',
            analysis_type='within_shuffled_vs_across_real',
            shuffle_id=0,
            ks_distance=0.3,
            w1_distance=0.25,
        ),
        _base_row(
            source_run_id='exchangeability_job_old',
            analysis_type='across_real_vs_across_shuffled',
            shuffle_id=1,
            ks_distance=0.1,
            w1_distance=0.1,
        ),
        _base_row(
            source_run_id='exchangeability_job_old',
            analysis_type='within_shuffled_vs_across_real',
            shuffle_id=1,
            ks_distance=0.8,
            w1_distance=0.7,
        ),
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    completed_reps, rows_written, file_mode = _prepare_resume_state(
        output_csv=str(csv_path),
        fieldnames=ANALYSIS_FIELDNAMES,
        shuffle_repeats=2,
        resume=True,
        width_sources={32: 'exchangeability_job_new'},
    )

    assert completed_reps == set()
    assert rows_written == 0
    assert file_mode == 'w'
    assert _read_csv_rows(csv_path) == []
