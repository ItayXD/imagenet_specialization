import csv
import os

from scripts.build_width_slurm_jobs import main


def _write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_builds_width_specific_manifests_and_submit_scripts(tmp_path, monkeypatch):
    manifest_path = tmp_path / 'conf' / 'exchangeability_manifest.csv'
    timing_path = tmp_path / 'timing_by_width.csv'
    manifest_out = tmp_path / 'conf' / 'manifests_by_width'
    slurm_out = tmp_path / 'conf' / 'slurm_jobs'

    _write_csv(
        str(manifest_path),
        fieldnames=['job_id', 'width', 'group_id', 'experiment_name'],
        rows=[
            {'job_id': '0', 'width': '32', 'group_id': '0', 'experiment_name': 'exchangeability_w32_g0'},
            {'job_id': '1', 'width': '32', 'group_id': '1', 'experiment_name': 'exchangeability_w32_g1'},
            {'job_id': '2', 'width': '64', 'group_id': '0', 'experiment_name': 'exchangeability_w64_g0'},
        ],
    )

    _write_csv(
        str(timing_path),
        fieldnames=['width', 'recommended_sbatch_time'],
        rows=[
            {'width': '32', 'recommended_sbatch_time': '06:00:00'},
            {'width': '64', 'recommended_sbatch_time': '07:30:00'},
        ],
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        'sys.argv',
        [
            'build_width_slurm_jobs.py',
            '--manifest',
            str(manifest_path),
            '--timing-width-csv',
            str(timing_path),
            '--manifest-output-dir',
            str(manifest_out),
            '--slurm-output-dir',
            str(slurm_out),
        ],
    )

    main()

    manifest_32 = manifest_out / 'exchangeability_manifest_w32.csv'
    manifest_64 = manifest_out / 'exchangeability_manifest_w64.csv'
    assert manifest_32.exists()
    assert manifest_64.exists()

    with open(manifest_32, 'r', encoding='utf-8', newline='') as f:
        rows_32 = list(csv.DictReader(f))
    with open(manifest_64, 'r', encoding='utf-8', newline='') as f:
        rows_64 = list(csv.DictReader(f))

    assert len(rows_32) == 2
    assert len(rows_64) == 1

    submit_32 = slurm_out / 'submit_exchangeability_w32.sbatch'
    submit_64 = slurm_out / 'submit_exchangeability_w64.sbatch'
    submit_all = slurm_out / 'submit_exchangeability_all_widths.sh'

    assert submit_32.exists()
    assert submit_64.exists()
    assert submit_all.exists()

    text_32 = submit_32.read_text(encoding='utf-8')
    text_64 = submit_64.read_text(encoding='utf-8')

    assert '#SBATCH --time=06:00:00' in text_32
    assert '#SBATCH --array=0-1' in text_32
    assert '#SBATCH --time=07:30:00' in text_64
    assert '#SBATCH --array=0-0' in text_64

    submit_all_text = submit_all.read_text(encoding='utf-8')
    assert 'sbatch "conf/slurm_jobs/submit_exchangeability_w32.sbatch"' in submit_all_text
    assert 'sbatch "conf/slurm_jobs/submit_exchangeability_w64.sbatch"' in submit_all_text


def test_builds_dataset_aware_names_when_manifest_has_dataset(tmp_path, monkeypatch):
    manifest_path = tmp_path / 'conf' / 'exchangeability_manifest.csv'
    timing_path = tmp_path / 'timing_by_width.csv'
    manifest_out = tmp_path / 'conf' / 'manifests_by_width'
    slurm_out = tmp_path / 'conf' / 'slurm_jobs'

    _write_csv(
        str(manifest_path),
        fieldnames=['job_id', 'dataset', 'width', 'group_id', 'experiment_name'],
        rows=[
            {'job_id': '0', 'dataset': 'cifar5m', 'width': '32', 'group_id': '0', 'experiment_name': 'cifar5m_exchangeability_w32_g0'},
        ],
    )
    _write_csv(
        str(timing_path),
        fieldnames=['width', 'recommended_sbatch_time'],
        rows=[{'width': '32', 'recommended_sbatch_time': '06:00:00'}],
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        'sys.argv',
        [
            'build_width_slurm_jobs.py',
            '--manifest',
            str(manifest_path),
            '--timing-width-csv',
            str(timing_path),
            '--manifest-output-dir',
            str(manifest_out),
            '--slurm-output-dir',
            str(slurm_out),
        ],
    )

    main()

    assert (manifest_out / 'cifar5m_exchangeability_manifest_w32.csv').exists()
    assert (slurm_out / 'submit_exchangeability_cifar5m_w32.sbatch').exists()
    submit_all = slurm_out / 'submit_exchangeability_cifar5m_all_widths.sh'
    assert submit_all.exists()
    assert 'submit_exchangeability_cifar5m_w32.sbatch' in submit_all.read_text(encoding='utf-8')
