import numpy as np

from scripts.diagnose_single_network_subpart_w1 import collect_diagnostic_rows
from scripts.diagnose_single_network_subpart_w1 import summarize_diagnostic_rows


def _write_group_artifact(group_dir, step: int, weights: np.ndarray, dataset: str = "imagenet") -> None:
    artifact_dir = group_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    np.savez(artifact_dir / f"first_layer_{step}.npz", first_layer_weights=weights.astype(np.float32))
    (group_dir / "metadata.json").write_text(f'{{"dataset": "{dataset}"}}', encoding="utf-8")


def test_collect_diagnostic_rows_marks_notebook_row_and_computes_summary(tmp_path):
    run_dir = tmp_path / "exchangeability_test"
    width_dir = run_dir / "width_32"

    weights_g0 = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.8, 0.2]],
            [[0.0, 0.0, 1.0], [0.0, 0.2, 0.8], [1.0, 0.0, 0.0], [0.8, 0.0, 0.2]],
        ],
        dtype=np.float32,
    )
    weights_g1 = np.asarray(
        [
            [[1.0, 1.0, 0.0], [0.9, 1.0, 0.1], [0.0, 1.0, 1.0], [0.1, 0.9, 1.0]],
            [[1.0, 0.0, 1.0], [1.0, 0.1, 0.9], [0.0, 1.0, 1.0], [0.1, 1.0, 0.9]],
        ],
        dtype=np.float32,
    )

    _write_group_artifact(width_dir / "group_0", 100, weights_g0)
    _write_group_artifact(width_dir / "group_1", 100, weights_g1)

    rows = collect_diagnostic_rows(
        base_save_dir=str(run_dir),
        run_id="",
        resolution_mode="auto",
        widths=[32],
        steps=[100],
        fractions=[0.5],
        repeats=4,
        seed=7,
        member_indices=None,
    )

    assert len(rows) == 4
    notebook_rows = [row for row in rows if int(row["is_notebook_row"]) == 1]
    assert len(notebook_rows) == 1
    assert int(notebook_rows[0]["group_id"]) == 0
    assert int(notebook_rows[0]["member_index"]) == 0

    for row in rows:
        assert row["dataset"] == "imagenet"
        assert int(row["width"]) == 32
        assert int(row["images_seen"]) == 100
        assert float(row["fraction"]) == 0.5
        assert int(row["subpart_width"]) == 2
        assert int(row["repeats"]) == 4
        assert np.isfinite(float(row["w1_mean"]))
        assert np.isfinite(float(row["pooled_subpart_w1"]))
        assert np.isfinite(float(row["width_times_w1_mean"]))

    summary_rows = summarize_diagnostic_rows(rows)
    assert len(summary_rows) == 1
    summary = summary_rows[0]
    assert int(summary["row_count"]) == 4
    assert int(summary["group_count"]) == 2
    assert int(summary["group_member_count"]) == 4
    assert int(summary["notebook_row_present"]) == 1
    assert int(summary["notebook_group_id"]) == 0
    assert int(summary["notebook_member_index"]) == 0
    assert np.isfinite(float(summary["all_rows_w1_mean_mean"]))
    assert np.isfinite(float(summary["group_mean_w1_mean"]))
