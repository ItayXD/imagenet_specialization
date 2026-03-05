import os
from pathlib import Path

import pytest

from scripts.analyze_exchangeability import _resolve_run_id, _resolve_width_dirs


def _make_dir(
    base: Path,
    name: str,
    mtime: int,
    with_run_layout: bool = True,
    widths: tuple[int, ...] = (32,),
) -> None:
    path = base / name
    path.mkdir()
    if with_run_layout:
        for width in widths:
            (path / f"width_{width}").mkdir()
            (path / f"width_{width}" / "group_0").mkdir()
    path.touch()
    path_ts = float(mtime)
    os.utime(path, (path_ts, path_ts))


def test_resolve_run_id_exact_mode_prefers_exact(tmp_path: Path) -> None:
    _make_dir(tmp_path, "exchangeability", 100)
    _make_dir(tmp_path, "exchangeability_job1", 200)
    assert _resolve_run_id(str(tmp_path), "exchangeability", "exact") == "exchangeability"


def test_resolve_run_id_latest_prefix_mode(tmp_path: Path) -> None:
    _make_dir(tmp_path, "exchangeability_job1", 100)
    _make_dir(tmp_path, "exchangeability_job2", 200)
    assert _resolve_run_id(str(tmp_path), "exchangeability", "latest_prefix") == "exchangeability_job2"


def test_resolve_run_id_auto_mode_picks_newest_across_exact_and_prefix(tmp_path: Path) -> None:
    _make_dir(tmp_path, "exchangeability", 100)
    _make_dir(tmp_path, "exchangeability_job9", 200)
    assert _resolve_run_id(str(tmp_path), "exchangeability", "auto") == "exchangeability_job9"


def test_resolve_run_id_latest_prefix_falls_back_to_exact(tmp_path: Path) -> None:
    _make_dir(tmp_path, "exchangeability", 100)
    assert _resolve_run_id(str(tmp_path), "exchangeability", "latest_prefix") == "exchangeability"


def test_resolve_run_id_exact_mode_errors_when_exact_missing(tmp_path: Path) -> None:
    _make_dir(tmp_path, "exchangeability_job1", 100)
    with pytest.raises(FileNotFoundError):
        _resolve_run_id(str(tmp_path), "exchangeability", "exact")


def test_resolve_run_id_latest_prefix_ignores_similarity_cache_when_exact_exists(tmp_path: Path) -> None:
    _make_dir(tmp_path, "exchangeability", 100, with_run_layout=True)
    # Cache folder has width_* but no group_* directories, so it must not be treated as a run.
    cache_dir = tmp_path / "exchangeability_metrics_w32_similarity"
    cache_width_dir = cache_dir / "width_32"
    cache_width_dir.mkdir(parents=True)
    path_ts = float(200)
    os.utime(cache_dir, (path_ts, path_ts))
    os.utime(cache_width_dir, (path_ts, path_ts))
    assert _resolve_run_id(str(tmp_path), "exchangeability", "latest_prefix") == "exchangeability"


def test_resolve_width_dirs_latest_prefix_picks_latest_per_requested_width(tmp_path: Path) -> None:
    _make_dir(tmp_path, "exchangeability_job1", 100, widths=(32,))
    _make_dir(tmp_path, "exchangeability_job2", 200, widths=(64,))
    _make_dir(tmp_path, "exchangeability_job3", 300, widths=(32,))

    width_dirs, width_sources = _resolve_width_dirs(
        str(tmp_path),
        "exchangeability",
        "latest_prefix",
        requested_widths=[32, 64],
    )

    assert set(width_dirs.keys()) == {32, 64}
    assert width_sources[32] == "exchangeability_job3"
    assert width_sources[64] == "exchangeability_job2"


def test_resolve_width_dirs_auto_uses_newest_source_per_width(tmp_path: Path) -> None:
    _make_dir(tmp_path, "exchangeability", 250, widths=(64,))
    _make_dir(tmp_path, "exchangeability_job1", 100, widths=(64,))
    _make_dir(tmp_path, "exchangeability_job2", 300, widths=(32,))

    width_dirs, width_sources = _resolve_width_dirs(
        str(tmp_path),
        "exchangeability",
        "auto",
        requested_widths=[32, 64],
    )

    assert set(width_dirs.keys()) == {32, 64}
    assert width_sources[32] == "exchangeability_job2"
    assert width_sources[64] == "exchangeability"
