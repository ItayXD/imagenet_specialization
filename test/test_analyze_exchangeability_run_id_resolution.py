import os
from pathlib import Path

import pytest

from scripts.analyze_exchangeability import _resolve_run_id


def _make_dir(base: Path, name: str, mtime: int) -> None:
    path = base / name
    path.mkdir()
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
