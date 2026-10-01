"""Ablation command: one switch per configuration, results of 2.0.0 stay valid, the end
points repeat the full run exactly, and a reused sample gives the same results as a fresh one.
"""
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlshare_rerun import ablation as A        # noqa: E402
from sqlshare_rerun import stages as ST         # noqa: E402
from sqlshare_rerun.config import make_settings  # noqa: E402
from sqlshare_rerun.selftest import SELFTEST_SETTINGS, build_fixture  # noqa: E402

WORKERS = int(os.environ.get("SQLSHARE_TEST_WORKERS", "2"))
THREADS = int(os.environ.get("SQLSHARE_TEST_THREADS", "4"))

#: Fingerprints written by version 2.0.0 (the published full run has the first two).
V200 = {("paper", False): "11cf10d670fa", ("corrected", False): "c9c9df996639",
        ("paper", True): "762beb346515", ("corrected", True): "ec3c890469e9"}


def test_results_of_version_2_0_0_stay_valid():
    for (proto, quick), fp in V200.items():
        assert make_settings(proto, ".", quick).fingerprint() == fp
    s = make_settings("paper", ".")
    assert s.exact_formulation() and not make_settings("corrected", ".").exact_formulation()


def test_each_configuration_changes_only_its_switch(tmp_path):
    lat = {"value_set": tmp_path / "a", "predicate": tmp_path / "b"}
    ends = {b: A._config(b, tmp_path, False, {}, b, tmp_path, lat, {}) for b in ("paper", "corrected")}
    configs = A.configurations(tmp_path, False, "both", 2)
    names = [c[0] for c in configs]
    assert len(names) == len(set(names)) == 2 + 2 * len(A.SWITCHES) + 4
    switch = {n: (pv, cv) for n, _, pv, cv, _ in A.SWITCHES}
    for name, base, ov, sw, direction in configs:
        s = A._config(base, tmp_path, False, ov, name, tmp_path, lat, {})
        diff = A._settings_diff(ends[base], s)
        if direction == "end point":
            assert diff == {}
        elif direction == "seed set":
            assert set(diff) == set(A.SAMPLING_SEED_FIELDS)
        else:
            want = switch[sw][1] if direction == "add" else switch[sw][0]
            changed = {k for k, v in want.items() if getattr(ends[base], k) != v}
            assert set(diff) == changed and changed, (name, diff)
            for k in changed:
                assert getattr(s, k) == want[k]
    with pytest.raises(ValueError):
        A.run_ablation(tmp_path, only=["no_such_switch"])


def test_ablation_end_points_and_sample_cache(tmp_path):
    (tmp_path / "data").mkdir()
    build_fixture(tmp_path / "data" / "sqlshare_data_release1.zip")
    common = dict(SELFTEST_SETTINGS, workers=WORKERS, threads=THREADS)
    for proto in ("paper", "corrected"):                      # the "full run"
        s = make_settings(proto, tmp_path, **common)
        ST.stage_main_market(s)
    out = A.run_ablation(tmp_path, False, "both", 1, ["C12_row_sampling", "alias_resolve", "C05_fit"],
                         common=common)
    assert out["checks"] == {"paper": "identical", "corrected": "identical"}
    adir = tmp_path / "results" / "ablation"
    for f in ("ABLATION.md", "ablation_long.csv", "ablation_headline.csv", "ablation_configurations.csv"):
        assert (adir / f).exists()
    long = pd.read_csv(adir / "ablation_long.csv")
    assert set(long.configuration) >= {"paper", "corrected", "add_C12_row_sampling", "drop_alias_resolve",
                                       "paper_seedset1", "corrected_seedset1"}
    # a configuration that reused cached draws equals the same configuration drawn fresh
    name = "add_C05_fit"
    cached = pd.read_csv(adir / name / "main_market" / "runs.csv")
    shutil.rmtree(adir / "_samples")
    shutil.rmtree(adir / name)
    lat = {"value_set": adir / "_lattice_value_set" / "lattice", "predicate": adir / "_lattice_predicate" / "lattice"}
    s = A._config("paper", tmp_path, False, {"fit": "breakpoints"}, name, adir, lat, common)
    s.extra.pop("sample_cache")
    ST.stage_main_market(s)
    fresh = pd.read_csv(adir / name / "main_market" / "runs.csv")
    keep = [c for c in cached.columns if c not in ("elapsed_s", "build_seconds", "seconds")]
    pd.testing.assert_frame_equal(cached[keep], fresh[keep], check_exact=True)
