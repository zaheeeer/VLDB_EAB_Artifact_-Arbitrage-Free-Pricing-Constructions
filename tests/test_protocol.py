"""Independent small reference cases for errors identified in the recovered code."""
import sys
from pathlib import Path
import numpy as np
import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "candidate"))
sys.path.insert(0, str(ROOT / "legacy"))
from protocol import (buyer_split, fit_scale_breakpoints, partition_entropy,
                      verified_effective_prices, delete_one_seeded)
from tpc_experiment import fit_scale as old_fit_scale


def test_fixed_distribution_reproducible_independent_splits():
    a = buyer_split(np.ones(12), 17, 120, 100)
    b = buyer_split(np.ones(12), 17, 120, 100)
    np.testing.assert_array_equal(a["train_demand"], a["test_demand"])
    for split in ("train", "test"):
        for x, y in zip(a[split], b[split]):
            np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a["train"][1][:100], a["test"][1])


def test_test_stream_does_not_depend_on_training_size():
    a = buyer_split(np.ones(12), 17, 120, 100)
    b = buyer_split(np.ones(12), 17, 240, 100)
    for x, y in zip(a["test"], b["test"]):
        np.testing.assert_array_equal(x, y)


def test_shift_is_explicit():
    a = buyer_split(np.ones(12), 17, shift=True)
    assert not np.array_equal(a["train_demand"], a["test_demand"])


def test_fixed_grid_counterexample_is_repaired():
    targets = np.zeros(4, dtype=int)
    values = np.full(4, 100.0)
    assert old_fit_scale(targets, values, np.ones(1)) == 60
    assert fit_scale_breakpoints(targets, values, np.ones(1)) == 100


def test_flat_price_matches_independent_uniform_bundle_reference():
    values = np.array([4., 9., 9., 15., 22., 22., 27., 71., 120.])
    expected = max(p * np.count_nonzero(values >= p) for p in np.unique(values))
    alpha = fit_scale_breakpoints(np.arange(len(values)), values, np.ones(len(values)))
    assert alpha * np.count_nonzero(values >= alpha) == pytest.approx(expected)


def test_scaled_price_matches_dense_independent_grid():
    # Integer scores and quarter-integer valuations put all optimum breakpoints
    # on this independent fine grid. Zero-score buyers cannot increase revenue.
    scores = np.array([0., 1., 2., 4.])
    target = np.array([0, 1, 1, 2, 2, 3, 3])
    values = np.array([100., 3., 8., 10., 21., 24., 36.])
    alpha = fit_scale_breakpoints(target, values, scores)
    revenue = lambda p: float((p * scores[target])[p * scores[target] <= values].sum())
    assert revenue(alpha) == pytest.approx(max(revenue(x) for x in np.arange(0, 100, .125)))


def test_free_scores_and_empty_training():
    assert fit_scale_breakpoints(np.array([0, 0]), np.array([10., 20.]), [0.]) == 0
    assert fit_scale_breakpoints(np.array([], dtype=int), [], [1.]) == 0


def test_entropy_merges_identical_answers_and_accepts_weights():
    assert partition_entropy([("count", 1), ("count", 1)]) == 0
    assert partition_entropy([("count", 1), ("count", 2)]) == 1
    assert partition_entropy(["same", "same", "different"], [1, 2, 1]) == pytest.approx(.8112781244591328)


def test_failed_unchecked_and_unavailable_plans_cannot_discount():
    actual = verified_effective_prices([10, 10, 10, 10, 10], [1, 1, 1, 1, np.inf],
                                       [True, False, None, "True", True])
    np.testing.assert_array_equal(actual, [1, 10, 10, 10, 10])


def test_seeded_deletion_handles_duplicate_null_rows_and_rolls_back():
    con = duckdb.connect()
    con.execute("CREATE TABLE sample(x INTEGER, y VARCHAR)")
    con.execute("INSERT INTO sample VALUES (1, NULL), (1, NULL), (2, 'a')")
    ids = []
    for _ in range(2):
        con.execute("BEGIN")
        ids.append(delete_one_seeded(con, ("main", "sample"), np.random.default_rng(3)))
        assert con.execute("SELECT count(*) FROM sample").fetchone()[0] == 2
        con.execute("ROLLBACK")
        assert con.execute("SELECT count(*) FROM sample").fetchone()[0] == 3
    assert ids[0] == ids[1]
    con.close()


def test_seeded_deletion_rejects_explicit_rowid_column():
    con = duckdb.connect()
    con.execute("CREATE TABLE sample(rowid INTEGER)")
    with pytest.raises(ValueError, match="stable-ID"):
        delete_one_seeded(con, ("main", "sample"), np.random.default_rng(0))
    con.close()
