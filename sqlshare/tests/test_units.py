"""Unit tests: every optimization must return exactly what the historical code returned,
and every alternative must do what its choice says. Run with: python -m pytest -q tests"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREADS = int(__import__("os").environ.get("SQLSHARE_TEST_THREADS", "4"))

from sqlshare_rerun import pricing as P                      # noqa: E402
from sqlshare_rerun.lattice import (SortedTargets, cover_structure, derive_pair,  # noqa: E402
                                    frames_equal, query_predicate, try_selection_predicate)
from sqlshare_rerun.planner import min_cost_row_cover, plan_all, verify_row_cover  # noqa: E402
from sqlshare_rerun.support import SupportSampler, base_fingerprints  # noqa: E402
from sqlshare_rerun.util import windows_name  # noqa: E402


# ---- historical reference implementations (verbatim from legacy/) ------------------------
def legacy_uip(edges, vals, n_items):
    v = np.asarray(vals, float)
    sizes = np.array([max(len(e), 1) for e in edges], float)
    q = v / sizes
    best_w, best_r = 0.0, -1.0
    for cand in np.unique(q):
        w = np.full(n_items, cand)
        r = P.revenue(P._edge_price(edges, w), v)
        if r > best_r:
            best_w, best_r = float(cand), r
    return best_w


def legacy_cip(edges, vals, n_items, eps=0.5):
    from scipy.optimize import linprog
    v = np.asarray(vals, float)
    m = len(edges)
    k_max = max(m, 2)
    best_w, best_r = np.zeros(n_items), -1.0
    k = 1.0
    while k <= k_max:
        rowsA, b = [], []
        for j in range(n_items):
            ind = np.array([1.0 if j in e else 0.0 for e in edges])
            if ind.any():
                rowsA.append(ind)
                b.append(k)
        if not rowsA:
            break
        res = linprog(-v, A_ub=np.array(rowsA), b_ub=np.array(b), bounds=[(0, 1)] * m, method="highs")
        if res.success and res.ineqlin is not None:
            duals = np.abs(res.ineqlin.marginals)
            w = np.zeros(n_items)
            jj = 0
            for j in range(n_items):
                if any(j in e for e in edges):
                    w[j] = duals[jj]
                    jj += 1
            r = P.revenue(P._edge_price(edges, w), v)
            if r > best_r:
                best_w, best_r = w, r
        k *= (1.0 + eps)
    return best_w


def legacy_plan_all(p, keys, ki, contrib, nrows, answers):
    cost = p.copy()
    vok = vbad = 0
    pm = {k: float(p[ki[k]]) for k in keys}
    for k, c_ in contrib.items():
        cc_, plan = min_cost_row_cover(c_, nrows[k], pm)
        if plan and cc_ < p[ki[k]] - 1e-9:
            if verify_row_cover(answers, k, plan):
                cost[ki[k]] = cc_
                vok += 1
            else:
                vbad += 1
    return cost, vok, vbad


def random_edges(rng, m, n_items, pmax=0.08):
    return [frozenset(np.flatnonzero(rng.random(n_items) < rng.uniform(0, pmax)).tolist()) for _ in range(m)]


# ---- optimizations are exact -----------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_uip_fast_path_equals_legacy(seed):
    rng = np.random.default_rng(seed)
    edges = random_edges(rng, 120, 60)
    vals = rng.lognormal(3, 0.6, 120)
    _, info = P.uip(edges, vals, 60)
    assert info["w"] == legacy_uip(edges, vals, 60)


@pytest.mark.parametrize("seed", range(3))
def test_cip_vectorized_equals_legacy(seed):
    rng = np.random.default_rng(seed)
    edges = random_edges(rng, 60, 40, 0.15)
    vals = rng.lognormal(3, 0.6, 60)
    _, info = P.cip(edges, vals, 40)
    np.testing.assert_array_equal(info["weights"], legacy_cip(edges, vals, 40))


def test_xos_reuses_lpip_and_cip():
    rng = np.random.default_rng(3)
    edges_asset = random_edges(rng, 30, 50, 0.2)
    tr_t = rng.integers(0, 30, 80)
    tr_v = rng.lognormal(3, 0.6, 80)
    out = P.chawla_all(edges_asset, tr_t, tr_v, 50, max_lps=10)
    edges_b = [edges_asset[t] for t in tr_t]
    _, i1 = P.lpip(edges_b, tr_v, 50, max_lps=10)
    _, i2 = P.cip(edges_b, tr_v, 50)
    ref = np.array([max(float(i1["weights"][list(e)].sum()) if len(e) else 0.0,
                        float(i2["weights"][list(e)].sum()) if len(e) else 0.0) for e in edges_asset])
    np.testing.assert_array_equal(out["chawla_xos"], ref)


def _cover_fixture(seed, n_assets=14, n_rows=12):
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({"a": np.arange(n_rows), "b": rng.integers(0, 3, n_rows)})
    answers = {}
    for k in range(n_assets):
        idx = np.sort(rng.choice(n_rows, size=int(rng.integers(1, n_rows + 1)), replace=False))
        answers[k] = base.iloc[idx].reset_index(drop=True)
    return answers


@pytest.mark.parametrize("seed", range(6))
def test_plan_filter_is_exact(seed):
    answers = _cover_fixture(seed)
    keys = sorted(answers)
    ki = {k: i for i, k in enumerate(keys)}
    contrib, nrows = cover_structure(answers, keys, 2000)
    rng = np.random.default_rng(100 + seed)
    for _ in range(5):
        p = rng.uniform(0, 10, len(keys))
        p[rng.random(len(keys)) < 0.2] = 0.0
        cost, _, vok, vbad = plan_all(p, keys, ki, contrib, nrows, answers)
        ref, rok, rbad = legacy_plan_all(p, keys, ki, contrib, nrows, answers)
        np.testing.assert_allclose(np.minimum(p, cost), np.minimum(p, ref))
        assert (vok, vbad) == (rok, rbad)


def test_sorted_target_cache_matches_frames_equal():
    rng = np.random.default_rng(0)
    T = pd.DataFrame({"x": rng.integers(0, 5, 30), "y": rng.normal(size=30).round(3)})
    cache = SortedTargets({"t": T})
    for _ in range(20):
        got = T.sample(frac=1.0, random_state=int(rng.integers(1e6))).reset_index(drop=True)
        if rng.random() < 0.5:
            got.loc[0, "y"] += 1.0
        assert cache.equal(got, "t") == frames_equal(got, T)


# ---- alternatives do what they claim --------------------------------------------------
def test_breakpoint_fit_is_optimal_and_uncapped():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.01, 1, 40)
    t = rng.integers(0, 40, 300)
    v = 100 * rng.lognormal(0, 0.6, 300) * scores[t] ** 0.7
    a = P.fit_scale_breakpoints(t, v, scores)
    rev = lambda x: float((x * scores[t])[x * scores[t] <= v].sum())
    dense = max(rev(x) for x in np.linspace(0, 2000, 40001))
    assert rev(a) >= dense - 1e-9
    assert a > 60                                    # the historical grid could not reach it
    assert P.fit_scale_grid(t, v, scores) <= 60


def test_flat_fit_equals_uniform_bundle_price():
    v = np.array([4., 9., 9., 15., 22., 22., 27., 71., 120.])
    a = P.fit_scale_breakpoints(np.arange(9), v, np.ones(9))
    assert a * np.count_nonzero(v >= a) == max(p * np.count_nonzero(v >= p) for p in v)


def test_shared_demand_and_shift_arm():
    info = np.linspace(0.1, 1, 25)
    (t1, _), (t2, _) = P.buyer_split(info, 7, 4000, 4000, shift=False)
    (s1, _), (s2, _) = P.buyer_split(info, 7, 4000, 4000, shift=True)
    c = lambda t: np.bincount(t, minlength=25) / len(t)
    assert np.abs(c(t1) - c(t2)).max() < 0.05          # same demand
    assert np.abs(c(s1) - c(s2)).max() > 0.05          # shifted demand


def test_exact_entropy_merges_equal_answers():
    n = 4
    singleton = P.qirana_from_edges([frozenset({0, 1, 2})], n)["qirana_shannon"][0]
    exact = P.exact_entropy({0: ("x", 1), 1: ("x", 1), 2: ("y", 2)}, n)
    assert exact < singleton
    assert exact == pytest.approx(1.5)


def test_predicate_selection_uses_query_not_answer():
    S = pd.DataFrame({"station": ["S1", "S1", "S2", "S3"], "depth": [5, 7, 9, 11]})
    T = pd.DataFrame({"station": ["S1", "S1"], "depth": [5, 7]})
    pred = query_predicate('SELECT station, depth FROM "446"."ctd.csv" c WHERE c.station = \'S1\'')
    assert pred is not None and "c." not in pred
    got = try_selection_predicate(S, T, pred)
    assert frames_equal(got, T)
    # a TOP-n target has no WHERE: the buyer cannot rebuild it, the value-set rule could
    assert query_predicate('SELECT * FROM "446"."ctd.csv" LIMIT 2') is None
    targets = SortedTargets({"t": T})
    assert derive_pair(S, T, "t", targets, "value_set") == "selection"
    assert derive_pair(S, T, "t", targets, "predicate", None) is None


def test_windows_names():
    assert windows_name("table_a:b?.csv") == "table_a_b_.csv"
    assert windows_name("table_x.csv. ") == "table_x.csv"


# ---- support sampling is deterministic under threads -----------------------------------
@pytest.fixture()
def small_db(tmp_path):
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    con.execute('CREATE SCHEMA "s"')
    con.execute('CREATE TABLE "s"."a" AS SELECT i % 13 AS g, i AS v FROM range(400) r(i)')
    con.execute('CREATE TABLE "s"."b" AS SELECT i % 5 AS g, i * 2 AS w FROM range(150) r(i)')
    con.execute('INSERT INTO "s"."b" SELECT * FROM "s"."b" LIMIT 20')    # duplicates
    yield con
    con.close()


@pytest.mark.parametrize("row_mode", ["unique_only", "any_row"])
def test_sampler_same_result_for_any_thread_count(small_db, row_mode):
    con = small_db
    schemas = ["s"]
    assets = {1: ("s", 'SELECT g, sum(v) AS sv FROM "s"."a" GROUP BY g'),
              2: ("s", 'SELECT DISTINCT g FROM "s"."b"'),
              3: ("s", 'SELECT a.g, count(*) AS n FROM "s"."a" a JOIN "s"."b" b ON a.g = b.g GROUP BY a.g')}
    readers = {("s", "a"): [1, 3], ("s", "b"): [2, 3]}
    fp = base_fingerprints(con, schemas, assets)
    targets = {("s", "a"): ("s", "a"), ("s", "b"): ("s", "b")}
    counts = {("s", "a"): 400, ("s", "b"): 170}
    tables = sorted(readers)
    res = []
    for th in sorted({1, THREADS}):
        smp = SupportSampler(con, schemas, assets, readers, fp, targets, counts, row_mode, th)
        res.append(smp.sample(tables, [1.0, 1.0], 120, 5, 6, record_fp=True))
    assert res[0].accepted == res[1].accepted
    assert res[0].incidence == res[1].incidence
    assert res[0].changed_fp == res[1].changed_fp
    assert (res[0].draws, res[0].skipped) == (res[1].draws, res[1].skipped)
    if row_mode == "unique_only":
        assert res[0].skipped > 0                       # duplicate rows are skipped
    assert con.execute('SELECT count(*) FROM "s"."b"').fetchone()[0] == 170   # rolled back
