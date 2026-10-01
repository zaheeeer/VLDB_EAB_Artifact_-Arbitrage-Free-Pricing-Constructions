"""Measure the derivability lattice that occurs naturally in the SQLShare workload.

On TPC the assets are a generated view family, so a lattice exists by construction. Here
the assets are the queries real users actually wrote, so the lattice either exists or it
does not. That is the point of the measurement.

Three rewrite families are attempted, each sound and each verified by comparing the
reconstructed answer against the independently evaluated one:

projection   the target is a duplicate-eliminated projection of the source;
rollup       the target sums the source's numeric columns over a coarser grouping;
selection    the target is the source restricted to a value set on one projected column.

A pair is recorded as derivable only when a reconstruction reproduces the target exactly,
so the count is a lower bound on true derivability, never an overstatement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROW_CAP = 50_000


def norm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def frames_equal(a, b, rtol=1e-9) -> bool:
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    if list(a.columns) != list(b.columns):
        return False
    a2 = a.sort_values(list(a.columns), kind="mergesort").reset_index(drop=True)
    b2 = b.sort_values(list(b.columns), kind="mergesort").reset_index(drop=True)
    for c in a2.columns:
        x, y = a2[c], b2[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.to_numpy(dtype=float), y.to_numpy(dtype=float),
                               rtol=rtol, atol=0, equal_nan=True):
                return False
        else:
            if not x.astype(str).equals(y.astype(str)):
                return False
    return True


def try_projection(S, T):
    cols = list(T.columns)
    return S.loc[:, cols].drop_duplicates().reset_index(drop=True)


def try_rollup(S, T):
    cols = list(T.columns)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(T[c])]
    key = [c for c in cols if c not in num]
    if not num or not key:
        return None
    out = S.groupby(key, dropna=False, as_index=False)[num].sum()
    return out.loc[:, cols].reset_index(drop=True)


def try_selection(S, T):
    """Restrict the source to the target's value set on one projected column."""
    cols = list(T.columns)
    sub = S.loc[:, cols]
    for c in cols:
        vals = set(T[c].dropna().unique())
        if not vals or len(vals) >= sub[c].nunique(dropna=True):
            continue
        cand = sub[sub[c].isin(vals)].drop_duplicates().reset_index(drop=True)
        if cand.shape[0] == T.shape[0]:
            return cand
    return None


ATTEMPTS = (("projection", try_projection), ("rollup", try_rollup), ("selection", try_selection))


def derive_pair(S, T):
    """Return the rewrite family that reproduces T from S, or None."""
    if not set(T.columns) <= set(S.columns):
        return None
    if len(T) > len(S):
        return None
    for name, fn in ATTEMPTS:
        try:
            got = fn(S, T)
        except Exception:
            continue
        if got is None:
            continue
        try:
            if frames_equal(got, T.reset_index(drop=True)):
                return name
        except Exception:
            continue
    return None


def row_keys(df: pd.DataFrame) -> set:
    """Hashable row identities, so subset and union tests are exact set operations."""
    return set(map(tuple, df.astype(str).itertuples(index=False, name=None)))


def partial_sources(answers: dict, target, cache: dict, target_cap=2000):
    """Sources whose projection onto the target's columns yields a subset of its rows.

    A source contributing every row is the single-source case; one contributing a strict
    subset can only help as part of a union, which is what makes a cover multi-source.
    """
    T = answers[target]
    if len(T) > target_cap:
        return None, None
    cols = tuple(T.columns)
    tkeys = row_keys(T)
    # Any stable index works; row keys mix types across columns so they are not ordered.
    idx = {k: i for i, k in enumerate(tkeys)}
    out = {}
    for s, S in answers.items():
        if s == target or not set(cols) <= set(S.columns):
            continue
        ck = (s, cols)
        if ck not in cache:
            cache[ck] = row_keys(S.loc[:, list(cols)].drop_duplicates())
        sk = cache[ck]
        if sk and sk <= tkeys:
            out[s] = np.array(sorted(idx[k] for k in sk), dtype=int)
    return out, len(tkeys)


def min_cost_row_cover(contrib: dict, n_rows: int, prices: dict, time_limit=5.0):
    """Minimum-cost set cover of the target's rows. Returns (cost, sources)."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    if not contrib or n_rows == 0:
        return float("inf"), []
    src = sorted(contrib)
    A = np.zeros((n_rows, len(src)))
    for c, s in enumerate(src):
        A[contrib[s], c] = 1.0
    if not np.all(A.sum(axis=1) > 0):
        return float("inf"), []
    cost = np.array([prices[s] for s in src], dtype=float)
    res = milp(c=cost, constraints=[LinearConstraint(A, lb=np.ones(n_rows), ub=np.inf)],
               integrality=np.ones(len(src)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if not res.success or res.x is None:
        return float("inf"), []
    chosen = [src[k] for k in range(len(src)) if res.x[k] > 0.5]
    return float(res.fun), chosen


def verify_row_cover(answers, target, plan) -> bool:
    """Re-execute the union and compare against the independently evaluated answer."""
    if not plan:
        return False
    T = answers[target]
    cols = list(T.columns)
    got = pd.concat([answers[s].loc[:, cols] for s in plan], ignore_index=True).drop_duplicates()
    return frames_equal(got.reset_index(drop=True), T.reset_index(drop=True))


def component_lattice(answers: dict):
    """All verified derivable pairs among a component's materialized answers."""
    keys = sorted(answers)
    verified, by_family = [], {}
    attempted = 0
    for i in keys:
        S = answers[i]
        for j in keys:
            if i == j:
                continue
            T = answers[j]
            if not set(T.columns) <= set(S.columns) or len(T) > len(S):
                continue
            attempted += 1
            fam = derive_pair(S, T)
            if fam:
                verified.append((i, j, fam))
                by_family[fam] = by_family.get(fam, 0) + 1
    return {"verified": verified, "by_family": by_family, "attempted": attempted,
            "n_assets": len(keys)}
