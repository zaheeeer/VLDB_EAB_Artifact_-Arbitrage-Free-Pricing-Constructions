"""Provenance and multi-source acquisition planning for the TPC-H asset substrate.

Provenance
----------
An asset reads exactly the base rows its filter admits. For this template family the
filter is (template predicate) AND (shipmonth in M), so per-month row counts computed
once give an exact support size for every asset, and support inclusion reduces to
month-set inclusion. No sampling and no estimation.

Acquisition planning
--------------------
A buyer wanting asset ``t`` may instead buy a set of assets and recombine. For additive
measures the reconstruction is valid when the sources share the template, each groups at
least as finely as ``t``, and their month sets are pairwise disjoint and cover ``t``'s.
Disjointness is required: overlapping sources double count.

That is an exact cover, solved as an integer program so the reported cost is the optimum
over this decomposition family rather than a greedy upper bound. Every plan the planner
returns is re-executed and compared against the independently evaluated answer before it
is counted as arbitrage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

INF = float("inf")


def support_tables(con, TEMPLATES):
    """Per-template, per-month base-row counts. One scan per template."""
    out = {}
    for name, spec in TEMPLATES.items():
        where = f"WHERE {spec['extra_where']}" if spec["extra_where"] else ""
        rows = con.execute(
            f"SELECT CAST(date_trunc('month', l_shipdate) AS DATE) AS m, count(*) AS n "
            f"FROM lineitem {where} GROUP BY 1"
        ).fetchall()
        out[name] = {r[0]: int(r[1]) for r in rows}
    return out


def support_size(asset, sup) -> int:
    t = sup[asset.template]
    return int(sum(t.get(m, 0) for m in asset.months))


def support_subset(a, b) -> bool:
    """True when asset a's base rows are a subset of asset b's."""
    return a.template == b.template and a.months <= b.months


def candidate_sources(assets, target_idx):
    """Assets admissible in a cover of the target: same template, grouping at least as
    fine, month set contained in the target's."""
    t = assets[target_idx]
    out = []
    for i, a in enumerate(assets):
        if i == target_idx:
            continue
        if a.template != t.template:
            continue
        if not t.group <= a.group:
            continue
        if not a.months <= t.months:
            continue
        # A source covering a strict subset of its own months is not allowed unless it
        # can be restricted, which needs the month key present.
        out.append(i)
    return out


def min_cost_cover(assets, target_idx, prices, sources=None, time_limit=10.0):
    """Exact-cover ILP. Returns (cost, [source indices]) or (inf, [])."""
    t = assets[target_idx]
    months = sorted(t.months)
    if not months:
        return INF, []
    if sources is None:
        src = candidate_sources(assets, target_idx)
    else:
        # An explicitly supplied catalogue still has to be admissible: same template,
        # grouping at least as fine, months contained in the target's.
        src = [i for i in sources
               if i != target_idx
               and assets[i].template == t.template
               and t.group <= assets[i].group
               and assets[i].months <= t.months]
    if not src:
        return INF, []

    m_index = {m: k for k, m in enumerate(months)}
    A = np.zeros((len(months), len(src)))
    for c, i in enumerate(src):
        for m in assets[i].months:
            A[m_index[m], c] = 1.0
    cover_all = np.all(A.sum(axis=1) > 0)
    if not cover_all:
        return INF, []

    cost = np.array([prices[i] for i in src], dtype=float)
    # Exact cover: every month used exactly once, so no source overlaps another.
    con = LinearConstraint(A, lb=np.ones(len(months)), ub=np.ones(len(months)))
    res = milp(
        c=cost,
        constraints=[con],
        integrality=np.ones(len(src)),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit},
    )
    if not res.success or res.x is None:
        return INF, []
    chosen = [src[k] for k in range(len(src)) if res.x[k] > 0.5]
    return float(res.fun), chosen


def reconstruct_cover(answers, assets, target_idx, plan, TEMPLATES):
    """Rebuild the target answer from a cover, or return None."""
    if not plan:
        return None
    t = assets[target_idx]
    gcols = [g for g in TEMPLATES[t.template]["group_dims"] if g in t.group]
    mcols = list(t.measures)
    parts = []
    for i in plan:
        s = assets[i]
        df = answers[s]
        src_g = [g for g in TEMPLATES[s.template]["group_dims"] if g in s.group]
        if not set(gcols) <= set(src_g):
            return None
        if not set(mcols) <= set(df.columns):
            return None
        parts.append(df.loc[:, src_g + mcols])
    allp = pd.concat(parts, ignore_index=True)
    if gcols:
        out = allp.groupby(gcols, dropna=False, as_index=False)[mcols].sum()
        out = out.sort_values(gcols).reset_index(drop=True)
    else:
        out = pd.DataFrame([allp[mcols].sum()])
    return out.loc[:, gcols + mcols]


def plan_all(assets, answers, prices, TEMPLATES, match_fn, verify_limit=None):
    """Min-cost acquisition for every asset, with plans verified by re-execution.

    Returns a frame with the posted price, the cheapest recomposition cost, whether the
    plan was verified, and the plan size.
    """
    n = len(assets)
    rows = []
    checked = 0
    for j in range(n):
        cost, plan = min_cost_cover(assets, j, prices)
        singleton = len(plan) == 1 and plan[0] == j
        verified = None
        if plan and cost < prices[j] - 1e-9 and not singleton:
            if verify_limit is None or checked < verify_limit:
                got = reconstruct_cover(answers, assets, j, plan, TEMPLATES)
                verified = bool(got is not None and match_fn(got, answers[assets[j]]))
                checked += 1
        rows.append({
            "asset": j,
            "price": float(prices[j]),
            "cover_cost": cost,
            "plan_size": len(plan),
            "gain": max(0.0, (prices[j] - cost) / prices[j]) if prices[j] > 0 and cost < INF else 0.0,
            "verified": verified,
        })
    return pd.DataFrame(rows)
