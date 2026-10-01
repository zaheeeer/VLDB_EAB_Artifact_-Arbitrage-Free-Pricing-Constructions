"""Feasibility probe: does the TPC-H template family yield a verified derivability lattice?

An asset is a view over one TPC-H template: a subset of that template's grouping keys,
a subset of the shipdate months the template filters on, and the template's measures.
Templates and their dimensions come from the TPC-H specification. The view family per
template is the object the view-based pricing formulation prices.

Derivability is verified, never declared. Each asset's answer is evaluated by its own SQL
against lineitem. A candidate pair is accepted only when re-aggregating the source answer
reproduces the target answer within tolerance.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

GROUP_DIMS = {
    "returnflag": "l_returnflag",
    "linestatus": "l_linestatus",
    "shipmonth": "date_trunc('month', l_shipdate)",
}

TEMPLATES = {
    # Q1: pricing summary report. Additive measures plus count; avg is recoverable.
    "q1": {
        "group_dims": ("returnflag", "linestatus", "shipmonth"),
        "measures": {
            "sum_qty": "sum(l_quantity)",
            "sum_base_price": "sum(l_extendedprice)",
            "sum_disc_price": "sum(l_extendedprice * (1 - l_discount))",
            "count_order": "count(*)",
        },
        "extra_where": "",
    },
    # Q6: forecasting revenue change. Single measure, spec discount and quantity filters.
    "q6": {
        "group_dims": ("shipmonth",),
        "measures": {"revenue": "sum(l_extendedprice * l_discount)"},
        "extra_where": "l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24",
    },
}


@dataclass(frozen=True)
class Asset:
    template: str
    group: frozenset
    months: frozenset

    @property
    def measures(self) -> tuple:
        return tuple(TEMPLATES[self.template]["measures"])

    def label(self) -> str:
        g = ",".join(sorted(self.group)) or "-"
        return f"{self.template}[g={g};m={len(self.months)}]"


def month_universe(con) -> list:
    rows = con.execute(
        "SELECT DISTINCT CAST(date_trunc('month', l_shipdate) AS DATE) AS m "
        "FROM lineitem ORDER BY 1"
    ).fetchall()
    return [r[0] for r in rows]


def asset_sql(a: Asset) -> str:
    spec = TEMPLATES[a.template]
    gcols = [g for g in spec["group_dims"] if g in a.group]
    sel = [f"{GROUP_DIMS[g]} AS {g}" for g in gcols]
    sel += [f"{expr} AS {name}" for name, expr in spec["measures"].items()]
    months = ", ".join(f"DATE '{m.isoformat()}'" for m in sorted(a.months))
    where = [f"date_trunc('month', l_shipdate) IN ({months})"]
    if spec["extra_where"]:
        where.append(spec["extra_where"])
    sql = f"SELECT {', '.join(sel)} FROM lineitem WHERE {' AND '.join(where)}"
    if gcols:
        sql += " GROUP BY " + ", ".join(GROUP_DIMS[g] for g in gcols)
    return sql


def evaluate(con, a: Asset) -> pd.DataFrame:
    df = con.execute(asset_sql(a)).fetchdf()
    gcols = [g for g in TEMPLATES[a.template]["group_dims"] if g in a.group]
    if gcols:
        df = df.sort_values(gcols).reset_index(drop=True)
    return df


def determines(src: Asset, tgt: Asset) -> bool:
    """Structural candidate condition for src to determine tgt.

    Roll-up requires the target's grouping keys to be a subset of the source's, and the
    measures to be additive, which holds for sum and count. Restricting the month set
    requires the month dimension to be visible in the source grouping, otherwise rows
    from different months cannot be told apart.
    """
    if src.template != tgt.template:
        return False
    if not tgt.group <= src.group:
        return False
    if not tgt.months <= src.months:
        return False
    if tgt.months != src.months and "shipmonth" not in src.group:
        return False
    return True


def derive(src_answer: pd.DataFrame, src: Asset, tgt: Asset) -> pd.DataFrame | None:
    """Re-aggregate the source answer into the target answer."""
    if not determines(src, tgt):
        return None
    df = src_answer
    if tgt.months != src.months:
        want = pd.to_datetime(sorted(tgt.months))
        keep = pd.to_datetime(df["shipmonth"]).isin(want)
        if not keep.any():
            return None
        df = df[keep.to_numpy()]
    gcols = [g for g in TEMPLATES[tgt.template]["group_dims"] if g in tgt.group]
    mcols = list(tgt.measures)
    if gcols:
        out = df.groupby(gcols, dropna=False, as_index=False)[mcols].sum()
        out = out.sort_values(gcols).reset_index(drop=True)
    else:
        out = pd.DataFrame([df[mcols].sum()])
    return out.loc[:, gcols + mcols]


def answers_match(a: pd.DataFrame, b: pd.DataFrame, rtol=1e-9, atol=1e-6) -> bool:
    if a is None or b is None or a.shape != b.shape:
        return False
    if list(a.columns) != list(b.columns):
        return False
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=rtol, atol=atol):
                return False
        else:
            if not a[c].astype(str).equals(b[c].astype(str)):
                return False
    return True


def build_assets(months: list, seed: int = 0, n_random: int = 24) -> list:
    rng = np.random.default_rng(seed)
    assets: list[Asset] = []
    all_m = frozenset(months)

    # Q1: every grouping subset at full range, plus per-year and per-quarter ranges.
    dims = TEMPLATES["q1"]["group_dims"]
    for k in range(len(dims) + 1):
        for g in itertools.combinations(dims, k):
            assets.append(Asset("q1", frozenset(g), all_m))
    years = sorted({m.year for m in months})
    for y in years:
        ym = frozenset(m for m in months if m.year == y)
        for g in [("returnflag", "linestatus", "shipmonth"), ("returnflag", "linestatus"), ()]:
            assets.append(Asset("q1", frozenset(g), ym))
    for y in years[:3]:
        for qtr in range(1, 5):
            qm = frozenset(m for m in months if m.year == y and (m.month - 1) // 3 + 1 == qtr)
            if qm:
                assets.append(Asset("q1", frozenset(("returnflag", "linestatus")), qm))

    # Q6: full range, per-year, per-month, with and without the month key exposed.
    assets.append(Asset("q6", frozenset(("shipmonth",)), all_m))
    assets.append(Asset("q6", frozenset(), all_m))
    for y in years:
        ym = frozenset(m for m in months if m.year == y)
        assets.append(Asset("q6", frozenset(("shipmonth",)), ym))
        assets.append(Asset("q6", frozenset(), ym))

    # Random month subsets, to break any regularity the curated sets introduce.
    for _ in range(n_random):
        tmpl = "q1" if rng.random() < 0.6 else "q6"
        k = int(rng.integers(1, 13))
        sub = frozenset(rng.choice(months, size=k, replace=False).tolist())
        gd = TEMPLATES[tmpl]["group_dims"]
        gk = int(rng.integers(0, len(gd) + 1))
        g = frozenset(rng.choice(gd, size=gk, replace=False).tolist()) if gk else frozenset()
        assets.append(Asset(tmpl, g, sub))

    seen, out = set(), []
    for a in assets:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def build_assets_n(months: list, n: int, seed: int = 0) -> list:
    """Asset family of target size n: a lattice spine plus random filler.

    The spine is the view family each template's own dimensions generate: every grouping
    subset at full range, then per-year, per-quarter and per-month ranges at the full and
    at the coarsest grouping. Filler adds arbitrary month subsets so the measured lattice
    is not an artifact of the spine's regularity.
    """
    rng = np.random.default_rng(seed)
    all_m = frozenset(months)
    years = sorted({m.year for m in months})
    out, seen = [], set()

    def add(t, g, ms):
        x = Asset(t, frozenset(g), frozenset(ms))
        if x not in seen and x.months:
            seen.add(x)
            out.append(x)

    for tmpl, spec in TEMPLATES.items():
        dims = spec["group_dims"]
        full = dims
        coarse = tuple(d for d in dims if d != "shipmonth")
        for k in range(len(dims) + 1):
            for g in itertools.combinations(dims, k):
                add(tmpl, g, all_m)
        for y in years:
            ym = [m for m in months if m.year == y]
            add(tmpl, full, ym)
            add(tmpl, coarse, ym)
        for y in years:
            for q in range(1, 5):
                qm = [m for m in months if m.year == y and (m.month - 1) // 3 + 1 == q]
                add(tmpl, full, qm)
                add(tmpl, coarse, qm)
        for m in months:
            add(tmpl, coarse, [m])

    spine = len(out)
    tmpls = list(TEMPLATES)
    guard = 0
    while len(out) < n and guard < 200 * n:
        guard += 1
        t = tmpls[int(rng.integers(0, len(tmpls)))]
        dims = TEMPLATES[t]["group_dims"]
        k = int(rng.integers(1, min(13, len(months)) + 1))
        ms = rng.choice(months, size=k, replace=False).tolist()
        gk = int(rng.integers(0, len(dims) + 1))
        g = rng.choice(dims, size=gk, replace=False).tolist() if gk else []
        add(t, g, ms)
    return out[:n], min(spine, n)


def run_at(con, n: int, seed: int = 0, months=None):
    months = months if months is not None else month_universe(con)
    assets, spine = build_assets_n(months, n, seed)

    t0 = time.perf_counter()
    answers = {x: evaluate(con, x) for x in assets}
    t_eval = time.perf_counter() - t0

    t0 = time.perf_counter()
    cands = [
        (i, j)
        for i, si in enumerate(assets)
        for j, sj in enumerate(assets)
        if i != j and determines(si, sj)
    ]
    t_cand = time.perf_counter() - t0

    t0 = time.perf_counter()
    verified, rejected = [], []
    for i, j in cands:
        got = derive(answers[assets[i]], assets[i], assets[j])
        (verified if answers_match(got, answers[assets[j]]) else rejected).append((i, j))
    t_verify = time.perf_counter() - t0

    return {
        "n_assets": len(assets),
        "n_spine": spine,
        "n_candidates": len(cands),
        "n_verified": len(verified),
        "n_rejected": len(rejected),
        "precision": len(verified) / max(len(cands), 1),
        "pairs_per_asset": len(verified) / max(len(assets), 1),
        "t_eval_s": t_eval,
        "t_candidate_s": t_cand,
        "t_verify_s": t_verify,
        "ms_per_asset": 1000 * t_eval / max(len(assets), 1),
        "ms_per_pair": 1000 * t_verify / max(len(cands), 1),
    }


def run(con):
    months = month_universe(con)
    assets = build_assets(months)

    t0 = time.perf_counter()
    answers = {}
    for a in assets:
        answers[a] = evaluate(con, a)
    t_eval = time.perf_counter() - t0

    cands = [
        (i, j)
        for i, si in enumerate(assets)
        for j, sj in enumerate(assets)
        if i != j and determines(si, sj)
    ]
    t0 = time.perf_counter()
    verified, rejected = [], []
    for i, j in cands:
        got = derive(answers[assets[i]], assets[i], assets[j])
        (verified if answers_match(got, answers[assets[j]]) else rejected).append((i, j))
    t_verify = time.perf_counter() - t0

    return {
        "assets": assets,
        "answers": answers,
        "months": months,
        "candidates": cands,
        "verified": verified,
        "rejected": rejected,
        "t_eval": t_eval,
        "t_verify": t_verify,
    }
