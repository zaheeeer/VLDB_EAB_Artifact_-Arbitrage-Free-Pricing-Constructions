"""The recomposing adversary and the fit, freeze, and evaluate runs.

``plan_all`` is the historical ``sqlshare_plan_all``: for every set-semantics target it
solves a min-cost row cover and accepts the plan only if it undercuts the posted price
and its re-executed union equals the target answer. A cover can undercut the price only
if every source it uses is itself cheaper than the target, so targets whose cheaper
sources cannot cover all rows are skipped without an ILP. This is exact: it changes no
outcome, only the number of solver calls.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .lattice import frames_equal
from .pricing import (audit_monotone, chawla_all, draw_buyers, enforce_monotone,
                      monopoly_prices, scaled_prices, score)
from .parallel import run_pool
from .util import load_pickle

INF = float("inf")


def min_cost_row_cover(contrib: dict, n_rows: int, prices: dict, time_limit=5.0):
    from scipy.optimize import Bounds, LinearConstraint, milp
    if not contrib or n_rows == 0:
        return INF, []
    src = sorted(contrib)
    A = np.zeros((n_rows, len(src)))
    for c, s in enumerate(src):
        A[contrib[s], c] = 1.0
    if not np.all(A.sum(axis=1) > 0):
        return INF, []
    cost = np.array([prices[s] for s in src], dtype=float)
    if not np.all(np.isfinite(cost)):
        keep = np.isfinite(cost)
        if not keep.any():
            return INF, []
        A, cost, src = A[:, keep], cost[keep], [s for s, k in zip(src, keep) if k]
        if not np.all(A.sum(axis=1) > 0):
            return INF, []
    res = milp(c=cost, constraints=[LinearConstraint(A, lb=np.ones(n_rows), ub=np.inf)],
               integrality=np.ones(len(src)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if not res.success or res.x is None:
        return INF, []
    chosen = [src[k] for k in range(len(src)) if res.x[k] > 0.5]
    return float(res.fun), chosen


def verify_row_cover(answers, target, plan) -> bool:
    if not plan:
        return False
    T = answers[target]
    cols = list(T.columns)
    got = pd.concat([answers[s].loc[:, cols] for s in plan], ignore_index=True).drop_duplicates()
    return frames_equal(got.reset_index(drop=True), T.reset_index(drop=True))


def plan_all(p, keys, ki, contrib, nrows, answers, time_limit=5.0, exact_formulation=True):
    """Effective prices under the recomposing adversary.

    Returns (cost vector, plan sizes, plans verified, plans failed). ``exact_formulation``
    solves the ILP over all candidate sources, as the historical run did, whenever a
    cheaper cover is possible; otherwise only over the cheaper sources (same optimum).
    """
    cost = np.asarray(p, float).copy()
    psize = np.ones(len(keys), int)
    vok = vbad = 0
    pm = {k: float(p[ki[k]]) for k in keys}
    for k, c_ in contrib.items():
        pk = pm[k]
        cheap = {s: idx for s, idx in c_.items() if pm[s] < pk - 1e-9}
        if not cheap:
            continue
        covered = np.zeros(nrows[k], bool)
        for idx in cheap.values():
            covered[idx] = True
        if not covered.all():
            continue
        cc_, plan = min_cost_row_cover(c_ if exact_formulation else cheap, nrows[k], pm, time_limit)
        if plan and cc_ < pk - 1e-9:
            try:
                ok = verify_row_cover(answers, k, plan)
            except Exception:
                ok = False
            if ok:
                cost[ki[k]] = cc_
                psize[ki[k]] = len(plan)
                vok += 1
            else:
                vbad += 1
    return cost, psize, vok, vbad


# ----------------------------------------------------------------------------------------
# one (arm, family, seed) run of the full protocol
# ----------------------------------------------------------------------------------------
_MARKET = {}


def _market(path):
    if path not in _MARKET:
        _MARKET.clear()
        _MARKET[path] = load_pickle(Path(path))
    return _MARKET[path]


def run_task(args):
    path, arm, family, seed = args
    d = _market(path)
    cfg = d["cfg"]
    keys, ki, n = d["keys"], d["ki"], len(d["keys"])
    info = d["info"][family]
    (tr_t, tr_v), (te_t, te_v) = draw_buyers(cfg["demand"], arm, info, 1000 * seed + 7,
                                             cfg["n_buyers"], cfg["n_buyers"])
    prices, alphas = {}, {}
    for name, sc in d["scale_scores"].items():
        prices[name], alphas[name] = scaled_prices(cfg["fit"], tr_t, tr_v, sc)
    mono = monopoly_prices(tr_t, tr_v, n)
    prices["monopoly_per_asset"] = mono
    prices["monotone_constrained"] = enforce_monotone(mono, d["pairs"])
    prices.update(chawla_all(d["edges"], tr_t, tr_v, d["support_n"], max_lps=cfg["max_lps"],
                             eps=cfg["cip_eps"]))
    ref = score(tr_t, tr_v, prices["monopoly_per_asset"])["revenue"]
    rows = []
    for name, p in prices.items():
        cost, psz, vok, vbad = plan_all(p, keys, ki, d["contrib"], d["nrows"], d["answers"],
                                        cfg["ilp_time_limit"], cfg["exact_formulation"])
        eff = np.minimum(p, cost)
        nai = score(te_t, te_v, p)
        stra = score(te_t, te_v, eff)
        ins = score(tr_t, tr_v, p)["revenue"]
        ma = audit_monotone(p, d["pairs"])
        with np.errstate(invalid="ignore", divide="ignore"):
            gain = np.where(np.isinf(p), np.where(np.isfinite(eff), 1.0, 0.0),
                            np.where(p > 0, (p - eff) / np.maximum(p, 1e-12), 0.0))
        a = alphas.get(name)
        rows.append({
            "arm": arm, "family": family, "seed": seed, "mechanism": name,
            "revenue_train_norm": ins / max(ref, 1e-12),
            "revenue_naive_norm": nai["revenue"] / max(ref, 1e-12),
            "revenue_strategic_norm": stra["revenue"] / max(ref, 1e-12),
            "surplus_naive_norm": nai["surplus"] / max(ref, 1e-12),
            "welfare_naive_norm": nai["welfare"] / max(ref, 1e-12),
            "surplus_strategic_norm": stra["surplus"] / max(ref, 1e-12),
            "welfare_strategic_norm": stra["welfare"] / max(ref, 1e-12),
            "generalization_gap": 1 - nai["revenue"] / max(ins, 1e-12),
            "mono_violation_rate": ma["violation_rate"],
            "comb_violation_rate": float(np.mean(gain > 1e-6)),
            "comb_mean_gain": float(np.mean(gain)),
            "comb_max_gain": float(gain.max()),
            "served_naive": nai["served"], "served_strategic": stra["served"],
            "plans_verified": int(vok), "plans_failed": int(vbad),
            "alpha": a if a is not None else np.nan,
            "alpha_at_grid_max": bool(a is not None and cfg["fit"] == "grid60" and a >= 60.0 - 1e-9),
            "unsellable_assets": int(np.isinf(p).sum()),
        })
    return rows


def run_protocol(market_pickle: Path, arms, families, seeds, workers: int, stall_s: float = 1800.0):
    tasks = [(str(market_pickle), a, f, s) for a in arms for f in families for s in seeds]
    try:
        out = run_pool(run_task, tasks, workers, "fit, freeze, and evaluate runs", stall_s)
    finally:
        _MARKET.clear()
    return pd.DataFrame([r for rows in out for r in rows])
