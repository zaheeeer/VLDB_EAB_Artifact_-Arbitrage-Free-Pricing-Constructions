"""Pricing mechanisms and the F1-F3 experiment on the TPC-H substrate.

Every mechanism produces a non-negative score per asset. A single free scale is fitted on
a training buyer sample that pays posted prices, then frozen and scored on an independent
test sample that may recompose. No mechanism receives more tuning than another, so the
revenue column compares pricing rules rather than calibration effort.

Mechanisms
----------
uniform              flat fee
size_proportional    answer cardinality
support_set          base rows read, the weighted-coverage family
compute_metered      measured evaluation time
monopoly_per_asset   per-asset revenue-maximizing price, no arbitrage constraint
monotone_constrained monopoly prices raised to satisfy the verified determinacy order
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def buyer_sample(n, info, rng, gamma=0.7, sigma=0.6, scale=100.0, demand_skew=1.0):
    k = len(info)
    rank = rng.permutation(k) + 1
    demand = 1.0 / np.power(rank, demand_skew)
    demand = demand / demand.sum()
    target = rng.choice(k, size=n, p=demand)
    theta = rng.lognormal(0.0, sigma, size=n)
    value = scale * theta * np.power(info[target], gamma)
    return target, value, demand


def settle(target, value, cost_per_asset):
    c = cost_per_asset[target]
    buys = c <= value
    return {
        "revenue": float(np.sum(c[buys])),
        "surplus": float(np.sum(value[buys] - c[buys])),
        "served": float(np.mean(buys)),
    }


def fit_scale(target, value, scores, grid=None):
    """Revenue-maximizing multiplier on a score vector, fitted on one buyer sample."""
    if grid is None:
        grid = np.concatenate([np.linspace(0.01, 5, 200), np.linspace(5.2, 60, 140)])
    s = scores[target]
    best_a, best_r = grid[0], -1.0
    for a in grid:
        c = a * s
        r = float(np.sum(c[c <= value]))
        if r > best_r:
            best_a, best_r = a, r
    return float(best_a)


def monopoly_prices(target, value, k):
    """Per-asset revenue-maximizing posted price from the training sample."""
    p = np.zeros(k)
    fallback = float(np.median(value)) if len(value) else 1.0
    for j in range(k):
        v = np.sort(value[target == j])[::-1]
        if len(v) == 0:
            p[j] = fallback
            continue
        rev = v * (np.arange(len(v)) + 1)
        p[j] = float(v[int(np.argmax(rev))])
    return p


def enforce_monotone(p, pairs, iters=200):
    """Raise prices until every verified pair (i determines j) satisfies p_i >= p_j.

    Raising rather than lowering keeps the constraint satisfiable without driving prices
    to zero. The order is a DAG, so repeated relaxation converges.
    """
    q = p.astype(float).copy()
    for _ in range(iters):
        changed = False
        for i, j in pairs:
            if q[j] > q[i] + 1e-12:
                q[i] = q[j]
                changed = True
        if not changed:
            break
    return q


def info_families(rowsz, ssz, ncols):
    """Valuation families. Buyer value has no public ground truth, so the family is a
    measured axis and the reported result is whether the mechanism ranking is invariant
    across it."""
    def nz(x):
        x = np.asarray(x, dtype=float)
        return x / max(x.max(), 1e-12)
    return {
        "answer_cells": nz(rowsz * ncols),
        "support_rows": nz(ssz),
        "rows_x_log_support": nz(rowsz * np.log1p(ssz)),
        "flat": np.ones(len(rowsz)),
    }


def run_experiment(A, ans, pairs, scores_by_mech, info_map, TEMPLATES, match_fn,
                   plan_all_fn, seeds=(0, 1, 2, 3, 4), n_train=800, n_test=800):
    out = []
    for fam, info in info_map.items():
        for seed in seeds:
            rng = np.random.default_rng(1000 * seed + 7)
            tr_t, tr_v, _ = buyer_sample(n_train, info, rng)
            te_t, te_v, _ = buyer_sample(n_test, info, rng)
            k = len(A)
            price_sets = {}
            for name, sc in scores_by_mech.items():
                # Normalize before fitting so one multiplier grid conditions every
                # mechanism equally. Prices stay proportional to the raw score.
                s = np.asarray(sc, dtype=float)
                s = s / max(s.max(), 1e-12)
                a = fit_scale(tr_t, tr_v, s)
                price_sets[name] = a * s
            mono = monopoly_prices(tr_t, tr_v, k)
            price_sets["monopoly_per_asset"] = mono
            price_sets["monotone_constrained"] = enforce_monotone(mono, pairs)

            ref_train = settle(tr_t, tr_v, price_sets["monopoly_per_asset"])["revenue"] * (n_test / n_train)
            for name, p in price_sets.items():
                pl = plan_all_fn(A, ans, p, TEMPLATES, match_fn)
                cover = pl.cover_cost.to_numpy()
                eff = np.minimum(p, np.where(np.isfinite(cover), cover, p))
                nai = settle(te_t, te_v, p)
                stra = settle(te_t, te_v, eff)
                ins = settle(tr_t, tr_v, p)["revenue"] * (n_test / n_train)
                mono_a = audit_monotone(p, pairs)
                gain = pl.gain.to_numpy()
                exercised = pl.verified.notna()
                out.append({
                    "family": fam, "seed": seed, "mechanism": name,
                    "revenue_train": ins, "revenue_naive": nai["revenue"],
                    "revenue_strategic": stra["revenue"],
                    "revenue_train_norm": ins / max(ref_train, 1e-12),
                    "revenue_naive_norm": nai["revenue"] / max(ref_train, 1e-12),
                    "revenue_strategic_norm": stra["revenue"] / max(ref_train, 1e-12),
                    "generalization_gap": 1.0 - nai["revenue"] / max(ins, 1e-12),
                    "leakage": (nai["revenue"] - stra["revenue"]) / max(nai["revenue"], 1e-12),
                    "served_naive": nai["served"], "served_strategic": stra["served"],
                    "surplus_strategic": stra["surplus"],
                    "mono_violation_rate": mono_a["violation_rate"],
                    "mono_max_excess": mono_a["max_excess"],
                    "sep_violation_rate": float(np.mean(gain > 1e-6)),
                    "sep_mean_gain": float(np.mean(gain)),
                    "sep_max_gain": float(gain.max()) if len(gain) else 0.0,
                    "plans_checked": int(exercised.sum()),
                    "plans_failed": int((exercised & (pl.verified == False)).sum()),
                    "multi_source_plans": int((pl.plan_size >= 2).sum()),
                })
    return pd.DataFrame(out)


def audit_monotone(prices, pairs, tol=1e-9):
    if not pairs:
        return {"n_pairs": 0, "violation_rate": 0.0, "max_excess": 0.0}
    viol, worst = 0, 0.0
    for i, j in pairs:
        excess = prices[j] - prices[i]
        if excess > tol * max(prices[i], 1.0):
            viol += 1
            worst = max(worst, excess / max(prices[i], 1e-12))
    return {"n_pairs": len(pairs), "violation_rate": viol / len(pairs), "max_excess": worst}
