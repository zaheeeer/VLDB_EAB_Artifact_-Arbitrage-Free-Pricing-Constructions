"""Full comparison: three published systems, baselines, welfare, arbitrage audits.

Every mechanism produces one non-negative price per asset and is then scored by the same
protocol: fit on a training buyer sample that pays posted prices, freeze, evaluate on an
independent test sample that may recompose. Revenue, buyer surplus, welfare and coverage
are recorded together, because a revenue-only table cannot distinguish a mechanism that
earns less from one that transfers the same value to buyers instead.

Scale-fitted mechanisms get one multiplier tuned on the training sample. The
Chawla et al. algorithms tune themselves on the training valuations, which is their
design, so no extra multiplier is applied to them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score(targets, values, prices):
    """Revenue, surplus, welfare and coverage for one posted-price vector."""
    c = np.asarray(prices, float)[targets]
    v = np.asarray(values, float)
    buys = c <= v
    rev = float(c[buys].sum())
    sur = float((v[buys] - c[buys]).sum())
    return {"revenue": rev, "surplus": sur, "welfare": rev + sur,
            "served": float(buys.mean())}


def chawla_prices(alg, edges_asset, tr_t, tr_v, n_items, **kw):
    """Run one revenue-maximization algorithm and lift it to a per-asset price vector."""
    import ports_chawla as C
    edges_b = [edges_asset[t] for t in tr_t]
    vals_b = np.asarray(tr_v, float)
    fn = C.ALGORITHMS[alg]
    if alg == "chawla_ubp":
        _, info = fn(edges_b, vals_b)
        return np.full(len(edges_asset), info["P"])
    if alg == "chawla_xos":
        _, info = fn(edges_b, vals_b, n_items, **kw)
        w1 = info["lpip"]["weights"]
        w2 = info["cip"]["weights"]
        return np.array([max(float(w1[list(e)].sum()) if len(e) else 0.0,
                             float(w2[list(e)].sum()) if len(e) else 0.0) for e in edges_asset])
    _, info = fn(edges_b, vals_b, n_items, **kw)
    w = info["weights"]
    return np.array([float(w[list(e)].sum()) if len(e) else 0.0 for e in edges_asset])


def run(assets, answers, pairs, scale_scores, edges_asset, support_n, info_map,
        plan_fn, audit_fn, fit_scale, buyer_sample, monopoly_prices, enforce_monotone,
        seeds=(0, 1, 2, 3, 4), n_buyers=800, chawla=("chawla_ubp", "chawla_uip", "chawla_lpip",
                                                     "chawla_cip", "chawla_layering", "chawla_xos"),
        max_lps=60):
    out = []
    n = len(assets)
    for fam, info in info_map.items():
        for seed in seeds:
            rng = np.random.default_rng(1000 * seed + 7)
            tr_t, tr_v, _ = buyer_sample(n_buyers, info, rng)
            te_t, te_v, _ = buyer_sample(n_buyers, info, rng)

            prices = {}
            for name, sc in scale_scores.items():
                s = np.asarray(sc, float)
                s = s / max(s.max(), 1e-12)
                prices[name] = fit_scale(tr_t, tr_v, s) * s
            mono = monopoly_prices(tr_t, tr_v, n)
            prices["monopoly_per_asset"] = mono
            prices["monotone_constrained"] = enforce_monotone(mono, pairs)
            for alg in chawla:
                kw = {"max_lps": max_lps} if alg in ("chawla_lpip", "chawla_xos") else {}
                prices[alg] = chawla_prices(alg, edges_asset, tr_t, tr_v, support_n, **kw)

            ref = score(tr_t, tr_v, prices["monopoly_per_asset"])["revenue"]
            for name, p in prices.items():
                eff, psize, vok, vbad = plan_fn(p)
                nai = score(te_t, te_v, p)
                stra = score(te_t, te_v, eff)
                ins = score(tr_t, tr_v, p)["revenue"]
                ma = audit_fn(p, pairs)
                gain = np.where(p > 0, (p - eff) / np.maximum(p, 1e-12), 0.0)
                out.append({
                    "family": fam, "seed": seed, "mechanism": name,
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
                })
    return pd.DataFrame(out)
