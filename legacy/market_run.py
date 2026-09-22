"""Per-market pricing run, so every finding can be reported with a spread across markets.

A single market is an anecdote. This module runs the reduced protocol on each connected
component of the query-to-table graph: revenue, welfare, coverage and the
information-arbitrage audit, without the min-cost cover planner, which dominates runtime
and is not needed to test whether a finding holds across markets.

Support sets are sampled with probability proportional to how many assets read a table,
not to table size. Section 7 reports what that choice is worth and what it costs.
"""
from __future__ import annotations

import collections

import numpy as np


def run_market(c, ctx, support_n=300, n_buyers=400, seeds=(0, 1, 2), max_lps=25,
               families=("support_rows", "answer_cells"), min_assets=8):
    """Return a list of result rows for one market, or a single status row."""
    (E2ok, ROW_CAP, db2, norm_frame, component_lattice, q2t, ent, fp_sql, delete_one,
     qirana_from_edges, info_families, buyer_sample, fit_scale, monopoly_prices,
     audit_monotone, chawla_prices, score) = ctx

    sub = E2ok[(E2ok.comp == c) & (E2ok.rows <= ROW_CAP)]
    answers, paths, ducks = {}, {}, {}
    for r in sub.itertuples():
        try:
            db2.execute(f"SET search_path='{r.own}'")
            answers[r.k] = norm_frame(db2.execute(r.duck).df())
            paths[r.k], ducks[r.k] = r.own, r.duck
        except Exception:
            pass
    db2.execute("SET search_path='main'")
    keys = sorted(answers)
    if len(keys) < min_assets:
        return [{"market": str(c), "assets": len(keys), "status": "too_small"}]

    lat = component_lattice(answers)
    idx = {k: i for i, k in enumerate(keys)}
    pairs = [(idx[s], idx[t]) for s, t, _fam in lat["verified"] if s in idx and t in idx]

    readers = collections.defaultdict(list)
    for k in keys:
        for b in q2t.get(k, ()):
            readers[b].append(k)
    tables = sorted(readers)
    if not tables:
        return [{"market": str(c), "assets": len(keys), "status": "no_base_tables"}]
    nread = np.array([len(readers[b]) for b in tables], float)
    weights = nread / nread.sum()

    base = {}
    for k in keys:
        db2.execute(f"SET search_path='{paths[k]}'")
        try:
            base[k] = db2.execute(fp_sql(ducks[k])).fetchone()
        except Exception:
            base[k] = None
    db2.execute("SET search_path='main'")

    incidence = {k: set() for k in keys}
    rg0 = np.random.default_rng(4242)
    used, guard = 0, 0
    while used < support_n and guard < support_n * 20:
        guard += 1
        b = tables[int(rg0.choice(len(tables), p=weights))]
        db2.execute("BEGIN TRANSACTION")
        try:
            ok = delete_one(db2, b, rg0)
        except Exception:
            ok = False
        if not ok:
            db2.execute("ROLLBACK")
            continue
        used += 1
        for k in readers.get(b, ()):
            if base[k] is None:
                continue
            db2.execute(f"SET search_path='{paths[k]}'")
            try:
                if db2.execute(fp_sql(ducks[k])).fetchone() != base[k]:
                    incidence[k].add(used - 1)
            except Exception:
                pass
        db2.execute("ROLLBACK")
    db2.execute("SET search_path='main'")

    edges = [frozenset(incidence[k]) for k in keys]
    sizes = np.array([len(e) for e in edges], float)
    rowsz = np.array([len(answers[k]) for k in keys], float)
    qs = qirana_from_edges(edges, support_n)

    derivable_from = collections.defaultdict(list)
    for s, t in pairs:
        derivable_from[t].append(s)
    catalogue = [j for j in range(len(keys)) if j not in derivable_from]
    catp = {j: float(rowsz[j]) for j in catalogue}
    big = max(catp.values()) * 1.5 if catp else 1.0
    qm = np.array([catp.get(j, min([catp.get(s, big) for s in derivable_from.get(j, [])] or [big]))
                   for j in range(len(keys))])

    falsified = sum(1 for s, t in pairs if sizes[t] > sizes[s] + 1e-9)
    info = info_families(rowsz, rowsz, np.ones(len(keys)))
    out = []
    for fam in families:
        for sd in seeds:
            rg = np.random.default_rng(100 * sd + 3)
            tt, tv, _ = buyer_sample(n_buyers, info[fam], rg)
            ht, hv, _ = buyer_sample(n_buyers, info[fam], rg)
            prices = {"uniform": fit_scale(tt, tv, np.ones(len(keys))) * np.ones(len(keys))}
            for nm, sc in (("qirana_weighted_coverage", qs["qirana_weighted_coverage"]),
                           ("querymarket_viewcover", qm)):
                s_ = sc / max(sc.max(), 1e-12)
                prices[nm] = fit_scale(tt, tv, s_) * s_
            prices["monopoly_per_asset"] = monopoly_prices(tt, tv, len(keys))
            prices["chawla_ubp"] = chawla_prices("chawla_ubp", edges, tt, tv, support_n)
            prices["chawla_lpip"] = chawla_prices("chawla_lpip", edges, tt, tv, support_n, max_lps=max_lps)
            ref = score(tt, tv, prices["monopoly_per_asset"])["revenue"]
            for nm, p in prices.items():
                sc_ = score(ht, hv, p)
                aud = audit_monotone(p, pairs)
                out.append({"market": str(c), "assets": len(keys), "pairs": len(pairs),
                            "falsified": falsified, "zero_priced": int((sizes == 0).sum()),
                            "family": fam, "seed": sd, "mechanism": nm,
                            "revenue_norm": sc_["revenue"] / max(ref, 1e-12),
                            "welfare_norm": sc_["welfare"] / max(ref, 1e-12),
                            "served": sc_["served"],
                            "mono_violation_rate": aud["violation_rate"], "status": "ok"})
    return out
