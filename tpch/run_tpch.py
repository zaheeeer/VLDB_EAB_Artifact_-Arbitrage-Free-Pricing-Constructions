"""TPC-H campaign: 368 assets, 1,046 verified pairs, 15 mechanisms, 4 valuation
families x 5 seeds (300 runs). Writes the per-run records behind the TPC-H columns of
Tables 4, 5 and 6 of the paper.

    python tpch/run_tpch.py                 # writes tpch/out/full_tpch_runs.csv
    python tpch/run_tpch.py --out somewhere

The steps below are the code of the original TPC-H run, collected in order into one
script. The modules in tpch/modules are executed into this script's namespace, as in
the original run, and the calls and arguments are unchanged:
  1. generate TPC-H at scale factor 1 with DuckDB's tpch extension;
  2. build the 368-asset catalog (Q1 and Q6 by month) and the 1,046 verified pairs;
  3. time each asset query once (the score of the compute-metered price);
  4. sample the 400-instance support set and the conflict sets (hyperedges);
  5. price, fit, freeze and evaluate every mechanism for every family and seed, with
     the acquisition planner re-executing at most eight cheaper plans per mechanism
     and run (verify_limit=8; Section 3.5 of the paper).

The published records are in results/tpch/full_tpch_runs.csv. Compute-metered prices
come from measured query times, so that mechanism's rows depend on the machine.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = os.path.join(HERE, "modules")


def load(name: str) -> None:
    """Execute one module into this script's globals, as the original run did."""
    path = os.path.join(MODULES, name)
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), globals())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "out"), help="output folder")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, MODULES)  # run_full.py imports ports_chawla as a module
    g = globals()

    # 1-2. Data, catalog and verified pairs
    t_start = time.perf_counter()
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute("CALL dbgen(sf=1)")
    for f in ("probe_tpch_lattice.py", "tpc_substrate.py", "tpc_experiment.py", "ports.py"):
        load(f)
    TEMPLATES = g["TEMPLATES"]
    TEMPLATES["q1"]["py_pred"] = None
    TEMPLATES["q6"]["py_pred"] = lambda qty, disc: (0.05 <= float(disc) <= 0.07) and (float(qty) < 24)
    months = month_universe(con)                                    # noqa: F821
    A, spine = build_assets_n(months, 200, seed=0)                  # noqa: F821
    views_new = [Asset(t, frozenset(s["group_dims"]), frozenset({m}))  # noqa: F821
                 for t, s in TEMPLATES.items() for m in months]
    A2 = list(A) + [v for v in views_new if v not in set(A)]
    ans = {a: evaluate(con, a) for a in A2}                         # noqa: F821
    sup = support_tables(con, TEMPLATES)                            # noqa: F821
    ssz2 = np.array([support_size(a, sup) for a in A2], float)      # noqa: F821
    rowsz2 = np.array([len(ans[a]) for a in A2], float)
    pairs2 = [(i, j) for i in range(len(A2)) for j in range(len(A2))
              if i != j and determines(A2[i], A2[j])]               # noqa: F821
    assert len(A2) == 368 and len(pairs2) == 1046, (len(A2), len(pairs2))
    print(f"catalog: {len(A2)} assets, {len(pairs2)} verified pairs")

    # 3. Query times for the compute-metered price
    et_path = os.path.join(args.out, "_A2_et.npy")
    if os.path.exists(et_path):
        et2 = np.load(et_path)
    else:
        et2 = np.zeros(len(A2))
        for i, a in enumerate(A2):
            t0 = time.perf_counter()
            con.execute(asset_sql(a)).fetchall()                     # noqa: F821
            et2[i] = time.perf_counter() - t0
        np.save(et_path, et2)

    # 4. Support set, conflict sets and valuation families
    load("ports_chawla.py")
    support = sample_support_set(con, 400, seed=0)                  # noqa: F821
    edges_asset = [frozenset(i for i, r in enumerate(support) if distinguishes(a, r, TEMPLATES))  # noqa: F821
                   for a in A2]
    info_map = info_families(rowsz2, ssz2, np.ones(len(A2)))        # noqa: F821
    print(f"support set |S| = {len(support)}, non-empty conflict sets "
          f"{sum(1 for e in edges_asset if e)}")

    # 5. Mechanisms, fit, freeze, evaluate
    load("run_full.py")
    qs = qirana_scores(A2, support, TEMPLATES)                      # noqa: F821
    view_idx = [i for i, a in enumerate(A2)
                if len(a.months) == 1 and a.group == frozenset(TEMPLATES[a.template]["group_dims"])]
    qm = querymarket_prices(A2, view_idx, {i: float(ssz2[i]) for i in view_idx},  # noqa: F821
                            min_cost_cover)                         # noqa: F821
    qm[~np.isfinite(qm)] = qm[np.isfinite(qm)].max() * 1.5
    scale_scores = {"uniform": np.ones(len(A2)), "size_proportional": rowsz2.copy(),
                    "compute_metered": et2.copy(),
                    "qirana_weighted_coverage": qs["qirana_weighted_coverage"].copy(),
                    "qirana_uniform_gain": qs["qirana_uniform_gain"].copy(),
                    "qirana_shannon": np.maximum(qs["qirana_shannon"], 0).copy(),
                    "querymarket_viewcover": qm.copy()}

    def plan_fn(p):
        pl = plan_all(A2, ans, p, TEMPLATES, answers_match, verify_limit=8)  # noqa: F821
        eff = np.minimum(p, pl.cover_cost.to_numpy())
        return eff, pl.plan_size.to_numpy(), int(pl.verified.sum()), int((pl.verified == False).sum())  # noqa: E712

    t0 = time.perf_counter()
    RF = run(A2, ans, pairs2, scale_scores, edges_asset, len(support), info_map,  # noqa: F821
             plan_fn, audit_monotone, fit_scale, buyer_sample, monopoly_prices,   # noqa: F821
             enforce_monotone)                                                     # noqa: F821
    out_csv = os.path.join(args.out, "full_tpch_runs.csv")
    RF.to_csv(out_csv, index=False)
    print(f"{len(RF)} runs in {time.perf_counter() - t0:.0f}s | plans re-executed "
          f"{int(RF.plans_verified.sum())} | failed {int(RF.plans_failed.sum())}")
    cols = ["revenue_strategic_norm", "surplus_strategic_norm", "welfare_strategic_norm",
            "served_strategic", "mono_violation_rate", "comb_violation_rate", "comb_max_gain",
            "generalization_gap"]
    summary = RF.groupby("mechanism", sort=False)[cols].mean()
    summary.to_csv(os.path.join(args.out, "full_tpch_summary.csv"))
    print(f"wrote {out_csv} ({time.perf_counter() - t_start:.0f}s in total)")


if __name__ == "__main__":
    main()
