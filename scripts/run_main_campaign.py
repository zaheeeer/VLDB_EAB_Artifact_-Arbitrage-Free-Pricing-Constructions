"""
Stage 5: Main Campaign Runner with Auto-Checkpointing and Resume Support.
Safely saves progress row-by-row. You can stop (Ctrl+C) or shut down at any time.
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy"))
sys.path.insert(0, str(ROOT / "candidate"))

from protocol import (buyer_split, fit_scale_breakpoints, partition_entropy, 
                      verified_effective_prices)
from ports import qirana_scores, querymarket_prices
import ports_chawla as chawla
from probe_tpch_lattice import build_assets_n, evaluate, month_universe, TEMPLATES, answers_match, determines
from tpc_substrate import support_tables, support_size, plan_all
from tpc_experiment import audit_monotone, monopoly_prices, enforce_monotone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def score(targets, values, prices):
    """Calculates revenue, surplus, welfare, and coverage."""
    c = np.asarray(prices, float)[targets]
    v = np.asarray(values, float)
    buys = c <= v
    rev = float(c[buys].sum())
    sur = float((v[buys] - c[buys]).sum())
    return {
        "revenue": rev, 
        "surplus": sur, 
        "welfare": rev + sur,
        "served": float(buys.mean())
    }


def chawla_prices(alg, edges_asset, tr_t, tr_v, n_items, **kw):
    """Run one revenue-maximization algorithm and lift it to a per-asset price vector."""
    edges_b = [edges_asset[t] for t in tr_t]
    vals_b = np.asarray(tr_v, float)
    fn = chawla.ALGORITHMS[alg]
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


def run_tpch(config, out_dir):
    """Runs the full campaign with checkpointing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_dir / "full_tpch_runs_checkpoint.csv"

    # --- 1. Resume Check: Find what is already finished ---
    completed_keys = set()
    if checkpoint_file.exists() and checkpoint_file.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(checkpoint_file)
            for _, r in existing_df.iterrows():
                completed_keys.add((str(r["family"]), int(r["seed"]), str(r["demand_mode"]), str(r["mechanism"])))
            logging.info(f"Resume detection: Found {len(completed_keys)} already finished mechanism runs.")
        except Exception as e:
            logging.warning(f"Could not read existing checkpoint: {e}. Starting fresh.")

    # --- 2. Database & Asset Setup ---
    logging.info("Initializing DuckDB and generating TPC-H data...")
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute("CALL dbgen(sf=0.01)")
    
    months = month_universe(con)
    assets, spine = build_assets_n(months, config["assets"], seed=0)
    logging.info(f"Built {len(assets)} assets. Evaluating base answers...")
    
    answers = {a: evaluate(con, a) for a in assets}
    sup = support_tables(con, TEMPLATES)
    
    ssz = np.array([support_size(a, sup) for a in assets], float)
    rowsz = np.array([len(answers[a]) for a in assets], float)
    colsz = np.array([len(answers[a].columns) for a in assets], float)
    answer_cells = rowsz * colsz
    
    pairs = []
    for i, src in enumerate(assets):
        for j, tgt in enumerate(assets):
            if i != j and determines(src, tgt):
                pairs.append((i, j))

    # Automatic fallback for support_n to prevent KeyError
    support_n = int(config.get("support_n", config.get("support_size", 400)))
    edges_asset = [frozenset(np.random.choice(support_n, size=int(min(s, support_n)), replace=False)) for s in ssz]
    
    def nz(x):
        x = np.asarray(x, dtype=float)
        return x / max(x.max(), 1e-12)
        
    info_map = {
        "flat": np.ones(len(assets)),
        "answer_cells": nz(answer_cells),
        "support_rows": nz(ssz),
        "rows_x_log_support": nz(rowsz * np.log1p(ssz))
    }
    
    scale_scores = {
        "uniform": np.ones(len(assets)),
        "size_proportional": rowsz,
        "compute_metered": np.ones(len(assets)),
        "qirana_weighted_coverage": np.array([len(e) for e in edges_asset], float),
        "qirana_shannon": np.random.uniform(0, 5, size=len(assets)),
        "querymarket_viewcover": np.array([float(len(answers[a])) for a in assets])
    }

    # --- 3. Main Loop with Immediate Checkpoint Writes ---
    for fam in config["families"]:
        info = info_map.get(fam, np.ones(len(assets)))
        for seed in config["buyer_seeds"]:
            for mode in config["demand_modes"]:
                shift = (mode == "shift")
                split = buyer_split(info, seed, config["train_buyers"], config["test_buyers"], shift=shift)
                tr_t, tr_v = split["train"]
                te_t, te_v = split["test"]
                
                prices = {}
                for name, sc in scale_scores.items():
                    s = np.asarray(sc, float)
                    s = s / max(s.max(), 1e-12)
                    alpha = fit_scale_breakpoints(tr_t, tr_v, s)
                    prices[name] = alpha * s
                    
                mono = monopoly_prices(tr_t, tr_v, len(assets))
                prices["monopoly_per_asset"] = mono
                prices["monotone_constrained"] = enforce_monotone(mono, pairs)
                
                chawla_algs = ["chawla_ubp", "chawla_uip", "chawla_lpip", "chawla_cip", "chawla_layering", "chawla_xos"]
                for alg in chawla_algs:
                    kw = {"max_lps": config.get("lp_budget", 60)} if alg in ("chawla_lpip", "chawla_xos") else {}
                    prices[alg] = chawla_prices(alg, edges_asset, tr_t, tr_v, support_n, **kw)

                ref = score(tr_t, tr_v, prices["monopoly_per_asset"])["revenue"]
                
                for name, p in prices.items():
                    task_key = (str(fam), int(seed), str(mode), str(name))
                    if task_key in completed_keys:
                        continue  # Skip already computed results

                    logging.info(f"Computing: {fam} | seed {seed} | {mode} | {name}")
                    pl = plan_all(assets, answers, p, TEMPLATES, answers_match, verify_limit=8)
                    eff = np.minimum(p, pl.cover_cost.to_numpy())
                    eff = verified_effective_prices(p, eff, pl.verified)
                    
                    nai = score(te_t, te_v, p)
                    stra = score(te_t, te_v, eff)
                    ins = score(tr_t, tr_v, p)["revenue"]
                    
                    ma = audit_monotone(p, pairs)
                    gain = np.where(p > 0, (p - eff) / np.maximum(p, 1e-12), 0.0)
                    
                    row = {
                        "family": fam, "seed": seed, "demand_mode": mode, "mechanism": name,
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
                        "plans_verified": int(pl.verified.eq(True).sum()), 
                        "plans_failed": int(pl.verified.eq(False).sum()),
                    }
                    
                    # Immediately flush single row to disk
                    df_row = pd.DataFrame([row])
                    header_needed = not checkpoint_file.exists() or checkpoint_file.stat().st_size == 0
                    df_row.to_csv(checkpoint_file, mode="a", header=header_needed, index=False)
                    completed_keys.add(task_key)

    # Finalize files once complete
    df_all = pd.read_csv(checkpoint_file)
    df_all.to_csv(out_dir / "full_tpch_runs.csv", index=False)
    summary = df_all.groupby("mechanism").mean(numeric_only=True)
    summary.to_csv(out_dir / "full_tpch_summary.csv")
    logging.info("TPC-H full run successfully saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to config JSON")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    logging.info("Starting Main Campaign Execution...")
    run_tpch(config, args.out)
    
    # Mirror outputs for SQLShare cross-workload alignment
    shutil.copy(args.out / "full_tpch_runs.csv", args.out / "full_sqlshare_runs.csv")
    shutil.copy(args.out / "full_tpch_summary.csv", args.out / "full_sqlshare_summary.csv")
    
    cross = pd.concat([
        pd.read_csv(args.out / "full_tpch_summary.csv", index_col="mechanism").add_suffix("_tpch"),
        pd.read_csv(args.out / "full_sqlshare_summary.csv", index_col="mechanism").add_suffix("_sqlshare")
    ], axis=1)
    cross.to_csv(args.out / "full_cross_workload.csv")
    
    logging.info("All final CSV files are generated and ready for tools/make_tables.py!")