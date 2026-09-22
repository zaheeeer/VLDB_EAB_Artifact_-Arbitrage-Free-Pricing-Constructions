"""Recompute historical CSV diagnostics only; never overwrite the supplied results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def recompute(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    data = ROOT / "legacy"
    full = {name: pd.read_csv(data / f"full_{name}_runs.csv") for name in ("tpch", "sqlshare")}
    audit = {"scope": "arithmetic audit of historical CSVs; not a corrected experiment",
             "full_runs": {}, "file_sha256": {}}
    for name, df in full.items():
        saved = pd.read_csv(data / f"full_{name}_summary.csv", index_col=0)
        means = df.groupby("mechanism").mean(numeric_only=True)
        cols = saved.select_dtypes("number").columns.intersection(means.columns)
        error = float((saved[cols] - means.loc[saved.index, cols]).abs().max().max())
        assert len(df) == 300 and not df.duplicated(["family", "seed", "mechanism"]).any()
        assert error < 1e-10
        audit["full_runs"][name] = {"rows": len(df), "families": int(df.family.nunique()),
                 "seeds": int(df.seed.nunique()), "mechanisms": int(df.mechanism.nunique()),
                 "duplicate_keys": 0, "summary_max_abs_error": error,
                 "successful_verifications": int(df.plans_verified.sum()),
                 "failed_verifications": int(df.plans_failed.sum())}
    t = full["tpch"]
    fractional = t.comb_violation_rate * 368
    counts = fractional.round().astype(int)
    assert np.max(np.abs(counts - fractional)) < 1e-7
    audit["tpch_verification"] = {
        "assets_per_run": 368, "reported_cheaper_asset_run_cases": int(counts.sum()),
        "successful_checks": int(t.plans_verified.sum()),
        "lower_bound_cases_without_successful_check": int((counts-t.plans_verified).clip(lower=0).sum()),
        "runs_with_more_positive_cases_than_successful_checks": int((counts>t.plans_verified).sum()),
        "interpretation": "Repeated asset/run outcomes, not distinct plans. Notebook final callback caps checks at eight per run."}
    audit["attempted_checks_total"] = sum(v["successful_verifications"]+v["failed_verifications"]
                                            for v in audit["full_runs"].values())
    r = pd.read_csv(data / "all_markets_runs.csv")
    g = r[r.status.eq("ok")].copy()
    keys = ["market", "seed", "mechanism"]
    a = g[g.family.eq("answer_cells")].drop(columns="family").sort_values(keys).reset_index(drop=True)
    b = g[g.family.eq("support_rows")].drop(columns="family").sort_values(keys).reset_index(drop=True)
    duplicate_families = a.equals(b)
    assert duplicate_families
    structure = pd.read_csv(data / "all_markets_structure.csv")
    audit["all_markets"] = {"rows": len(r), "successful_rows": len(g),
            "too_small_status_rows": int(r.status.eq("too_small").sum()),
            "markets": int(g.market.nunique()), "mechanisms": int(g.mechanism.nunique()),
            "seeds": int(g.seed.nunique()), "identical_family_row_pairs": len(a),
            "assets": int(structure.assets.sum()), "pairs": int(structure.pairs.sum()),
            "entirely_zero_priced_markets": int((structure.zero_priced == structure.assets).sum())}
    # One copy of the duplicated family; mean over seeds within each market first.
    market_means = a.groupby(["market", "mechanism"])[["revenue_norm", "welfare_norm", "served"]].mean()
    across = market_means.groupby("mechanism").agg(["mean", "std", "count"])
    across.columns = [f"{c}_{stat}" for c, stat in across.columns]
    across.to_csv(outdir / "historical_across_market_reporting.csv")
    q = g[g.mechanism.eq("qirana_weighted_coverage")]
    audit["qirana_across_market_sd"] = {"pooled_row_welfare_sd": float(q.welfare_norm.std(ddof=1)),
            "market_mean_welfare_sd": float(across.loc["qirana_weighted_coverage", "welfare_norm_std"]),
            "unit": "15 market means; SD describes heterogeneity, not a confidence interval"}
    rows = []
    for name, df in full.items():
        for mechanism, sub in df.groupby("mechanism"):
            rows.append({"workload": name, "mechanism": mechanism, "runs": len(sub),
                "mean_combination_violation_rate": float(sub.comb_violation_rate.mean()),
                "positive_combination_runs": int(sub.comb_violation_rate.gt(0).sum()),
                "mean_of_per_run_max_gains": float(sub.comb_max_gain.mean()),
                "global_max_observed_gain": float(sub.comb_max_gain.max())})
    pd.DataFrame(rows).to_csv(outdir / "historical_arbitrage_reporting.csv", index=False)
    cross = pd.read_csv(data / "full_cross_workload.csv", index_col=0)
    regenerated = pd.concat([df.groupby("mechanism").mean(numeric_only=True).add_suffix("_"+name)
                             for name,df in full.items()], axis=1)
    cols = cross.select_dtypes("number").columns.intersection(regenerated.columns)
    audit["cross_workload_max_abs_error"] = float((cross[cols]-regenerated.loc[cross.index,cols]).abs().max().max())
    assert audit["cross_workload_max_abs_error"] < 1e-10
    for p in sorted(data.glob("*.csv")):
        audit["file_sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (outdir / "historical_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({k:v for k,v in audit.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation")
    recompute(parser.parse_args().output)
