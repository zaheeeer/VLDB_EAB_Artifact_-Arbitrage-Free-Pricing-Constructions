"""Rebuild Tables 4 to 9, the data of Figure 2, and the numbers quoted in the text of
"Arbitrage-Free Query Pricing under Recomposition" from the result files, and check
every value against the paper.

    python reproduce/make_tables.py
    python reproduce/make_tables.py --tpch tpch/out --sqlshare sqlshare/results   # a fresh run

Inputs (defaults are the published results in results/):
  --tpch      folder with full_tpch_runs.csv and oracle_stress_confusion.csv
  --sqlshare  folder with paper/ (the setting of Table 2), corrected/ (every alternative
              of Table 2 on), and ablation/ (the 30 configurations of Section 5.3)

Outputs in reproduce/out/: table4_cross.csv ... table9_sensitivity.csv,
figure2a.csv, figure2b_markets.csv, figure2c.csv, and text_numbers.csv. The script
prints one line per table and per text number and exits with status 1 if any value
differs from the paper at the precision the paper prints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Rows of Tables 4 to 6, in the paper's order: (label in the paper, name in the files).
MECHANISMS = [
    ("flat fee (= UBP)", "uniform"),
    ("size proportional", "size_proportional"),
    ("compute metered", "compute_metered"),
    ("per-asset monopoly", "monopoly_per_asset"),
    ("monotone fitted", "monotone_constrained"),
    ("Qirana weighted coverage", "qirana_weighted_coverage"),
    ("Qirana uniform gain", "qirana_uniform_gain"),
    ("Qirana entropy", "qirana_shannon"),
    ("QueryMarket view cover", "querymarket_viewcover"),
    ("uniform bundle (UBP)", "chawla_ubp"),
    ("uniform item (UIP)", "chawla_uip"),
    ("LP item (LPIP)", "chawla_lpip"),
    ("capacity item (CIP)", "chawla_cip"),
    ("layering", "chawla_layering"),
    ("XOS", "chawla_xos"),
]
TEN = [m for _, m in MECHANISMS[5:]]          # the ten published constructions
ALL15 = [m for _, m in MECHANISMS]
TABLE8 = ["uniform", "monopoly_per_asset", "qirana_weighted_coverage",
          "querymarket_viewcover", "chawla_ubp", "chawla_lpip"]
FOUR = ["qirana_weighted_coverage", "querymarket_viewcover", "chawla_ubp", "chawla_lpip"]
SEEDS_SETTING = ["paper", "paper_seedset1", "paper_seedset2"]
SEEDS_ALL = ["corrected", "corrected_seedset1", "corrected_seedset2"]


# ---------------------------------------------------------------------------- inputs
def tpch_means(folder):
    runs = pd.read_csv(os.path.join(folder, "full_tpch_runs.csv"))
    return runs, runs.groupby("mechanism").mean(numeric_only=True)


def sqlshare_means(folder):
    s = pd.read_csv(os.path.join(folder, "paper", "main_market", "summary_by_mechanism.csv"))
    s = s[s.arm == "published"].set_index("mechanism")
    return s


def ablation(folder):
    long = pd.read_csv(os.path.join(folder, "ablation", "ablation_long.csv"))
    piv = long.pivot_table(index="configuration", columns="quantity", values="value",
                           aggfunc="first")
    return piv.apply(pd.to_numeric, errors="coerce")


def numbers(folder, protocol):
    with open(os.path.join(folder, protocol, "paper_numbers.json"), encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------- tables
def table4(t, s):
    rows = []
    for label, m in MECHANISMS:
        rows.append({"mechanism": label,
                     "revenue_tpch": t.loc[m, "revenue_strategic_norm"],
                     "revenue_sqlshare": s.loc[m, "revenue_strategic_norm_mean"],
                     "welfare_tpch": t.loc[m, "welfare_strategic_norm"],
                     "welfare_sqlshare": s.loc[m, "welfare_strategic_norm_mean"],
                     "served_tpch": t.loc[m, "served_strategic"],
                     "served_sqlshare": s.loc[m, "served_strategic_mean"]})
    return pd.DataFrame(rows)


def table5(t, s):
    return pd.DataFrame([{"mechanism": label, "gap_tpch": t.loc[m, "generalization_gap"],
                          "gap_sqlshare": s.loc[m, "generalization_gap_mean"]}
                         for label, m in MECHANISMS])


def table6(t, s):
    rows = []
    for label, m in MECHANISMS:
        rows.append({"mechanism": label,
                     "info_arb_tpch": t.loc[m, "mono_violation_rate"],
                     "info_arb_sqlshare": s.loc[m, "mono_violation_rate_mean"],
                     "comb_arb_tpch": t.loc[m, "comb_violation_rate"],
                     "comb_arb_sqlshare": s.loc[m, "comb_violation_rate_mean"],
                     "max_gain_tpch": t.loc[m, "comb_max_gain"],
                     "max_gain_sqlshare": s.loc[m, "comb_max_gain_mean"]})
    return pd.DataFrame(rows)


def table7(sq, ab):
    t6 = pd.read_csv(os.path.join(sq, "paper", "main_market", "table6_support_sampling.csv"))
    size, read = t6.iloc[0], t6.iloc[1]
    rows = []
    for label, q_size, q_read, v_size, v_read in [
            ("priceable assets (of %d)" % int(size.assets), "t6_priceable_size",
             "t6_priceable_read", size.assets_priceable, read.assets_priceable),
            ("median |C(a)|", "t6_median_size", "t6_median_read",
             size.median_conflict_set, read.median_conflict_set),
            ("price levels", "t6_levels_size", "t6_levels_read",
             size.distinct_price_levels, read.distinct_price_levels)]:
        rs = ab.loc[SEEDS_SETTING, q_size]
        rr = ab.loc[SEEDS_SETTING, q_read]
        rows.append({"row": label, "size_proportional": v_size, "size_min": rs.min(),
                     "size_max": rs.max(), "readership_weighted": v_read,
                     "read_min": rr.min(), "read_max": rr.max()})
    rows.append({"row": "build time (s)", "size_proportional": size.build_seconds,
                 "readership_weighted": read.build_seconds})
    return pd.DataFrame(rows)


def table8(sq):
    t7 = pd.read_csv(os.path.join(sq, "paper", "markets", "table7_markets.csv")).set_index("mechanism")
    label = dict((m, l) for l, m in MECHANISMS)
    return pd.DataFrame([{"mechanism": label[m], "revenue_mean": t7.loc[m, "revenue_mean"],
                          "revenue_sd": t7.loc[m, "revenue_sd_across_markets"],
                          "welfare_mean": t7.loc[m, "welfare_mean"],
                          "welfare_sd": t7.loc[m, "welfare_sd_across_markets"]} for m in TABLE8])


# (row label, quantity, choice, switch, show seed range)
TABLE9 = [
    ("claimed edges", "claimed_edges", "selection rewrite", "C07_selection", False),
    ("refuted edges", "false_edges_total", "re-executed queries", "alias_resolve", True),
    ("zero-priced assets", "zero_priced_readership", "re-executed queries", "alias_resolve", True),
    ("priceable assets, readership weighted", "t6_priceable_read", "re-executed queries",
     "alias_resolve", True),
    ("Qirana maximum gain", "qirana_max_gain", "re-executed queries", "alias_resolve", True),
    ("share of fits at the grid cap", "alpha_at_grid_max_share", "scale fit", "C05_fit", False),
    ("QueryMarket revenue", "qm_revenue", "scale fit", "C05_fit", False),
    ("QueryMarket welfare", "qm_welfare", "scale fit", "C05_fit", False),
    ("Qirana weighted coverage welfare", "welfare:qirana_weighted_coverage", "scale fit",
     "C05_fit", False),
    ("negative generalization gaps (of 15)", "gaps_negative", "test demand", "C04_demand", False),
    ("Qirana information arbitrage", "info_arb:qirana_weighted_coverage", "edges removed",
     "remove_all_falsified", False),
]


def table9(ab):
    rows = []
    for label, q, choice, sw, rng in TABLE9:
        r = {"quantity": label, "setting": ab.loc["paper", q],
             "all_alternatives": ab.loc["corrected", q], "choice": choice,
             "added_alone": ab.loc["add_" + sw, q] - ab.loc["paper", q],
             "added_last": ab.loc["corrected", q] - ab.loc["drop_" + sw, q]}
        if rng:
            r.update(setting_min=ab.loc[SEEDS_SETTING, q].min(), setting_max=ab.loc[SEEDS_SETTING, q].max(),
                     all_min=ab.loc[SEEDS_ALL, q].min(), all_max=ab.loc[SEEDS_ALL, q].max())
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- figure data
def figure_data(t4, sq, t7, out):
    fa = pd.DataFrame({"mechanism": t4.mechanism,
                       "revenue": (t4.revenue_tpch + t4.revenue_sqlshare) / 2,
                       "welfare": (t4.welfare_tpch + t4.welfare_sqlshare) / 2})
    fa.to_csv(os.path.join(out, "figure2a.csv"), index=False)
    fb = pd.read_csv(os.path.join(sq, "paper", "markets", "figure2b_markets.csv"))
    fb.to_csv(os.path.join(out, "figure2b_markets.csv"), index=False)
    fc = t7[t7.row != "build time (s)"][["row", "size_proportional", "readership_weighted"]]
    fc.to_csv(os.path.join(out, "figure2c.csv"), index=False)


# ---------------------------------------------------------------------------- text numbers
def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def text_numbers(runs_t, oracle, sq, ab):
    P, C = numbers(sq, "paper"), numbers(sq, "corrected")
    lat, lat_c, main, main_c, mk, mk_c = (P["lattice"], C["lattice"], P["main"], C["main"],
                                          P["markets"], C["markets"])
    cases = (runs_t.comb_violation_rate * 368).round().astype(int)
    by_mech = runs_t.assign(c=cases).groupby("mechanism").c.sum()
    rev = lambda c, m: ab.loc[c, "revenue:" + m]                              # noqa: E731
    wel = lambda c, m: ab.loc[c, "welfare:" + m]                              # noqa: E731
    configs = list(ab.index)
    shared = [c for c in configs if c == "add_C04_demand"
              or (c.startswith("corrected")) or (c.startswith("drop_") and c != "drop_C04_demand")]
    pm = {}
    for prot in ("paper", "corrected"):
        d = pd.read_csv(os.path.join(sq, prot, "markets", "per_market_means.csv"))
        r = d.pivot_table(index="rank", columns="mechanism", values="revenue_norm")
        w = d.pivot_table(index="rank", columns="mechanism", values="welfare_norm")
        pm[prot] = (int((r[FOUR].idxmax(axis=1) == "chawla_ubp").sum()),
                    int(((r.chawla_lpip < r.chawla_ubp) & (w.chawla_lpip > w.chawla_ubp)).sum()))
    move = lambda a, b, kinds: max(abs(ab.loc[a, k + m] - ab.loc[b, k + m])   # noqa: E731
                                   for k in kinds for m in ALL15)
    sp = {c: spearman([rev(c, m) for m in ALL15], [wel(c, m) for m in ALL15]) for c in configs}
    rate = ab.false_edges_total / ab.claimed_edges
    n = {
        # Section 4.1 and 5.2.1: lattice
        "SQLShare verified pairs": (lat["verified_pairs"], 2969),
        "SQLShare materialized assets": (lat["materialized_assets"], 806),
        "SQLShare pairs per asset": (lat["pairs_per_asset"], 3.68),
        "projection share %": (lat["family_share_pct"]["projection"], 71.4),
        "selection share %": (lat["family_share_pct"]["selection"], 27.8),
        "roll-up share %": (lat["family_share_pct"]["rollup"], 0.7),
        "pairs with the predicate rewrite": (lat_c["verified_pairs"], 2667),
        "projection share %, predicate rewrite": (lat_c["family_share_pct"]["projection"], 79.5),
        "selection share %, predicate rewrite": (lat_c["family_share_pct"]["selection"], 19.6),
        "roll-up share %, predicate rewrite": (lat_c["family_share_pct"]["rollup"], 0.8),
        # Section 5.2.2: falsification on the largest market
        "claimed edges": (main["claimed_edges"], 2047),
        "inverted at 30,000 instances": (main["falsified_curve"], 33),
        "witnessed in the 4,000-instance sample": (main["witnessed"], 20),
        "refuted by the pricing support": (main["falsified_by_pricing_support"], 12),
        "refuted edges, setting": (main["false_edges_total"], 45),
        "refuted edges, alias readers re-executed": (ab.loc["add_alias_resolve", "false_edges_total"], 16),
        "refuted edges, every alternative on": (main_c["false_edges_total"], 14),
        "refuted edges, min over 30 configurations": (ab.false_edges_total.min(), 14),
        "refuted edges, max over 30 configurations": (ab.false_edges_total.max(), 45),
        "refuted rate %, min over 30 configurations": (100 * rate.min(), 0.7),
        "refuted rate %, max over 30 configurations": (100 * rate.max(), 2.3),
        # Section 5.2.3: samplers
        "corr(table size, readership)": (main["corr_size_vs_readership"], -0.088),
        "zero-priced assets, readership sampler": (main["zero_priced_readership"], 60),
        "zero-priced assets, alias readers re-executed": (ab.loc["add_alias_resolve", "zero_priced_readership"], 40),
        "build cost ratio": (main["build_cost_ratio"], 113),
        # Sections 1.1, 3.5 and 6.5: plans
        "SQLShare plans re-executed": (main["plans_verified"] + main["plans_failed"], 3240),
        "SQLShare plans failed": (main["plans_failed"], 12),
        "SQLShare plans failed, layering": (main["plans_failed_by_mechanism"].get("chawla_layering", 0), 8),
        "SQLShare plans failed, LPIP": (main["plans_failed_by_mechanism"].get("chawla_lpip", 0), 4),
        "TPC-H plans re-executed": (int(runs_t.plans_verified.sum()), 387),
        "TPC-H plans failed": (int(runs_t.plans_failed.sum()), 0),
        "plans re-executed, both workloads": (main["plans_verified"] + main["plans_failed"]
                                              + int(runs_t.plans_verified.sum()), 3627),
        "TPC-H cheaper asset/run cases": (int(cases.sum()), 1541),
        "TPC-H cases, compute metered": (int(by_mech.get("compute_metered", 0)), 100),
        "TPC-H cases, per-asset monopoly": (int(by_mech.get("monopoly_per_asset", 0)), 1235),
        "TPC-H cases, monotone fitted": (int(by_mech.get("monotone_constrained", 0)), 206),
        # Section 3.2: TPC-H oracle
        "TPC-H oracle pairs tested": (int(oracle.pairs.sum()), 54080),
        "TPC-H oracle precision": (float(oracle.precision.min()), 1.0),
        "TPC-H oracle recall": (float(oracle.recall.min()), 1.0),
        # Section 5.2.4: markets
        "markets": (mk["markets_ok"], 15),
        "assets in the per-market runs": (mk["assets"], 812),
        "pairs in the per-market runs": (mk["pairs"], 2971),
        "refuted pairs across markets": (mk["falsified"], 42),
        "markets with refuted pairs": (mk["markets_with_falsified"], 4),
        "refuted pairs across markets, every alternative on": (mk_c["falsified"], 20),
        "pairs across markets, every alternative on": (mk_c["pairs"], 2669),
        "markets with refuted pairs, every alternative on": (mk_c["markets_with_falsified"], 2),
        "welfare wins of 15": (mk["welfare_wins_published"], 15),
        "revenue wins of flat pricing of 15": (mk["revenue_wins_flat"], 13),
        "welfare wins of 15, every alternative on": (mk_c["welfare_wins_published"], 10),
        "revenue wins of 15, every alternative on": (mk_c["revenue_wins_flat"], 11),
        "median zero-priced share %": (mk["zero_priced_median_pct"], 31.7),
        "median zero-priced share %, every alternative on": (mk_c["zero_priced_median_pct"], 10.5),
        "markets entirely zero-priced": (mk["markets_all_zero_priced"], 4),
        "markets where UBP tops the four constructions": (pm["paper"][0], 11),
        "same, every alternative on": (pm["corrected"][0], 10),
        "markets where LPIP trades revenue for welfare": (pm["paper"][1], 12),
        "same, every alternative on ": (pm["corrected"][1], 12),
        # Section 5.3: sensitivity analysis
        "configurations": (len(configs), 30),
        "fits at the grid cap": (ab.loc["paper", "alpha_at_grid_max_share"], 0.237),
        "UBP top revenue of the ten, configurations": (
            sum(rev(c, "chawla_ubp") >= max(rev(c, m) for m in TEN) for c in configs), 30),
        "UBP revenue, min": (min(rev(c, "chawla_ubp") for c in configs), 0.554),
        "UBP revenue, max": (max(rev(c, "chawla_ubp") for c in configs), 0.616),
        "LPIP more welfare, more served, less revenue than UBP, configurations": (
            sum(wel(c, "chawla_lpip") > wel(c, "chawla_ubp")
                and ab.loc[c, "served:chawla_lpip"] > ab.loc[c, "served:chawla_ubp"]
                and rev(c, "chawla_lpip") < rev(c, "chawla_ubp") for c in configs), 30),
        "readership sampler prices more assets, configurations": (
            int((ab.t6_priceable_read > ab.t6_priceable_size).sum()), 30),
        "Qirana welfare above UBP, configurations": (
            sum(wel(c, "qirana_weighted_coverage") > wel(c, "chawla_ubp") for c in configs), 28),
        "Spearman revenue vs welfare, min": (min(sp.values()), -0.82),
        "Spearman revenue vs welfare, max": (max(sp.values()), 0.53),
        "Spearman revenue vs welfare, setting": (sp["paper"], -0.48),
        "negative gaps, setting": (ab.loc["paper", "gaps_negative"], 7),
        "mean gap, setting": (ab.loc["paper", "gap_mean"], 0.034),
        "mean gap, shared demand": (ab.loc["add_C04_demand", "gap_mean"], 0.129),
        "negative gaps restored from the other end": (ab.loc["drop_C04_demand", "gaps_negative"], 4),
        "shared-demand configurations with no negative gap": (
            sum(ab.loc[c, "gaps_negative"] == 0 for c in shared), 14),
        "shared-demand configurations": (len(shared), 15),
        "monopoly has the largest gap, configurations": (
            sum(max(ALL15, key=lambda m: ab.loc[c, "gap:" + m]) == "monopoly_per_asset"
                for c in configs), 30),
        "monopoly gap, min": (ab["gap:monopoly_per_asset"].min(), 0.420),
        "monopoly gap, max": (ab["gap:monopoly_per_asset"].max(), 0.481),
        "LPIP gap, max over configurations": (ab["gap:chawla_lpip"].max(), 0.131),
        "CIP gap, max over configurations": (ab["gap:chawla_cip"].max(), 0.074),
        "QueryMarket revenue, exact fit": (ab.loc["add_C05_fit", "qm_revenue"], 0.529),
        "QueryMarket welfare, exact fit": (ab.loc["add_C05_fit", "qm_welfare"], 1.069),
        "Qirana welfare, exact fit": (ab.loc["add_C05_fit", "welfare:qirana_weighted_coverage"], 1.193),
        "UBP welfare, exact fit": (ab.loc["add_C05_fit", "welfare:chawla_ubp"], 1.289),
        "largest move of the selection rewrite": (max(move("add_C07_selection", "paper",
                                                            ["revenue:", "welfare:", "served:"]),
                                                       move("drop_C07_selection", "corrected",
                                                            ["revenue:", "welfare:", "served:"])), 0.009),
        "largest move of answer cells": (max(move("add_C11_answer_cells", "paper", ["revenue:", "welfare:"]),
                                             move("drop_C11_answer_cells", "corrected",
                                                  ["revenue:", "welfare:"])), 0.081),
    }
    return n


# ---------------------------------------------------------------------------- checks
NUM = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")


def paper_rows(name):
    """Numbers in each body row of a paper table: {first cell: [numeric tokens]}."""
    text = open(os.path.join(HERE, "paper_tables", name), encoding="utf-8").read()
    body = text.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    rows = {}
    for line in body.strip().splitlines():
        if "&" not in line:
            continue
        cells = [c.strip() for c in line.split(r"\\")[0].split("&")]
        label = cells[0].replace(r"\qirana{}", "Qirana").replace(r"\querymarket{}", "QueryMarket")
        label = re.sub(r"\$\^\{\\dagger\}\$", "", label).replace("$|", "|").replace("|$", "|")
        label = label.replace(r"\cset{a}", "C(a)").replace("$", "").strip()
        tokens = []
        for c in cells[1:]:
            tokens += NUM.findall(c.replace(r"\pm", " ").replace("$", ""))
        rows[label] = tokens
    return rows


def close(value, token):
    d = len(token.split(".")[1]) if "." in token else 0
    want = float(token.replace(",", ""))
    return abs(round(float(value), d) - want) <= 0.5 * 10 ** (-d) + 1e-9


def check_table(name, frame, label_col, value_cols):
    paper = paper_rows(name)
    bad = []
    if len(paper) != len(frame):
        bad.append(f"{len(frame)} rows rebuilt, {len(paper)} rows in the paper table")
    for _, r in frame.iterrows():
        label = r[label_col]
        want = paper.get(label)
        if want is None:
            bad.append(f"{label}: row not found in the paper table")
            continue
        got = [r[c] for c in value_cols if c in r and not (isinstance(r[c], float) and math.isnan(r[c]))]
        if len(got) != len(want) or not all(close(g, w) for g, w in zip(got, want)):
            bad.append(f"{label}: paper {want}, files {[round(float(g), 4) for g in got]}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tpch", default=os.path.join(ROOT, "results", "tpch"))
    ap.add_argument("--sqlshare", default=os.path.join(ROOT, "results", "sqlshare"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs_t, t = tpch_means(args.tpch)
    oracle = pd.read_csv(os.path.join(args.tpch, "oracle_stress_confusion.csv"))
    s = sqlshare_means(args.sqlshare)
    ab = ablation(args.sqlshare)

    t4, t5, t6 = table4(t, s), table5(t, s), table6(t, s)
    t7, t8, t9 = table7(args.sqlshare, ab), table8(args.sqlshare), table9(ab)
    for name, frame in [("table4_cross", t4), ("table5_generalization", t5), ("table6_audits", t6),
                        ("table7_sampling", t7), ("table8_markets", t8), ("table9_sensitivity", t9)]:
        frame.to_csv(os.path.join(args.out, name + ".csv"), index=False)
    figure_data(t4, args.sqlshare, t7, args.out)

    checks = [
        ("Table 4", check_table("tab_cross.tex", t4, "mechanism",
                                ["revenue_tpch", "revenue_sqlshare", "welfare_tpch",
                                 "welfare_sqlshare", "served_tpch", "served_sqlshare"])),
        ("Table 5", check_table("tab_generalization.tex", t5, "mechanism",
                                ["gap_tpch", "gap_sqlshare"])),
        ("Table 6", check_table("tab_audits.tex", t6, "mechanism",
                                ["info_arb_tpch", "info_arb_sqlshare", "comb_arb_tpch",
                                 "comb_arb_sqlshare", "max_gain_tpch", "max_gain_sqlshare"])),
        ("Table 7", check_table("tab_sampling.tex", t7, "row",
                                ["size_proportional", "size_min", "size_max",
                                 "readership_weighted", "read_min", "read_max"])),
        ("Table 8", check_table("tab_markets.tex", t8, "mechanism",
                                ["revenue_mean", "revenue_sd", "welfare_mean", "welfare_sd"])),
        ("Table 9", check_table("tab_sensitivity.tex", t9, "quantity",
                                ["setting", "setting_min", "setting_max", "all_alternatives",
                                 "all_min", "all_max", "added_alone", "added_last"])),
    ]
    failed = 0
    for name, bad in checks:
        print(f"{name}: {'matches the paper' if not bad else 'DIFFERS'}")
        for b in bad:
            print("   ", b)
        failed += bool(bad)

    nums = text_numbers(runs_t, oracle, args.sqlshare, ab)
    rows = []
    for key, (got, want) in nums.items():
        token = f"{want}"
        ok = close(got, token) if isinstance(want, float) else int(round(float(got))) == want
        rows.append({"number": key.strip(), "from_files": got, "in_paper": want, "match": ok})
    tn = pd.DataFrame(rows)
    tn.to_csv(os.path.join(args.out, "text_numbers.csv"), index=False)
    n_bad = int((~tn.match).sum())
    print(f"Text numbers: {len(tn) - n_bad} of {len(tn)} match the paper")
    for _, r in tn[~tn.match].iterrows():
        print(f"    {r.number}: paper {r.in_paper}, files {r.from_files}")
    failed += n_bad > 0
    print(f"wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
