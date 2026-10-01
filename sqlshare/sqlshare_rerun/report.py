"""Collect every SQLShare number the paper cites, write paper-ready tables, and compare
a new run with the values printed in the paper and with the other protocol."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import expected_paper as V
from .config import Settings
from .util import LOG, read_csv, read_json, stage_done, write_csv, write_json, write_text

STAGE_MARKERS = {"lattice": "lattice.done", "main_market": "main.done", "markets": "markets.done"}


def _current(s: Settings, stage: str) -> bool:
    """True when the stage's outputs were produced with the current settings and data.
    Outputs of other settings (for example an earlier -Quick run) are never reported."""
    prep = s.work_dir / "prepare.done"
    if not prep.exists():
        return False
    try:
        key = f"{s.fingerprint()}|{read_json(prep)['key']}"
    except Exception:
        return False
    ok = stage_done(s.results_dir / stage / STAGE_MARKERS[stage], key)
    if not ok and (s.results_dir / stage).exists():
        LOG.warning("results/%s/%s were made with other settings (for example -Quick); they are not "
                    "reported. Run that stage again with the current settings.", s.protocol, stage)
    return ok


def _exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def _fmt(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def collect(s: Settings) -> dict:
    r = s.results_dir
    nums = {"protocol": s.protocol, "settings": s.as_dict()}
    if _exists(s.work_dir / "prepare_summary.json"):
        nums["prepare"] = read_json(s.work_dir / "prepare_summary.json")
        nums["funnel"] = read_csv(s.work_dir / "funnel.csv").to_dict("records")
    if _exists(r / "lattice" / "summary.json") and _current(s, "lattice"):
        nums["lattice"] = read_json(r / "lattice" / "summary.json")
    if _exists(r / "main_market" / "summary.json") and _current(s, "main_market"):
        nums["main"] = read_json(r / "main_market" / "summary.json")
    if _exists(r / "markets" / "summary.json") and _current(s, "markets"):
        nums["markets"] = read_json(r / "markets" / "summary.json")
    return nums


def _main_arm(summ: pd.DataFrame) -> str:
    for a in ("published", "stationary"):
        if a in set(summ.arm):
            return a
    return summ.arm.iloc[0]


def paper_tables(s: Settings) -> dict:
    """SQLShare columns of Tables 3-7 as CSV and LaTeX rows."""
    r = s.results_dir
    tdir = r / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    out = {}
    main_ok, markets_ok = _current(s, "main_market"), _current(s, "markets")
    stale = ([] if main_ok else ["table3_*", "table4_*", "table5_*", "table6.*"]) + ([] if markets_ok else ["table7.*"])
    for pattern in stale:               # never leave tables from other settings next to current ones
        for f in tdir.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass
    sp = r / "main_market" / "summary_by_mechanism.csv"
    if _exists(sp) and main_ok:
        summ = read_csv(sp)
        for arm in sorted(set(summ.arm)):
            g = summ[summ.arm == arm].set_index("mechanism")
            mechs = [m for m in V.MECH_ORDER if m in g.index]
            t3 = pd.DataFrame({"mechanism": mechs,
                               "revenue": [g.loc[m, "revenue_strategic_norm_mean"] for m in mechs],
                               "revenue_sd": [g.loc[m, "revenue_strategic_norm_sd"] for m in mechs],
                               "welfare": [g.loc[m, "welfare_strategic_norm_mean"] for m in mechs],
                               "welfare_sd": [g.loc[m, "welfare_strategic_norm_sd"] for m in mechs],
                               "served": [g.loc[m, "served_strategic_mean"] for m in mechs]})
            t4 = pd.DataFrame({"mechanism": mechs,
                               "generalization_gap": [g.loc[m, "generalization_gap_mean"] for m in mechs],
                               "generalization_gap_sd": [g.loc[m, "generalization_gap_sd"] for m in mechs]})
            t5 = pd.DataFrame({"mechanism": mechs,
                               "information_arbitrage": [g.loc[m, "mono_violation_rate_mean"] for m in mechs],
                               "combination_arbitrage": [g.loc[m, "comb_violation_rate_mean"] for m in mechs],
                               "max_gain_mean_of_run_maxima": [g.loc[m, "comb_max_gain_mean"] for m in mechs],
                               "max_gain_global": [g.loc[m, "comb_max_gain_global"] for m in mechs]})
            for name, df in (("table3", t3), ("table4", t4), ("table5", t5)):
                write_csv(df, tdir / f"{name}_sqlshare_{arm}.csv")
            tex = []
            for m in mechs:
                tex.append(f"{V.PAPER_NAME[m]} & {_fmt(g.loc[m, 'revenue_strategic_norm_mean'])} & "
                           f"{_fmt(g.loc[m, 'welfare_strategic_norm_mean'])} & "
                           f"{_fmt(g.loc[m, 'served_strategic_mean'], 2)} \\\\")
            write_text(tdir / f"table3_sqlshare_{arm}.tex", "\n".join(tex) + "\n")
            tex = [f"{V.PAPER_NAME[m]} & {_fmt(g.loc[m, 'generalization_gap_mean'], sign=True)} \\\\" for m in mechs]
            write_text(tdir / f"table4_sqlshare_{arm}.tex", "\n".join(tex) + "\n")
            tex = [f"{V.PAPER_NAME[m]} & {_fmt(g.loc[m, 'mono_violation_rate_mean'])} & "
                   f"{_fmt(g.loc[m, 'comb_violation_rate_mean'])} & {_fmt(g.loc[m, 'comb_max_gain_mean'])} \\\\"
                   for m in mechs]
            write_text(tdir / f"table5_sqlshare_{arm}.tex", "\n".join(tex) + "\n")
            out[arm] = {"table3": t3, "table4": t4, "table5": t5}
    t6p = r / "main_market" / "table6_support_sampling.csv"
    if _exists(t6p) and main_ok:
        t6 = read_csv(t6p)
        write_csv(t6, tdir / "table6.csv")
        tex = [f"{row.sampling} & {row.assets_priceable} / {row.assets} & {row.median_conflict_set:.0f} & "
               f"{row.distinct_price_levels} & {row.build_seconds:.0f} \\\\" for row in t6.itertuples()]
        write_text(tdir / "table6.tex", "\n".join(tex) + "\n")
        out["table6"] = t6
    t7p = r / "markets" / "table7_markets.csv"
    if _exists(t7p) and markets_ok:
        t7 = read_csv(t7p)
        sd = "across_markets" if s.report_sd == "across_markets" else "pooled"
        tex = [f"{V.PAPER_NAME[row.mechanism]} & {row.revenue_mean:.3f} $\\pm$ {getattr(row, 'revenue_sd_' + sd):.3f} & "
               f"{row.welfare_mean:.3f} $\\pm$ {getattr(row, 'welfare_sd_' + sd):.3f} \\\\" for row in t7.itertuples()]
        write_text(tdir / "table7.tex", "\n".join(tex) + "\n")
        write_csv(t7, tdir / "table7.csv")
        out["table7"] = t7
    return out


def compare_paper(s: Settings, nums: dict, tabs: dict) -> pd.DataFrame:
    rows = []

    def add(key, ref, new):
        diff = None
        try:
            if ref is not None and new is not None:
                diff = float(new) - float(ref)
        except (TypeError, ValueError):
            pass
        rows.append({"quantity": key, "paper": ref, "rerun": new, "difference": diff})

    f = {x["stage"]: x["n"] for x in nums.get("funnel", [])}
    add("funnel: queries in release", 11121, f.get("queries in release"))
    add("funnel: all relations resolvable", 6398, f.get("all relations resolvable"))
    add("funnel: in a selected market component", 2660, f.get("in a selected market component"))
    add("funnel: binds", 1135, f.get("binds against loaded schema"))
    add("funnel: executes", 1051, f.get("executes successfully"))
    add("funnel: non-empty answers (the 975 queries)", 975, f.get("returns a non-empty answer"))
    lat = nums.get("lattice", {})
    add("lattice: markets with >=20 survivors", 15, lat.get("markets_structural"))
    add("lattice: materialized assets", 806, lat.get("materialized_assets"))
    add("lattice: verified pairs", 2969, lat.get("verified_pairs"))
    add("lattice: pairs per asset", 3.68, lat.get("pairs_per_asset"))
    sh = lat.get("family_share_pct", {})
    add("lattice: projection share %", 71.4, sh.get("projection"))
    add("lattice: selection share %", 27.8, sh.get("selection"))
    add("lattice: roll-up share %", 0.7, sh.get("rollup"))
    m = nums.get("main", {})
    add("main: assets", 222, m.get("assets"))
    add("main: claimed edges", 2047, m.get("claimed_edges"))
    add("main: falsified at 30,000", 33, m.get("falsified_curve"))
    add("main: witnessed in a 4,000 sample", 20, m.get("witnessed"))
    add("main: falsified by pricing support", 12, m.get("falsified_by_pricing_support"))
    add("main: false edges total", 45, m.get("false_edges_total"))
    add("main: false-edge rate %", 2.2, m.get("false_edge_rate_pct"))
    add("main: corr(size, readership)", -0.088, m.get("corr_size_vs_readership"))
    add("main: zero-priced under readership sampling", 60, m.get("zero_priced_readership"))
    add("main: build-cost ratio", 113, m.get("build_cost_ratio"))
    add("main: plans re-executed", 3240, m.get("plans_verified", 0) + m.get("plans_failed", 0) if m else None)
    add("main: plans failed", 12, m.get("plans_failed"))
    mk = nums.get("markets", {})
    add("markets: ok", 15, mk.get("markets_ok"))
    add("markets: assets", 812, mk.get("assets"))
    add("markets: pairs", 2971, mk.get("pairs"))
    add("markets: falsified pairs", 42, mk.get("falsified"))
    add("markets: with a falsified pair", 4, mk.get("markets_with_falsified"))
    add("markets: published wins welfare", 15, mk.get("welfare_wins_published"))
    add("markets: flat wins revenue", 13, mk.get("revenue_wins_flat"))
    add("markets: zero-priced median %", 31.7, mk.get("zero_priced_median_pct"))
    add("markets: all zero-priced", 4, mk.get("markets_all_zero_priced"))
    arm = "published" if "published" in tabs else ("stationary" if "stationary" in tabs else None)
    if arm:
        t3 = tabs[arm]["table3"].set_index("mechanism")
        t4 = tabs[arm]["table4"].set_index("mechanism")
        t5 = tabs[arm]["table5"].set_index("mechanism")
        for mm in V.MECH_ORDER:
            if mm not in t3.index:
                continue
            e3, e4, e5 = V.TABLE3[mm], V.TABLE4[mm], V.TABLE5[mm]
            add(f"Table 4 revenue: {V.PAPER_NAME[mm]}", e3[0], round(t3.loc[mm, "revenue"], 3))
            add(f"Table 4 welfare: {V.PAPER_NAME[mm]}", e3[1], round(t3.loc[mm, "welfare"], 3))
            add(f"Table 4 served: {V.PAPER_NAME[mm]}", e3[2], round(t3.loc[mm, "served"], 2))
            add(f"Table 5 gap: {V.PAPER_NAME[mm]}", e4, round(t4.loc[mm, "generalization_gap"], 3))
            add(f"Table 6 info-arb: {V.PAPER_NAME[mm]}", e5[0], round(t5.loc[mm, "information_arbitrage"], 3))
            add(f"Table 6 comb-arb: {V.PAPER_NAME[mm]}", e5[1], round(t5.loc[mm, "combination_arbitrage"], 3))
            add(f"Table 6 max gain: {V.PAPER_NAME[mm]}", e5[2], round(t5.loc[mm, "max_gain_mean_of_run_maxima"], 3))
    if "table6" in tabs:
        t6 = tabs["table6"]
        for i, key in enumerate(("size_proportional", "readership_weighted")):
            if i < len(t6):
                e = V.TABLE6[key]
                add(f"Table 7 priceable: {key}", e[0], int(t6.iloc[i].assets_priceable))
                add(f"Table 7 median |C|: {key}", e[1], float(t6.iloc[i].median_conflict_set))
                add(f"Table 7 price levels: {key}", e[2], int(t6.iloc[i].distinct_price_levels))
                add(f"Table 7 build seconds (machine-dependent): {key}", e[3], float(t6.iloc[i].build_seconds))
    if "table7" in tabs:
        t7 = tabs["table7"].set_index("mechanism")
        for mm, e in V.TABLE7.items():
            if mm in t7.index:
                add(f"Table 8 revenue mean: {V.PAPER_NAME[mm]}", e[0], round(t7.loc[mm, "revenue_mean"], 3))
                add(f"Table 8 revenue sd across markets: {V.PAPER_NAME[mm]}", e[1], round(t7.loc[mm, "revenue_sd_across_markets"], 3))
                add(f"Table 8 welfare mean: {V.PAPER_NAME[mm]}", e[2], round(t7.loc[mm, "welfare_mean"], 3))
                add(f"Table 8 welfare sd across markets: {V.PAPER_NAME[mm]}", e[3], round(t7.loc[mm, "welfare_sd_across_markets"], 3))
    return pd.DataFrame(rows)


def stage_report(s: Settings) -> dict:
    nums = collect(s)
    tabs = paper_tables(s)
    cmp = compare_paper(s, nums, tabs)
    s.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(cmp, s.results_dir / "compare_with_paper.csv")
    write_json(nums, s.results_dir / "paper_numbers.json")
    LOG.info("report (%s): %s", s.protocol, s.results_dir / "compare_with_paper.csv")
    return {"numbers": nums, "comparison": cmp}


def cross_protocol(root: Path) -> Path | None:
    a = root / "results" / "paper" / "compare_with_paper.csv"
    b = root / "results" / "corrected" / "compare_with_paper.csv"
    if not (a.exists() and b.exists()):
        return None
    pa, pc = read_csv(a), read_csv(b)
    m = pa.merge(pc[["quantity", "rerun"]], on="quantity", how="left", suffixes=("_paper", "_corrected"))
    m = m[["quantity", "paper", "rerun_paper", "rerun_corrected"]].rename(
        columns={"rerun_paper": "paper protocol", "rerun_corrected": "corrected protocol"})
    out = root / "results" / "protocol_comparison.csv"
    write_csv(m, out)
    def cell(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        try:
            f = float(x)
        except (TypeError, ValueError):
            return str(x)
        return f"{int(f):,}" if f.is_integer() and abs(f) >= 10 else f"{f:.3f}".rstrip("0").rstrip(".")

    lines = ["# SQLShare run: values in the paper, the setting, and every alternative on", "",
             "The 'paper' column holds the values the paper prints for the setting of Table 2.",
             "Machine-dependent values (build seconds, compute-metered prices) differ between machines.",
             "With every alternative on ('corrected' in the code), values differ by design (Section 5.3);",
             "its Tables 4 to 6 values are the arm in which test buyers share the training demand.", "",
             "| quantity | paper | setting (paper protocol) | every alternative on |", "|---|---|---|---|"]
    quick = any(read_json(p).get("quick") for p in (root / "results" / "paper" / "settings.json",
                                                    root / "results" / "corrected" / "settings.json")
                if p.exists())
    if quick:
        lines[2:2] = ["**QUICK RUN: sample sizes are scaled down. Do not use these numbers in the paper.**", ""]
    for row in m.itertuples(index=False):
        lines.append(f"| {row[0]} | {cell(row[1])} | {cell(row[2])} | {cell(row[3])} |")
    write_text(root / "results" / "REPORT.md", "\n".join(lines) + "\n")
    return out
