"""Sensitivity analysis on the main market (the largest SQLShare market), Section 5.3.

Each evaluation choice is switched to its alternative alone, starting from the paper protocol ("add"), and
switched off alone, starting from the corrected protocol ("drop"). Every configuration
runs the whole main-market stage: resolution curve, falsification, witnesses, Table 6,
and Tables 3 to 5. Two checks come with it:

* The two end points ("paper" and "corrected") are run again and compared with the full
  run in results/paper and results/corrected: identical numbers show that the ablation
  measures the same thing, and that a rerun on the same machine repeats every number.
* Replicates with other sampling seeds ("seed sets") give a sampling range for the
  sampled quantities (falsified edges, Table 6, Qirana prices). A switch whose effect on a
  sampled quantity stays inside that range cannot be told apart from sampling noise.

Draws of the support sampler are cached on disk (results/ablation/_samples), keyed by
every input of the draw, so a switch that does not change sampling costs minutes. The
three-timing measurement for compute-metered prices is also taken once and shared, so
timing noise never shows up as the effect of a switch.
Outputs: results/ablation/<configuration>/main_market/, ablation_long.csv,
ablation_headline.csv, and ABLATION.md.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings, make_settings
from .stages import stage_lattice, stage_main_market, stage_prepare
from .util import LOG, read_csv, read_json, stage_done, write_csv, write_json, write_text

# name, label, value in the paper protocol, value in the corrected protocol, what it changes
SWITCHES = [
    ("C04_demand", "C04", {"demand": "redraw", "shift_arm": False}, {"demand": "shared"},
     "test buyers share the training demand instead of a redrawn popularity ranking"),
    ("C05_fit", "C05", {"fit": "grid60"}, {"fit": "breakpoints"},
     "exact, uncapped scale fit instead of a grid capped at 60"),
    ("C06_entropy", "C06", {"entropy": "singleton"}, {"entropy": "exact"},
     "Qirana entropy over real answer classes"),
    ("C07_selection", "C07", {"selection": "value_set"}, {"selection": "predicate"},
     "selection rewrite from the target's WHERE clause instead of its answer values"),
    ("C09_querymarket", "C09", {"querymarket": "single_predecessor"}, {"querymarket": "cover"},
     "QueryMarket prices by the cheapest catalogue cover"),
    ("C11_answer_cells", "C11", {"answer_cells": "rows"}, {"answer_cells": "rows_x_cols"},
     "answer cells counted as rows x columns"),
    ("C12_row_sampling", "C12", {"row_sampling": "unique_only"}, {"row_sampling": "any_row"},
     "any physical row can be deleted, duplicates included"),
    ("alias_resolve", "alias", {"alias_delete": "skip"}, {"alias_delete": "resolve"},
     "a deletion is checked against queries that read the table through an alias view"),
    ("remove_all_falsified", "edges", {"remove_falsified": "curve"}, {"remove_falsified": "all"},
     "edges refuted by the pricing support are also removed"),
    ("metered_x3", "timing", {"metered_repeats": 1}, {"metered_repeats": 3},
     "compute-metered prices from the median of three timings"),
    ("witness_seed", "seed", {"witness_seed": 11}, {"witness_seed": 12},
     "witness sample drawn independently of the curve sample"),
    ("ilp_formulation", "solver", {"ilp_formulation": "all_sources"}, {"ilp_formulation": "cheaper_sources"},
     "cover ILP over the cheaper sources only (same optimum; a check)"),
]
SAMPLING_SEED_FIELDS = ("curve_seed", "witness_seed", "table6_seed")
SEED_STEP = 1000

HEADLINE = [
    ("claimed_edges", "claimed edges"), ("falsified_curve", "falsified in the curve sample"),
    ("witnessed", "witnessed"), ("falsified_by_pricing_support", "falsified by the pricing support"),
    ("false_edges_total", "false edges total"), ("zero_priced_readership", "zero-priced (readership)"),
    ("t6_priceable_size", "Table 6 priceable (size)"), ("t6_priceable_read", "Table 6 priceable (readership)"),
    ("plans_failed", "plans failed"), ("alpha_at_grid_max_share", "fits at the grid cap"),
    ("gaps_negative", "Table 4 gaps below zero"), ("gap_mean", "Table 4 mean gap"),
    ("qm_revenue", "QueryMarket revenue"), ("qm_welfare", "QueryMarket welfare"),
    ("qirana_max_gain", "Qirana max gain"),
]
SAMPLED = {"falsified_curve", "witnessed", "falsified_by_pricing_support", "false_edges_total",
           "zero_priced_readership", "t6_priceable_size", "t6_priceable_read", "t6_median_size",
           "t6_median_read", "t6_levels_size", "t6_levels_read"}


def _config(base: str, root, quick: bool, overrides: dict, name: str, adir: Path, lat: dict,
            common: dict) -> Settings:
    s = make_settings(base, root, quick, **common)
    for k, v in overrides.items():
        setattr(s, k, v)
    s.extra = {"results_dir": str(adir / name), "lattice_dir": str(lat[s.selection]),
               "lattice_ready": True, "sample_cache": str(adir / "_samples")}
    return s


def configurations(root, quick: bool, mode: str, replicates: int, only=None, common=None) -> list:
    """(name, base protocol, overrides, switch, direction) for every configuration."""
    common = common or {}
    out = [("paper", "paper", {}, "", "end point"), ("corrected", "corrected", {}, "", "end point")]
    for name, concern, pv, cv, _ in SWITCHES:
        if only and name not in only:
            continue
        if mode in ("add", "both"):
            out.append((f"add_{name}", "paper", dict(cv), name, "add"))
        if mode in ("drop", "both"):
            out.append((f"drop_{name}", "corrected", dict(pv), name, "drop"))
    base_seeds = make_settings("paper", root, quick, **common)
    corr_seeds = make_settings("corrected", root, quick, **common)
    for r in range(1, replicates + 1):
        for base, ref in (("paper", base_seeds), ("corrected", corr_seeds)):
            ov = {f: getattr(ref, f) + SEED_STEP * r for f in SAMPLING_SEED_FIELDS}
            out.append((f"{base}_seedset{r}", base, ov, "", "seed set"))
    return out


def _settings_diff(a: Settings, b: Settings) -> dict:
    da, db = a.as_dict(), b.as_dict()
    skip = {"extra", "protocol", "root", "zip_path"}
    return {k: (da[k], db[k]) for k in da if k not in skip and da[k] != db[k]}


def _metrics(res_dir: Path) -> dict:
    mm = res_dir / "main_market"
    summ = read_json(mm / "summary.json")
    m = {k: summ.get(k) for k in ("claimed_edges", "falsified_curve", "witnessed",
                                   "falsified_by_pricing_support", "false_edges_total",
                                   "zero_priced_readership", "plans_verified", "plans_failed",
                                   "alpha_at_grid_max_share", "querymarket_catalogue", "clean_pairs_used")}
    t6 = read_csv(mm / "table6_support_sampling.csv")
    for tag, row in (("size", t6.iloc[0]), ("read", t6.iloc[1])):
        m[f"t6_priceable_{tag}"] = int(row.assets_priceable)
        m[f"t6_median_{tag}"] = float(row.median_conflict_set)
        m[f"t6_levels_{tag}"] = int(row.distinct_price_levels)
    sm = read_csv(mm / "summary_by_mechanism.csv")
    arm = summ.get("main_arm") or ("published" if "published" in set(sm.arm) else "stationary")
    g = sm[sm.arm == arm].set_index("mechanism")
    for mech in g.index:
        m[f"revenue:{mech}"] = float(g.loc[mech, "revenue_strategic_norm_mean"])
        m[f"welfare:{mech}"] = float(g.loc[mech, "welfare_strategic_norm_mean"])
        m[f"served:{mech}"] = float(g.loc[mech, "served_strategic_mean"])
        m[f"gap:{mech}"] = float(g.loc[mech, "generalization_gap_mean"])
        m[f"info_arb:{mech}"] = float(g.loc[mech, "mono_violation_rate_mean"])
        m[f"comb_arb:{mech}"] = float(g.loc[mech, "comb_violation_rate_mean"])
        m[f"max_gain:{mech}"] = float(g.loc[mech, "comb_max_gain_mean"])
    gaps = [m[f"gap:{x}"] for x in g.index]
    m["gaps_negative"] = int(sum(1 for x in gaps if x < 0))
    m["gap_mean"] = float(np.mean(gaps)) if gaps else float("nan")
    m["qm_revenue"] = m.get("revenue:querymarket_viewcover")
    m["qm_welfare"] = m.get("welfare:querymarket_viewcover")
    m["qirana_max_gain"] = m.get("max_gain:qirana_weighted_coverage")
    m["main_arm"] = arm
    return m


def _same_as_full_run(s_end: Settings, adir: Path, name: str, common: dict) -> str:
    """Compare an end point with the full run of the same protocol (timings excluded)."""
    full = s_end.root / "results" / name / "main_market"
    mine = adir / name / "main_market"
    ref = make_settings(name, s_end.root, s_end.quick, **common)
    prep = s_end.work_dir / "prepare.done"
    try:
        key = f"{ref.fingerprint()}|{read_json(prep)['key']}"
    except Exception:
        return "no full run to compare with"
    if not stage_done(full / "main.done", key):
        return "no full run with the same settings to compare with"
    bad = []
    a, b = read_json(full / "summary.json"), read_json(mine / "summary.json")
    skip = {"timings_s", "build_cost_ratio"}
    if ref.metered_repeats > 1:        # plan counts include compute-metered prices, which are
        skip |= {"plans_verified", "plans_verified_all_arms", "plans_failed",   # re-measured
                 "plans_failed_by_mechanism"}
    for k in a:
        if k in skip:
            continue
        if a.get(k) != b.get(k):
            bad.append(f"summary.{k}")
    timing = {"elapsed_s", "build_seconds", "seconds"}
    for f in ("runs.csv", "falsified_edges.csv", "resolution_curve.csv", "table6_support_sampling.csv",
              "plan_verification.csv"):
        x, y = read_csv(full / f, keep_default_na=False), read_csv(mine / f, keep_default_na=False)
        x = x[[c for c in x.columns if c not in timing]]
        y = y[[c for c in y.columns if c not in timing]]
        if "mechanism" in x.columns and ref.metered_repeats > 1:   # re-measured timings
            x = x[x.mechanism != "compute_metered"].reset_index(drop=True)
            y = y[y.mechanism != "compute_metered"].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=True)
        except AssertionError:
            bad.append(f)
    return "identical" if not bad else "DIFFERS: " + ", ".join(bad)


def _fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float) and not float(v).is_integer():
        return f"{v:.3f}"
    return f"{int(v):,}" if isinstance(v, (int, float, np.integer)) else str(v)


def run_ablation(root, quick: bool = False, mode: str = "both", replicates: int = 2, only=None,
                 common=None) -> dict:
    names = [n for n, *_ in SWITCHES]
    unknown = [x for x in (only or []) if x not in names]
    if unknown:
        raise ValueError(f"unknown ablation switch {unknown}; choose from {names}")
    common = {k: v for k, v in (common or {}).items() if v not in (None, 0, "")}
    root = Path(root)
    base = make_settings("paper", root, quick, **common)
    stage_prepare(base)
    adir = base.root / "results" / "ablation"
    adir.mkdir(parents=True, exist_ok=True)
    # one lattice per selection rule, shared by every configuration that uses it
    lat = {}
    for sel in ("value_set", "predicate"):
        ls = make_settings("paper", root, quick, **common)
        ls.selection = sel
        ls.extra = {"results_dir": str(adir / f"_lattice_{sel}")}
        stage_lattice(ls)
        lat[sel] = ls.lattice_dir
    configs = configurations(root, quick, mode, replicates, only, common)
    LOG.info("ablation: %d configurations on the main market (mode %s, %d seed sets)",
             len(configs), mode, replicates)
    ends = {"paper": _config("paper", root, quick, {}, "paper", adir, lat, common),
            "corrected": _config("corrected", root, quick, {}, "corrected", adir, lat, common)}
    rows, meta = [], []
    for i, (name, b, ov, switch, direction) in enumerate(configs, 1):
        s = _config(b, root, quick, ov, name, adir, lat, common)
        diff = _settings_diff(ends[b], s)
        LOG.info("ablation %d/%d: %s (%s)", i, len(configs), name,
                 ", ".join(f"{k}={v[1]}" for k, v in diff.items()) or "no change")
        write_json(s.as_dict(), s.results_dir / "settings.json")
        stage_main_market(s)
        m = _metrics(s.results_dir)
        meta.append({"configuration": name, "base": b, "switch": switch, "direction": direction,
                     "changed_settings": "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in diff.items()),
                     "main_arm": m.pop("main_arm")})
        for q, v in m.items():
            rows.append({"configuration": name, "quantity": q, "value": v})
    long = pd.DataFrame(rows)
    info = pd.DataFrame(meta)
    wide = long.pivot(index="quantity", columns="configuration", values="value")

    # deltas against the base of each configuration, and the sampling range per base
    out_rows = []
    for rec in meta:
        name, b = rec["configuration"], rec["base"]
        seeds = [c for c in wide.columns if c == b or c.startswith(f"{b}_seedset")]
        for q in wide.index:
            v, ref = wide.at[q, name], wide.at[q, b]
            vals = pd.to_numeric(wide.loc[q, seeds], errors="coerce").dropna()
            rng = float(vals.max() - vals.min()) if len(vals) > 1 else float("nan")
            try:
                d = float(v) - float(ref)
            except (TypeError, ValueError):
                d = float("nan")
            sampled = q in SAMPLED or q.split(":")[0] in ("revenue", "welfare", "served", "gap", "info_arb",
                                                           "comb_arb", "max_gain")
            beyond = (None if not sampled or math.isnan(rng) or math.isnan(d)
                      else bool(abs(d) > rng + 1e-12))
            out_rows.append({**rec, "quantity": q, "value": v, "base_value": ref, "delta": d,
                             "seed_range_of_base": rng, "beyond_seed_range": beyond})
    res = pd.DataFrame(out_rows)
    write_csv(res, adir / "ablation_long.csv")
    head = wide.loc[[q for q, _ in HEADLINE if q in wide.index]].copy()
    head.index = [dict(HEADLINE)[q] for q in head.index]
    order = [c for c, *_ in configs if c in head.columns]
    head = head[order]
    write_csv(head.reset_index().rename(columns={"index": "quantity"}), adir / "ablation_headline.csv")
    write_csv(info, adir / "ablation_configurations.csv")

    checks = {n: _same_as_full_run(ends[n], adir, n, common) for n in ("paper", "corrected")}
    lines = ["# Ablation on the main market", "",
             "Each evaluation choice switched to its alternative alone from the setting (paper, add_*) "
             "and switched back alone from every alternative on (corrected, drop_*). Seed sets repeat the end points with other "
             "sampling seeds; buyer seeds stay the same, so the seed range measures sampling noise only.",
             "", "End points against the full run (timings excluded):", ""]
    lines += [f"- {n}: {v}" for n, v in checks.items()]
    lines += ["", "Switches:", ""]
    lines += [f"- `{n}` ({c}): {what}" for n, c, _, _, what in SWITCHES]
    lines += ["", "Not ablated here, because they act only on the other markets (Table 7, Figure 2b): "
              "support family of the market runs (C11), empty tables in market samplers, the Table 7 SD "
              "(C14), and the full protocol on every market (C15)."]
    for b in ("paper", "corrected"):
        cols = [c for c in order if info.set_index("configuration").at[c, "base"] == b]
        lines += ["", f"## From the {b} protocol", "",
                  "| quantity | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for q in head.index:
            lines.append(f"| {q} | " + " | ".join(_fmt(head.at[q, c]) for c in cols) + " |")
    lines += ["", "ablation_long.csv has every quantity (Tables 3 to 5 per mechanism included), the "
              "change against the base, the seed range of the base, and whether the change is larger "
              "than that range (sampled quantities only). With few seed sets the range is a rough "
              "guide, not a test."]
    write_text(adir / "ABLATION.md", "\n".join(lines) + "\n")
    LOG.info("ablation report: %s", adir / "ABLATION.md")
    for n, v in checks.items():
        LOG.info("ablation end point %s against the full run: %s", n, v)
    return {"configurations": len(configs), "checks": checks}
