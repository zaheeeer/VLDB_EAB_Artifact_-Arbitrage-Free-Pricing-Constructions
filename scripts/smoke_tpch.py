"""New SF0.001 validation fixture, NOT a reproduction of the SF1 experiment."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "legacy"), str(ROOT / "candidate")]
from probe_tpch_lattice import (Asset, TEMPLATES, evaluate, answers_match,
                                 determines, derive)
from tpc_substrate import plan_all
from protocol import buyer_split, fit_scale_breakpoints, verified_effective_prices


def run(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute("CALL dbgen(sf=0.001)")
    # Choose two months that have positive Q6 support, avoiding empty-sum behavior
    # in this narrow fixture. Empty and NULL aggregate semantics remain a gate.
    months = [r[0] for r in con.execute(
        "SELECT CAST(date_trunc('month',l_shipdate) AS DATE) AS m FROM lineitem "
        "WHERE l_discount BETWEEN .05 AND .07 AND l_quantity < 24 "
        "GROUP BY 1 HAVING count(*) > 0 ORDER BY 1 LIMIT 2").fetchall()]
    assert len(months) == 2
    assets = []
    for template in ("q1", "q6"):
        dims = frozenset(TEMPLATES[template]["group_dims"])
        assets.extend([Asset(template, dims, frozenset([months[0]])),
                       Asset(template, dims, frozenset([months[1]])),
                       Asset(template, frozenset(), frozenset(months)),
                       Asset(template, dims, frozenset(months))])
    answers = {asset: evaluate(con, asset) for asset in assets}
    prices = np.array([2., 3., 20., 25., 2., 3., 20., 25.])
    plans = plan_all(assets, answers, prices, TEMPLATES, answers_match, verify_limit=None)
    assert plans.verified.eq(True).sum() == 4
    assert plans.verified.eq(False).sum() == 0
    effective = verified_effective_prices(prices, plans.cover_cost, plans.verified)
    np.testing.assert_array_equal(effective, [2., 3., 5., 5., 2., 3., 5., 5.])
    # Deliberately corrupt one reconstruction source; the cheaper invalid plan
    # must be rejected by verification and leave the target's posted price intact.
    bad_answers = {a: df.copy(deep=True) for a, df in answers.items()}
    bad_answers[assets[0]].loc[0, "sum_qty"] += 100
    bad = plan_all(assets, bad_answers, prices, TEMPLATES, answers_match, verify_limit=None)
    assert bad.loc[2, "verified"] == False
    safe = verified_effective_prices(prices, bad.cover_cost, bad.verified)
    assert safe[2] == prices[2]
    capped = plan_all(assets, answers, prices, TEMPLATES, answers_match, verify_limit=1)
    safe_capped = verified_effective_prices(prices, capped.cover_cost, capped.verified)
    assert safe_capped[6] == prices[6]
    pair_checks = []
    for i, src in enumerate(assets):
        for j, tgt in enumerate(assets):
            if i != j and determines(src, tgt):
                pair_checks.append(bool(answers_match(derive(answers[src], src, tgt), answers[tgt])))
    assert len(pair_checks) == 6 and all(pair_checks)
    split = buyer_split(np.ones(len(assets)), 37, 40, 40)
    alpha = fit_scale_breakpoints(*split["train"], np.ones(len(assets)))
    plans.to_csv(outdir / "smoke_tpch_plans.csv", index=False)
    result = {"purpose": "small correctness fixture; not paper evidence or a full rerun",
              "scale_factor": .001, "lineitem_rows": con.execute("SELECT count(*) FROM lineitem").fetchone()[0],
              "assets": len(assets), "templates": list(TEMPLATES), "months": [str(m) for m in months],
              "cheaper_plans_verified": int(plans.verified.eq(True).sum()),
              "cheaper_plans_failed": int(plans.verified.eq(False).sum()),
              "single_source_pair_checks": len(pair_checks),
              "corrupted_plan_rejected": bool(safe[2] == prices[2]),
              "unchecked_plan_excluded": bool(safe_capped[6] == prices[6]),
              "flat_scale_smoke": alpha, "elapsed_seconds": time.perf_counter()-start,
              "threads": 1, "python": platform.python_version(), "duckdb": duckdb.__version__,
              "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
              "platform": platform.platform(),
              "limitations": ["eight assets, two nonempty months, one small database",
                              "does not validate all mechanisms or published guarantees",
                              "legacy solver cannot distinguish timeout from infeasibility",
                              "NULL/empty aggregate semantics not covered",
                              "full buyer protocol and sampler not integrated into original drivers"]}
    (outdir / "smoke_tpch_result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation")
    run(parser.parse_args().output)
