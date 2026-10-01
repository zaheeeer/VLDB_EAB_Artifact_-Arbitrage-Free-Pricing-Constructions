"""Stress test of the TPC-H derivability oracle (Section 3.2 of the paper).

    python tpch/run_oracle_stress.py        # writes tpch/out/oracle_stress_confusion.csv

For two catalog sizes (120 and 200 assets, 14,280 + 39,800 = 54,080 ordered pairs) the
structural rule's candidate pairs are compared against executed reconstructions, which
gives the rule's confusion matrix. The steps are the code of the original run, collected
into one script; the published file is results/tpch/oracle_stress_confusion.csv.
"""
from __future__ import annotations

import argparse
import os

import duckdb
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = os.path.join(HERE, "modules")


def load(name: str) -> None:
    path = os.path.join(MODULES, name)
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), globals())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "out"), help="output folder")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute("CALL dbgen(sf=1)")
    load("probe_tpch_lattice.py")
    load("stress_oracle.py")
    months = month_universe(con)                                    # noqa: F821
    rows = []
    for n in (120, 200):
        A, spine = build_assets_n(months, n, seed=0)                # noqa: F821
        ans = {x: evaluate(con, x) for x in A}                      # noqa: F821
        c = confusion(ans, A, determines, answers_match, TEMPLATES)  # noqa: F821
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in c.items() if k not in ("fp_examples", "fn_examples")})
        rows.append({"test": "candidate rule confusion", "scale": f"{n} assets",
                     "pairs": c["pairs_tested"], "tp": c["true_positive"],
                     "fp": c["false_positive"], "fn": c["false_negative"],
                     "precision": c["precision"], "recall": c["recall"]})
    out_csv = os.path.join(args.out, "oracle_stress_confusion.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
