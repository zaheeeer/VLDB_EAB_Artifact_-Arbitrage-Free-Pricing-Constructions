# Arbitrage-Free Query Pricing under Recomposition: artifact

Code, result records, and reproduction scripts for the PVLDB paper *Arbitrage-Free
Query Pricing under Recomposition: An Experimental Evaluation of Published Constructions
[Experiment, Analysis & Benchmark]*.

The paper evaluates ten arbitrage-free query-pricing constructions from three published
systems (QueryMarket view-cover pricing, Qirana conflict-set pricing, and the
revenue-maximizing algorithms of Chawla et al.) and five baselines on TPC-H and on the
SQLShare query log, and measures how ten evaluation choices change the results.

## Contents

| folder | what it holds |
|---|---|
| `results/tpch/` | TPC-H records: 300 runs (4 valuation families x 5 seeds x 15 mechanisms) and the oracle stress test |
| `results/sqlshare/` | SQLShare records: `paper/` (the setting of Table 2, used in Sections 5.1 and 5.2), `corrected/` (every alternative of Table 2 on), and `ablation/` (the 30 configurations of Section 5.3) |
| `sqlshare/` | the SQLShare pipeline: download, load, lattice, falsification, samplers, all mechanisms, per-market runs, sensitivity analysis; 68 tests ([sqlshare/README.md](sqlshare/README.md)) |
| `tpch/` | the TPC-H campaign (`run_tpch.py`) and oracle stress test (`run_oracle_stress.py`), with the modules they execute in `tpch/modules/` |

## Quick check: rebuild every table and figure from the records (about a minute)

```bash
pip install -r requirements.txt
python reproduce/make_tables.py
python reproduce/make_figures.py
```

`make_tables.py` prints one line per table and ends with `Text numbers: 86 of 86 match
the paper`. It exits with status 1 if any value differs from the paper at the precision
the paper prints. Outputs go to `reproduce/out/`.

## One command per table and figure

| paper | records | command | output in `reproduce/out/` |
|---|---|---|---|
| Table 4 (revenue, welfare, coverage) | `results/tpch/full_tpch_runs.csv`, `results/sqlshare/paper/main_market/summary_by_mechanism.csv` | `python reproduce/make_tables.py` | `table4_cross.csv` |
| Table 5 (generalization gap) | same | same | `table5_generalization.csv` |
| Table 6 (arbitrage audits) | same | same | `table6_audits.csv` |
| Table 7 (support-set sampling) | `results/sqlshare/paper/main_market/table6_support_sampling.csv`; seed ranges from `results/sqlshare/ablation/ablation_long.csv` | same | `table7_sampling.csv` |
| Table 8 (15 markets) | `results/sqlshare/paper/markets/table7_markets.csv` | same | `table8_markets.csv` |
| Table 9 (sensitivity) | `results/sqlshare/ablation/ablation_long.csv` | same | `table9_sensitivity.csv` |
| numbers in the text | `results/sqlshare/*/paper_numbers.json`, `ablation_long.csv`, `per_market_means.csv`, TPC-H records | same | `text_numbers.csv` |
| Figure 1 (taxonomy) | none | `python reproduce/make_figures.py` | `fig_taxonomy.pdf` |
| Figure 2 (comparisons) | `figure2a.csv`, `figure2b_markets.csv`, `figure2c.csv` | same | `fig_results.pdf` |

Tables 1 to 3 (notation, evaluation choices, port fidelity) hold no measured values.

## Rerunning the experiments

Python 3.11 to 3.13 (not 3.14), with the versions pinned in `requirements.txt`.

### SQLShare (Sections 5.2 and 5.3)

```bash
cd sqlshare
python -m sqlshare_rerun all         # both protocols; on Windows: run_all.ps1
python -m sqlshare_rerun ablation    # the 30 configurations of Section 5.3
python ../reproduce/make_tables.py --sqlshare results
```

The pipeline downloads the public SQLShare release (3.6 GB), needs about 15 GB of disk,
and took about 15 hours plus 4.5 hours for the sensitivity analysis on a 12-core laptop.
[sqlshare/README.md](sqlshare/README.md) gives the Windows commands, the mapping from the
code's switch names to the rows of Table 2, and troubleshooting.

### TPC-H (Section 5.1 and Section 3.2)

```bash
python tpch/run_tpch.py              # 300 runs; writes tpch/out/full_tpch_runs.csv
python tpch/run_oracle_stress.py     # writes tpch/out/oracle_stress_confusion.csv
python reproduce/make_tables.py --tpch tpch/out
```

Both scripts generate TPC-H at scale factor 1 with DuckDB's `tpch` extension, which
DuckDB downloads on first use. `run_tpch.py` builds the 368-asset catalog (Q1 and Q6 by
month) and its 1,046 verified pairs, samples the 400-instance support set, and runs every
mechanism for 4 valuation families and 5 seeds; the acquisition planner re-executes at
most eight cheaper plans per mechanism and run (Section 3.5). The scripts collect the
code of the original TPC-H run, unchanged, into two commands; they were checked end to
end on a small TPC-H instance. The published records in `results/tpch/` come from the
original run at scale factor 1. Compute-metered prices come from measured query times,
so that mechanism's rows depend on the machine.

## Naming

`paper` and `corrected` are the SQLShare code's labels for the two ends of the
sensitivity analysis, used in folder names and on the command line. `paper` is the
setting of the main results (the middle column of Table 2); `corrected` has every
alternative of Table 2 switched on (the right column). Section 5.3 switches one choice
at a time between the two ends and reports which results move.

## Data

- TPC-H is generated by DuckDB's `tpch` extension at scale factor 1; the TPC-H
  specification and data generator are available from https://www.tpc.org.
- SQLShare: Jain et al., SIGMOD 2016 (doi:10.1145/2882903.2882957). The release
  (https://uwescience.github.io/sqlshare/data_release.html) is downloaded by the
  pipeline and is not redistributed here. The SHA-256 of the release used is recorded in
  `results/sqlshare/paper/paper_numbers.json`.
