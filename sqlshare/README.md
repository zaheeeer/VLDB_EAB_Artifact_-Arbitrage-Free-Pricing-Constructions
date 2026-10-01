# SQLShare pipeline

This package rebuilds every SQLShare number in the paper from the public SQLShare query
log release: the query funnel (975 queries), the verified derivability lattice (806
assets, 2,969 pairs), the falsification of derivability claims, the two support-set
samplers, the full evaluation protocol on the largest market (222 assets), the 15
per-market runs, and the sensitivity analysis of Section 5.3.

## Requirements

- Python 3.11, 3.12 or 3.13, 64-bit (not 3.14: pandas 2.2.3 has no wheel for it).
- About 15 GB of free disk, 8 GB of RAM (16 GB recommended).
- Internet access for the packages and the release (3.6 GB). The release is downloaded
  from https://shrquerylogs.s3.amazonaws.com/public/sqlshare_data_release1.zip and is
  not redistributed here.

Package versions are pinned in `requirements.txt`.

## Run on Windows (PowerShell)

From this folder, in a normal (not administrator) PowerShell window:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Check      # Python, packages, computer check
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -SelfTest   # 68 tests + synthetic end-to-end run
powershell -ExecutionPolicy Bypass -File .\run_all.ps1             # full run, both protocols
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Ablation   # Section 5.3 (after the full run)
```

`-Quick` runs a scaled-down check on the real data. Other options: `-Protocol paper` or
`-Protocol corrected`, `-Zip <path to a downloaded release>`, `-Workers N`, `-Threads N`,
`-Serial`, `-MemoryLimit 8GB`, `-PipIndexUrl <mirror>`, `-SkipTests`.

## Run on Linux or macOS

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest                                   # 68 tests
python -m sqlshare_rerun selftest                  # synthetic end-to-end run
python -m sqlshare_rerun doctor --for-full-run     # computer check
python -m sqlshare_rerun all                       # full run, both protocols
python -m sqlshare_rerun ablation                  # Section 5.3 (after "all")
```

Every command resumes where it stopped; Ctrl+C is safe. `python -m sqlshare_rerun --help`
lists the stages (`download`, `prepare`, `lattice`, `main`, `markets`, `report`) and options.

## Time

The published results were produced on a Windows 11 laptop with 12 logical cores
(Python 3.12, DuckDB 1.5.5), 11 worker processes and 12 sampling threads: about 15
hours for both protocols, most of it in the per-market runs with every alternative on,
and 4.5 hours for the sensitivity analysis.

## Outputs

The run writes to `results/` in this folder (the published copy is `../results/sqlshare/`):

```
results/paper/         the settings of Table 2 of the paper ("the setting")
results/corrected/     every alternative of Table 2 switched on ("all alternatives")
    paper_numbers.json              funnel, lattice, main-market and market summaries
    lattice/summary.json, pairs.csv verified pairs and rewrite-family shares
    main_market/summary.json        claimed, inverted, witnessed and refuted edges; plans
    main_market/runs.csv            one row per arm x valuation family x seed x mechanism
    main_market/summary_by_mechanism.csv   Tables 4 to 6, SQLShare columns
    main_market/table6_support_sampling.csv   Table 7 of the paper
    markets/table7_markets.csv      Table 8 of the paper
    markets/figure2b_markets.csv    Figure 2(b)
    markets/per_market_means.csv    per-market means behind Table 8
results/ablation/      the 30 configurations of Section 5.3 (ablation_long.csv)
results/REPORT.md      the values printed in the paper beside both ends
```

The file names inside the package follow its own table numbering: its Tables 3 to 7 are
Tables 4 to 8 of the paper. To rebuild the paper's tables from a fresh run:

```bash
python ../reproduce/make_tables.py --sqlshare results
```

## The two protocols and the sensitivity switches

`paper` and `corrected` are the code's labels for the two ends of the sensitivity
analysis: `paper` is the setting of the main results (Table 2, middle column) and
`corrected` has every alternative switched on (Table 2, right column). The `ablation` command switches one choice at a time; the switch names
map to the rows of Table 2 as follows:

| switch | Table 2 row | setting | alternative |
|---|---|---|---|
| `C04_demand` | test demand | new popularity ranking | training ranking |
| `C05_fit` | scale fit | grid, multiplier at most 60 | exact, uncapped |
| `C06_entropy` | entropy classes | one per change | one per answer |
| `C07_selection` | selection rewrite | target's value set | target's predicate |
| `C09_querymarket` | QueryMarket substitute | cheapest predecessor | cheapest cover |
| `C11_answer_cells` | answer cells | rows | rows x columns |
| `C12_row_sampling` | deletable rows | rows without a duplicate | every row |
| `alias_resolve` | re-executed queries | those naming the table | also alias readers |
| `remove_all_falsified` | edges removed | falsification sample | also pricing support |
| `metered_x3` | metered time | one timing | median of three |
| `witness_seed`, `ilp_formulation` | checks | | |

`ablation_long.csv` has one row per configuration and quantity: `add_<switch>` is the
setting with one alternative switched on, `drop_<switch>` is every alternative on with
one switched back, and `*_seedset1`, `*_seedset2` repeat the two ends with other sampling
seeds.

## Tests

`python -m pytest` runs 68 tests: unit tests, safety tests (interruption, resumption,
thread and process counts), the sensitivity harness, and an equivalence test.
`tests/test_legacy_equivalence.py` runs the modules of the original pipeline (copies in
`tests/legacy_ref/`) next to this package on a synthetic SQLShare-format release and
requires identical output at every stage, given the same seeded row draws.

## Reproducibility notes

- All sampling is seeded. The same machine gives the same numbers with any number of
  workers or threads and after any number of interruptions (both are tested). A second
  run of the largest market repeated every number of the full run, except timings and
  the compute-metered prices derived from them.
- Build times and compute-metered prices are timings; they depend on the machine.
- The SHA-256 of the release used for the published results is recorded in
  `../results/sqlshare/paper/paper_numbers.json` (`prepare.release.sha256`).
- `sqlshare_rerun/expected_paper.py` and `report.py` hold the SQLShare values printed in
  the paper; the `report` command prints them beside a new run (`compare_with_paper.csv`).

## Troubleshooting

- "running scripts is disabled": start with `powershell -ExecutionPolicy Bypass -File .\run_all.ps1`.
- Download blocked or slow: download the release in a browser and pass `-Zip <path>`
  (Windows) or `--zip <path>` (Linux, macOS).
- Worker processes or threads stopped: run again with `-Serial` (or `--workers 1 --threads 1`).
- Out of memory: `-MemoryLimit 6GB -Workers 2` (or `--memory-limit 6GB --workers 2`).
- Start over: delete `work/`, `results/` and `logs/`; keep `data/` to avoid downloading again.
