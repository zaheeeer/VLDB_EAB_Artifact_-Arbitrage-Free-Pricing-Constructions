Reference code of the original SQLShare pipeline, used only by
tests/test_legacy_equivalence.py.

These files are unmodified copies of the modules of the original pipeline
(market_run.py, run_full.py, ports_chawla.py, pipeline.py, sqlshare_lattice.py,
sqlshare_load.py, sqlshare_retention.py, sqlshare_translate.py, tpc_experiment.py) and
of helper functions taken from the notebooks of the original run
(recovered_original_helpers.py, each function marked with its source cell).

Do not edit them. The tests run this package and this code side by side on a synthetic
SQLShare release and require identical results.
