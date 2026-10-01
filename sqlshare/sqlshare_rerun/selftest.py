"""A small synthetic release in the exact SQLShare layout, and an end-to-end run on it.

The fixture exercises every code path the real data uses: T-SQL translation (TOP, ISNULL,
CONVERT, string '+'), views, alias names, unsupported PATINDEX, missing relations,
duplicate rows, projections, selections, roll-ups, joins, and two markets.
"""
from __future__ import annotations

import csv
import faulthandler
import io
import os
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

REL = "sqlshare_data_release1"
SEP = "\n\n\n" + "_" * 40 + "\n\n\n"


def _csv(header, rows, delim=","):
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delim, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _extra_owner(o: int, rng, scale: int):
    """Files, views and queries of one more synthetic owner, for stress runs (scale > 1)."""
    own = f"9{o:03d}"
    sites = [f"P{i}" for i in range(1, 6 + o % 5)]
    n = 400 * scale
    obs = [[sites[i % len(sites)], int(rng.integers(1990, 2020)), int(1 + i % 12),
            round(float(rng.normal(10, 4)), 3), int(rng.integers(0, 40)),
            ("q" + str(int(rng.integers(0, 4)))) if i % 11 else ""] for i in range(n)]
    obs += obs[: 3 + o % 4]                                    # duplicates
    meta = [[s, f"basin-{j % 3}", round(30 + j * 0.37, 2)] for j, s in enumerate(sites)]
    tiny = [[s, 1 if j == 0 else 0] for s in sites for j in range(2)]
    files = {
        f"{REL}/data/{own}/table_obs_{o}.csv": _csv(["site", "yr", "mon", "val", "cnt", "flag"], obs),
        f"{REL}/data/{own}/table_meta_{o}.tab": _csv(["site", "basin", "lat"], meta),
        f"{REL}/data/{own}/table_tiny_{o}.csv": _csv(["site", "active"], tiny),
        f"{REL}/data/{own}/table_empty_{o}.csv": _csv(["site", "z"], []),
    }
    obs_t, meta_t, tiny_t = f"[{own}].[obs_{o}.csv]", f"[{own}].[meta_{o}.tab]", f"[{own}].[tiny_{o}.csv]"
    views = [f"create view [{own}].[recent_{o}] ([site],[yr],[val]) as select site, yr, val from {obs_t} "
             f"where yr >= 2005"]
    q = [f"SELECT * FROM {obs_t}", f"SELECT site, yr FROM {obs_t}", f"SELECT DISTINCT site FROM {obs_t}",
         f"SELECT site, yr, mon FROM {obs_t}", f"SELECT site, sum(val) AS v FROM {obs_t} GROUP BY site",
         f"SELECT site, yr, sum(cnt) AS c FROM {obs_t} GROUP BY site, yr",
         f"SELECT yr, count(*) AS n FROM {obs_t} GROUP BY yr", f"SELECT TOP 25 * FROM {obs_t}",
         f"SELECT site, ISNULL(flag, 'none') AS f FROM {obs_t}", f"SELECT * FROM [{own}].[recent_{o}]",
         f"SELECT site FROM [{own}].[recent_{o}] WHERE val > 12",
         f"SELECT o.site, m.basin FROM {obs_t} o JOIN {meta_t} m ON o.site = m.site",
         f"SELECT m.basin, sum(o.cnt) AS c FROM {obs_t} o JOIN {meta_t} m ON o.site = m.site GROUP BY m.basin",
         f"SELECT * FROM {meta_t}", f"SELECT site, lat FROM {meta_t}", f"SELECT DISTINCT site FROM {tiny_t} WHERE active = 1",
         f"SELECT site FROM {meta_t}", f"SELECT * FROM [{own}].[empty_{o}.csv]",
         f"SELECT site, 'y' + CONVERT(varchar(8), yr) AS label FROM {obs_t}",
         f"SELECT PATINDEX('%1%', site) AS p FROM {obs_t}"]
    for s in sites:
        q.append(f"SELECT site, yr, val FROM {obs_t} WHERE site = '{s}'")
        q.append(f"SELECT yr, mon FROM {obs_t} WHERE site = '{s}' AND yr > 2000")
    for y in range(1990, 2020, 3 + (o % 3)):
        q.append(f"SELECT site, yr FROM {obs_t} WHERE yr = {y}")
        q.append(f"SELECT site, sum(val) AS v FROM {obs_t} WHERE yr >= {y} GROUP BY site")
    return files, views, q


def build_fixture(zip_path: Path, seed: int = 0, scale: int = 1) -> Path:
    """The synthetic release. ``scale`` > 1 adds ``4 * scale`` more owners with larger
    tables and about 50 queries each, for stress runs; scale 1 is the self-test fixture."""
    rng = np.random.default_rng(seed)
    stations = [f"S{i}" for i in range(1, 9)]
    species = list("ABCDEF")
    ctd = []
    for i in range(240):
        st = stations[i % len(stations)]
        ctd.append([st, int(rng.integers(1, 120)), round(float(rng.normal(12, 3)), 3),
                    round(float(rng.normal(34, 1)), 3) if i % 17 else "", int(1 + i % 6)])
    ctd += [ctd[3], ctd[3], ctd[10]]                     # exact duplicate rows
    bio = [[stations[i % 8], species[int(rng.integers(0, 6))], int(rng.integers(1, 50)),
            int(rng.integers(1, 120))] for i in range(180)]
    key = [[st, round(40 + 0.5 * i, 2), round(-120 - 0.7 * i, 2)] for i, st in enumerate(stations)]
    # An edge that holds on this instance but not in general: every species appears in
    # species_list, and each appears exactly once with flag = 1 in tiny. The projection of
    # species_list equals the flagged species of tiny, until a flagged row is deleted.
    species_list = [[sp, f"name-{sp}"] for sp in species]
    tiny = [[sp, 1 if j == 0 else 0, j] for sp in species for j in range(3)]
    regions, products = ["north", "south", "east", "west"], ["p1", "p2", "p3", "p4", "p5"]
    sales = [[regions[i % 4], products[int(rng.integers(0, 5))], int(rng.integers(1, 20)),
              round(float(rng.uniform(1, 9)), 2), int(1 + i % 12)] for i in range(300)]

    files = {
        f"{REL}/data/446/table_ctd.csv": _csv(["station", "depth", "temp", "salinity", "month"], ctd),
        f"{REL}/data/446/table_bio.csv": _csv(["station", "species", "count", "depth"], bio),
        f"{REL}/data/446/table_key.csv": _csv(["station", "lat", "lon"], key),
        f"{REL}/data/446/table_species_list.csv": _csv(["species", "label"], species_list),
        f"{REL}/data/446/table_tiny.csv": _csv(["species", "flag", "rep"], tiny),
        # a .tab name holding comma-separated text, as the real release does
        f"{REL}/data/777/table_sales.tab": _csv(["region", "product", "qty", "price", "month"], sales),
        f"{REL}/data/999/table_unused.csv": _csv(["a", "b"], [[1, 2], [3, 4]]),
        # file-name quirks seen in the real release
        f"{REL}/data/446/table_event1002.csvB3D88": _csv(["station", "event", "depth"],
                                                         [[stations[i % 8], i, 10 + i] for i in range(40)]),
        f"{REL}/data/446/materialized_Snapshot of ctd": _csv(["station", "depth"], [r[:2] for r in ctd[:60]]),
        f"{REL}/data/446/table_odd:name?.csv": _csv(["station", "x"], [[s, i] for i, s in enumerate(stations)]),
        f"{REL}/data/446/table_Case.csv": _csv(["v"], [[1], [2]]),
        f"{REL}/data/446/table_case.csv": _csv(["v"], [[7], [8], [9]]),
        f"{REL}/data/446/table_semi.csv": _csv(["station", "q"], [[s, i] for i, s in enumerate(stations)], ";"),
        f"{REL}/data/446/table_empty.csv": _csv(["station", "y"], []),
        # the historical loader could not read a file with a quote in its name, and the
        # historical delete could not remove a row from a table with a quote in a column name
        f"{REL}/data/446/table_bob's data.csv": _csv(["station", "n"], [[s, i] for i, s in enumerate(stations)]),
        f"{REL}/data/446/table_quoted.csv": _csv(["station", 'q"x', "v"],
                                                 [[stations[i % 8], i % 3, i] for i in range(30)]),
        f"{REL}/data/1059/table_seaflow.tab": _csv(["lat", "lon", "temp"],
                                                  [[40 + i / 10, -120 - i / 10, 10 + i % 5] for i in range(50)]),
    }
    views = [
        "create view [446].[ctd_view] ([station],[depth],[temp]) as select station, depth, temp\n"
        "from [446].[ctd.csv]",
        "create view [446].[deep_view] ([station],[depth]) as select station, depth from [446].[ctd_view] "
        "where depth > 60",
        "create view [777].[north_sales] ([product],[qty]) as select product, qty from [777].[sales.tab] "
        "where region = 'north'",
        # views that fail, as many did in the real release
        "create view [446].[bad_view] ([x]) as select nosuchcol as x from [446].[ctd.csv]",
        "create view [446].[pat_view] ([p]) as select PATINDEX('%1%', station) as p from [446].[ctd.csv]",
    ]
    q446 = [
        "SELECT * FROM [446].[ctd.csv]",
        "SELECT station, depth FROM [446].[ctd.csv]",
        "SELECT DISTINCT station FROM [446].[ctd.csv]",
        "SELECT station, depth, temp FROM [446].[ctd.csv] WHERE station = 'S1'",
        "SELECT station, depth FROM [446].[ctd.csv] WHERE station = 'S2'",
        "SELECT station, depth FROM [446].[ctd.csv] WHERE station IN ('S1','S2')",
        "SELECT station, sum(temp) AS t FROM [446].[ctd.csv] GROUP BY station",
        "SELECT station, month, sum(temp) AS t FROM [446].[ctd.csv] GROUP BY station, month",
        "SELECT month, count(*) AS n FROM [446].[ctd.csv] GROUP BY month",
        "SELECT TOP 10 * FROM [446].[ctd.csv]",
        "SELECT station, ISNULL(salinity, 0) AS s FROM [446].[ctd.csv]",
        "SELECT 'st-' + station AS label FROM [446].[ctd.csv]",
        "SELECT CONVERT(varchar(10), depth) AS d FROM [446].[ctd.csv]",
        "SELECT * FROM [446].[ctd_view]",
        "SELECT station, depth FROM [446].[ctd_view] WHERE depth > 10",
        "SELECT * FROM [446].[deep_view]",
        "SELECT station FROM [446].[deep_view]",
        "SELECT b.species, k.lat FROM [446].[bio.csv] b JOIN [446].[key.csv] k ON b.station = k.station",
        "SELECT species, sum(count) AS n FROM [446].[bio.csv] GROUP BY species",
        "SELECT station, species, sum(count) AS n FROM [446].[bio.csv] GROUP BY station, species",
        "SELECT station, species FROM [446].[bio.csv] WHERE species IN ('A','B')",
        "SELECT station, species FROM [446].[bio.csv]",
        "SELECT * FROM [446].[bio.csv] WHERE depth < 50",
        "SELECT * FROM [446].[bio.csv]",
        "SELECT station, lat FROM [446].[key.csv]",
        "SELECT * FROM [446].[key.csv]",
        "SELECT c.station, c.temp, k.lon FROM [446].[ctd.csv] c JOIN [446].[key.csv] k ON c.station = k.station",
        "SELECT station, depth FROM [446].[table_ctd]",
        "SELECT species FROM [446].[species_list.csv]",
        "SELECT DISTINCT species FROM [446].[tiny.csv] WHERE flag = 1",
        "SELECT t.species, s.label FROM [446].[tiny.csv] t JOIN [446].[species_list.csv] s ON t.species = s.species",
        "SELECT b.station, s.label FROM [446].[bio.csv] b JOIN [446].[species_list.csv] s ON b.species = s.species",
        "SELECT station, depth FROM [446].[event1002.csvB3D88]",
        "SELECT station FROM [446].[event1002.csvB3D88] WHERE depth > 30",
        "SELECT e.station, c.temp FROM [446].[event1002.csvB3D88] e JOIN [446].[ctd.csv] c ON e.station = c.station",
        "SELECT * FROM [446].[Snapshot of ctd]",
        "SELECT station FROM [446].[Snapshot of ctd] WHERE depth > 50",
        "SELECT s.station, c.depth FROM [446].[Snapshot of ctd] s JOIN [446].[ctd.csv] c ON s.station = c.station",
        "SELECT station, x FROM [446].[odd:name?.csv]",
        "SELECT o.station, c.depth FROM [446].[odd:name?.csv] o JOIN [446].[ctd.csv] c ON o.station = c.station",
        "SELECT * FROM [446].[semi.csv]",
        "SELECT s.station, c.depth FROM [446].[semi.csv] s JOIN [446].[ctd.csv] c ON s.station = c.station",
        "SELECT * FROM [446].[empty.csv]",
        "SELECT e.station, c.depth FROM [446].[empty.csv] e JOIN [446].[ctd.csv] c ON e.station = c.station",
        "SELECT station FROM [446].[ctd.csv] WHERE CAST(station AS INTEGER) > 3",   # binds, fails to run
        "SELECT * FROM [446].[bad_view]",
        "SELECT * FROM [446].[pat_view]",
        "SELECT * FROM [446].[bob's data.csv]",
        "SELECT b.station, c.depth FROM [446].[bob's data.csv] b JOIN [446].[ctd.csv] c ON b.station = c.station",
        "SELECT * FROM [446].[quoted.csv]",
        "SELECT station, v FROM [446].[quoted.csv]",
        "SELECT station, sum(v) AS sv FROM [446].[quoted.csv] GROUP BY station",
        "SELECT q.station, k.lat FROM [446].[quoted.csv] q JOIN [446].[key.csv] k ON q.station = k.station",
        "WITH d AS (SELECT station, max(depth) AS md FROM [446].[ctd.csv] GROUP BY station) SELECT * FROM d",
        "SELECT c.station, s.lat FROM [446].[ctd.csv] c JOIN [1059].[seaflow.tab] s ON c.depth = s.temp",
        "SELECT PATINDEX('%1%', station) AS p FROM [446].[ctd.csv]",
        "SELECT * FROM [446].[missing.csv]",
        "SELECT station FROM [446].[ctd.csv] WHERE depth > 1000",
    ]
    q777 = [
        "SELECT * FROM [777].[sales.tab]",
        "SELECT region, product FROM [777].[sales.tab]",
        "SELECT DISTINCT region FROM [777].[sales.tab]",
        "SELECT region, product, qty FROM [777].[sales.tab] WHERE region = 'north'",
        "SELECT region, product FROM [777].[sales.tab] WHERE region = 'south'",
        "SELECT region, sum(qty) AS q FROM [777].[sales.tab] GROUP BY region",
        "SELECT region, product, sum(qty) AS q FROM [777].[sales.tab] GROUP BY region, product",
        "SELECT product, sum(qty) AS q FROM [777].[sales.tab] GROUP BY product",
        "SELECT * FROM [777].[north_sales]",
        "SELECT product FROM [777].[north_sales]",
        "SELECT region, month FROM [777].[sales.tab] WHERE month <= 6",
        "SELECT region, product, price FROM [777].[sales.tab] WHERE price > 5",
    ]
    queries = q446 + q777 + ["SELECT 1", "SELECT * FROM [999].[nothing]"]
    if scale > 1:
        extra_rng = np.random.default_rng(seed + 1)
        for o in range(4 * scale):
            f_, v_, q_ = _extra_owner(o, extra_rng, scale)
            files.update(f_)
            views += v_
            queries += q_
    files[f"{REL}/queries.txt"] = SEP.join(queries) + SEP
    files[f"{REL}/view_script.txt"] = SEP.join(views) + SEP
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
    return zip_path


SELFTEST_SETTINGS = dict(
    min_component_queries=5, min_lattice_survivors=3, market_min_assets=3,
    curve_checkpoints=(50, 300), witness_n=200, table6_n=80, run_seeds=(0, 1), n_buyers=200,
    max_lps=8, market_support_n=40, market_seeds=(0,), market_buyers=100, market_max_lps=5,
)


def _remove_tree(path: Path) -> Path:
    """Remove an old run folder; if Windows keeps a file locked, use a fresh folder instead."""
    for _ in range(3):
        if not path.exists():
            return path
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return path
        time.sleep(2)
    return path.with_name(f"{path.name}_{time.strftime('%Y%m%d_%H%M%S')}")


def _too_long(root: Path, minutes: float) -> None:
    msg = (f"SELFTEST FAILED: it took more than {minutes:.0f} minutes (it normally takes 1 to 5). "
           f"Please send {root / 'results' / 'run.log'} and {root / 'results' / 'stack_dumps.log'}.")
    try:
        with open(root / "results" / "stack_dumps.log", "a", encoding="utf-8") as fh:
            fh.write("\n==== selftest time limit reached ====\n")
            faulthandler.dump_traceback(file=fh, all_threads=True)
    except Exception:
        pass
    print(msg, flush=True)
    sys.stdout.flush()
    os._exit(4)


def run_selftest(a, scale: int = 1, limit_minutes: float = 30.0) -> int:
    from .cli import run
    from .support import SamplingStalled
    from .util import LOG, setup_logging

    root = _remove_tree(Path(a.root) / ("selftest_run" if scale == 1 else f"stress_run_x{scale}"))
    (root / "data").mkdir(parents=True)
    build_fixture(root / "data" / "sqlshare_data_release1.zip", scale=scale)
    a.root, a.command, a.zip = str(root), "all", None
    t0 = time.perf_counter()
    timer = threading.Timer(limit_minutes * 60, _too_long, args=(root, limit_minutes))
    timer.daemon = True
    timer.start()
    try:
        rc = run(a, **SELFTEST_SETTINGS)
    except SamplingStalled:
        raise
    except Exception:
        LOG.exception("the self-test run stopped with an error")
        rc = 1
    finally:
        timer.cancel()
    setup_logging(None)
    checks = []
    for proto in (["paper", "corrected"] if a.protocol == "both" else [a.protocol]):
        r = root / "results" / proto
        for f in ("lattice/summary.json", "main_market/runs.csv", "main_market/table6_support_sampling.csv",
                  "markets/table7_markets.csv", "compare_with_paper.csv", "tables/table7.tex"):
            checks.append((f"{proto}/{f}", (r / f).exists()))
    if a.protocol == "both":
        checks.append(("REPORT.md", (root / "results" / "REPORT.md").exists()))
    ok = rc == 0 and all(c[1] for c in checks)
    for name, good in checks:
        LOG.info("selftest %-55s %s", name, "ok" if good else "MISSING")
    LOG.info("SELFTEST %s in %.1f min (outputs in %s)", "PASSED" if ok else "FAILED",
             (time.perf_counter() - t0) / 60, root)
    return 0 if ok else 1
