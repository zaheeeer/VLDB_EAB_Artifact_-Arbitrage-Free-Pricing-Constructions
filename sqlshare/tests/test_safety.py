"""Safety paths: each must work, so that nothing can hang or silently break.

Worker processes that die or freeze, a sampling thread that freezes, a dropped or
finished download, a damaged zip, a locked database, markets too small to price, a
wrong -Zip path, and the preflight check.
"""
import http.server
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlshare_rerun import corpus as C          # noqa: E402
from sqlshare_rerun import report as R          # noqa: E402
from sqlshare_rerun import stopflag             # noqa: E402
from sqlshare_rerun import database as D        # noqa: E402
from sqlshare_rerun import stages as ST         # noqa: E402
from sqlshare_rerun.cli import main as cli_main  # noqa: E402
from sqlshare_rerun.config import make_settings  # noqa: E402
from sqlshare_rerun.parallel import run_pool    # noqa: E402
from sqlshare_rerun.selftest import SELFTEST_SETTINGS, build_fixture  # noqa: E402
from sqlshare_rerun.support import SamplingStalled, SupportSampler, base_fingerprints  # noqa: E402

WORKERS = int(os.environ.get("SQLSHARE_TEST_WORKERS", "2"))
needs_workers = pytest.mark.skipif(WORKERS < 2, reason="worker processes are disabled on this computer")


# ---- tasks for worker processes (module level, so a fresh process can import them) -------
def _square(x):
    return x * x


def _die_in_worker(x):
    if multiprocessing.current_process().name != "MainProcess":
        os._exit(1)                              # a worker killed from outside
    return x * x


def _freeze_in_worker(x):
    if multiprocessing.current_process().name != "MainProcess":
        time.sleep(3600)                         # a worker frozen by security software
    return x * x


def _bug(x):
    raise ValueError("a real bug must not be hidden")


# ---- worker pools --------------------------------------------------------------------
@needs_workers
def test_pool_gives_same_results_as_serial():
    assert run_pool(_square, range(9), 2, "t") == [x * x for x in range(9)]


@needs_workers
def test_pool_recovers_when_a_worker_dies():
    assert run_pool(_die_in_worker, range(6), 2, "t") == [x * x for x in range(6)]


@needs_workers
def test_pool_recovers_when_workers_freeze():
    t0 = time.perf_counter()
    assert run_pool(_freeze_in_worker, range(4), 2, "t", stall_s=5) == [0, 1, 4, 9]
    assert time.perf_counter() - t0 < 120


@needs_workers
def test_pool_does_not_hide_a_bug():
    with pytest.raises(ValueError):
        run_pool(_bug, range(3), 2, "t")


# ---- a frozen sampling thread ------------------------------------------------------------
def test_sampler_reports_a_freeze_instead_of_waiting(tmp_path):
    con = duckdb.connect(str(tmp_path / "samp.duckdb"))
    con.execute('CREATE SCHEMA "s"')
    con.execute('CREATE TABLE "s"."a" AS SELECT i AS v FROM range(50) r(i)')
    assets = {1: ("s", 'SELECT sum(v) AS s FROM "s"."a"')}
    fp = base_fingerprints(con, ["s"], assets)
    release = threading.Event()
    smp = SupportSampler(con, ["s"], assets, {("s", "a"): [1]}, fp, {("s", "a"): ("s", "a")},
                         {("s", "a"): 50}, "any_row", threads=2, stall_s=3)
    smp._evaluate = lambda cand: release.wait(60)          # every deletion freezes
    t0 = time.perf_counter()
    with pytest.raises(SamplingStalled):
        smp.sample([("s", "a")], [1.0], 5, 1, 2)
    assert time.perf_counter() - t0 < 60
    release.set()


@pytest.mark.parametrize("threads", [1, 4])
def test_same_row_drawn_many_times_in_one_batch(tmp_path, threads):
    """A one-row table: every candidate is the same row. With threads, two deletions of one
    row at once would conflict; each distinct row is evaluated once per batch instead."""
    con = duckdb.connect(str(tmp_path / "one.duckdb"))
    con.execute('CREATE SCHEMA "s"')
    con.execute('CREATE TABLE "s"."a" AS SELECT 7 AS v')
    assets = {1: ("s", 'SELECT sum(v) AS s FROM "s"."a"')}
    fp = base_fingerprints(con, ["s"], assets)
    smp = SupportSampler(con, ["s"], assets, {("s", "a"): [1]}, fp, {("s", "a"): ("s", "a")},
                         {("s", "a"): 1}, "any_row", threads=threads, stall_s=60)
    t0 = time.perf_counter()
    res = smp.sample([("s", "a")], [1.0], 60, 1, 2)
    assert res.n == 60 and res.incidence == {1: list(range(60))}
    assert time.perf_counter() - t0 < 30
    assert con.execute('SELECT count(*) FROM "s"."a"').fetchone()[0] == 1


def test_ctrl_c_is_never_recorded_as_a_failed_query(tmp_path):
    con = duckdb.connect()
    import pandas as pd
    assets = pd.DataFrame({"k": [1], "own": ["main"], "duck": ["SELECT * FROM no_such_table"]})
    stopflag.STOP.set()
    try:
        with pytest.raises(KeyboardInterrupt):
            D.materialize(con, assets, 60)
    finally:
        stopflag.STOP.clear()
    answers, _ = D.materialize(con, assets, 60)           # without Ctrl+C: left out, as historically
    assert answers == {}


def test_results_from_other_settings_are_not_reported(tmp_path):
    (tmp_path / "data").mkdir()
    build_fixture(tmp_path / "data" / "sqlshare_data_release1.zip")
    s = make_settings("paper", tmp_path, **dict(SELFTEST_SETTINGS, workers=1, threads=1))
    ST.stage_lattice(s)
    assert "lattice" in R.collect(s)
    other = make_settings("paper", tmp_path, **dict(SELFTEST_SETTINGS, workers=1, threads=1, quick=True))
    assert "lattice" not in R.collect(other)               # made with other settings: not reported
    (s.results_dir / "tables").mkdir(parents=True, exist_ok=True)
    stale = s.results_dir / "tables" / "table7.tex"
    stale.write_text("old", encoding="utf-8")
    R.paper_tables(other)
    assert not stale.exists()                              # stale tables are removed


def test_sampler_empty_input_is_safe(tmp_path):
    con = duckdb.connect(str(tmp_path / "e.duckdb"))
    smp = SupportSampler(con, [], {}, {}, {}, {}, {}, "any_row", threads=2)
    res = smp.sample([], [], 10, 1, 2)
    assert res.n == 0 and res.edges([1, 2]) == [frozenset(), frozenset()]


# ---- the download ----------------------------------------------------------------------
class _Server(http.server.BaseHTTPRequestHandler):
    data, drop_after, starts, forbid = b"", None, [], False

    def do_GET(self):
        if type(self).forbid:                        # a network that blocks the site
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        rng, start = self.headers.get("Range"), 0
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(self.data):
                self.send_response(416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        body = self.data[start:]
        self.send_response(206 if rng else 200)
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(self.data) - 1}/{len(self.data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        type(self).starts.append(start)
        cut = type(self).drop_after
        if cut is not None:                          # drop the connection part-way, once
            type(self).drop_after = None
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server(tmp_path):
    z = tmp_path / "src.zip"
    build_fixture(z)
    _Server.data, _Server.drop_after, _Server.starts, _Server.forbid = z.read_bytes(), None, [], False
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    old = urllib.request._opener
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    yield f"http://127.0.0.1:{srv.server_address[1]}/sqlshare_data_release1.zip"
    urllib.request._opener = old
    srv.shutdown()
    srv.server_close()


def test_download_resumes_after_a_dropped_connection(server, tmp_path):
    _Server.drop_after = len(_Server.data) // 3
    dest = tmp_path / "dl" / "sqlshare_data_release1.zip"
    C.download(server, dest, retries=5)
    assert dest.read_bytes() == _Server.data
    assert len(_Server.starts) == 2 and _Server.starts[1] > 0          # the second request resumed


def test_blocked_download_stops_quickly_with_instructions(server, tmp_path):
    _Server.forbid = True
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="-Zip"):
        C.download(server, tmp_path / "dl" / "sqlshare_data_release1.zip")
    assert time.perf_counter() - t0 < 60


def test_download_finishes_a_complete_part_file(server, tmp_path):
    dest = tmp_path / "dl" / "sqlshare_data_release1.zip"
    dest.parent.mkdir(parents=True)
    dest.with_suffix(".zip.part").write_bytes(_Server.data)            # killed just before the rename
    C.download(server, dest, retries=3)
    assert dest.read_bytes() == _Server.data


def test_damaged_zip_is_set_aside_and_downloaded_again(server, tmp_path):
    s = make_settings("paper", tmp_path / "run", zip_url=server)
    s.zip_file.parent.mkdir(parents=True)
    s.zip_file.write_bytes(_Server.data[: len(_Server.data) // 2])     # a cut-off file
    with pytest.raises(RuntimeError):
        ST.stage_download(s)
    assert s.zip_file.with_suffix(".zip.broken").exists() and not s.zip_file.exists()
    info = ST.stage_download(s)                                        # the next run downloads it again
    assert info["entries"] > 10 and s.zip_file.read_bytes() == _Server.data


def test_missing_user_zip_is_a_clear_error(tmp_path):
    rc = cli_main(["download", "--root", str(tmp_path), "--zip", str(tmp_path / "no_such.zip")])
    assert rc == 1
    assert "does not exist" in (tmp_path / "results" / "run.log").read_text(encoding="utf-8")


# ---- extraction names are stable -----------------------------------------------------------
def test_extracted_names_do_not_depend_on_order(tmp_path):
    z = tmp_path / "r.zip"
    build_fixture(z)
    with zipfile.ZipFile(z) as zz:
        names = sorted(n for n in zz.namelist() if "/data/" in n)
    m1 = C.extract(z, names, tmp_path / "a")
    m2 = C.extract(z, list(reversed(names)), tmp_path / "b")
    loc = lambda m, root: {x["zip_name"]: Path(x["local"]).relative_to(root).as_posix() for x in m}  # noqa: E731
    assert loc(m1, tmp_path / "a") == loc(m2, tmp_path / "b")
    assert len(set(loc(m1, tmp_path / "a").values())) == len(names)


# ---- a locked database -----------------------------------------------------------------------
def test_locked_database_gives_a_plain_message(tmp_path):
    db = tmp_path / "x.duckdb"
    duckdb.connect(str(db)).close()
    code = ("import duckdb, sys, time\ncon = duckdb.connect(sys.argv[1])\nprint('ready', flush=True)\n"
            "time.sleep(60)\n")
    holder = subprocess.Popen([sys.executable, "-c", code, str(db)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        with pytest.raises(RuntimeError, match="another program is using it"):
            D.connect(db, lock_retries=2, lock_wait=0.2)
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()


# ---- markets too small to price --------------------------------------------------------------
def test_markets_stage_with_no_market_large_enough(tmp_path):
    (tmp_path / "data").mkdir()
    build_fixture(tmp_path / "data" / "sqlshare_data_release1.zip")
    kw = dict(SELFTEST_SETTINGS, market_min_assets=10 ** 6, workers=1, threads=1)
    s = make_settings("paper", tmp_path, **kw)
    summary = ST.stage_markets(s)
    assert summary["markets_ok"] == 0 and summary["assets"] == 0
    assert (s.results_dir / "markets" / "table7_markets.csv").exists()


# ---- the preflight check ----------------------------------------------------------------------
def test_doctor_runs_and_writes_advice(tmp_path):
    rc = cli_main(["doctor", "--root", str(tmp_path)])
    out = json.loads((tmp_path / "logs" / "doctor.json").read_text(encoding="utf-8"))
    assert rc in (0, 1) and set(out["recommend"]) == {"workers", "threads"}
    names = {c["check"] for c in out["checks"]}
    assert {"Python", "write files", "DuckDB with threads", "worker processes"} <= names
    assert not out["fail"], [c for c in out["checks"] if c["level"] == "FAIL"]
