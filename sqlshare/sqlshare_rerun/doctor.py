"""Preflight check of this computer: ``python -m sqlshare_rerun doctor``.

Runs in about a minute and tests, on this machine, everything that could stop or freeze
the run: Python and package versions, memory, free disk and file system, writing files,
DuckDB transactions from several threads, worker processes, and network access to the
release. It prints one line per check and writes logs/doctor.json with the settings to
use; run_all.ps1 applies them automatically.
"""
from __future__ import annotations

import json
import os
import platform
import random
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

from .parallel import SPAWN, _stop

PINNED = {"duckdb": "1.5.5", "numpy": "2.3.5", "pandas": "2.2.3", "scipy": "1.17.0", "sqlglot": "30.18.0"}
ZIP_URL = "https://shrquerylogs.s3.amazonaws.com/public/sqlshare_data_release1.zip"


def _probe_task(x: int) -> int:
    """Runs in a worker process: import what the real workers import."""
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy.optimize  # noqa: F401
    return x * x


def _ram_gb():
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception:
        return None


def _filesystem(path: Path):
    if os.name != "nt":
        return None
    try:
        import ctypes
        drive = os.path.splitdrive(str(path.resolve()))[0]
        if not drive:
            return None
        buf = ctypes.create_unicode_buffer(64)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(ctypes.c_wchar_p(drive + "\\"), None, 0, None,
                                                          None, None, buf, len(buf))
        return buf.value if ok else None
    except Exception:
        return None


def _duckdb_threads(db: Path, n_threads: int = 8, per_thread: int = 30, timeout: float = 180.0):
    """Several threads delete, re-query, and roll back at once, as the support sampler does."""
    import duckdb
    con = duckdb.connect(str(db))
    con.execute("CREATE OR REPLACE TABLE t AS SELECT i % 97 AS g, i AS v, 'x' || (i % 13) AS s "
                "FROM range(20000) r(i)")
    con.execute("CREATE OR REPLACE TABLE u AS SELECT i % 7 AS g, i * 2 AS w FROM range(300) r(i)")
    q = ("SELECT count(*) AS n, sum(hash(to_json(qq))::HUGEINT) AS h FROM "
         "(SELECT t.g, sum(v) AS sv, count(u.w) AS c FROM t JOIN u ON t.g = u.g GROUP BY t.g) qq")
    base = con.execute(q).fetchone()
    curs = [con.cursor() for _ in range(n_threads)]
    errors, finished = [], []

    def work(i):
        rng, cur = random.Random(i), curs[i]
        try:
            for _ in range(per_thread):
                tab, size = ("t", 20000) if rng.random() < 0.7 else ("u", 300)
                row = rng.randrange(size)
                for _attempt in range(1000):
                    try:
                        cur.execute("BEGIN TRANSACTION")
                        cur.execute(f"DELETE FROM {tab} WHERE rowid = {row}")
                        break
                    except duckdb.TransactionException:
                        try:
                            cur.execute("ROLLBACK")
                        except Exception:
                            pass
                        time.sleep(0.002)
                cur.execute(q).fetchone()
                cur.execute("ROLLBACK")
            finished.append(i)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    ths = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(n_threads)]
    t0 = time.perf_counter()
    for th in ths:
        th.start()
    for th in ths:
        th.join(max(0.0, timeout - (time.perf_counter() - t0)))
    stuck = any(th.is_alive() for th in ths)
    if stuck:
        return False, True, f"threads did not finish within {timeout:.0f} s"
    intact = con.execute("SELECT (SELECT count(*) FROM t), (SELECT count(*) FROM u)").fetchone() == (20000, 300)
    same = con.execute(q).fetchone() == base
    for c in curs:
        c.close()
    con.close()
    if errors or not intact or not same:
        return False, False, (errors[0] if errors else "rollback did not restore the tables")
    return True, False, f"{n_threads} threads x {per_thread} delete/rollback rounds in {time.perf_counter() - t0:.1f} s"


def _workers(timeout: float = 180.0):
    t0 = time.perf_counter()
    ex = ProcessPoolExecutor(max_workers=2, mp_context=SPAWN)
    try:
        futs = [ex.submit(_probe_task, i) for i in range(4)]
        done, pending = wait(futs, timeout=timeout)
        if pending:
            _stop(ex)
            return False, f"worker processes did not answer within {timeout:.0f} s"
        if [f.result() for f in futs] != [0, 1, 4, 9]:
            _stop(ex)
            return False, "worker processes returned wrong results"
        _stop(ex, grace=10.0)
        return True, f"2 worker processes started and answered in {time.perf_counter() - t0:.1f} s"
    except Exception as e:  # noqa: BLE001
        _stop(ex)
        return False, f"{type(e).__name__}: {e}"


def run_doctor(root: Path, zip_path: Path | None = None, full_run: bool = False) -> int:
    root = Path(root).resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    checks, stuck = [], False
    rec = {"workers": None, "threads": None}

    def report(level, what, detail):
        checks.append({"level": level, "check": what, "detail": detail})
        print(f"[{level:^4}] {what}: {detail}", flush=True)

    # ---- Python and packages ------------------------------------------------------------
    v = sys.version_info
    bits = 64 if sys.maxsize > 2 ** 32 else 32
    ok_py = (3, 11) <= v[:2] <= (3, 13) and bits == 64
    report("OK" if ok_py else "FAIL", "Python", f"{platform.python_version()} {bits}-bit at {sys.executable}"
           + ("" if ok_py else " (need 64-bit Python 3.11, 3.12 or 3.13)"))
    for mod, want in PINNED.items():
        try:
            have = __import__(mod).__version__
            report("OK" if have == want else "WARN", f"package {mod}",
                   have if have == want else f"{have} installed, {want} pinned (results may differ)")
        except Exception as e:  # noqa: BLE001
            report("FAIL", f"package {mod}", f"missing ({type(e).__name__})")

    # ---- machine ----------------------------------------------------------------------
    if os.name == "nt":
        try:
            import ctypes
            if ctypes.windll.shell32.IsUserAnAdmin():
                report("WARN", "window", "PowerShell runs as administrator. Please use a normal window "
                       "for every step; mixing the two can lock test folders.")
        except Exception:
            pass
    report("OK", "CPU", f"{os.cpu_count()} logical cores")
    ram = _ram_gb()
    if ram is not None:
        report("OK" if ram >= 7.5 else "WARN", "memory",
               f"{ram:.1f} GB" + ("" if ram >= 7.5 else " (below 8 GB: use -Workers 2 -MemoryLimit 4GB)"))
    s = str(root)
    report("WARN" if "onedrive" in s.lower() else "OK", "folder",
           s + (" (inside OneDrive: move it to a plain folder such as C:\\sqlshare)"
                if "onedrive" in s.lower() else ""))
    if len(s) > 120:
        report("WARN", "folder path length", f"{len(s)} characters; a shorter path is safer on Windows")
    fs = _filesystem(root)
    if fs:
        report("WARN" if fs.upper().startswith("FAT") else "OK", "file system",
               fs + (" (FAT32 cannot hold files over 4 GB; use an NTFS or exFAT drive)"
                     if fs.upper() == "FAT32" else ""))
    free = shutil.disk_usage(root).free / 2 ** 30
    need = 15 if full_run else 1
    report("OK" if free >= need else ("FAIL" if full_run else "WARN"), "free disk",
           f"{free:.1f} GB free" + ("" if free >= 15 else " (the full run needs about 15 GB)"))

    # ---- writing files ------------------------------------------------------------------
    scratch = root / "work" / "doctor"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        a, b = scratch / "a.bin", scratch / "b.bin"
        a.write_bytes(os.urandom(1 << 20))
        os.replace(a, b)
        assert b.read_bytes()[:16]
        b.unlink()
        report("OK", "write files", str(scratch))
    except Exception as e:  # noqa: BLE001
        report("FAIL", "write files", f"cannot write in {scratch}: {type(e).__name__}: {e}")

    # ---- DuckDB from several threads ----------------------------------------------------
    try:
        db = scratch / "doctor.duckdb"
        for p in (db, db.with_suffix(".duckdb.wal")):
            if p.exists():
                p.unlink()
        ok, hung, detail = _duckdb_threads(db)
        stuck = stuck or hung
        if ok:
            report("OK", "DuckDB with threads", detail)
        else:
            rec["threads"] = 1
            report("WARN", "DuckDB with threads", detail + " -> the run will use -Threads 1")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "DuckDB", f"{type(e).__name__}: {e}")

    # ---- worker processes -----------------------------------------------------------------
    ok, detail = _workers()
    if ok:
        report("OK", "worker processes", detail)
    else:
        rec["workers"] = 1
        report("WARN", "worker processes", detail + " -> the run will use -Workers 1 "
               "(if you use security software, allow python.exe)")

    # ---- network (only needed when the release is not here yet) ---------------------------
    zp = Path(zip_path) if zip_path else root / "data" / "sqlshare_data_release1.zip"
    if zp.exists():
        report("OK", "release zip", f"{zp} ({zp.stat().st_size / 1e9:.2f} GB)")
    elif zip_path:
        report("FAIL", "release zip", f"the file given with -Zip does not exist: {zp}")
    else:
        why = None
        for _ in range(2):
            try:
                req = urllib.request.Request(ZIP_URL, method="HEAD", headers={"User-Agent": "sqlshare-rerun/2.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    size = int(resp.headers.get("Content-Length") or 0)
                report("OK", "download site", f"reachable, release is {size / 1e9:.2f} GB")
                why = None
                break
            except Exception as e:  # noqa: BLE001
                why = f"{type(e).__name__}: {e}"
        if why is not None:
            hard = any(s in why for s in ("403", "Forbidden", "Tunnel connection failed", "407"))
            level = "FAIL" if (hard and full_run) else "WARN"
            report(level, "download site", f"not reachable ({why}). Download the zip in a browser from "
                   f"{ZIP_URL} , save it, and run again with -Zip <path to the zip>")

    fail = any(c["level"] == "FAIL" for c in checks)
    out = {"fail": fail, "recommend": rec, "checks": checks, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(logs / "doctor.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(("DOCTOR FAILED: fix the FAIL lines above, then run again." if fail else "DOCTOR PASSED"), flush=True)
    code = 1 if fail else 0
    if stuck:                      # a stuck DuckDB thread would keep the process from exiting
        sys.stdout.flush()
        os._exit(code)
    try:
        shutil.rmtree(scratch, ignore_errors=True)
    except Exception:
        pass
    return code
