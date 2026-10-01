"""Build the DuckDB substrate: load tables, create aliases and views, bind and execute.

Mirrors legacy/sqlshare_load.py and legacy/pipeline.py. The loader tries three parses
per file and keeps the one with the most columns, exactly as the historical run did.
"""
from __future__ import annotations

import collections
import re
import threading
import time

import duckdb
import pandas as pd

from . import stopflag
from .translate import PATIDX, classify_error, transpile_tsql
from .util import LOG, quote_ident, sql_str


def _fatal(e: BaseException, what: str) -> None:
    """Stop on Ctrl+C. Any other error is the query's own and is recorded as in the
    historical run; running out of memory is named in the log, because it depends on the
    computer and can change the funnel."""
    stopflag.check()
    if stopflag.out_of_memory(e):
        LOG.warning("out of memory while %s; recorded as a failure (a computer with more memory "
                    "may give a different funnel)", what)


PREFIXES = ("table_", "materialized_")
DELIMS = (",", "\t", ";", "|", " ")
HDR2 = re.compile(r"(?is)^create\s+view\s+\[([^\]]+)\]\s*\.\s*\[([^\]]+)\]\s*(\(.*?\))?\s*as\s+(.*)$")


def object_name(filename: str) -> str:
    for p in PREFIXES:
        if filename.lower().startswith(p):
            return filename[len(p):]
    return filename


def connect(path, threads: int = 0, memory_limit: str = "", temp_dir=None, read_only=False,
            lock_retries: int = 12, lock_wait: float = 5.0):
    """Open the database. Another program holding it (a second run, or a scanner) is
    retried for a minute, then reported plainly."""
    last = None
    for attempt in range(lock_retries):
        try:
            con = duckdb.connect(str(path), read_only=read_only)
            break
        except duckdb.IOException as e:
            msg = str(e).lower()
            if not any(s in msg for s in ("lock", "already open", "used by another process", "being used")):
                raise
            last = e
            time.sleep(lock_wait)
    else:
        raise RuntimeError(f"cannot open the database {path}: another program is using it. Close other "
                           f"PowerShell windows running this package (or restart the computer) and "
                           f"run the same command again. DuckDB said: {str(last)[:300]}")
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    if memory_limit:
        con.execute(f"SET memory_limit = {sql_str(memory_limit)}")
    if temp_dir is not None:
        con.execute(f"SET temp_directory = {sql_str(str(temp_dir).replace(chr(92), '/'))}")
    return con


def fp_sql(q: str) -> str:
    return f"SELECT count(*) AS n, sum(hash(to_json(qq))::HUGEINT) AS h FROM ({q}) qq"


class Watchdog:
    """Interrupt a DuckDB connection if one statement runs longer than ``seconds``, and say
    so in the log when a statement is still running after five minutes."""

    def __init__(self, con, seconds: float, label: str = ""):
        self.con, self.seconds, self.label, self.timers = con, seconds, label, []

    def _note(self):
        LOG.info("still running after 5 min: %s (it will be stopped after %.0f min)",
                 self.label or "a query", self.seconds / 60)

    def __enter__(self):
        if self.seconds and self.seconds > 0:
            for delay, fn in ((self.seconds, self.con.interrupt), (300.0, self._note)):
                if delay <= self.seconds:
                    tm = threading.Timer(delay, fn)
                    tm.daemon = True
                    tm.start()
                    self.timers.append(tm)
        return self

    def __exit__(self, *exc):
        for tm in self.timers:
            tm.cancel()
        return False


# ----------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------
def _col_count(con, reader: str) -> int:
    try:
        return len(con.execute(f"SELECT * FROM {reader} LIMIT 0").description)
    except Exception as e:
        _fatal(e, "reading a file header")
        return 0


def load_file(con, owner: str, disk_name: str, local_path: str) -> dict:
    path = local_path.replace("\\", "/").replace("'", "''")
    tname = object_name(disk_name)
    rec = {"owner": owner, "file": disk_name, "table": tname, "mode": None,
           "rows": 0, "cols": 0, "delim": "", "error": ""}
    if "'" in owner or "'" in disk_name:
        # The historical loader pasted the file path into SQL without escaping, so every
        # parse of a file whose owner or name holds a single quote failed. Kept, so the
        # same tables exist and the same queries bind.
        rec["error"] = "no parse produced columns"
        return rec
    base = [
        ("auto", f"read_csv('{path}', auto_detect=true, null_padding=true, "
                 f"ignore_errors=true, sample_size=-1)"),
        ("all_varchar", f"read_csv('{path}', auto_detect=true, all_varchar=true, "
                        f"null_padding=true, ignore_errors=true)"),
    ]
    best = None
    for mode, reader in base:
        c = _col_count(con, reader)
        if c and (best is None or c > best[2]):
            best = (mode, reader, c, "")
    if best is None or best[2] <= 1:
        for cand in DELIMS:
            dd = "\\t" if cand == "\t" else cand
            reader = (f"read_csv('{path}', auto_detect=true, null_padding=true, "
                      f"ignore_errors=true, sample_size=-1, delim='{dd}')")
            c = _col_count(con, reader)
            if c and (best is None or c > best[2]):
                best = ("delim", reader, c, cand)
    if best is None:
        rec["error"] = "no parse produced columns"
        return rec
    mode, reader, ncols, chosen = best
    try:
        con.execute(f"CREATE OR REPLACE TABLE {quote_ident(owner)}.{quote_ident(tname)} AS SELECT * FROM {reader}")
        n = con.execute(f"SELECT count(*) FROM {quote_ident(owner)}.{quote_ident(tname)}").fetchone()[0]
        rec.update(mode=mode, rows=int(n), cols=ncols, delim=chosen, error="")
    except Exception as e:
        _fatal(e, f"loading {owner}/{disk_name}")
        rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return rec


def windows_disk_view(manifest: list, zip_order: dict) -> list:
    """Collapse members that shared one file on the historical Windows disk.

    Names that differ only in case land on the same NTFS file: the first extracted name
    survives, the last extracted content wins. Returns one record per on-disk file.
    """
    groups = collections.OrderedDict()
    for m in sorted(manifest, key=lambda m: zip_order.get(m["zip_name"], 0)):
        key = (m["owner"].lower(), m["disk_name"].lower())
        if key not in groups:
            groups[key] = dict(m)
        else:
            groups[key]["local"] = m["local"]
            groups[key]["collapsed"] = groups[key].get("collapsed", 1) + 1
    return list(groups.values())


def load_all(con, disk_files: list) -> pd.DataFrame:
    by_owner = collections.defaultdict(list)
    for m in disk_files:
        by_owner[m["owner"]].append(m)
    recs = []
    t0 = time.perf_counter()
    for n, owner in enumerate(sorted(by_owner)):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(owner)}")
        for m in sorted(by_owner[owner], key=lambda m: m["disk_name"]):
            recs.append(load_file(con, owner, m["disk_name"], m["local"]))
        LOG.info("loaded owner %s (%d/%d), %d files so far, %.0f s", owner, n + 1,
                 len(by_owner), len(recs), time.perf_counter() - t0)
    return pd.DataFrame(recs)


def create_aliases(con, load_df: pd.DataFrame):
    """Register every file under all names a query might use. Returns (count, alias map)."""
    objs = {(s.lower(), t.lower()) for s, t in con.execute(
        "SELECT table_schema, table_name FROM information_schema.tables").fetchall()}
    made, alias_map = 0, {}
    for r in load_df[load_df["mode"].notna()].itertuples():
        variants = {r.file, object_name(r.file), r.table, r.file.rsplit(".", 1)[0],
                    object_name(r.file).rsplit(".", 1)[0]}
        for v in sorted(variants):
            if not v or (r.owner.lower(), v.lower()) in objs:
                continue
            try:
                con.execute(f"CREATE OR REPLACE VIEW {quote_ident(r.owner)}.{quote_ident(v)} AS "
                            f"SELECT * FROM {quote_ident(r.owner)}.{quote_ident(r.table)}")
                objs.add((r.owner.lower(), v.lower()))
                alias_map[(r.owner.lower(), v.lower())] = (r.owner, r.table)
                made += 1
            except Exception as e:
                _fatal(e, "creating alias views")
    return made, alias_map


def schema_list(con) -> list:
    return sorted({s.lower() for s, _ in con.execute(
        "SELECT table_schema, table_name FROM information_schema.tables").fetchall()})


def search_path(own: str, schemas: list) -> str:
    return ",".join([own] + [s for s in schemas if s != own])


def view_statement(body: str):
    """DuckDB statement for one view body (the historical ``build_view2``), or None."""
    m = HDR2.match(body)
    if not m:
        return None
    own, name, cols, sel = m.group(1), m.group(2), m.group(3), m.group(4)
    duck = transpile_tsql(sel)
    colspec = ""
    if cols:
        cl = [c.strip() for c in re.findall(r"\[([^\]]+)\]", cols)]
        if cl:
            colspec = "(" + ",".join(f'"{c}"' for c in cl) + ")"
    # Historical formatting kept verbatim (no identifier escaping), so the same views
    # fail and succeed as in the historical run.
    return f'CREATE OR REPLACE VIEW "{own}"."{name}" {colspec} AS {duck}'


def build_views(con, corpus, need_v: set, schemas: list, rounds: int = 15):
    def build_view2(o):
        return view_statement(corpus.vbody[o])

    made, failed, msg = set(), {}, {}
    for _ in range(rounds):
        progress = 0
        for o in sorted(need_v - made):
            if PATIDX.search(corpus.vbody[o]):
                failed[o] = "unsupported:patindex"
                continue
            try:
                stmt = build_view2(o)
                if stmt is None:
                    failed[o] = "header_unparsed"
                    continue
                con.execute(f"SET search_path={sql_str(search_path(o[0], schemas))}")
                con.execute(stmt)
                made.add(o)
                failed.pop(o, None)
                progress += 1
            except Exception as e:
                _fatal(e, "creating views")
                failed[o] = classify_error(str(e))
                msg[o] = str(e)[:160]
        if not progress:
            break
    con.execute("SET search_path='main'")
    return made, failed, msg


def bind_and_execute(con, corpus, sel_q: list, schemas: list, timeout: float):
    stages, execs = [], []
    t0 = time.perf_counter()
    for n, k in enumerate(sel_q):
        sql = corpus.rows[k]["text"]
        own = corpus.rows[k]["owner"] if corpus.rows[k]["owner"] in schemas else schemas[0]
        if PATIDX.search(sql):
            stages.append({"k": k, "stage": "unsupported_patindex"})
            continue
        try:
            duck = transpile_tsql(sql)
        except Exception:
            stopflag.check()
            stages.append({"k": k, "stage": "transpile_error"})
            continue
        try:
            con.execute(f"SET search_path={sql_str(search_path(own, schemas))}")
            with Watchdog(con, timeout, f"binding query {k}"):
                con.execute("EXPLAIN " + duck)
        except Exception as e:
            _fatal(e, f"binding query {k}")
            stages.append({"k": k, "stage": "bind_" + classify_error(str(e))})
            continue
        stages.append({"k": k, "stage": "bind_ok"})
        try:
            with Watchdog(con, timeout, f"executing query {k}"):
                nrows = con.execute(f"SELECT count(*) FROM ({duck}) _x").fetchone()[0]
            execs.append({"k": k, "ok": True, "rows": int(nrows), "own": own, "duck": duck, "err": ""})
        except Exception as e:
            _fatal(e, f"executing query {k}")
            execs.append({"k": k, "ok": False, "rows": 0, "own": own, "duck": duck,
                          "err": classify_error(str(e))})
        if (n + 1) % 250 == 0:
            LOG.info("bound/executed %d/%d queries (%.0f s)", n + 1, len(sel_q), time.perf_counter() - t0)
    con.execute("SET search_path='main'")
    return pd.DataFrame(stages), pd.DataFrame(execs)


# ----------------------------------------------------------------------------------------
# answers
# ----------------------------------------------------------------------------------------
def norm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def materialize(con, assets: pd.DataFrame, timeout: float, repeats: int = 1):
    """Materialize answers with the owner-only search path the historical run used.

    Returns (answers, seconds). ``seconds`` is the single timed run (paper) or the median
    of ``repeats`` timed runs after the first (corrected), used by compute metering.
    """
    answers, seconds, failed = {}, {}, 0
    for r in assets.itertuples():
        try:
            con.execute(f"SET search_path={sql_str(r.own)}")
            t0 = time.perf_counter()
            with Watchdog(con, timeout, f"materializing query {r.k}"):
                df = con.execute(r.duck).df()
            dt = time.perf_counter() - t0
            answers[r.k] = norm_frame(df)
            if repeats > 1:
                ts = []
                for _ in range(repeats):
                    t1 = time.perf_counter()
                    with Watchdog(con, timeout, f"timing query {r.k}"):
                        con.execute(r.duck).df()
                    ts.append(time.perf_counter() - t1)
                dt = sorted(ts)[len(ts) // 2]
            seconds[r.k] = dt
        except Exception as e:
            _fatal(e, f"materializing query {r.k}")
            failed += 1
            if failed <= 20:
                LOG.warning("query %s could not be materialized (%s); it is left out, as in the "
                            "historical run", r.k, classify_error(str(e)))
    con.execute("SET search_path='main'")
    return answers, seconds
