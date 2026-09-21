"""Rebuild the SQLShare substrate end to end into a fresh database.

Expects these names in the caller's namespace: root, infos, ent, views, vbody, rows,
sel_q, need_v, load_market, object_name, transpile_tsql, PATIDX, classify_error.

Writes results/sqlshare_pipeline_*.csv and leaves the connection in ``DB2``.
"""
import collections
import os
import time

import duckdb
import pandas as pd

DB_PATH = "data/sqlshare/sqlshare2.duckdb"


def run_pipeline():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = duckdb.connect(DB_PATH)

    owners = sorted({i.filename.split("/")[2] for i in infos
                     if len(i.filename.split("/")) >= 4 and i.filename.split("/")[1] == "data"
                     and os.path.exists("data/sqlshare/" + i.filename)})
    t0 = time.perf_counter()
    recs = []
    for o in owners:
        recs += load_market(con, root, o)
    load_df = pd.DataFrame(recs)
    t_load = time.perf_counter() - t0

    # Register every file under all names a query might use: raw filename, prefix
    # stripped, and each with the trailing extension removed.
    objs = {(s.lower(), t.lower()) for s, t in con.execute(
        "SELECT table_schema, table_name FROM information_schema.tables").fetchall()}
    aliases = 0
    for r in load_df[load_df["mode"].notna()].itertuples():
        real = r.table
        variants = {r.file, object_name(r.file), real,
                    r.file.rsplit(".", 1)[0], object_name(r.file).rsplit(".", 1)[0]}
        for v in variants:
            if not v or (r.owner.lower(), v.lower()) in objs:
                continue
            try:
                con.execute(f'CREATE OR REPLACE VIEW "{r.owner}"."{v}" AS '
                            f'SELECT * FROM "{r.owner}"."{real}"')
                objs.add((r.owner.lower(), v.lower()))
                aliases += 1
            except Exception:
                pass

    schemas = sorted({s for s, _ in objs})

    # Build the views the selected queries depend on, iterating so that views resting on
    # other views converge.
    made, failed, msg = set(), {}, {}
    for _ in range(15):
        progress = 0
        for o in sorted(need_v - made):
            if PATIDX.search(vbody[o]):
                failed[o] = "unsupported:patindex"
                continue
            try:
                stmt = build_view2(o)
                if stmt is None:
                    failed[o] = "header_unparsed"
                    continue
                path = ",".join([o[0]] + [s for s in schemas if s != o[0]])
                con.execute(f"SET search_path='{path}'")
                con.execute(stmt)
                made.add(o)
                failed.pop(o, None)
                progress += 1
            except Exception as e:
                failed[o] = classify_error(str(e))
                msg[o] = str(e)[:160]
        if not progress:
            break
    con.execute("SET search_path='main'")

    # Bind, then execute.
    stages, execs = [], []
    for k in sel_q:
        sql = rows[k]["text"]
        own = rows[k]["owner"] if rows[k]["owner"] in schemas else schemas[0]
        if PATIDX.search(sql):
            stages.append({"k": k, "stage": "unsupported_patindex"})
            continue
        try:
            duck = transpile_tsql(sql)
        except Exception as e:
            stages.append({"k": k, "stage": "transpile_error"})
            continue
        path = ",".join([own] + [s for s in schemas if s != own])
        try:
            con.execute(f"SET search_path='{path}'")
            con.execute("EXPLAIN " + duck)
        except Exception as e:
            stages.append({"k": k, "stage": "bind_" + classify_error(str(e))})
            continue
        stages.append({"k": k, "stage": "bind_ok"})
        try:
            n = con.execute(f"SELECT count(*) FROM ({duck}) _x").fetchone()[0]
            execs.append({"k": k, "ok": True, "rows": int(n), "own": own, "duck": duck})
        except Exception as e:
            execs.append({"k": k, "ok": False, "rows": 0, "own": own,
                          "err": classify_error(str(e))})
    con.execute("SET search_path='main'")

    return {
        "con": con, "load": load_df, "aliases": aliases, "schemas": schemas,
        "views_made": made, "views_failed": failed, "view_msg": msg,
        "stages": pd.DataFrame(stages), "execs": pd.DataFrame(execs),
        "t_load": t_load,
    }
