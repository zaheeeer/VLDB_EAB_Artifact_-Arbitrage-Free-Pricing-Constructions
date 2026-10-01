"""Load extracted SQLShare market files into a DuckDB database.

One schema per owner. Each file becomes a table whose name is exactly the object name a
query references, which is the filename with its ``table_`` or ``materialized_`` prefix
removed and nothing stripped from the right.

Real user uploads are irregular, so loading is attempted in three passes of decreasing
strictness and the pass that succeeded is recorded per table. Nothing is silently
dropped: every failure is returned with its error so the attrition can be reported.
"""
from __future__ import annotations

import os

PREFIXES = ("table_", "materialized_")


def object_name(filename: str) -> str:
    for p in PREFIXES:
        if filename.lower().startswith(p):
            return filename[len(p):]
    return filename


#: Delimiters tried when the sniffer yields a single column. The filename extension is
#: NOT trusted: SQLShare dataset names keep the extension of the file the user happened
#: to upload, and comma-separated content is routinely stored under a ``.tab`` name.
DELIMS = (",", "\t", ";", "|", " ")


def _col_count(con, reader: str) -> int:
    try:
        return len(con.execute(f"SELECT * FROM {reader} LIMIT 0").description)
    except Exception:
        return 0


def load_market(con, root: str, owner: str):
    """Load every file in ``root/data/<owner>`` into schema ``<owner>``."""
    market_dir = os.path.join(root, "data", owner)
    if not os.path.isdir(market_dir):
        return []
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{owner}"')
    out = []
    for fname in sorted(os.listdir(market_dir)):
        path = os.path.join(market_dir, fname).replace("\\", "/")
        tname = object_name(fname)
        rec = {"owner": owner, "file": fname, "table": tname, "mode": None,
               "rows": 0, "cols": 0, "delim": "", "error": ""}

        base = [
            ("auto", f"read_csv('{path}', auto_detect=true, null_padding=true, "
                     f"ignore_errors=true, sample_size=-1)"),
            ("all_varchar", f"read_csv('{path}', auto_detect=true, all_varchar=true, "
                            f"null_padding=true, ignore_errors=true)"),
        ]
        # Pick the parse that splits the file into the most columns. A single-column
        # result means the delimiter was missed, since the column name then holds the
        # whole header row.
        best = None
        for mode, reader in base:
            c = _col_count(con, reader)
            if c and (best is None or c > best[2]):
                best = (mode, reader, c, "")
        if best is None or best[2] <= 1:
            for cand_delim in DELIMS:
                dd = "\\t" if cand_delim == "\t" else cand_delim
                reader = (f"read_csv('{path}', auto_detect=true, null_padding=true, "
                          f"ignore_errors=true, sample_size=-1, delim='{dd}')")
                c = _col_count(con, reader)
                if c and (best is None or c > best[2]):
                    best = ("delim", reader, c, cand_delim)

        if best is None:
            rec["error"] = "no parse produced columns"
            out.append(rec)
            continue

        mode, reader, ncols, chosen_delim = best
        try:
            con.execute(f'CREATE OR REPLACE TABLE "{owner}"."{tname}" AS SELECT * FROM {reader}')
            n = con.execute(f'SELECT count(*) FROM "{owner}"."{tname}"').fetchone()[0]
            rec.update(mode=mode, rows=n, cols=ncols, delim=chosen_delim, error="")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        out.append(rec)
    return out
