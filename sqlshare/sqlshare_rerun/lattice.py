"""Verified derivability between materialized answers (legacy/sqlshare_lattice.py).

Three rewrite families are tried in order and a pair is kept only when the rewrite
reproduces the target answer exactly:

projection  the target is a duplicate-free projection of the source;
rollup      the target sums the source's numeric columns over a coarser grouping;
selection   paper protocol: restrict one source column to the TARGET'S VALUE SET, which
            a buyer cannot know before buying (concern C07);
            corrected protocol: apply the WHERE predicate of the target QUERY to the
            source answer, which a buyer can do from the query text alone.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .parallel import run_pool
from .util import load_pickle, save_pickle

RID = "__rerun_rid__"


# ----------------------------------------------------------------------------------------
# equality
# ----------------------------------------------------------------------------------------
def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(list(df.columns), kind="mergesort").reset_index(drop=True)


def _columns_equal(a2: pd.DataFrame, b2: pd.DataFrame, rtol: float) -> bool:
    for c in a2.columns:
        x, y = a2[c], b2[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.to_numpy(dtype=float), y.to_numpy(dtype=float),
                               rtol=rtol, atol=0, equal_nan=True):
                return False
        else:
            if not x.astype(str).equals(y.astype(str)):
                return False
    return True


def frames_equal(a, b, rtol=1e-9) -> bool:
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    if list(a.columns) != list(b.columns):
        return False
    return _columns_equal(_sorted(a), _sorted(b), rtol)


class SortedTargets:
    """Caches each target's sorted frame; sorting the target is half of every check."""

    def __init__(self, answers: dict):
        self.answers, self.cache = answers, {}

    def get(self, key):
        if key not in self.cache:
            try:
                self.cache[key] = ("ok", _sorted(self.answers[key].reset_index(drop=True)))
            except Exception as e:  # the historical check raised here too
                self.cache[key] = ("err", e)
        status, val = self.cache[key]
        if status == "err":
            raise val
        return val

    def equal(self, got, key, rtol=1e-9) -> bool:
        T = self.answers[key]
        if got is None or got.shape != T.shape or list(got.columns) != list(T.columns):
            return False
        a2 = _sorted(got)
        return _columns_equal(a2, self.get(key), rtol)


# ----------------------------------------------------------------------------------------
# rewrites
# ----------------------------------------------------------------------------------------
def try_projection(S, T):
    return S.loc[:, list(T.columns)].drop_duplicates().reset_index(drop=True)


def try_rollup(S, T):
    cols = list(T.columns)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(T[c])]
    key = [c for c in cols if c not in num]
    if not num or not key:
        return None
    out = S.groupby(key, dropna=False, as_index=False)[num].sum()
    return out.loc[:, cols].reset_index(drop=True)


def try_selection(S, T):
    """Paper protocol: restrict the source to the target's value set on one column."""
    cols = list(T.columns)
    sub = S.loc[:, cols]
    for c in cols:
        vals = set(T[c].dropna().unique())
        if not vals or len(vals) >= sub[c].nunique(dropna=True):
            continue
        cand = sub[sub[c].isin(vals)].drop_duplicates().reset_index(drop=True)
        if cand.shape[0] == T.shape[0]:
            return cand
    return None


def query_predicate(duck_sql: str):
    """WHERE clause of a plain SELECT, qualifiers removed, or None.

    Only a single SELECT without grouping, aggregation, windows, set operations, or
    subqueries qualifies. The buyer holds the query text, so this predicate is
    available to a buyer without the answer.
    """
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(duck_sql, read="duckdb")
    except Exception:
        return None
    if not isinstance(tree, exp.Select):
        return None
    for arg in ("group", "having", "qualify", "with", "distinct"):
        if tree.args.get(arg):
            if arg == "distinct" and tree.args["distinct"].args.get("on") is None:
                continue
            return None
    if any(e.find(exp.AggFunc) or e.find(exp.Window) for e in tree.expressions):
        return None
    where = tree.args.get("where")
    if where is None:
        return None
    cond = where.this.copy()
    if cond.find(exp.Subquery) or cond.find(exp.Select) or cond.find(exp.Exists):
        return None
    for col in cond.find_all(exp.Column):
        for part in ("table", "db", "catalog"):
            if col.args.get(part) is not None:
                col.set(part, None)
    try:
        return cond.sql(dialect="duckdb")
    except Exception:
        return None


_DUCK = {}


def _duck():
    import duckdb
    pid = os.getpid()
    if pid not in _DUCK:
        _DUCK[pid] = duckdb.connect()
        _DUCK[pid].execute("SET threads = 1")
    return _DUCK[pid]


def try_selection_predicate(S, T, pred, src_rid=None):
    """Corrected protocol: apply the target query's WHERE predicate to the source.

    DuckDB only computes which source rows pass; the rows themselves are taken from the
    source frame, so values and dtypes are exactly those of the source answer.
    """
    if not pred:
        return None
    con = _duck()
    if src_rid is None:
        src_rid = S.copy()
        src_rid.insert(0, RID, np.arange(len(src_rid)))
    name = "__rerun_src__"
    try:
        con.register(name, src_rid)
        rid = con.execute(f"SELECT {RID} FROM {name} WHERE {pred}").fetchnumpy()[RID]
    except Exception:
        return None
    finally:
        try:
            con.unregister(name)
        except Exception:
            pass
    cand = S.iloc[np.asarray(rid, dtype=int)].loc[:, list(T.columns)]
    return cand.drop_duplicates().reset_index(drop=True)


def derive_pair(S, T, tkey, targets: SortedTargets, mode: str, pred=None, src_rid=None):
    if not set(T.columns) <= set(S.columns):
        return None
    if len(T) > len(S):
        return None
    attempts = (("projection", try_projection), ("rollup", try_rollup))
    for name, fn in attempts:
        try:
            got = fn(S, T)
        except Exception:
            continue
        if got is None:
            continue
        try:
            if targets.equal(got, tkey):
                return name
        except Exception:
            continue
    try:
        if mode == "value_set":
            got = try_selection(S, T)
        else:
            got = try_selection_predicate(S, T, pred, src_rid)
    except Exception:
        got = None
    if got is not None:
        try:
            if targets.equal(got, tkey):
                return "selection"
        except Exception:
            pass
    return None


# ----------------------------------------------------------------------------------------
# lattice over one market
# ----------------------------------------------------------------------------------------
_WORKER = {}


def _load(path):
    if path not in _WORKER:
        _WORKER.clear()
        data = load_pickle(Path(path))
        data["targets"] = SortedTargets(data["answers"])
        _WORKER[path] = data
    return _WORKER[path]


def _lattice_chunk(args):
    path, sources = args
    d = _load(path)
    answers, keys, mode, preds, targets = d["answers"], d["keys"], d["mode"], d["preds"], d["targets"]
    verified, attempted = [], 0
    for i in sources:
        S = answers[i]
        scols = set(S.columns)
        src_rid = None
        if mode != "value_set":
            src_rid = S.copy()
            src_rid.insert(0, RID, np.arange(len(src_rid)))
        for j in keys:
            if i == j:
                continue
            T = answers[j]
            if not set(T.columns) <= scols or len(T) > len(S):
                continue
            attempted += 1
            fam = derive_pair(S, T, j, targets, mode, preds.get(j), src_rid)
            if fam:
                verified.append((i, j, fam))
    return verified, attempted


def market_lattice(answers: dict, mode: str, preds: dict | None = None, workers: int = 1,
                   tmp_dir: Path | None = None, stall_s: float = 1800.0) -> dict:
    keys = sorted(answers)
    preds = preds or {}
    payload = {"answers": answers, "keys": keys, "mode": mode, "preds": preds}
    if len(keys) < 2:
        return {"verified": [], "by_family": {}, "attempted": 0, "n_assets": len(keys)}
    tmp_dir = Path(tmp_dir or tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".pkl", dir=str(tmp_dir))
    os.close(fd)
    save_pickle(payload, Path(path))
    try:
        if workers <= 1 or len(keys) < 40:
            results = [_lattice_chunk((path, keys))]
        else:
            n_chunks = min(len(keys), workers * 4)
            chunks = [keys[c::n_chunks] for c in range(n_chunks)]
            results = run_pool(_lattice_chunk, [(path, c) for c in chunks], workers,
                               "lattice", stall_s)
    finally:
        _WORKER.clear()
        try:
            os.remove(path)
        except OSError:
            pass
    verified = sorted((v for r in results for v in r[0]), key=lambda t: (t[0], t[1]))
    attempted = sum(r[1] for r in results)
    by_family = {}
    for _, _, f in verified:
        by_family[f] = by_family.get(f, 0) + 1
    return {"verified": verified, "by_family": by_family, "attempted": attempted,
            "n_assets": len(keys)}


# ----------------------------------------------------------------------------------------
# cover structure for the recomposing adversary
# ----------------------------------------------------------------------------------------
def row_keys(df: pd.DataFrame) -> set:
    return set(map(tuple, df.astype(str).itertuples(index=False, name=None)))


def partial_sources(answers: dict, target, cache: dict, target_cap: int = 2000):
    """Sources whose projection onto the target's columns is a subset of its rows."""
    T = answers[target]
    if len(T) > target_cap:
        return None, None
    cols = tuple(T.columns)
    tkeys = row_keys(T)
    idx = {k: i for i, k in enumerate(sorted(tkeys))}
    out = {}
    for s, S in answers.items():
        if s == target or not set(cols) <= set(S.columns):
            continue
        ck = (s, cols)
        if ck not in cache:
            cache[ck] = row_keys(S.loc[:, list(cols)].drop_duplicates())
        sk = cache[ck]
        if sk and sk <= tkeys:
            out[s] = np.array(sorted(idx[k] for k in sk), dtype=int)
    return out, len(tkeys)


def cover_structure(answers: dict, keys: list, target_cap: int):
    """Set-semantics cover targets, as in the historical main run (bag targets excluded)."""
    cache, contrib, nrows = {}, {}, {}
    for k in keys:
        c_, n_ = partial_sources(answers, k, cache, target_cap)
        if c_ is not None and len(answers[k]) == len(answers[k].drop_duplicates()):
            contrib[k], nrows[k] = c_, n_
    return contrib, nrows
