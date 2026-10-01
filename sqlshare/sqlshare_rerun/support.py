"""Support sets built by literal execution: delete one base row, re-run every query that
reads the table, compare answer fingerprints, roll back.

Deletions are drawn from two seeded random streams (tables, then rows), so a run is
reproducible. Candidates are evaluated by a pool of threads, each with its own DuckDB
cursor and transaction; the accepted support set is the first ``n`` successful
candidates in draw order, so the result does not depend on thread timing.

Paper protocol: a row is deletable only if no other row has identical values (the
historical value-match delete skipped duplicates, concern C12); alias views are never
deletable. Corrected protocol: any physical row is deletable; alias views resolve to the
base table they alias.
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field

import duckdb
import numpy as np

from . import stopflag
from .database import fp_sql, search_path
from .util import LOG, quote_ident, sql_str


def _stop_on(e: BaseException) -> None:
    """Ctrl+C stops the run. Running out of memory also stops it here: with several
    threads it would depend on timing, so the support set could depend on thread count."""
    stopflag.check()
    if stopflag.out_of_memory(e):
        raise RuntimeError("out of memory during support sampling. Run the same command again "
                           "with -Threads 2 (or -Serial); it continues where it stopped.") from e


class _AllRows:
    """Marks a table none of whose rows the historical delete could remove."""

    def __contains__(self, item):
        return True


_ALL_ROWS = _AllRows()


def table_catalog(con) -> dict:
    """(schema, name) lowercased -> table type, for every table and view."""
    rows = con.execute("SELECT table_schema, table_name, table_type FROM information_schema.tables").fetchall()
    return {(s.lower(), t.lower()): (s, t, typ) for s, t, typ in rows}


def resolve_table(b, catalog: dict, alias_map: dict, alias_delete: str):
    """Return (schema, name) of the base table behind key ``b``, or None if not deletable."""
    hit = catalog.get((b[0].lower(), b[1].lower()))
    if hit is None:
        return None
    schema, name, typ = hit
    if typ == "BASE TABLE":
        return (schema, name)
    if alias_delete == "resolve":
        real = alias_map.get((b[0].lower(), b[1].lower()))
        if real is not None:
            hit2 = catalog.get((real[0].lower(), real[1].lower()))
            if hit2 is not None and hit2[2] == "BASE TABLE":
                return (hit2[0], hit2[1])
    return None


@dataclass
class SampleResult:
    n: int
    accepted: list                     # (table key, rowid) in support order
    incidence: dict                    # asset -> sorted list of support indices
    changed_fp: dict                   # asset -> {support index: fingerprint} (if recorded)
    draws: int
    skipped: int
    seconds: float
    checkpoint_seconds: dict = field(default_factory=dict)

    def counts(self, keys, upto=None) -> np.ndarray:
        if upto is None:
            return np.array([len(self.incidence.get(k, ())) for k in keys], float)
        return np.array([sum(1 for i in self.incidence.get(k, ()) if i < upto) for k in keys], float)

    def edges(self, keys) -> list:
        return [frozenset(self.incidence.get(k, ())) for k in keys]


class SamplingStalled(RuntimeError):
    """Support sampling made no progress for a long time (see SupportSampler.sample)."""


class SupportSampler:
    def __init__(self, con, schemas, assets: dict, readers: dict, base_fp: dict,
                 targets: dict, counts: dict, row_mode: str, threads: int = 4,
                 stall_s: float = 1800.0):
        """
        assets:  key -> (owner, duck sql)
        readers: table key -> list of asset keys that read it
        base_fp: key -> fingerprint of the unmodified answer (None = skip)
        targets: table key -> (schema, name) of the deletable base table, or None
        counts:  table key -> row count used for sampling rows
        """
        self.con, self.schemas = con, schemas
        self.assets, self.readers, self.base_fp = assets, readers, base_fp
        self.targets, self.counts = targets, counts
        self.row_mode, self.threads = row_mode, max(1, int(threads))
        self.stall_s = float(stall_s)
        self._local = threading.local()
        self._dups, self._contig = {}, {}
        self._lock = threading.Lock()
        self._cursor_lock = threading.Lock()
        self._cursors = []

    # ---- cursors ---------------------------------------------------------------------
    def _new_cursor(self):
        with self._cursor_lock:                # creating cursors is not guaranteed thread-safe
            cur = self.con.cursor()
            self._cursors.append(cur)
        try:                                   # never look up Python variables from worker threads
            cur.execute("SET python_enable_replacements = false")
        except Exception:
            pass
        return cur

    def _cur(self):
        cur = getattr(self._local, "cur", None)
        if cur is None:
            cur = self._new_cursor()
            self._local.cur = cur
        return cur

    def close(self):
        """Close every cursor this sampler opened (the database file stays open otherwise)."""
        with self._cursor_lock:
            curs, self._cursors = self._cursors, []
        for cur in curs:
            try:
                cur.close()
            except Exception:
                pass
        self._local = threading.local()

    @staticmethod
    def _rollback(cur):
        try:
            cur.execute("ROLLBACK")
        except Exception:
            pass

    # ---- table facts (computed once, under a lock) ---------------------------------
    def _table_facts(self, tbl):
        with self._lock:
            if tbl not in self._contig:
                name = f"{quote_ident(tbl[0])}.{quote_ident(tbl[1])}"
                cur = self._new_cursor()
                n, mx = cur.execute(f"SELECT count(*), max(rowid) FROM {name}").fetchone()
                contiguous = (n == 0) or (mx == n - 1)
                if self.row_mode == "unique_only":
                    # Equivalent to the historical value-match test: a row is deletable only
                    # if no other row has the same values in every column (NULLs equal).
                    cols = [d[0] for d in cur.execute(f"SELECT * FROM {name} LIMIT 0").description]
                    if any('"' in c for c in cols):
                        # The historical test quoted column names without escaping, so it
                        # failed on every row of such a table.
                        self._dups[tbl] = _ALL_ROWS
                    else:
                        part = ", ".join(quote_ident(c) for c in cols)
                        dup = cur.execute(
                            f"SELECT rowid FROM (SELECT rowid, count(*) OVER (PARTITION BY {part}) AS __c "
                            f"FROM {name}) WHERE __c > 1").fetchall()
                        self._dups[tbl] = {int(r[0]) for r in dup}
                self._contig[tbl] = contiguous
                cur.close()
                with self._cursor_lock:
                    if cur in self._cursors:
                        self._cursors.remove(cur)
            return self._contig[tbl], self._dups.get(tbl, set())

    # ---- one candidate -------------------------------------------------------------
    def _evaluate(self, cand):
        b, r, track, record = cand
        tbl = self.targets.get(b)
        if tbl is None or r < 0:
            return None
        contiguous, dups = self._table_facts(tbl)
        cur = self._cur()
        name = f"{quote_ident(tbl[0])}.{quote_ident(tbl[1])}"
        if contiguous:
            rid = int(r)
        else:
            got = cur.execute(f"SELECT rowid FROM {name} ORDER BY rowid LIMIT 1 OFFSET {int(r)}").fetchone()
            if got is None:
                return None
            rid = int(got[0])
        if self.row_mode == "unique_only" and rid in dups:
            return None
        t0, attempt = time.perf_counter(), 0
        while True:
            try:
                cur.execute("BEGIN TRANSACTION")
                deleted = cur.execute(f"DELETE FROM {name} WHERE rowid = {rid}").fetchone()[0]
                break
            except duckdb.TransactionException:
                self._rollback(cur)             # another transaction holds this row: wait for it
                stopflag.check()
                if time.perf_counter() - t0 > self.stall_s:
                    raise SamplingStalled(f"row {rid} of {name} stayed locked for "
                                          f"{self.stall_s / 60:.0f} minutes") from None
                time.sleep(0.002 * (1 + attempt % 25))
                attempt += 1
            except Exception as e:
                self._rollback(cur)
                _stop_on(e)
                return None
        try:
            if deleted != 1:
                return None
            changed = {}
            for k in self.readers.get(b, ()):
                if track is not None and k not in track:
                    continue
                base = self.base_fp.get(k)
                if base is None:
                    continue
                own, duck = self.assets[k]
                try:
                    cur.execute(f"SET search_path={sql_str(search_path(own, self.schemas))}")
                    fp = cur.execute(fp_sql(duck)).fetchone()
                except Exception as e:          # the query's own error: unchanged, as historically
                    _stop_on(e)
                    continue
                if fp != base:
                    changed[k] = fp if record else True
            return changed
        finally:
            self._rollback(cur)

    # ---- a support set -------------------------------------------------------------
    def _result(self, fut, futs, clock, label):
        """Wait for one candidate. The stall clock restarts whenever any candidate of the
        batch finishes, so slow but moving work is never cut off."""
        poll = min(30.0, max(0.5, self.stall_s / 4))
        while True:
            try:
                return fut.result(timeout=poll)
            except FuturesTimeout:
                n_done = sum(g.done() for g in futs)
                now = time.perf_counter()
                if n_done > clock[0]:
                    clock[0], clock[1] = n_done, now
                elif now - clock[1] > self.stall_s:
                    raise SamplingStalled(
                        f"{label}: no deletion finished for {self.stall_s / 60:.0f} minutes") from None

    def sample(self, tables: list, weights, n: int, table_seed: int, row_seed: int,
               checkpoints=(), track=None, record_fp=False, max_draw_factor=20,
               label="support") -> SampleResult:
        """Accept the first ``n`` candidates, in draw order, whose deletion succeeds.

        With ``threads == 1`` candidates are evaluated in this thread, one at a time;
        otherwise a thread pool evaluates each batch ahead of time. A (table, row) drawn
        twice in one batch is evaluated once. The accepted set, and every count reported,
        is the same for any number of threads. If no candidate finishes for ``stall_s``
        seconds, SamplingStalled is raised.
        """
        accepted, incidence, changed_fp = [], {}, {}
        drawn = evaluated = skipped = 0
        cps = sorted(set(int(c) for c in checkpoints if c <= n))
        cp_sec = {}
        t0 = time.perf_counter()
        w = np.asarray(weights, float)
        if len(tables) == 0 or n <= 0 or not np.isfinite(w).all() or w.sum() <= 0:
            LOG.warning("%s: nothing to sample from", label)
            return SampleResult(n=0, accepted=[], incidence={}, changed_fp={}, draws=0, skipped=0,
                                seconds=0.0, checkpoint_seconds={})
        w = w / w.sum()
        rng_t = np.random.default_rng(table_seed)
        rng_r = np.random.default_rng(row_seed)
        track = set(track) if track is not None else None
        last_log = t0
        ex = ThreadPoolExecutor(max_workers=self.threads) if self.threads > 1 else None
        abandon = False
        try:
            while len(accepted) < n and drawn < n * max_draw_factor:
                rate = len(accepted) / evaluated if evaluated else 0.7
                need = n - len(accepted)
                size = int(min(self.threads * 16, max(self.threads, math.ceil(need / max(rate, 0.05) * 1.1))))
                size = min(size, n * max_draw_factor - drawn)
                batch = []
                for _ in range(size):
                    b = tables[int(rng_t.choice(len(tables), p=w))]
                    cnt = self.counts.get(b, 0)
                    r = int(rng_r.integers(cnt)) if cnt > 0 else -1
                    batch.append((b, r, track, record_fp))
                drawn += len(batch)
                futmap, cache, futs, clock = {}, {}, [], [0, time.perf_counter()]
                if ex is not None:
                    for cand in batch:
                        if (cand[0], cand[1]) not in futmap:
                            futmap[(cand[0], cand[1])] = ex.submit(self._evaluate, cand)
                    futs = list(futmap.values())
                try:
                    for cand in batch:
                        if len(accepted) >= n:
                            break
                        key = (cand[0], cand[1])
                        if ex is not None:
                            res = self._result(futmap[key], futs, clock, label)
                        else:
                            if key not in cache:
                                cache[key] = self._evaluate(cand)
                            res = cache[key]
                        evaluated += 1
                        if res is None:
                            skipped += 1
                            continue
                        idx = len(accepted)
                        accepted.append(key)
                        for k, v in res.items():
                            incidence.setdefault(k, []).append(idx)
                            if record_fp:
                                changed_fp.setdefault(k, {})[idx] = v
                        if cps and idx + 1 in cps:
                            cp_sec[idx + 1] = round(time.perf_counter() - t0, 1)
                finally:
                    for f in futs:               # candidates past the n-th are not needed
                        f.cancel()
                running = [f for f in futs if not f.done()]
                if running:                      # let started ones roll back, but never wait forever
                    _, not_done = wait(running, timeout=self.stall_s)
                    if not_done:
                        raise SamplingStalled(f"{label}: {len(not_done)} deletions did not finish "
                                              f"within {self.stall_s / 60:.0f} minutes")
                now = time.perf_counter()
                if now - last_log > 30:
                    last_log = now
                    LOG.info("%s: %d/%d neighbours accepted (%d candidates, %.0f s)", label,
                             len(accepted), n, evaluated, now - t0)
        except BaseException:
            abandon = True                       # never wait on a thread that may be stuck
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if not abandon:
                if ex is not None:
                    ex.shutdown(wait=True, cancel_futures=True)
                self.close()
        if len(accepted) < n:
            LOG.warning("%s: only %d of %d neighbours after %d candidates", label, len(accepted), n, evaluated)
        return SampleResult(n=len(accepted), accepted=accepted, incidence=incidence,
                            changed_fp=changed_fp, draws=evaluated, skipped=skipped,
                            seconds=round(time.perf_counter() - t0, 1), checkpoint_seconds=cp_sec)


def base_fingerprints(con, schemas, assets: dict) -> dict:
    out = {}
    for k, (own, duck) in assets.items():
        try:
            con.execute(f"SET search_path={sql_str(search_path(own, schemas))}")
            out[k] = con.execute(fp_sql(duck)).fetchone()
        except Exception:
            stopflag.check()
            out[k] = None
    con.execute("SET search_path='main'")
    return out


def table_counts(con, keys, targets: dict | None = None) -> dict:
    """Row counts per table key; keys that fail to count are left out (historical)."""
    out = {}
    for b in keys:
        try:
            out[b] = int(con.execute(f"SELECT count(*) FROM {quote_ident(b[0])}.{quote_ident(b[1])}").fetchone()[0])
        except Exception:
            stopflag.check()
    return out
