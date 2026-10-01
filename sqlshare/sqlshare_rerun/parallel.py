"""Worker processes that cannot hang the run.

Every parallel step (lattice chunks, fit-freeze-evaluate runs) goes through ``run_pool``.
Workers always start fresh ("spawn", the Windows default), never by fork. If workers
cannot be started (for example because security software blocks python.exe), if a worker
dies, or if no task finishes for ``stall_s`` seconds (a frozen worker), the pool is
stopped and the remaining tasks run in this process. Tasks are pure functions of their
inputs, so the results are identical; only the speed changes. A task's own error is a
bug and is raised as usual.
"""
from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

from .util import LOG

SPAWN = multiprocessing.get_context("spawn")


def _stop(ex: ProcessPoolExecutor, grace: float = 0.0) -> None:
    """Shut the pool down without ever waiting long: idle workers get ``grace`` seconds to
    exit on their own, then every remaining worker is terminated."""
    procs = list((getattr(ex, "_processes", None) or {}).values())
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    deadline = time.perf_counter() + grace
    for p in procs:
        try:
            p.join(max(0.0, deadline - time.perf_counter()))
        except Exception:
            pass
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
                p.join(5)
        except Exception:
            pass


def run_pool(fn, tasks, workers: int, label: str, stall_s: float = 1800.0) -> list:
    """``[fn(t) for t in tasks]``, in worker processes when ``workers > 1``."""
    tasks = list(tasks)
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    results, done = [None] * len(tasks), [False] * len(tasks)
    reason = None
    try:
        ex = ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=SPAWN)
    except (OSError, ValueError) as e:
        LOG.warning("%s: worker processes are not available (%s); running in this process", label, e)
        return [fn(t) for t in tasks]
    try:
        futs = {}
        try:
            for i, t in enumerate(tasks):
                futs[ex.submit(fn, t)] = i
        except (OSError, BrokenProcessPool, RuntimeError) as e:
            reason = f"worker processes could not be started ({type(e).__name__}: {e})"
        pending = set(futs)
        t0 = last = time.perf_counter()
        while pending and reason is None:
            finished, pending = wait(pending, timeout=stall_s, return_when=FIRST_COMPLETED)
            if not finished:
                reason = f"no task finished for {stall_s / 60:.0f} min"
                break
            for f in finished:
                i = futs[f]
                results[i] = f.result()       # a task's own error is raised here, as a bug
                done[i] = True
            now = time.perf_counter()
            if now - last > 60 and pending:
                last = now
                LOG.info("%s: %d of %d tasks done (%.0f s)", label, done.count(True), len(tasks), now - t0)
    except BrokenProcessPool as e:
        reason = f"a worker process stopped unexpectedly ({e})"
    except BaseException:
        _stop(ex)
        raise
    _stop(ex, grace=10.0 if reason is None else 0.0)
    if reason is None:
        return results
    LOG.warning("%s: %s. Finishing the remaining %d tasks in this process (same results, slower). "
                "If you see this message again, rerun with -Workers 1.", label, reason, done.count(False))
    for i, t in enumerate(tasks):
        if not done[i]:
            results[i] = fn(t)
    return results
