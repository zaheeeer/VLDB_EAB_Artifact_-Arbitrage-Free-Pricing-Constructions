"""Command line: python -m sqlshare_rerun <command> [options]

Commands
  doctor     one-minute check of this computer (run this first)
  download   fetch the SQLShare release (3.6 GB, resumable)
  prepare    parse, select markets, extract, load DuckDB, bind and execute (shared)
  lattice    verified derivability lattice          (per protocol)
  main       largest market: Table 6, falsification, Tables 3-5 SQLShare columns
  markets    all markets: Table 7, Figure 2(b), cross-market falsification
  report     paper-ready tables and a comparison with the values in the paper
  all        everything above, for --protocol paper, corrected, or both (default)
  ablation   main market with each evaluation choice switched alone (after "all")
  selftest   end-to-end run on a small synthetic release (a few minutes)
"""
from __future__ import annotations

import argparse
import faulthandler
import multiprocessing
import os
import sys
import time
from pathlib import Path

from . import stopflag
from .config import PROTOCOLS, make_settings
from .util import LOG, environment_info, setup_logging, write_json

_DUMP_FILES = []            # [crash log, stack dump log] of the current run
_DUMP_ROOT = []


def _args(argv=None):
    p = argparse.ArgumentParser(prog="python -m sqlshare_rerun", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["doctor", "download", "prepare", "lattice", "main", "markets",
                                       "report", "all", "ablation", "selftest"])
    p.add_argument("--protocol", default="both", choices=list(PROTOCOLS) + ["both"])
    p.add_argument("--root", default=".", help="working folder (default: current folder)")
    p.add_argument("--zip", default=None, help="path to an already downloaded release zip")
    p.add_argument("--workers", type=int, default=0, help="processes (default: cores - 1; 1 = none)")
    p.add_argument("--threads", type=int, default=0, help="sampling threads (default: cores; 1 = none)")
    p.add_argument("--duckdb-threads", type=int, default=0)
    p.add_argument("--memory-limit", default="", help='DuckDB memory limit, e.g. "8GB"')
    p.add_argument("--quick", action="store_true", help="scaled-down run for a fast check")
    p.add_argument("--force", action="store_true", help="redo the named stage even if complete")
    p.add_argument("--for-full-run", action="store_true", help="doctor: require the disk space of a full run")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--mode", default="both", choices=["both", "add", "drop"],
                   help="ablation: switch each choice to its alternative from paper (add), back from corrected (drop), or both")
    p.add_argument("--replicates", type=int, default=2,
                   help="ablation: extra sampling-seed sets for the two end points (default 2)")
    p.add_argument("--only", default="", help="ablation: comma-separated switch names (default: all)")
    p.add_argument("--scale", type=int, default=1, help=argparse.SUPPRESS)   # selftest: stress size
    return p.parse_args(argv)


def _settings(a, protocol, **extra):
    return make_settings(protocol, root=Path(a.root), quick=a.quick,
                         zip_path=Path(a.zip) if a.zip else None, workers=a.workers,
                         threads=a.threads, duckdb_threads=a.duckdb_threads,
                         memory_limit=a.memory_limit, **extra)


def _stack_dumps(root: Path) -> None:
    """Write every thread's stack to results/stack_dumps.log every 30 minutes, and on a
    hard crash to results/crash.log. If the run ever seems stuck, those files show where."""
    root = Path(root).resolve()
    if _DUMP_ROOT and _DUMP_ROOT[0] == root:
        try:
            faulthandler.dump_traceback_later(1800, repeat=True, file=_DUMP_FILES[1])
        except Exception:
            pass
        return
    _close_dump_files()
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    crash = open(out / "crash.log", "a", encoding="utf-8")
    dumps = open(out / "stack_dumps.log", "a", encoding="utf-8")
    _DUMP_FILES.extend([crash, dumps])
    _DUMP_ROOT.append(root)
    try:
        faulthandler.enable(file=crash, all_threads=True)
        dumps.write(f"\n==== run started {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        dumps.flush()
        faulthandler.dump_traceback_later(1800, repeat=True, file=dumps)
    except Exception:
        pass


def _stop_stack_dumps() -> None:
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass


def _close_dump_files() -> None:
    _stop_stack_dumps()
    if _DUMP_FILES:
        try:
            faulthandler.enable(file=sys.__stderr__ or sys.stderr, all_threads=True)
        except Exception:
            pass
    for fh in _DUMP_FILES:
        try:
            fh.close()
        except Exception:
            pass
    _DUMP_FILES.clear()
    _DUMP_ROOT.clear()


def run(a, **extra) -> int:
    from . import stages as S
    from .report import cross_protocol, stage_report

    root = Path(a.root)
    setup_logging(root / "results" / "run.log", a.verbose)
    _stack_dumps(root)
    protocols = list(PROTOCOLS) if a.protocol == "both" else [a.protocol]
    LOG.info("sqlshare_rerun | command %s | protocols %s | %s", a.command, protocols, environment_info())
    t0 = time.perf_counter()
    base = _settings(a, protocols[0], **extra)
    LOG.info("folder %s | workers %d | sampling threads %d", root.resolve(), base.n_workers(), base.n_threads())
    if a.command == "download":
        S.stage_download(base)
        return 0
    if a.command == "ablation":
        from .ablation import run_ablation
        only = [x.strip() for x in a.only.split(",") if x.strip()] or None
        run_ablation(root, a.quick, a.mode, max(0, a.replicates), only,
                     common={"zip_path": Path(a.zip) if a.zip else None, "workers": a.workers,
                             "threads": a.threads, "duckdb_threads": a.duckdb_threads,
                             "memory_limit": a.memory_limit, **extra})
        LOG.info("finished in %.1f min", (time.perf_counter() - t0) / 60)
        return 0
    if a.command in ("prepare", "all"):
        S.stage_prepare(base, force=a.force and a.command == "prepare")
        if a.command == "prepare":
            return 0
    for proto in protocols:
        s = _settings(a, proto, **extra)
        write_json(s.as_dict(), s.results_dir / "settings.json")
        if a.command in ("lattice", "all"):
            S.stage_lattice(s, force=a.force and a.command == "lattice")
        if a.command in ("main", "all"):
            S.stage_main_market(s, force=a.force and a.command == "main")
        if a.command in ("markets", "all"):
            S.stage_markets(s, force=a.force and a.command == "markets")
        if a.command in ("report", "all", "main", "markets"):
            stage_report(s)
    out = cross_protocol(root)
    if out is not None:
        LOG.info("side-by-side comparison: %s and %s", out, root / "results" / "REPORT.md")
    LOG.info("finished in %.1f min", (time.perf_counter() - t0) / 60)
    return 0


def guarded(fn, a, **extra) -> int:
    """Run ``fn`` and turn every failure into a clear message plus a log entry."""
    from .support import SamplingStalled
    try:
        return fn(a, **extra)
    except KeyboardInterrupt:
        LOG.warning("stopped by the user; run the same command again to continue where it stopped")
        for h in LOG.handlers:
            h.flush()
        sys.stdout.flush()
        os._exit(130)               # DuckDB may still be finishing an interrupted query
    except SamplingStalled as e:
        LOG.error("%s. Run the same command again with -Threads 1 (it continues where it stopped). "
                  "Please send results\\run.log and results\\stack_dumps.log.", e)
        faulthandler.dump_traceback(file=_DUMP_FILES[1] if len(_DUMP_FILES) > 1 else sys.stderr,
                                    all_threads=True)
        for h in LOG.handlers:
            h.flush()
        os._exit(3)                 # a stuck database thread would block a normal exit
    except Exception:
        LOG.exception("stopped with an error. Please send results\\run.log. Running the same command "
                      "again continues from the last finished stage.")
        return 1
    finally:
        _close_dump_files()


def main(argv=None) -> int:
    multiprocessing.freeze_support()
    stopflag.install()
    a = _args(argv)
    if a.command == "doctor":
        from .doctor import run_doctor
        return run_doctor(Path(a.root), Path(a.zip) if a.zip else None, a.for_full_run)
    if a.command == "selftest":
        from .selftest import run_selftest
        return guarded(lambda args, **kw: run_selftest(args, scale=max(1, args.scale)), a)
    return guarded(run, a)


if __name__ == "__main__":
    sys.exit(main())
