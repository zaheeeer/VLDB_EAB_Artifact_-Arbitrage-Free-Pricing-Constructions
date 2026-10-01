"""Logging, checkpoints, and small helpers shared by every stage."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LOG = logging.getLogger("sqlshare_rerun")


def setup_logging(log_file: Path | None = None, verbose: bool = False) -> None:
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    for stream in (sys.stdout, sys.stderr):
        try:                       # never crash on a character the console cannot show
            stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        path = str(log_file.resolve())
        for h in [h for h in LOG.handlers if isinstance(h, logging.FileHandler) and h.baseFilename != path]:
            LOG.removeHandler(h)            # one log file per run; release the old one
            h.close()
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == path for h in LOG.handlers):
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(fmt)
            # The file comes first: if the console window is paused (a click in an old
            # PowerShell window), the log file still records everything.
            LOG.handlers.insert(0, fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in LOG.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        LOG.addHandler(sh)
    # sqlglot prints one warning per unsupported CONVERT style; keep the console readable.
    logging.getLogger("sqlglot").setLevel(logging.ERROR)


@contextmanager
def timed(label: str, sink: dict | None = None):
    t0 = time.perf_counter()
    LOG.info("%s ...", label)
    yield
    dt = time.perf_counter() - t0
    if sink is not None:
        sink[label] = round(dt, 2)
    LOG.info("%s done in %.1f s", label, dt)


def replace_file(src, dst, max_wait: float = 900.0) -> None:
    """os.replace that waits out locks: an antivirus scan (seconds) or a results file left
    open in Excel or WPS (until it is closed; a note is logged every 30 s)."""
    t0 = last = time.perf_counter()
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            now = time.perf_counter()
            if now - t0 > max_wait:
                raise PermissionError(f"cannot write {dst}: another program keeps it open. Close it "
                                      f"(Excel, WPS, an editor) and run the same command again.") from None
            if now - last > 30:
                last = now
                LOG.warning("waiting to write %s: close it if it is open in Excel, WPS or an editor", dst)
            time.sleep(1.0)


def _sync(fh) -> None:
    fh.flush()
    try:
        os.fsync(fh.fileno())            # the data is on disk before the rename (power cuts)
    except OSError:
        pass


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        _sync(fh)
    replace_file(tmp, path)


def write_csv(df, path: Path, **kw) -> None:
    """DataFrame.to_csv, written to a temporary file first, then moved into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    kw.setdefault("index", False)
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, **kw)
        _sync(fh)
    replace_file(tmp, path)


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        _sync(fh)
    replace_file(tmp, path)


def load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def read_csv(path, **kw):
    """pandas.read_csv with exact float parsing (the default parser can be 1 ulp off)."""
    import pandas as pd
    return pd.read_csv(path, float_precision="round_trip", **kw)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
        _sync(fh)
    replace_file(tmp, path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def stage_done(marker: Path, key: str) -> bool:
    """A stage is complete when its marker exists and was written for the same settings."""
    if not marker.exists():
        return False
    try:
        return read_json(marker).get("key") == key
    except Exception:
        return False


def mark_done(marker: Path, key: str, extra: dict | None = None) -> None:
    write_json({"key": key, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}, marker)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count()}
    for mod in ("duckdb", "numpy", "pandas", "scipy", "sqlglot"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "missing"
    return info


_ILLEGAL = str.maketrans(':<>|"?*', "_______")


def windows_name(name: str) -> str:
    """The file name the historical Windows extraction produced for a zip member.

    CPython's zipfile replaces each of ``:<>|"?*`` with ``_`` and strips trailing dots and
    spaces on Windows. The historical tables were named after these on-disk names, so the
    rerun uses the same names on every operating system.
    """
    return name.translate(_ILLEGAL).rstrip(" .")


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def sql_str(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"
