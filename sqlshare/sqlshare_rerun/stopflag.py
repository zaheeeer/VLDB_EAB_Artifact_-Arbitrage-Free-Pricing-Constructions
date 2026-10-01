"""Ctrl+C that always stops the run.

DuckDB turns Ctrl+C during a query into an ordinary ``RuntimeError("Query interrupted")``,
which an error handler around a query would record as a failed query, and the run would
carry on with a wrong result. The signal handler below records the request, and
``check()``, called in every such handler, turns it back into KeyboardInterrupt.
"""
import signal
import threading

STOP = threading.Event()


def _handler(signum, frame):
    STOP.set()
    raise KeyboardInterrupt


def install() -> None:
    for name in ("SIGINT", "SIGBREAK"):          # SIGBREAK: Ctrl+Break on Windows
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def check() -> None:
    """Raise KeyboardInterrupt if the user asked to stop."""
    if STOP.is_set():
        raise KeyboardInterrupt


def out_of_memory(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "outofmemory" in text.replace(" ", "") or "out of memory" in text
