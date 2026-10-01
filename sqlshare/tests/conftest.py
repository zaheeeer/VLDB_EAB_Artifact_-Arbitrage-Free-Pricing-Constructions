"""Test-session guards.

A test that runs longer than TEST_LIMIT_S is treated as hung: every thread's stack is
written to logs/test_hang.txt (the name of the test is in logs/current_test.txt) and the
session stops with an error instead of waiting forever. run_all.ps1 prints that file.
"""
import faulthandler
import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
TEST_LIMIT_S = float(os.environ.get("SQLSHARE_TEST_LIMIT_S", "900"))
_HANG = []


def pytest_sessionstart(session):
    LOGS.mkdir(exist_ok=True)
    _HANG.append(open(LOGS / "test_hang.txt", "w", encoding="utf-8"))   # empty unless a test hangs


def pytest_sessionfinish(session, exitstatus):
    for fh in _HANG:
        try:
            fh.close()
        except Exception:
            pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    try:
        (LOGS / "current_test.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {item.nodeid}\n",
                                               encoding="utf-8")
    except OSError:
        pass
    if _HANG:
        faulthandler.dump_traceback_later(TEST_LIMIT_S, exit=True, file=_HANG[0])
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
