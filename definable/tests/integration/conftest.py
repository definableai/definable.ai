"""
Integration-level conftest — rate-limit handling and temp DB fixtures.

This conftest adds:
  - pytest_runtest_makereport hook: auto-skip tests that hit 429 rate limits
  - Temp file fixtures for SQLite databases
"""

import contextlib
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Rate-limit 429 auto-skip hook
# ---------------------------------------------------------------------------


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
  """Skip tests that hit rate limits (429) instead of failing.

  Checks three sources: exception message, traceback, and captured stdout/stderr.
  Agent logs often appear in captured output, so all three are necessary.
  """
  outcome = yield
  report = outcome.get_result()

  if report.when == "call" and report.failed:
    if call.excinfo is not None:
      error_msg = str(call.excinfo.value)
      full_repr = str(report.longrepr)
      sections_text = " ".join(content for _, content in report.sections)
      combined = (error_msg + full_repr + sections_text).lower()

      rate_limit_indicators = ["429", "rate limit", "rate_limit", "quota", "resource_exhausted"]
      if any(p in combined for p in rate_limit_indicators):
        report.outcome = "skipped"
        report.longrepr = ("", -1, "Skipped: rate limit (429)")


# ---------------------------------------------------------------------------
# Temp DB fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_storage_db_file():
  """Function-scoped temp SQLite file. Cleaned up after test."""
  fd, path = tempfile.mkstemp(suffix=".db", prefix="definable_test_storage_")
  os.close(fd)
  yield path
  with contextlib.suppress(OSError):
    os.unlink(path)


@pytest.fixture
def temp_memory_db_file():
  """Function-scoped temp SQLite file for memory store. Cleaned up after test."""
  fd, path = tempfile.mkstemp(suffix=".db", prefix="definable_test_memory_")
  os.close(fd)
  yield path
  with contextlib.suppress(OSError):
    os.unlink(path)
