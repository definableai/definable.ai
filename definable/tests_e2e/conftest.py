"""Shared fixtures for E2E tests."""

from os import getenv
from typing import Callable

import pytest

from definable.models.message import Message


def requires_env(var_name: str):
  """Skip decorator for tests requiring specific env vars."""
  return pytest.mark.skipif(
    not getenv(var_name),
    reason=f"{var_name} environment variable not set",
  )


@pytest.fixture(autouse=True)
def reset_async_client():
  """Reset global async httpx client between tests to avoid event loop issues."""
  yield
  # Cleanup after each test to prevent event loop closed errors
  try:
    from definable.utils.http import _async_client_lock
    import definable.utils.http as http_module

    with _async_client_lock:
      if http_module._global_async_client is not None:
        # Don't await close - just set to None so a new one is created
        http_module._global_async_client = None
  except Exception:
    pass


@pytest.fixture
def simple_messages() -> list[Message]:
  """Return a simple user message for basic invocation tests."""
  return [Message(role="user", content="What is 2+2? Answer with just the number.")]


@pytest.fixture
def assistant_message() -> Callable[[], Message]:
  """Factory fixture for creating empty assistant messages."""

  def _create() -> Message:
    return Message(role="assistant", content=None)

  return _create
