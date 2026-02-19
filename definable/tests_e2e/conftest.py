"""
Shared fixtures for the entire test suite.

Architecture:
  - Unit tests: use only pure Python fixtures (no external services)
  - Integration tests: use session-scoped real API clients for speed
  - Behavioral tests: use fresh agent per test (function-scoped)
  - Contract tests: each implementation gets its own fixture

ALL tests that require API credentials will FAIL (not skip) when keys are missing.
Set keys by running: source .env.test
"""

import os

import pytest

from definable.media import File
from definable.model.message import Message

# Load .env.test if present (handles "export KEY=value" shell format)
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env.test")
if os.path.isfile(_env_path):
  try:
    with open(_env_path) as _f:
      for _line in _f:
        _line = _line.strip()
        if not _line or _line.startswith("#"):
          continue
        _line = _line.removeprefix("export ").strip()
        if "=" in _line:
          _k, _, _v = _line.partition("=")
          _v = _v.strip().strip('"').strip("'")
          os.environ.setdefault(_k.strip(), _v)
  except Exception:
    pass


# ---------------------------------------------------------------------------
# Auto-use cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_async_clients():
  """
  Reset all cached async clients between tests.

  Session-scoped model fixtures (openai_model, etc.) cache AsyncOpenAI clients.
  Each test runs in a function-scoped event loop. When a test ends its loop closes,
  leaving the cached client bound to a dead event loop. The next test's API call
  triggers an HTTP/2 connection teardown on the dead loop → 'Event loop is closed'.

  Fix: clear all model async_client caches after each test so the next test
  creates a fresh client in the current event loop.
  """
  yield
  # Reset global async httpx client
  try:
    from definable.utils.http import _async_client_lock
    import definable.utils.http as http_module

    with _async_client_lock:
      if http_module._global_async_client is not None:
        http_module._global_async_client = None
  except Exception:
    pass
  # Reset cached async clients on all live model instances (OpenAI, DeepSeek, etc.)
  # They cache AsyncOpenAI/AsyncClient objects that are bound to the previous event loop.
  try:
    import gc
    from definable.model.base import Model  # base class for all providers

    for obj in gc.get_objects():
      if isinstance(obj, Model):
        try:
          if getattr(obj, "async_client", None) is not None:
            obj.async_client = None  # type: ignore[assignment,attr-defined]
        except Exception:
          pass
  except Exception:
    pass


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def user_message():
  """Single user message for minimal agent invocations."""
  return [Message(role="user", content="What is 2+2? Answer with just the number.")]


@pytest.fixture
def assistant_message_factory():
  """Factory for empty assistant message placeholders."""

  def _create():
    return Message(role="assistant", content=None)

  return _create


# ---------------------------------------------------------------------------
# Real model fixtures (session-scoped — expensive to create)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def openai_model():
  """Real OpenAI model (gpt-4o-mini). FAILS if OPENAI_API_KEY not set."""
  from definable.model.openai import OpenAIChat

  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    pytest.fail("OPENAI_API_KEY not set — required for e2e tests. Run: source .env.test")
  return OpenAIChat(api_key=api_key, id="gpt-4o-mini")


@pytest.fixture(scope="session")
def deepseek_model():
  """Real DeepSeek model. FAILS if DEEPSEEK_API_KEY not set."""
  from definable.model.deepseek import DeepSeekChat

  api_key = os.getenv("DEEPSEEK_API_KEY")
  if not api_key:
    pytest.fail("DEEPSEEK_API_KEY not set — required for e2e tests. Run: source .env.test")
  return DeepSeekChat(api_key=api_key)


@pytest.fixture(scope="session")
def moonshot_model():
  """Real Moonshot model. FAILS if MOONSHOT_API_KEY not set."""
  from definable.model.moonshot import MoonshotChat

  api_key = os.getenv("MOONSHOT_API_KEY")
  if not api_key:
    pytest.fail("MOONSHOT_API_KEY not set — required for e2e tests. Run: source .env.test")
  return MoonshotChat(api_key=api_key)


@pytest.fixture(scope="session")
def xai_model():
  """Real xAI model. FAILS if XAI_API_KEY not set."""
  from definable.model.xai import xAI

  api_key = os.getenv("XAI_API_KEY")
  if not api_key:
    pytest.fail("XAI_API_KEY not set — required for e2e tests. Run: source .env.test")
  return xAI(api_key=api_key, id="grok-3")


# ---------------------------------------------------------------------------
# Real embedder fixtures (session-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def openai_embedder():
  """Real OpenAI embedder (text-embedding-3-small). FAILS if OPENAI_API_KEY not set."""
  from definable.knowledge.embedder.openai import OpenAIEmbedder

  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    pytest.fail("OPENAI_API_KEY not set — required for e2e tests. Run: source .env.test")
  return OpenAIEmbedder(api_key=api_key, id="text-embedding-3-small", dimensions=1536)


@pytest.fixture(scope="session")
def voyage_embedder():
  """Real VoyageAI embedder. FAILS if VOYAGEAI_API_KEY not set."""
  from definable.knowledge.embedder.voyageai import VoyageAIEmbedder

  api_key = os.getenv("VOYAGEAI_API_KEY")
  if not api_key:
    pytest.fail("VOYAGEAI_API_KEY not set — required for e2e tests. Run: source .env.test")
  return VoyageAIEmbedder(api_key=api_key)


# ---------------------------------------------------------------------------
# VectorDB fixtures (function-scoped — each test gets a clean store)
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_vectordb(openai_embedder):
  """Fresh InMemoryVectorDB per test with a real embedder."""
  from definable.vectordb import InMemoryVectorDB

  db = InMemoryVectorDB(embedder=openai_embedder)
  yield db
  db.drop()


# ---------------------------------------------------------------------------
# Synthetic file fixtures (for reader tests)
# ---------------------------------------------------------------------------


import struct
import zlib


@pytest.fixture
def sample_csv_file():
  """Minimal CSV file."""
  data = b"product,units,revenue\nWidget A,150,4500\nWidget B,200,8000\n"
  return File(content=data, filename="sales.csv", mime_type="text/csv")


@pytest.fixture
def sample_json_file():
  """Minimal JSON config file."""
  data = b'{"app": "definable", "version": "0.2.8", "debug": true}'
  return File(content=data, filename="config.json", mime_type="application/json")


@pytest.fixture
def sample_text_file():
  """Minimal plain-text file."""
  data = b"Hello, world! This is a sample text document for testing."
  return File(content=data, filename="readme.txt", mime_type="text/plain")


@pytest.fixture
def sample_png_bytes():
  """Minimal valid 1×1 white PNG (67 bytes)."""
  signature = b"\x89PNG\r\n\x1a\n"
  ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
  ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
  ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
  raw_data = zlib.compress(b"\x00\xff\xff\xff")
  idat_crc = zlib.crc32(b"IDAT" + raw_data) & 0xFFFFFFFF
  idat = struct.pack(">I", len(raw_data)) + b"IDAT" + raw_data + struct.pack(">I", idat_crc)
  iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
  iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
  return signature + ihdr + idat + iend
