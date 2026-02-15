"""Shared fixtures for E2E tests."""

import struct
import zlib
from os import getenv
from typing import Callable

import pytest

from definable.media import File
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


# ---------------------------------------------------------------------------
# Shared model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_model():
  """OpenAI model for testing. Skips if no API key."""
  from definable.models.openai import OpenAIChat

  api_key = getenv("OPENAI_API_KEY")
  if not api_key:
    pytest.skip("OPENAI_API_KEY environment variable not set")
  return OpenAIChat(api_key=api_key, id="gpt-4o-mini")


@pytest.fixture
def deepseek_model():
  """DeepSeek model for testing. Skips if no API key."""
  from definable.models.deepseek import DeepSeekChat

  api_key = getenv("DEEPSEEK_API_KEY")
  if not api_key:
    pytest.skip("DEEPSEEK_API_KEY environment variable not set")
  return DeepSeekChat(api_key=api_key)


@pytest.fixture
def moonshot_model():
  """Moonshot model for testing. Skips if no API key."""
  from definable.models.moonshot import MoonshotChat

  api_key = getenv("MOONSHOT_API_KEY")
  if not api_key:
    pytest.skip("MOONSHOT_API_KEY environment variable not set")
  return MoonshotChat(api_key=api_key)


@pytest.fixture
def xai_model():
  """xAI model for testing. Skips if no API key."""
  from definable.models.xai import xAI

  api_key = getenv("XAI_API_KEY")
  if not api_key:
    pytest.skip("XAI_API_KEY environment variable not set")
  return xAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Synthetic file data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv_file():
  """Sample CSV file for testing."""
  csv_data = b"product,units,revenue\nWidget A,150,$4500\nWidget B,200,$8000\nWidget C,75,$3000\n"
  return File(content=csv_data, filename="sales.csv", mime_type="text/csv")


@pytest.fixture
def sample_json_file():
  """Sample JSON config file for testing."""
  json_data = b'{"app": "test-app", "version": "1.0.0", "debug": true, "max_retries": 3}'
  return File(content=json_data, filename="config.json", mime_type="application/json")


@pytest.fixture
def sample_text_file():
  """Sample plain text file for testing."""
  return File(
    content=b"Hello, world! This is a sample text document for testing purposes.",
    filename="readme.txt",
    mime_type="text/plain",
  )


@pytest.fixture
def sample_png_bytes():
  """Minimal valid 1x1 white PNG (67 bytes)."""

  def _make_png():
    signature = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk: 1x1, 8-bit RGB
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    # IDAT chunk: single white pixel
    raw_data = zlib.compress(b"\x00\xff\xff\xff")
    idat_crc = zlib.crc32(b"IDAT" + raw_data) & 0xFFFFFFFF
    idat = struct.pack(">I", len(raw_data)) + b"IDAT" + raw_data + struct.pack(">I", idat_crc)
    # IEND chunk
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return signature + ihdr + idat + iend

  return _make_png()


@pytest.fixture
def sample_wav_bytes():
  """Minimal valid WAV file (0.1s silence, 8kHz mono 8-bit)."""

  def _make_wav():
    sample_rate = 8000
    duration_s = 0.1
    num_samples = int(sample_rate * duration_s)
    data = b"\x80" * num_samples  # silence for unsigned 8-bit PCM
    data_size = len(data)
    # RIFF header
    riff = b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
    # fmt sub-chunk: PCM, mono, 8kHz, 8-bit
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate, 1, 8)
    # data sub-chunk
    data_chunk = b"data" + struct.pack("<I", data_size) + data
    return riff + fmt + data_chunk

  return _make_wav()
