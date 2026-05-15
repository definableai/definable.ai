from __future__ import annotations

import socket

import pytest

from definable.db import close_all
from definable.observability.registry import AgentRegistry
from definable.observability.server import ObservabilityServer, _pick_free_port


@pytest.fixture(autouse=True)
async def _clean():
  AgentRegistry.get().clear()
  ObservabilityServer._instance = None
  await close_all()
  yield
  ObservabilityServer._instance = None
  AgentRegistry.get().clear()
  await close_all()


def test_singleton_returns_same_instance() -> None:
  a = ObservabilityServer.singleton()
  b = ObservabilityServer.singleton()
  assert a is b


def test_pick_free_port_skips_occupied() -> None:
  # Bind one socket to occupy the preferred port; the picker should jump.
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
    occupied.bind(("127.0.0.1", 0))
    busy_port = occupied.getsockname()[1]
    chosen = _pick_free_port("127.0.0.1", busy_port)
  assert chosen != busy_port


def test_url_property_before_open() -> None:
  s = ObservabilityServer.singleton()
  assert s.url.startswith("http://127.0.0.1:")
