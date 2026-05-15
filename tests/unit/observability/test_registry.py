from __future__ import annotations

from dataclasses import dataclass

import pytest

from definable.observability.registry import AgentRegistry


@dataclass
class _StubAgent:
  name: str


@pytest.fixture(autouse=True)
def _clean_registry():
  AgentRegistry.get().clear()
  yield
  AgentRegistry.get().clear()


def test_singleton_returns_same_instance() -> None:
  assert AgentRegistry.get() is AgentRegistry.get()


def test_register_and_lookup() -> None:
  reg = AgentRegistry.get()
  a = _StubAgent(name="agent-1")
  reg.register(a)
  assert reg.lookup("agent-1") is a


def test_list_returns_all() -> None:
  reg = AgentRegistry.get()
  reg.register(_StubAgent(name="a"))
  reg.register(_StubAgent(name="b"))
  names = sorted(x.name for x in reg.list())
  assert names == ["a", "b"]


def test_unregister() -> None:
  reg = AgentRegistry.get()
  reg.register(_StubAgent(name="gone"))
  reg.unregister("gone")
  assert reg.lookup("gone") is None


def test_lookup_unknown_returns_none() -> None:
  assert AgentRegistry.get().lookup("nope") is None


def test_reregister_replaces() -> None:
  reg = AgentRegistry.get()
  first = _StubAgent(name="dup")
  second = _StubAgent(name="dup")
  reg.register(first)
  reg.register(second)
  assert reg.lookup("dup") is second
  assert len(reg.list()) == 1
