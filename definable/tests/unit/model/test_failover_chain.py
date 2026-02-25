"""Tests for FailoverChain — ordered provider failover."""

from definable.model.resilience.failover import FailoverChain, FailoverEntry
from definable.model.resilience.key_pool import KeyPool


class FakeModel:
  """Minimal fake for testing FailoverEntry."""

  def __init__(self, model_id: str = "fake"):
    self.id = model_id


class TestFailoverEntry:
  def test_defaults(self):
    entry = FailoverEntry(model=FakeModel())  # type: ignore[arg-type]
    assert entry.key_pool is None
    assert entry.priority == 0

  def test_with_key_pool(self):
    pool = KeyPool(keys=["sk-1"])
    entry = FailoverEntry(model=FakeModel(), key_pool=pool, priority=5)  # type: ignore[arg-type]
    assert entry.key_pool is pool
    assert entry.priority == 5


class TestFailoverChainOrdering:
  def test_single_entry(self):
    chain = FailoverChain([FailoverEntry(model=FakeModel("a"), priority=0)])  # type: ignore[arg-type]
    assert len(chain) == 1
    assert chain.primary.model.id == "a"  # type: ignore[attr-defined]

  def test_priority_ordering(self):
    entries = [
      FailoverEntry(model=FakeModel("c"), priority=2),  # type: ignore[arg-type]
      FailoverEntry(model=FakeModel("a"), priority=0),  # type: ignore[arg-type]
      FailoverEntry(model=FakeModel("b"), priority=1),  # type: ignore[arg-type]
    ]
    chain = FailoverChain(entries)
    ids = [e.model.id for e in chain]  # type: ignore[attr-defined]
    assert ids == ["a", "b", "c"]

  def test_same_priority_preserves_order(self):
    entries = [
      FailoverEntry(model=FakeModel("x"), priority=0),  # type: ignore[arg-type]
      FailoverEntry(model=FakeModel("y"), priority=0),  # type: ignore[arg-type]
    ]
    chain = FailoverChain(entries)
    ids = [e.model.id for e in chain]  # type: ignore[attr-defined]
    assert ids == ["x", "y"]

  def test_entries_returns_copy(self):
    entries = [FailoverEntry(model=FakeModel("a"), priority=0)]  # type: ignore[arg-type]
    chain = FailoverChain(entries)
    copy = chain.entries
    copy.append(FailoverEntry(model=FakeModel("b"), priority=1))  # type: ignore[arg-type]
    assert len(chain) == 1  # Original unchanged
