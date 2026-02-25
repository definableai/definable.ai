"""FailoverChain — ordered provider failover list."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional

if TYPE_CHECKING:
  from definable.model.base import Model
  from definable.model.resilience.key_pool import KeyPool


@dataclass
class FailoverEntry:
  """A single entry in a failover chain.

  Args:
    model: The model provider for this entry.
    key_pool: Optional KeyPool for key rotation on this provider.
    priority: Lower priority values are tried first (default 0).
  """

  model: "Model"
  key_pool: Optional["KeyPool"] = None
  priority: int = 0


class FailoverChain:
  """Ordered list of failover providers, sorted by priority.

  Lower priority values are tried first. Entries with the same
  priority preserve insertion order.

  Args:
    entries: List of FailoverEntry instances.

  Example::

    chain = FailoverChain([
      FailoverEntry(model=primary_model, key_pool=pool, priority=0),
      FailoverEntry(model=backup_model, priority=1),
    ])
    for entry in chain:
      try:
        return entry.model.invoke(...)
      except:
        continue
  """

  def __init__(self, entries: List[FailoverEntry]) -> None:
    if not entries:
      raise ValueError("FailoverChain requires at least one entry")
    self._entries = sorted(entries, key=lambda e: e.priority)

  @property
  def primary(self) -> FailoverEntry:
    """The highest-priority (lowest number) entry."""
    return self._entries[0]

  @property
  def entries(self) -> List[FailoverEntry]:
    """Return a copy of the sorted entries list."""
    return list(self._entries)

  def __len__(self) -> int:
    return len(self._entries)

  def __iter__(self) -> Iterator[FailoverEntry]:
    return iter(self._entries)

  def __repr__(self) -> str:
    return f"<FailoverChain entries={len(self._entries)}>"
