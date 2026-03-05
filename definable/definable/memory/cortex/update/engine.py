"""Update engine — active CRUD operations on memory records."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from definable.memory.cortex.record.types import MemoryRecord, MemorySource
from definable.utils.log import log_debug

if TYPE_CHECKING:
  from definable.memory.cortex.config import CortexConfig
  from definable.memory.cortex.index.graph import GraphIndex
  from definable.memory.cortex.index.tags import TagIndex
  from definable.memory.cortex.record.scratchpad import Scratchpad
  from definable.memory.cortex.store import CortexStore
  from definable.memory.cortex.update.cascade import CascadePropagator


class UpdateEngine:
  """Handles updates, soft-deletes, and scratchpad modifications.

  All updates go through this engine to ensure cascade propagation
  and index consistency.
  """

  def __init__(
    self,
    store: "CortexStore",
    config: "CortexConfig",
    graph_index: Optional["GraphIndex"] = None,
    tag_index: Optional["TagIndex"] = None,
    cascade: Optional["CascadePropagator"] = None,
  ):
    self._store = store
    self._config = config
    self._graph_index = graph_index
    self._tag_index = tag_index
    self._cascade = cascade

  async def update_content(self, record_id: str, new_content: str, reason: str = "") -> Optional[MemoryRecord]:
    """Update a record's content. Creates a new record and supersedes the old one.

    This is a soft-update: the old record is marked as superseded,
    and a new record is created with the updated content.
    """
    old = await self._store.get_record(record_id)
    if old is None:
      return None

    # Create replacement record
    new_record = MemoryRecord(
      session_id=old.session_id,
      user_id=old.user_id,
      source=MemorySource.CONSOLIDATION,
      raw_content=new_content,
      role=old.role,
      turn_index=old.turn_index,
      narrative=old.narrative,
      facts=old.facts,
      tags=old.tags,
      embedding=old.embedding,
    )

    # Soft-delete old record
    old.superseded_by = new_record.record_id
    old.updated_at = time.time()
    await self._store.update_record(old)
    await self._store.add_record(new_record)

    # Propagate staleness through graph
    if self._cascade:
      await self._cascade.propagate(record_id)

    log_debug(f"UpdateEngine: {record_id[:8]} → {new_record.record_id[:8]} ({reason})", log_level=2)
    return new_record

  async def forget(self, record_id: str, reason: str = "") -> bool:
    """Soft-delete a record (set superseded_by to 'forgotten').

    Returns True if the record was found and deleted.
    """
    record = await self._store.get_record(record_id)
    if record is None:
      return False

    record.superseded_by = "forgotten"
    record.updated_at = time.time()
    await self._store.update_record(record)

    # Clean up indexes
    if self._tag_index:
      await self._tag_index.remove_tags(record_id)
    if self._graph_index:
      await self._graph_index.remove_node(record_id)

    # Propagate staleness
    if self._cascade:
      await self._cascade.propagate(record_id)

    log_debug(f"UpdateEngine: forgot {record_id[:8]} ({reason})", log_level=2)
    return True

  async def set_belief(self, key: str, value: object, session_id: str = "default", user_id: str = "default") -> None:
    """Update a scratchpad belief."""
    scratchpad = await self._store.get_scratchpad(session_id, user_id)
    scratchpad.set_belief(key, value)
    await self._store.save_scratchpad(scratchpad)

  async def remove_belief(self, key: str, session_id: str = "default", user_id: str = "default") -> None:
    """Remove a scratchpad belief."""
    scratchpad = await self._store.get_scratchpad(session_id, user_id)
    scratchpad.remove_belief(key)
    await self._store.save_scratchpad(scratchpad)

  async def get_scratchpad(self, session_id: str = "default", user_id: str = "default") -> "Scratchpad":
    """Get the current scratchpad."""
    return await self._store.get_scratchpad(session_id, user_id)
