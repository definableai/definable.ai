"""Tests for CortexStore."""

import pytest
from definable.memory.cortex.record.scratchpad import Scratchpad
from definable.memory.cortex.record.types import (
  Fact,
  MemoryRecord,
  MemorySource,
  NarrativeEpisode,
)
from definable.memory.cortex.store import CortexStore


@pytest.fixture
async def store(tmp_path):
  s = CortexStore(db_path=str(tmp_path / "test_cortex.db"))
  await s.initialize()
  yield s
  await s.close()


@pytest.mark.asyncio
class TestCortexStoreRecords:
  async def test_add_and_get(self, store):
    r = MemoryRecord(raw_content="hello", session_id="s1", user_id="u1")
    await store.add_record(r)
    fetched = await store.get_record(r.record_id)
    assert fetched is not None
    assert fetched.raw_content == "hello"
    assert fetched.session_id == "s1"

  async def test_get_records_session(self, store):
    for i in range(3):
      await store.add_record(MemoryRecord(raw_content=f"msg-{i}", session_id="s1", user_id="u1"))
    await store.add_record(MemoryRecord(raw_content="other", session_id="s2", user_id="u1"))
    records = await store.get_records("s1", "u1")
    assert len(records) == 3

  async def test_active_only_filter(self, store):
    r1 = MemoryRecord(raw_content="active", session_id="s1")
    r2 = MemoryRecord(raw_content="superseded", session_id="s1", superseded_by="xxx")
    await store.add_record(r1)
    await store.add_record(r2)
    active = await store.get_records("s1", active_only=True)
    assert len(active) == 1
    assert active[0].raw_content == "active"
    all_recs = await store.get_records("s1", active_only=False)
    assert len(all_recs) == 2

  async def test_update_record(self, store):
    r = MemoryRecord(raw_content="original", session_id="s1")
    await store.add_record(r)
    r.raw_content = "updated"
    r.staleness = 0.5
    await store.update_record(r)
    fetched = await store.get_record(r.record_id)
    assert fetched is not None
    assert fetched.raw_content == "updated"
    assert fetched.staleness == 0.5

  async def test_delete_record(self, store):
    r = MemoryRecord(raw_content="to-delete", session_id="s1")
    await store.add_record(r)
    await store.delete_record(r.record_id)
    assert await store.get_record(r.record_id) is None

  async def test_count(self, store):
    for i in range(5):
      await store.add_record(MemoryRecord(raw_content=f"m-{i}", session_id="s1"))
    assert await store.count_records("s1") == 5

  async def test_full_record_roundtrip(self, store):
    r = MemoryRecord(
      raw_content="Full record",
      source=MemorySource.OBSERVATION,
      narrative=NarrativeEpisode(content="A narrative"),
      facts=[Fact(content="fact-a"), Fact(content="fact-b")],
      tags=["work", "technical"],
      signature=b"\xff\x00\xab",
      embedding=[0.1, 0.2, 0.3],
      session_id="s1",
    )
    await store.add_record(r)
    fetched = await store.get_record(r.record_id)
    assert fetched is not None
    assert fetched.narrative is not None
    assert fetched.narrative.content == "A narrative"
    assert len(fetched.facts) == 2
    assert fetched.tags == ["work", "technical"]
    assert fetched.signature == b"\xff\x00\xab"
    assert fetched.embedding == [0.1, 0.2, 0.3]

  async def test_get_all_records(self, store):
    await store.add_record(MemoryRecord(raw_content="a", session_id="s1", user_id="u1"))
    await store.add_record(MemoryRecord(raw_content="b", session_id="s2", user_id="u1"))
    all_recs = await store.get_all_records(user_id="u1")
    assert len(all_recs) == 2

  async def test_delete_session(self, store):
    await store.add_record(MemoryRecord(raw_content="a", session_id="s1"))
    await store.add_record(MemoryRecord(raw_content="b", session_id="s1"))
    await store.delete_session("s1")
    assert await store.count_records("s1") == 0


@pytest.mark.asyncio
class TestCortexStoreScratchpad:
  async def test_get_empty(self, store):
    sp = await store.get_scratchpad("s1", "u1")
    assert sp.beliefs == {}
    assert sp.session_id == "s1"

  async def test_save_and_get(self, store):
    sp = Scratchpad(session_id="s1", user_id="u1", beliefs={"name": "test"}, active_topics=["cortex"])
    await store.save_scratchpad(sp)
    fetched = await store.get_scratchpad("s1", "u1")
    assert fetched.beliefs == {"name": "test"}
    assert fetched.active_topics == ["cortex"]

  async def test_upsert(self, store):
    sp1 = Scratchpad(session_id="s1", user_id="u1", beliefs={"a": 1})
    await store.save_scratchpad(sp1)
    sp2 = Scratchpad(session_id="s1", user_id="u1", beliefs={"a": 2, "b": 3})
    await store.save_scratchpad(sp2)
    fetched = await store.get_scratchpad("s1", "u1")
    assert fetched.beliefs == {"a": 2, "b": 3}

  async def test_delete_session_clears_scratchpad(self, store):
    sp = Scratchpad(session_id="s1", user_id="u1", beliefs={"x": 1})
    await store.save_scratchpad(sp)
    await store.delete_session("s1", "u1")
    fetched = await store.get_scratchpad("s1", "u1")
    assert fetched.beliefs == {}


@pytest.mark.asyncio
class TestCortexStoreLifecycle:
  async def test_context_manager(self, tmp_path):
    async with CortexStore(db_path=str(tmp_path / "ctx.db")) as store:
      r = MemoryRecord(raw_content="ctx", session_id="s1")
      await store.add_record(r)
      assert await store.count_records("s1") == 1

  async def test_double_init(self, tmp_path):
    store = CortexStore(db_path=str(tmp_path / "double.db"))
    await store.initialize()
    await store.initialize()  # should be idempotent
    await store.close()
