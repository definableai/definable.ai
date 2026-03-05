"""Tests for Cortex record types."""

from definable.memory.cortex.record.types import (
  Edge,
  EdgeType,
  Fact,
  MemoryRecord,
  MemorySource,
  NarrativeEpisode,
)


class TestNarrativeEpisode:
  def test_auto_id_and_timestamp(self):
    ep = NarrativeEpisode(content="User discussed Python architecture")
    assert ep.episode_id
    assert ep.created_at > 0

  def test_roundtrip(self):
    ep = NarrativeEpisode(
      content="Built a memory system",
      participants=["user", "agent"],
      emotional_tone="focused",
      causal_chain=["research", "design", "implement"],
    )
    d = ep.to_dict()
    restored = NarrativeEpisode.from_dict(d)
    assert restored.content == ep.content
    assert restored.participants == ep.participants
    assert restored.emotional_tone == ep.emotional_tone
    assert restored.causal_chain == ep.causal_chain


class TestFact:
  def test_auto_id(self):
    f = Fact(content="User prefers 2-space indentation")
    assert f.fact_id
    assert f.confidence == 1.0

  def test_roundtrip(self):
    f = Fact(content="Definable uses ruff", confidence=0.9, entities=["ruff", "definable"])
    d = f.to_dict()
    restored = Fact.from_dict(d)
    assert restored.content == f.content
    assert restored.confidence == f.confidence
    assert restored.entities == f.entities


class TestEdge:
  def test_defaults(self):
    e = Edge(source_id="a", target_id="b")
    assert e.edge_type == EdgeType.SEMANTIC
    assert e.weight == 1.0

  def test_roundtrip(self):
    e = Edge(source_id="a", target_id="b", edge_type=EdgeType.CAUSAL, weight=0.8, label="caused_by")
    d = e.to_dict()
    restored = Edge.from_dict(d)
    assert restored.edge_type == EdgeType.CAUSAL
    assert restored.label == "caused_by"


class TestMemoryRecord:
  def test_auto_fields(self):
    r = MemoryRecord(raw_content="Hello world")
    assert r.record_id
    assert r.created_at > 0
    assert r.is_active

  def test_superseded(self):
    r = MemoryRecord(raw_content="old", superseded_by="new-id")
    assert not r.is_active

  def test_roundtrip_minimal(self):
    r = MemoryRecord(raw_content="test", session_id="s1", user_id="u1")
    d = r.to_dict()
    restored = MemoryRecord.from_dict(d)
    assert restored.raw_content == "test"
    assert restored.session_id == "s1"

  def test_roundtrip_full(self):
    r = MemoryRecord(
      raw_content="Complex record",
      source=MemorySource.OBSERVATION,
      narrative=NarrativeEpisode(content="A story"),
      facts=[Fact(content="fact1"), Fact(content="fact2")],
      tags=["work/project", "technical"],
      signature=b"\x01\x02\x03",
      embedding=[0.1, 0.2, 0.3],
    )
    d = r.to_dict()
    restored = MemoryRecord.from_dict(d)
    assert restored.source == MemorySource.OBSERVATION
    assert restored.narrative is not None
    assert restored.narrative.content == "A story"
    assert len(restored.facts) == 2
    assert restored.tags == ["work/project", "technical"]
    assert restored.signature == b"\x01\x02\x03"
    assert restored.embedding == [0.1, 0.2, 0.3]

  def test_source_enum(self):
    for s in MemorySource:
      r = MemoryRecord(source=s)
      d = r.to_dict()
      assert MemoryRecord.from_dict(d).source == s
