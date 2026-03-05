"""Tests for Cortex ingestion pipeline."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from definable.memory.cortex.config import CortexConfig
from definable.memory.cortex.index.signature import SignatureBuilder
from definable.memory.cortex.ingestion.fast import FastPathProcessor
from definable.memory.cortex.ingestion.facts import FactExtractor
from definable.memory.cortex.ingestion.narrative import NarrativeBuilder
from definable.memory.cortex.ingestion.pipeline import IngestionPipeline
from definable.memory.cortex.ingestion.slow import SlowPathProcessor
from definable.memory.cortex.ingestion.tags import TagGenerator
from definable.memory.cortex.record.types import MemorySource
from definable.memory.cortex.store import CortexStore


class TestFastPathProcessor:
  def test_extracts_emails(self):
    fast = FastPathProcessor()
    result = fast.process("Contact me at john@example.com or jane@test.org")
    assert "john@example.com" in result.entities
    assert "jane@test.org" in result.entities

  def test_extracts_urls(self):
    fast = FastPathProcessor()
    result = fast.process("Check https://github.com/definable for details")
    assert any("github.com" in e for e in result.entities)

  def test_extracts_mentions(self):
    fast = FastPathProcessor()
    result = fast.process("Hey @johndoe can you review this?")
    assert "@johndoe" in result.entities

  def test_extracts_camelcase(self):
    fast = FastPathProcessor()
    result = fast.process("We should use CortexMemory and SignatureBuilder")
    assert "CortexMemory" in result.entities
    assert "SignatureBuilder" in result.entities

  def test_builds_signature(self):
    builder = SignatureBuilder(dims=512)
    fast = FastPathProcessor(signature_builder=builder)
    result = fast.process("hello world")
    assert result.signature is not None
    assert len(result.signature) == 512 // 8

  def test_no_signature_without_builder(self):
    fast = FastPathProcessor()
    result = fast.process("hello world")
    assert result.signature is None

  def test_empty_text(self):
    fast = FastPathProcessor()
    result = fast.process("")
    assert result.entities == []


def _make_mock_model(response_text: str):
  """Create a mock model that returns a fixed response."""
  model = AsyncMock()
  mock_response = MagicMock()
  mock_response.content = response_text
  model.ainvoke = AsyncMock(return_value=mock_response)
  return model


class TestNarrativeBuilder:
  @pytest.mark.asyncio
  async def test_builds_narrative(self):
    response = json.dumps({
      "content": "The user discussed Python architecture preferences.",
      "participants": ["user", "assistant"],
      "emotional_tone": "focused",
      "causal_chain": ["asked about patterns", "discussed options"],
    })
    model = _make_mock_model(response)
    builder = NarrativeBuilder(model=model)
    result = await builder.build("Let's talk about Python architecture")
    assert result is not None
    assert "architecture" in result.content
    assert result.emotional_tone == "focused"

  @pytest.mark.asyncio
  async def test_no_model(self):
    builder = NarrativeBuilder(model=None)
    result = await builder.build("some text")
    assert result is None

  @pytest.mark.asyncio
  async def test_handles_bad_json(self):
    model = _make_mock_model("not valid json")
    builder = NarrativeBuilder(model=model)
    result = await builder.build("text")
    assert result is None


class TestFactExtractor:
  @pytest.mark.asyncio
  async def test_extracts_facts(self):
    response = json.dumps([
      {"content": "The user prefers 2-space indentation", "confidence": 0.9, "entities": ["indentation"]},
      {"content": "The project uses Python 3.12", "confidence": 1.0, "entities": ["Python"]},
    ])
    model = _make_mock_model(response)
    extractor = FactExtractor(model=model)
    facts = await extractor.extract("I prefer 2-space indentation and we use Python 3.12")
    assert len(facts) == 2
    assert facts[0].confidence == 0.9

  @pytest.mark.asyncio
  async def test_no_model(self):
    extractor = FactExtractor(model=None)
    assert await extractor.extract("text") == []

  @pytest.mark.asyncio
  async def test_handles_bad_json(self):
    model = _make_mock_model("not valid json")
    extractor = FactExtractor(model=model)
    assert await extractor.extract("text") == []


class TestTagGenerator:
  @pytest.mark.asyncio
  async def test_generates_tags(self):
    response = json.dumps(["technical/python", "activity/coding", "topic/memory"])
    model = _make_mock_model(response)
    gen = TagGenerator(model=model)
    tags = await gen.generate("Building a memory system in Python")
    assert len(tags) == 3
    assert "technical/python" in tags

  @pytest.mark.asyncio
  async def test_no_model(self):
    gen = TagGenerator(model=None)
    assert await gen.generate("text") == []


class TestSlowPathProcessor:
  @pytest.mark.asyncio
  async def test_parallel_processing(self):
    narrative_resp = json.dumps({
      "content": "A discussion about testing.",
      "participants": ["user"],
      "emotional_tone": "curious",
      "causal_chain": [],
    })
    facts_resp = json.dumps([{"content": "Tests are important", "confidence": 1.0, "entities": []}])
    tags_resp = json.dumps(["technical/testing"])

    call_count = 0

    async def mock_ainvoke(messages, assistant_message):
      nonlocal call_count
      call_count += 1
      resp = MagicMock()
      if call_count == 1:
        resp.content = narrative_resp
      elif call_count == 2:
        resp.content = facts_resp
      else:
        resp.content = tags_resp
      return resp

    model = AsyncMock()
    model.ainvoke = mock_ainvoke

    proc = SlowPathProcessor(model=model)
    result = await proc.process("Let's talk about testing")
    assert result.narrative is not None or len(result.facts) > 0 or len(result.tags) > 0


@pytest.fixture
async def pipeline_fixtures(tmp_path):
  store = CortexStore(db_path=str(tmp_path / "pipeline.db"))
  await store.initialize()
  config = CortexConfig(slow_path_enabled=False)  # Disable slow path for unit tests
  pipeline = IngestionPipeline(store=store, config=config)
  yield store, config, pipeline
  await store.close()


@pytest.mark.asyncio
class TestIngestionPipeline:
  async def test_ingest_creates_record(self, pipeline_fixtures):
    store, config, pipeline = pipeline_fixtures
    record = await pipeline.ingest("Hello world", session_id="s1", user_id="u1")
    assert record.record_id
    assert record.raw_content == "Hello world"
    fetched = await store.get_record(record.record_id)
    assert fetched is not None

  async def test_turn_counter_increments(self, pipeline_fixtures):
    store, config, pipeline = pipeline_fixtures
    r1 = await pipeline.ingest("first", session_id="s1")
    r2 = await pipeline.ingest("second", session_id="s1")
    assert r2.turn_index > r1.turn_index

  async def test_with_signature(self, tmp_path):
    import aiosqlite

    store = CortexStore(db_path=str(tmp_path / "sig_pipeline.db"))
    await store.initialize()
    config = CortexConfig(slow_path_enabled=False, enable_signatures=True)
    sig_builder = SignatureBuilder(dims=512)

    from definable.memory.cortex.index.signature import SignatureIndex

    db = await aiosqlite.connect(str(tmp_path / "sig_pipeline.db"))
    sig_idx = SignatureIndex()
    await sig_idx.initialize(db)

    pipeline = IngestionPipeline(
      store=store,
      config=config,
      signature_builder=sig_builder,
      signature_index=sig_idx,
    )
    record = await pipeline.ingest("Python programming tutorial")
    assert record.signature is not None
    await db.close()
    await store.close()

  async def test_ingest_source(self, pipeline_fixtures):
    store, config, pipeline = pipeline_fixtures
    record = await pipeline.ingest("observed pattern", source=MemorySource.OBSERVATION)
    assert record.source == MemorySource.OBSERVATION
