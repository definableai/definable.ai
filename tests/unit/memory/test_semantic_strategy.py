"""Tests for SemanticStrategy — extract atomic memory units from conversation."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.memory.strategies.semantic import SemanticStrategy, _extract_json_array
from definable.memory.types import MemoryEntry


def _make_entries(count: int, session_id: str = "s1") -> list[MemoryEntry]:
  """Create alternating user/assistant entries."""
  entries = []
  for i in range(count):
    role = "user" if i % 2 == 0 else "assistant"
    entries.append(
      MemoryEntry(
        memory_id=f"m-{i}",
        session_id=session_id,
        role=role,
        content=f"Message {i}",
        created_at=float(i),
        updated_at=float(i),
      )
    )
  return entries


def _make_mock_model(atoms: list[dict] | None = None) -> MagicMock:
  """Create a mock model that returns a JSON array of atoms."""
  if atoms is None:
    atoms = [
      {
        "lossless_content": "User discussed topic A with the assistant.",
        "keywords": ["topic", "discussion"],
        "entities": [],
        "persons": ["User"],
        "topic": "topic discussion",
      }
    ]
  model = MagicMock()
  response = MagicMock()
  response.content = json.dumps(atoms)
  model.ainvoke = AsyncMock(return_value=response)
  return model


# --- SemanticStrategy Tests ---


class TestSemanticStrategy:
  @pytest.mark.asyncio
  async def test_no_op_below_threshold(self):
    """If entries <= pin_count + recent_count, return as-is."""
    strategy = SemanticStrategy(pin_count=2, recent_count=3)
    entries = _make_entries(5)
    model = _make_mock_model()

    result = await strategy.optimize(entries, model)
    assert len(result) == 5
    model.ainvoke.assert_not_called()

  @pytest.mark.asyncio
  async def test_basic_extraction(self):
    """Middle entries get replaced with atoms; pin + recent preserved."""
    atoms = [
      {
        "lossless_content": "User sent Message 2 about greetings.",
        "keywords": ["greeting", "message"],
        "entities": [],
        "persons": [],
        "topic": "greetings",
      },
      {
        "lossless_content": "Assistant replied with Message 3.",
        "keywords": ["reply", "assistant"],
        "entities": [],
        "persons": [],
        "topic": "reply",
      },
    ]
    strategy = SemanticStrategy(pin_count=2, recent_count=3)
    entries = _make_entries(10)
    model = _make_mock_model(atoms)

    result = await strategy.optimize(entries, model)
    model.ainvoke.assert_called_once()

    # pin(2) + atoms(2) + recent(3) = 7
    assert len(result) == 7

    # Verify pinned entries preserved.
    assert result[0].content == "Message 0"
    assert result[1].content == "Message 1"

    # Verify atom entries.
    assert result[2].entry_type == "atom"
    assert result[2].role == "atom"
    assert result[2].lossless_content == "User sent Message 2 about greetings."
    assert result[2].content == "User sent Message 2 about greetings."
    assert result[2].keywords == ["greeting", "message"]
    assert result[2].topic == "greetings"

    assert result[3].entry_type == "atom"
    assert result[3].lossless_content == "Assistant replied with Message 3."

    # Verify recent entries preserved.
    assert result[4].content == "Message 7"
    assert result[5].content == "Message 8"
    assert result[6].content == "Message 9"

  @pytest.mark.asyncio
  async def test_atom_inherits_session_metadata(self):
    """Atoms inherit session_id and user_id from source entries."""
    atoms = [{"lossless_content": "A fact.", "keywords": [], "entities": [], "persons": [], "topic": "fact"}]
    strategy = SemanticStrategy(pin_count=1, recent_count=1)
    entries = _make_entries(5, session_id="test-session")
    for e in entries:
      e.user_id = "alice"
    model = _make_mock_model(atoms)

    result = await strategy.optimize(entries, model)
    atom_entries = [e for e in result if e.entry_type == "atom"]
    assert len(atom_entries) == 1
    assert atom_entries[0].session_id == "test-session"
    assert atom_entries[0].user_id == "alice"

  @pytest.mark.asyncio
  async def test_tool_call_boundary_pin(self):
    """Tool results at the start of middle get pulled into pinned."""
    strategy = SemanticStrategy(pin_count=2, recent_count=2)
    entries = [
      MemoryEntry(memory_id="m0", session_id="s1", role="user", content="Hi", created_at=0.0, updated_at=0.0),
      MemoryEntry(memory_id="m1", session_id="s1", role="assistant", content="Calling tool", created_at=1.0, updated_at=1.0),
      MemoryEntry(memory_id="m2", session_id="s1", role="tool", content="Tool result", created_at=2.0, updated_at=2.0),
      MemoryEntry(memory_id="m3", session_id="s1", role="user", content="Thanks", created_at=3.0, updated_at=3.0),
      MemoryEntry(memory_id="m4", session_id="s1", role="assistant", content="Welcome", created_at=4.0, updated_at=4.0),
      MemoryEntry(memory_id="m5", session_id="s1", role="user", content="Bye", created_at=5.0, updated_at=5.0),
      MemoryEntry(memory_id="m6", session_id="s1", role="assistant", content="Goodbye", created_at=6.0, updated_at=6.0),
    ]
    atoms = [{"lossless_content": "Fact from mid.", "keywords": [], "entities": [], "persons": [], "topic": "mid"}]
    model = _make_mock_model(atoms)

    result = await strategy.optimize(entries, model)

    # Tool result at index 2 should be pulled into pinned.
    pinned_content = [e.content for e in result if e.entry_type == "message" and e.role != "atom"]
    assert "Tool result" in pinned_content[:4]

  @pytest.mark.asyncio
  async def test_tool_call_boundary_recent(self):
    """Tool results at start of recent pull preceding entries from middle."""
    strategy = SemanticStrategy(pin_count=1, recent_count=2)
    entries = [
      MemoryEntry(memory_id="m0", session_id="s1", role="user", content="Start", created_at=0.0, updated_at=0.0),
      MemoryEntry(memory_id="m1", session_id="s1", role="assistant", content="Middle1", created_at=1.0, updated_at=1.0),
      MemoryEntry(memory_id="m2", session_id="s1", role="user", content="Middle2", created_at=2.0, updated_at=2.0),
      MemoryEntry(memory_id="m3", session_id="s1", role="assistant", content="Calling", created_at=3.0, updated_at=3.0),
      MemoryEntry(memory_id="m4", session_id="s1", role="tool", content="Result", created_at=4.0, updated_at=4.0),
      MemoryEntry(memory_id="m5", session_id="s1", role="user", content="Final", created_at=5.0, updated_at=5.0),
    ]
    atoms = [{"lossless_content": "Middle fact.", "keywords": [], "entities": [], "persons": [], "topic": "mid"}]
    model = _make_mock_model(atoms)

    result = await strategy.optimize(entries, model)

    # Recent starts at [m4(tool), m5]. Tool → pull m3 into recent.
    all_content = [e.content for e in result]
    assert "Calling" in all_content
    assert "Result" in all_content
    assert "Final" in all_content

  @pytest.mark.asyncio
  async def test_fallback_on_model_failure(self):
    """If LLM fails, produce a fallback atom."""
    strategy = SemanticStrategy(pin_count=1, recent_count=1)
    entries = _make_entries(5)
    model = MagicMock()
    model.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))

    result = await strategy.optimize(entries, model)

    atoms = [e for e in result if e.entry_type == "atom"]
    assert len(atoms) == 1
    assert "3 messages" in atoms[0].content
    assert atoms[0].importance == 0.3

  @pytest.mark.asyncio
  async def test_fallback_on_empty_json(self):
    """If LLM returns empty array, produce a fallback atom."""
    strategy = SemanticStrategy(pin_count=1, recent_count=1)
    entries = _make_entries(5)
    model = _make_mock_model([])

    result = await strategy.optimize(entries, model)

    atoms = [e for e in result if e.entry_type == "atom"]
    assert len(atoms) == 1
    assert "3 messages" in atoms[0].content

  @pytest.mark.asyncio
  async def test_fallback_on_invalid_json(self):
    """If LLM returns non-JSON, produce a fallback atom."""
    strategy = SemanticStrategy(pin_count=1, recent_count=1)
    entries = _make_entries(5)
    model = MagicMock()
    response = MagicMock()
    response.content = "I cannot extract any facts from this conversation."
    model.ainvoke = AsyncMock(return_value=response)

    result = await strategy.optimize(entries, model)

    atoms = [e for e in result if e.entry_type == "atom"]
    assert len(atoms) == 1

  @pytest.mark.asyncio
  async def test_sliding_windows(self):
    """Large middle sections are split into windows."""
    atoms_call_1 = [{"lossless_content": "Fact A.", "keywords": ["a"], "entities": [], "persons": [], "topic": "a"}]
    atoms_call_2 = [{"lossless_content": "Fact B.", "keywords": ["b"], "entities": [], "persons": [], "topic": "b"}]

    call_count = 0

    async def side_effect(**kwargs):
      nonlocal call_count
      resp = MagicMock()
      if call_count == 0:
        resp.content = json.dumps(atoms_call_1)
      else:
        resp.content = json.dumps(atoms_call_2)
      call_count += 1
      return resp

    model = MagicMock()
    model.ainvoke = AsyncMock(side_effect=side_effect)

    # window_size=5, overlap=2 → step=3. Middle of 10 entries → windows at [0:5], [3:8], [6:10].
    strategy = SemanticStrategy(pin_count=1, recent_count=1, window_size=5, overlap_size=2)
    entries = _make_entries(12)

    result = await strategy.optimize(entries, model)

    # Should have called LLM multiple times (one per window).
    assert model.ainvoke.call_count >= 2

    # All atoms should be present.
    atoms = [e for e in result if e.entry_type == "atom"]
    assert len(atoms) >= 2

  @pytest.mark.asyncio
  async def test_dedup_context_passed_between_windows(self):
    """Previous window's atoms are passed as dedup context to the next window."""
    atoms_1 = [{"lossless_content": "Fact from window 1.", "keywords": ["w1"], "entities": [], "persons": [], "topic": "w1"}]
    atoms_2 = [{"lossless_content": "Fact from window 2.", "keywords": ["w2"], "entities": [], "persons": [], "topic": "w2"}]

    calls = []

    async def side_effect(**kwargs):
      calls.append(kwargs)
      resp = MagicMock()
      if len(calls) == 1:
        resp.content = json.dumps(atoms_1)
      else:
        resp.content = json.dumps(atoms_2)
      return resp

    model = MagicMock()
    model.ainvoke = AsyncMock(side_effect=side_effect)

    strategy = SemanticStrategy(pin_count=1, recent_count=1, window_size=4, overlap_size=1)
    entries = _make_entries(10)

    await strategy.optimize(entries, model)

    # Second LLM call should contain dedup context from first window's atoms.
    assert len(calls) >= 2
    second_prompt = calls[1]["messages"][0].content
    assert "Fact from window 1" in second_prompt

  @pytest.mark.asyncio
  async def test_empty_middle_after_adjustments(self):
    """If tool-call adjustments consume all middle entries, return original."""
    strategy = SemanticStrategy(pin_count=2, recent_count=2)
    entries = [
      MemoryEntry(memory_id="m0", session_id="s1", role="user", content="Hi", created_at=0.0, updated_at=0.0),
      MemoryEntry(memory_id="m1", session_id="s1", role="assistant", content="Call", created_at=1.0, updated_at=1.0),
      MemoryEntry(memory_id="m2", session_id="s1", role="tool", content="Result", created_at=2.0, updated_at=2.0),
      MemoryEntry(memory_id="m3", session_id="s1", role="user", content="Ok", created_at=3.0, updated_at=3.0),
      MemoryEntry(memory_id="m4", session_id="s1", role="assistant", content="Done", created_at=4.0, updated_at=4.0),
    ]
    model = _make_mock_model()

    result = await strategy.optimize(entries, model)

    model.ainvoke.assert_not_called()
    assert len(result) == 5

  @pytest.mark.asyncio
  async def test_atoms_skip_empty_lossless_content(self):
    """Atoms with empty lossless_content are filtered out."""
    atoms = [
      {"lossless_content": "Valid fact.", "keywords": ["valid"], "entities": [], "persons": [], "topic": "valid"},
      {"lossless_content": "", "keywords": ["empty"], "entities": [], "persons": [], "topic": "empty"},
      {"lossless_content": "Another fact.", "keywords": ["another"], "entities": [], "persons": [], "topic": "another"},
    ]
    strategy = SemanticStrategy(pin_count=1, recent_count=1)
    entries = _make_entries(5)
    model = _make_mock_model(atoms)

    result = await strategy.optimize(entries, model)

    atom_entries = [e for e in result if e.entry_type == "atom"]
    assert len(atom_entries) == 2
    assert atom_entries[0].lossless_content == "Valid fact."
    assert atom_entries[1].lossless_content == "Another fact."

  @pytest.mark.asyncio
  async def test_count_tokens(self):
    strategy = SemanticStrategy()
    entries = [
      MemoryEntry(session_id="s1", content="Hello world", created_at=1.0, updated_at=1.0),
      MemoryEntry(session_id="s1", content="How are you?", created_at=2.0, updated_at=2.0),
    ]
    tokens = strategy.count_tokens(entries)
    assert tokens > 0
    assert tokens == (len("Hello world") + len("How are you?")) // 4

  @pytest.mark.asyncio
  async def test_prior_summary_included_in_prompt(self):
    """Existing summaries in pinned section are passed as prior context."""
    atoms = [{"lossless_content": "Fact.", "keywords": [], "entities": [], "persons": [], "topic": "t"}]
    strategy = SemanticStrategy(pin_count=2, recent_count=2)
    entries = [
      MemoryEntry(memory_id="m0", session_id="s1", role="summary", content="Previous: weather chat", created_at=0.0, updated_at=0.0),
      MemoryEntry(memory_id="m1", session_id="s1", role="user", content="Continue", created_at=1.0, updated_at=1.0),
      MemoryEntry(memory_id="m2", session_id="s1", role="assistant", content="Mid1", created_at=2.0, updated_at=2.0),
      MemoryEntry(memory_id="m3", session_id="s1", role="user", content="Mid2", created_at=3.0, updated_at=3.0),
      MemoryEntry(memory_id="m4", session_id="s1", role="assistant", content="End1", created_at=4.0, updated_at=4.0),
      MemoryEntry(memory_id="m5", session_id="s1", role="user", content="End2", created_at=5.0, updated_at=5.0),
    ]
    model = _make_mock_model(atoms)

    await strategy.optimize(entries, model)

    call_args = model.ainvoke.call_args
    prompt_content = call_args[1]["messages"][0].content
    assert "Previous: weather chat" in prompt_content

  @pytest.mark.asyncio
  async def test_multiple_atoms_with_rich_metadata(self):
    """Atoms preserve all metadata fields from LLM output."""
    atoms = [
      {
        "lossless_content": "Alice from Acme Corp requested a meeting on 2026-03-15.",
        "keywords": ["meeting", "schedule", "request"],
        "entities": ["Acme Corp"],
        "persons": ["Alice"],
        "topic": "meeting scheduling",
      },
      {
        "lossless_content": "The Q1 budget was approved at $500,000.",
        "keywords": ["budget", "Q1", "approval", "finance"],
        "entities": ["Q1 budget"],
        "persons": [],
        "topic": "budget approval",
      },
    ]
    strategy = SemanticStrategy(pin_count=1, recent_count=1)
    entries = _make_entries(5)
    model = _make_mock_model(atoms)

    result = await strategy.optimize(entries, model)

    atom_entries = [e for e in result if e.entry_type == "atom"]
    assert len(atom_entries) == 2

    a1 = atom_entries[0]
    assert a1.persons == ["Alice"]
    assert a1.entities == ["Acme Corp"]
    assert a1.keywords == ["meeting", "schedule", "request"]
    assert a1.topic == "meeting scheduling"

    a2 = atom_entries[1]
    assert a2.persons == []
    assert a2.entities == ["Q1 budget"]
    assert "budget" in a2.keywords


# --- JSON Extraction Tests ---


class TestExtractJsonArray:
  def test_raw_json_array(self):
    result = _extract_json_array('[{"lossless_content": "fact"}]')
    assert len(result) == 1
    assert result[0]["lossless_content"] == "fact"

  def test_fenced_json(self):
    text = '```json\n[{"lossless_content": "fact"}]\n```'
    result = _extract_json_array(text)
    assert len(result) == 1

  def test_fenced_without_lang(self):
    text = '```\n[{"lossless_content": "fact"}]\n```'
    result = _extract_json_array(text)
    assert len(result) == 1

  def test_json_with_surrounding_text(self):
    text = 'Here are the facts:\n[{"lossless_content": "fact"}]\nDone.'
    result = _extract_json_array(text)
    assert len(result) == 1

  def test_empty_array(self):
    result = _extract_json_array("[]")
    assert result == []

  def test_non_json_text(self):
    result = _extract_json_array("I cannot extract any facts.")
    assert result == []

  def test_nested_brackets(self):
    text = '[{"lossless_content": "a [bracketed] thing", "keywords": ["a", "b"]}]'
    result = _extract_json_array(text)
    assert len(result) == 1
    assert result[0]["keywords"] == ["a", "b"]

  def test_whitespace_handling(self):
    text = '  \n  [{"lossless_content": "fact"}]  \n  '
    result = _extract_json_array(text)
    assert len(result) == 1


# --- MemoryEntry Semantic Fields Tests ---


class TestMemoryEntrySemanticFields:
  def test_defaults(self):
    entry = MemoryEntry(session_id="s1")
    assert entry.entry_type == "message"
    assert entry.lossless_content is None
    assert entry.keywords == []
    assert entry.entities == []
    assert entry.persons == []
    assert entry.topic is None
    assert entry.importance == 0.5
    assert entry.superseded_by is None

  def test_atom_creation(self):
    entry = MemoryEntry(
      session_id="s1",
      role="atom",
      content="Alice met Bob on 2026-03-15.",
      entry_type="atom",
      lossless_content="Alice met Bob on 2026-03-15.",
      keywords=["meeting", "alice", "bob"],
      entities=[],
      persons=["Alice", "Bob"],
      topic="meeting",
      importance=0.8,
    )
    assert entry.entry_type == "atom"
    assert entry.lossless_content == "Alice met Bob on 2026-03-15."
    assert entry.persons == ["Alice", "Bob"]
    assert entry.importance == 0.8

  def test_to_dict_message_compact(self):
    """Regular messages don't include semantic fields in serialized form."""
    entry = MemoryEntry(session_id="s1", role="user", content="Hello")
    d = entry.to_dict()
    assert "entry_type" not in d  # Default "message" is omitted
    assert "lossless_content" not in d
    assert "keywords" not in d
    assert "entities" not in d
    assert "persons" not in d
    assert "topic" not in d
    assert "importance" not in d
    assert "superseded_by" not in d

  def test_to_dict_atom_includes_semantic_fields(self):
    """Atom entries include all semantic fields."""
    entry = MemoryEntry(
      session_id="s1",
      entry_type="atom",
      lossless_content="A fact.",
      keywords=["fact"],
      entities=["Corp"],
      persons=["Alice"],
      topic="facts",
      importance=0.9,
    )
    d = entry.to_dict()
    assert d["entry_type"] == "atom"
    assert d["lossless_content"] == "A fact."
    assert d["keywords"] == ["fact"]
    assert d["entities"] == ["Corp"]
    assert d["persons"] == ["Alice"]
    assert d["topic"] == "facts"
    assert d["importance"] == 0.9

  def test_from_dict_backward_compatible(self):
    """Old serialized data without semantic fields deserializes correctly."""
    data = {
      "memory_id": "m1",
      "session_id": "s1",
      "role": "user",
      "content": "Hello",
      "created_at": 1000.0,
      "updated_at": 1000.0,
    }
    entry = MemoryEntry.from_dict(data)
    assert entry.entry_type == "message"
    assert entry.lossless_content is None
    assert entry.keywords == []
    assert entry.importance == 0.5

  def test_roundtrip_atom(self):
    """Atom entry survives to_dict → from_dict roundtrip."""
    entry = MemoryEntry(
      session_id="s1",
      role="atom",
      content="Fact about X.",
      entry_type="atom",
      lossless_content="Fact about X.",
      keywords=["x", "fact"],
      entities=["X Corp"],
      persons=["Alice"],
      topic="facts",
      importance=0.7,
    )
    restored = MemoryEntry.from_dict(entry.to_dict())
    assert restored.entry_type == "atom"
    assert restored.lossless_content == "Fact about X."
    assert restored.keywords == ["x", "fact"]
    assert restored.entities == ["X Corp"]
    assert restored.persons == ["Alice"]
    assert restored.topic == "facts"
    assert restored.importance == 0.7

  def test_superseded_by_roundtrip(self):
    entry = MemoryEntry(session_id="s1", entry_type="atom", lossless_content="Old.", superseded_by="new-id")
    d = entry.to_dict()
    assert d["superseded_by"] == "new-id"
    restored = MemoryEntry.from_dict(d)
    assert restored.superseded_by == "new-id"
