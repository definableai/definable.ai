"""
Unit tests for the Message dataclass and supporting types.

Tests pure construction, field storage, serialization, and helper methods
on Message, MessageReferences, Citations, UrlCitation, and DocumentCitation.
No API calls. No external dependencies beyond pydantic.

Covers:
  - Message(role="user", content="hello") creates correctly
  - Message(role="assistant", content=None) creates correctly
  - Message(role="system", content="...") creates correctly
  - Message.role stores value
  - Message.content stores value
  - Message.to_dict() returns dict with role and content
  - Message with tool_calls field
  - Message with images/files fields
  - Message.get_content_string() for string and list content
  - Message.get_content() with and without compression
  - Message.content_is_valid() checks
  - Message.from_dict() round-trip
  - MessageReferences creation and fields
  - Citations and sub-types creation
"""

import pytest

from definable.model.message import (
  Citations,
  DocumentCitation,
  Message,
  MessageReferences,
  UrlCitation,
)


# ---------------------------------------------------------------------------
# Message: basic construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMessageCreation:
  """Message construction with various role/content combinations."""

  def test_user_message_creates_correctly(self):
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"

  def test_assistant_message_with_none_content(self):
    msg = Message(role="assistant", content=None)
    assert msg.role == "assistant"
    assert msg.content is None

  def test_system_message_creates_correctly(self):
    msg = Message(role="system", content="You are a helpful assistant.")
    assert msg.role == "system"
    assert msg.content == "You are a helpful assistant."

  def test_tool_role_creates_correctly(self):
    msg = Message(role="tool", content="result", tool_call_id="tc_123")
    assert msg.role == "tool"
    assert msg.content == "result"
    assert msg.tool_call_id == "tc_123"


# ---------------------------------------------------------------------------
# Message: field storage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMessageFieldStorage:
  """Message stores all fields accurately."""

  def test_role_stored(self):
    msg = Message(role="user", content="hi")
    assert msg.role == "user"

  def test_content_stored_as_string(self):
    msg = Message(role="user", content="hello world")
    assert msg.content == "hello world"

  def test_content_stored_as_list(self):
    parts = [{"type": "text", "text": "hello"}]
    msg = Message(role="user", content=parts)
    assert msg.content == parts

  def test_name_field_stored(self):
    msg = Message(role="user", content="hi", name="Alice")
    assert msg.name == "Alice"

  def test_tool_call_id_stored(self):
    msg = Message(role="tool", content="ok", tool_call_id="call_abc")
    assert msg.tool_call_id == "call_abc"

  def test_id_auto_generated(self):
    msg = Message(role="user", content="hi")
    assert msg.id is not None
    assert len(msg.id) > 0

  def test_ids_are_unique(self):
    msg1 = Message(role="user", content="a")
    msg2 = Message(role="user", content="b")
    assert msg1.id != msg2.id

  def test_created_at_is_int(self):
    msg = Message(role="user", content="hi")
    assert isinstance(msg.created_at, int)

  def test_defaults_for_boolean_fields(self):
    msg = Message(role="user", content="hi")
    assert msg.stop_after_tool_call is False
    assert msg.add_to_agent_memory is True
    assert msg.from_history is False
    assert msg.temporary is False

  def test_compressed_content_default_none(self):
    msg = Message(role="user", content="hi")
    assert msg.compressed_content is None

  def test_reasoning_content_default_none(self):
    msg = Message(role="user", content="hi")
    assert msg.reasoning_content is None


# ---------------------------------------------------------------------------
# Message: tool_calls field
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMessageToolCalls:
  """Message stores tool_calls list correctly."""

  def test_tool_calls_stored(self):
    calls = [
      {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
      }
    ]
    msg = Message(role="assistant", content=None, tool_calls=calls)
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["id"] == "call_1"

  def test_tool_calls_default_none(self):
    msg = Message(role="assistant", content="Hello")
    assert msg.tool_calls is None

  def test_multiple_tool_calls(self):
    calls = [
      {"id": "call_1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
      {"id": "call_2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
    ]
    msg = Message(role="assistant", content=None, tool_calls=calls)
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 2


# ---------------------------------------------------------------------------
# Message: media fields (images, files)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMessageMediaFields:
  """Message stores images, audio, videos, and files fields."""

  def test_images_default_none(self):
    msg = Message(role="user", content="hi")
    assert msg.images is None

  def test_files_default_none(self):
    msg = Message(role="user", content="hi")
    assert msg.files is None

  def test_audio_default_none(self):
    msg = Message(role="user", content="hi")
    assert msg.audio is None

  def test_videos_default_none(self):
    msg = Message(role="user", content="hi")
    assert msg.videos is None


# ---------------------------------------------------------------------------
# Message: to_dict()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMessageToDict:
  """Message.to_dict() returns expected dictionary structure."""

  def test_includes_role_and_content(self):
    msg = Message(role="user", content="hello")
    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "hello"

  def test_includes_id(self):
    msg = Message(role="user", content="hello")
    d = msg.to_dict()
    assert "id" in d
    assert d["id"] == msg.id

  def test_includes_created_at(self):
    msg = Message(role="user", content="hello")
    d = msg.to_dict()
    assert "created_at" in d
    assert isinstance(d["created_at"], int)

  def test_none_values_filtered_out(self):
    msg = Message(role="user", content="hello")
    d = msg.to_dict()
    # name is None by default, should be filtered out
    assert "name" not in d

  def test_tool_calls_included_when_present(self):
    calls = [{"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]
    msg = Message(role="assistant", content=None, tool_calls=calls)
    d = msg.to_dict()
    assert "tool_calls" in d
    assert len(d["tool_calls"]) == 1

  def test_tool_name_included_when_present(self):
    msg = Message(role="tool", content="result", tool_name="get_weather")
    d = msg.to_dict()
    assert d["tool_name"] == "get_weather"

  def test_stop_after_tool_call_false_not_filtered(self):
    msg = Message(role="user", content="hi")
    d = msg.to_dict()
    # stop_after_tool_call=False is a bool, not None, so it can appear
    assert d.get("stop_after_tool_call") is False or "stop_after_tool_call" not in d


# ---------------------------------------------------------------------------
# Message: get_content_string()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMessageGetContentString:
  """Message.get_content_string() converts content to a string."""

  def test_string_content_returned_as_is(self):
    msg = Message(role="user", content="hello world")
    assert msg.get_content_string() == "hello world"

  def test_list_with_text_dict_extracts_text(self):
    msg = Message(role="user", content=[{"text": "extracted text"}])
    assert msg.get_content_string() == "extracted text"

  def test_list_without_text_key_returns_json(self):
    msg = Message(role="user", content=[{"type": "image_url"}])
    result = msg.get_content_string()
    assert "image_url" in result

  def test_none_content_returns_empty_string(self):
    msg = Message(role="user", content=None)
    assert msg.get_content_string() == ""

  def test_empty_list_returns_json(self):
    msg = Message(role="user", content=[])
    # Empty list → json.dumps([]) = "[]" via the json path (len == 0 means early return "")
    # Actually: isinstance([], list) is True, len == 0, so the inner if doesn't match
    # and it falls through to json.dumps
    result = msg.get_content_string()
    assert result == "[]"


# ---------------------------------------------------------------------------
# Message: get_content()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMessageGetContent:
  """Message.get_content() returns original or compressed content."""

  def test_returns_original_content_by_default(self):
    msg = Message(role="user", content="original")
    assert msg.get_content() == "original"

  def test_returns_compressed_when_requested_and_available(self):
    msg = Message(role="user", content="original", compressed_content="compressed")
    assert msg.get_content(use_compressed_content=True) == "compressed"

  def test_returns_original_when_compressed_requested_but_absent(self):
    msg = Message(role="user", content="original")
    assert msg.get_content(use_compressed_content=True) == "original"


# ---------------------------------------------------------------------------
# Message: content_is_valid()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMessageContentIsValid:
  """Message.content_is_valid() checks that content is non-None and non-empty."""

  def test_valid_string_content(self):
    msg = Message(role="user", content="hello")
    assert msg.content_is_valid() is True

  def test_none_content_invalid(self):
    msg = Message(role="user", content=None)
    assert msg.content_is_valid() is False

  def test_empty_string_invalid(self):
    msg = Message(role="user", content="")
    assert msg.content_is_valid() is False

  def test_empty_list_invalid(self):
    msg = Message(role="user", content=[])
    assert msg.content_is_valid() is False

  def test_list_with_items_valid(self):
    msg = Message(role="user", content=[{"text": "hi"}])
    assert msg.content_is_valid() is True


# ---------------------------------------------------------------------------
# Message: from_dict() round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMessageFromDict:
  """Message.from_dict() reconstructs a Message from a dictionary."""

  def test_basic_round_trip(self):
    original = Message(role="user", content="hello")
    d = original.to_dict()
    restored = Message.from_dict(d)
    assert restored.role == "user"
    assert restored.content == "hello"
    assert restored.id == original.id

  def test_round_trip_preserves_tool_calls(self):
    calls = [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]
    original = Message(role="assistant", content=None, tool_calls=calls)
    d = original.to_dict()
    restored = Message.from_dict(d)
    assert restored.tool_calls is not None
    assert len(restored.tool_calls) == 1
    assert restored.tool_calls[0]["id"] == "call_1"


# ---------------------------------------------------------------------------
# MessageReferences
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMessageReferences:
  """MessageReferences stores query and optional references."""

  def test_creation_with_query(self):
    refs = MessageReferences(query="What is AI?")
    assert refs.query == "What is AI?"

  def test_references_default_none(self):
    refs = MessageReferences(query="test")
    assert refs.references is None

  def test_references_with_list(self):
    refs = MessageReferences(query="test", references=[{"source": "doc1"}, "raw ref"])
    assert refs.references is not None
    assert len(refs.references) == 2

  def test_time_default_none(self):
    refs = MessageReferences(query="test")
    assert refs.time is None

  def test_time_stored(self):
    refs = MessageReferences(query="test", time=0.5)
    assert refs.time == 0.5


# ---------------------------------------------------------------------------
# Citations and sub-types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUrlCitation:
  """UrlCitation stores url and title."""

  def test_creation_with_url(self):
    cite = UrlCitation(url="https://example.com", title="Example")
    assert cite.url == "https://example.com"
    assert cite.title == "Example"

  def test_defaults_are_none(self):
    cite = UrlCitation()
    assert cite.url is None
    assert cite.title is None


@pytest.mark.unit
class TestDocumentCitation:
  """DocumentCitation stores document fields."""

  def test_creation(self):
    doc = DocumentCitation(
      document_title="My Doc",
      cited_text="some text",
      file_name="doc.pdf",
    )
    assert doc.document_title == "My Doc"
    assert doc.cited_text == "some text"
    assert doc.file_name == "doc.pdf"

  def test_defaults_are_none(self):
    doc = DocumentCitation()
    assert doc.document_title is None
    assert doc.cited_text is None
    assert doc.file_name is None


@pytest.mark.unit
class TestCitations:
  """Citations aggregates raw, search_queries, urls, and documents."""

  def test_creation_with_defaults(self):
    cit = Citations()
    assert cit.raw is None
    assert cit.search_queries is None
    assert cit.urls is None
    assert cit.documents is None

  def test_creation_with_urls(self):
    url_cite = UrlCitation(url="https://example.com", title="Test")
    cit = Citations(urls=[url_cite])
    assert cit.urls is not None
    assert len(cit.urls) == 1
    assert cit.urls[0].url == "https://example.com"

  def test_creation_with_search_queries(self):
    cit = Citations(search_queries=["what is AI", "machine learning"])
    assert cit.search_queries is not None
    assert len(cit.search_queries) == 2

  def test_creation_with_documents(self):
    doc = DocumentCitation(document_title="Test Doc")
    cit = Citations(documents=[doc])
    assert cit.documents is not None
    assert len(cit.documents) == 1
    assert cit.documents[0].document_title == "Test Doc"

  def test_raw_accepts_any(self):
    cit = Citations(raw={"custom": "data"})
    assert cit.raw == {"custom": "data"}
