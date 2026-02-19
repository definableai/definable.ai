"""
Unit tests for the Document dataclass.

Tests pure Python logic: creation, serialization, field defaults.
No API calls. No external dependencies.

Covers:
  - Document creates with required and optional fields
  - to_dict() / from_dict() round-trip
  - from_json() parsing
  - embed() raises ValueError without embedder
  - async_embed() raises ValueError without embedder
  - Knowledge pipeline fields (source, source_type, chunk_index, etc.)
  - meta_data defaults to empty dict
"""

import json

import pytest

from definable.knowledge.document import Document


@pytest.mark.unit
class TestDocumentCreation:
  """Document dataclass construction and field defaults."""

  def test_minimal_creation(self):
    doc = Document(content="Hello world")
    assert doc.content == "Hello world"
    assert doc.id is None
    assert doc.name is None
    assert doc.meta_data == {}
    assert doc.embedding is None
    assert doc.source is None
    assert doc.source_type is None
    assert doc.chunk_index is None
    assert doc.chunk_total is None
    assert doc.parent_id is None

  def test_full_field_creation(self):
    doc = Document(
      content="Full doc",
      id="doc-001",
      name="My Document",
      meta_data={"category": "test", "version": 1},
      source="/path/to/file.txt",
      source_type="file",
      chunk_index=2,
      chunk_total=5,
      parent_id="parent-001",
    )
    assert doc.id == "doc-001"
    assert doc.name == "My Document"
    assert doc.meta_data["category"] == "test"
    assert doc.source == "/path/to/file.txt"
    assert doc.source_type == "file"
    assert doc.chunk_index == 2
    assert doc.chunk_total == 5
    assert doc.parent_id == "parent-001"

  def test_meta_data_is_independent_per_instance(self):
    """Each Document should have its own meta_data dict (not shared)."""
    doc1 = Document(content="Doc 1")
    doc2 = Document(content="Doc 2")
    doc1.meta_data["key"] = "value"
    assert "key" not in doc2.meta_data

  def test_embedding_field_stores_floats(self):
    embedding = [0.1, 0.2, 0.3]
    doc = Document(content="embedded", embedding=embedding)
    assert doc.embedding == embedding

  def test_reranking_score_field(self):
    doc = Document(content="reranked", reranking_score=0.95)
    assert doc.reranking_score == 0.95

  def test_content_id_and_origin_fields(self):
    doc = Document(content="test", content_id="cid-1", content_origin="upload")
    assert doc.content_id == "cid-1"
    assert doc.content_origin == "upload"

  def test_size_field(self):
    doc = Document(content="test", size=42)
    assert doc.size == 42

  def test_usage_field_default_none(self):
    doc = Document(content="test")
    assert doc.usage is None


@pytest.mark.unit
class TestDocumentSerialization:
  """Document serialization and deserialization."""

  def test_to_dict_contains_required_fields(self):
    doc = Document(content="Test content", name="Test Name", meta_data={"k": "v"})
    d = doc.to_dict()
    assert "content" in d
    assert d["content"] == "Test content"

  def test_to_dict_always_includes_content(self):
    doc = Document(content="")
    d = doc.to_dict()
    assert "content" in d

  def test_to_dict_includes_name_when_set(self):
    doc = Document(content="test", name="MyDoc")
    d = doc.to_dict()
    assert d["name"] == "MyDoc"

  def test_to_dict_includes_meta_data_when_set(self):
    doc = Document(content="test", meta_data={"key": "value"})
    d = doc.to_dict()
    assert d["meta_data"] == {"key": "value"}

  def test_to_dict_omits_none_name(self):
    doc = Document(content="test")
    d = doc.to_dict()
    assert "name" not in d

  def test_from_dict_round_trip(self):
    original = Document(content="Round trip doc", name="MyDoc", meta_data={"x": 42})
    d = original.to_dict()
    restored = Document.from_dict(d)
    assert restored.content == original.content
    assert restored.name == original.name
    assert restored.meta_data == original.meta_data

  def test_from_json_round_trip(self):
    original = Document(content="JSON doc", name="JsonDoc", meta_data={"flag": True})
    d = original.to_dict()
    json_str = json.dumps(d)
    restored = Document.from_json(json_str)
    assert restored.content == original.content
    assert restored.name == original.name

  def test_from_dict_minimal(self):
    doc = Document.from_dict({"content": "Minimal"})
    assert doc.content == "Minimal"

  def test_from_json_minimal(self):
    doc = Document.from_json('{"content": "JSON minimal"}')
    assert doc.content == "JSON minimal"


@pytest.mark.unit
class TestDocumentEmbedMethod:
  """Document.embed() and async_embed() raise without embedder."""

  def test_embed_raises_without_embedder(self):
    doc = Document(content="No embedder here")
    with pytest.raises(ValueError, match="No embedder provided"):
      doc.embed()

  @pytest.mark.asyncio
  async def test_async_embed_raises_without_embedder(self):
    doc = Document(content="No async embedder here")
    with pytest.raises(ValueError, match="No embedder provided"):
      await doc.async_embed()

  def test_embed_uses_provided_embedder(self):
    """embed() stores the embedding returned by the embedder."""

    class _FakeEmbedder:
      def get_embedding_and_usage(self, text):
        return ([0.1, 0.2, 0.3], {"tokens": 3})

    doc = Document(content="Embed me")
    doc.embed(embedder=_FakeEmbedder())  # type: ignore[arg-type]
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.usage == {"tokens": 3}

  def test_embed_uses_instance_embedder_if_no_argument(self):
    """embed() uses self.embedder when no argument is passed."""

    class _FakeEmbedder:
      def get_embedding_and_usage(self, text):
        return ([0.5, 0.6], None)

    doc = Document(content="Instance embedder", embedder=_FakeEmbedder())  # type: ignore[arg-type]
    doc.embed()
    assert doc.embedding == [0.5, 0.6]

  @pytest.mark.asyncio
  async def test_async_embed_uses_provided_embedder(self):
    """async_embed() stores the embedding returned by the embedder."""

    class _FakeAsyncEmbedder:
      async def async_get_embedding_and_usage(self, text):
        return ([0.7, 0.8], {"tokens": 2})

    doc = Document(content="Async embed me")
    await doc.async_embed(embedder=_FakeAsyncEmbedder())  # type: ignore[arg-type]
    assert doc.embedding == [0.7, 0.8]
    assert doc.usage == {"tokens": 2}

  def test_embed_with_mock_embedder_fixture(self, mock_embedder):
    """embed() works with the conftest HashEmbedder fixture."""
    doc = Document(content="Hello world")
    doc.embed(embedder=mock_embedder)
    assert doc.embedding is not None
    assert len(doc.embedding) == 128

  def test_embed_overwrites_previous_embedding(self):
    """Calling embed() a second time replaces the old embedding."""

    class _FakeEmbedder:
      def __init__(self, vec):
        self._vec = vec

      def get_embedding_and_usage(self, text):
        return (self._vec, None)

    doc = Document(content="test")
    doc.embed(embedder=_FakeEmbedder([1.0, 2.0]))  # type: ignore[arg-type]
    assert doc.embedding == [1.0, 2.0]
    doc.embed(embedder=_FakeEmbedder([3.0, 4.0]))  # type: ignore[arg-type]
    assert doc.embedding == [3.0, 4.0]


@pytest.mark.unit
class TestDocumentMetaData:
  """meta_data field behavior (NOT metadata -- uses meta_data)."""

  def test_meta_data_default_empty_dict(self):
    doc = Document(content="test")
    assert doc.meta_data == {}
    assert isinstance(doc.meta_data, dict)

  def test_meta_data_with_values(self):
    doc = Document(content="test", meta_data={"key": "value", "num": 42})
    assert doc.meta_data["key"] == "value"
    assert doc.meta_data["num"] == 42

  def test_meta_data_mutable(self):
    doc = Document(content="test")
    doc.meta_data["added"] = True
    assert doc.meta_data["added"] is True

  def test_meta_data_not_shared_between_instances(self):
    """Regression: ensure default_factory creates separate dicts."""
    docs = [Document(content=f"doc{i}") for i in range(3)]
    docs[0].meta_data["only_on_first"] = True
    assert "only_on_first" not in docs[1].meta_data
    assert "only_on_first" not in docs[2].meta_data
