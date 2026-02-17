"""Tests for Document metadata alias — issue #11.

Verifies that Document accepts both ``metadata`` (new standard) and
``meta_data`` (legacy) as parameter names, with full backwards compatibility.
"""

import json

import pytest

from definable.knowledge.document import Document


class TestDocumentMetadataAlias:
  """Document accepts both metadata and meta_data."""

  # ------------------------------------------------------------------
  # Construction
  # ------------------------------------------------------------------

  def test_metadata_kwarg_works(self):
    """Document(metadata=...) sets the metadata field."""
    doc = Document(content="Hello", metadata={"source": "test"})
    assert doc.metadata == {"source": "test"}

  def test_meta_data_kwarg_works(self):
    """Document(meta_data=...) still works for backwards compatibility."""
    doc = Document(content="Hello", meta_data={"source": "legacy"})
    assert doc.metadata == {"source": "legacy"}

  def test_default_metadata_is_empty_dict(self):
    """Document() without metadata defaults to empty dict."""
    doc = Document(content="Hello")
    assert doc.metadata == {}

  def test_both_metadata_and_meta_data_raises_type_error(self):
    """Passing both metadata= and meta_data= raises TypeError."""
    with pytest.raises(TypeError, match="both"):
      Document(content="Hello", metadata={"a": 1}, meta_data={"b": 2})

  # ------------------------------------------------------------------
  # Property alias
  # ------------------------------------------------------------------

  def test_meta_data_property_reads_metadata(self):
    """doc.meta_data returns the same value as doc.metadata."""
    doc = Document(content="Hello", metadata={"key": "value"})
    assert doc.meta_data == {"key": "value"}
    assert doc.meta_data is doc.metadata  # same object

  def test_meta_data_property_setter_updates_metadata(self):
    """Setting doc.meta_data updates doc.metadata."""
    doc = Document(content="Hello")
    doc.meta_data = {"updated": True}
    assert doc.metadata == {"updated": True}

  def test_metadata_setter_reflected_in_meta_data(self):
    """Setting doc.metadata is reflected in doc.meta_data."""
    doc = Document(content="Hello")
    doc.metadata = {"direct": True}
    assert doc.meta_data == {"direct": True}

  # ------------------------------------------------------------------
  # Serialization
  # ------------------------------------------------------------------

  def test_to_dict_uses_metadata_key(self):
    """to_dict() outputs 'metadata', not 'meta_data'."""
    doc = Document(content="Hello", metadata={"k": "v"}, name="test")
    d = doc.to_dict()
    assert "metadata" in d
    assert "meta_data" not in d
    assert d["metadata"] == {"k": "v"}

  def test_to_dict_omits_empty_metadata(self):
    """to_dict() includes metadata even when empty (it's not None)."""
    doc = Document(content="Hello")
    d = doc.to_dict()
    # metadata is {} (not None), so it should be included
    assert "metadata" in d
    assert d["metadata"] == {}

  def test_from_dict_with_metadata_key(self):
    """from_dict() accepts 'metadata' key."""
    doc = Document.from_dict({"content": "Hello", "metadata": {"k": "v"}})
    assert doc.metadata == {"k": "v"}

  def test_from_dict_with_meta_data_key(self):
    """from_dict() accepts legacy 'meta_data' key."""
    doc = Document.from_dict({"content": "Hello", "meta_data": {"k": "v"}})
    assert doc.metadata == {"k": "v"}

  def test_from_json_with_metadata_key(self):
    """from_json() accepts 'metadata' key."""
    data = json.dumps({"content": "Hello", "metadata": {"j": 1}})
    doc = Document.from_json(data)
    assert doc.metadata == {"j": 1}

  def test_from_json_with_meta_data_key(self):
    """from_json() accepts legacy 'meta_data' key."""
    data = json.dumps({"content": "Hello", "meta_data": {"j": 2}})
    doc = Document.from_json(data)
    assert doc.metadata == {"j": 2}

  # ------------------------------------------------------------------
  # Round-trip
  # ------------------------------------------------------------------

  def test_roundtrip_to_dict_from_dict(self):
    """Document -> to_dict -> from_dict preserves metadata."""
    original = Document(content="Hello", metadata={"round": "trip"}, name="rt")
    restored = Document.from_dict(original.to_dict())
    assert restored.metadata == original.metadata
    assert restored.content == original.content

  def test_roundtrip_to_dict_from_dict_with_legacy_construction(self):
    """Document(meta_data=...) -> to_dict -> from_dict preserves data."""
    original = Document(content="Hello", meta_data={"legacy": "round"}, name="rt")
    d = original.to_dict()
    # to_dict now uses 'metadata' key
    assert "metadata" in d
    restored = Document.from_dict(d)
    assert restored.metadata == {"legacy": "round"}

  # ------------------------------------------------------------------
  # Integration: existing patterns still work
  # ------------------------------------------------------------------

  def test_spread_meta_data_dict(self):
    """Spreading doc.meta_data in a new Document still works (chunker pattern)."""
    parent = Document(content="Parent", meta_data={"source": "file.txt"})
    child = Document(
      content="Child chunk",
      meta_data={**parent.meta_data, "chunk_index": 0},
    )
    assert child.metadata == {"source": "file.txt", "chunk_index": 0}

  def test_spread_metadata_dict(self):
    """Spreading doc.metadata in a new Document works with new API."""
    parent = Document(content="Parent", metadata={"source": "file.txt"})
    child = Document(
      content="Child chunk",
      metadata={**parent.metadata, "chunk_index": 0},
    )
    assert child.metadata == {"source": "file.txt", "chunk_index": 0}

  def test_meta_data_get_method(self):
    """doc.meta_data.get() works (InMemoryVectorDB filter pattern)."""
    doc = Document(content="Hello", metadata={"topic": "ai"})
    assert doc.meta_data.get("topic") == "ai"
    assert doc.meta_data.get("missing") is None
