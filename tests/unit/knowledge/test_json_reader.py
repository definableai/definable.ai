"""
Unit tests for JSONReader.

Tests pure logic: does the reader parse JSON correctly?
No API calls. No external dependencies.

Covers:
  - Read JSON file (single object, array, primitive)
  - read_string() for in-memory JSON
  - content_key extraction
  - metadata_keys extraction
  - flatten nested objects
  - Error handling (file not found, invalid JSON)
  - can_read() file extension check
  - Chunk metadata (source, source_type, index)
"""

import json
import tempfile
from pathlib import Path

import pytest

from definable.knowledge.reader.json_reader import JSONReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_json(data, suffix=".json") -> Path:
  """Write JSON data to a temporary file and return the path."""
  f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
  json.dump(data, f)
  f.close()
  return Path(f.name)


# ---------------------------------------------------------------------------
# Read from file
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONReaderFile:
  """Reading JSON files."""

  def test_read_single_object(self):
    path = write_json({"name": "Alice", "age": 30})
    reader = JSONReader()
    docs = reader.read(path)
    assert len(docs) == 1
    assert "Alice" in docs[0].content
    assert docs[0].source_type == "json"

  def test_read_array_of_objects(self):
    path = write_json([{"text": "Hello"}, {"text": "World"}, {"text": "Foo"}])
    reader = JSONReader()
    docs = reader.read(path)
    assert len(docs) == 3

  def test_read_empty_array(self):
    path = write_json([])
    reader = JSONReader()
    docs = reader.read(path)
    assert len(docs) == 0

  def test_read_primitive_string(self):
    path = write_json("just a string")
    reader = JSONReader()
    docs = reader.read(path)
    assert len(docs) == 1
    assert docs[0].content == "just a string"

  def test_read_primitive_number(self):
    path = write_json(42)
    reader = JSONReader()
    docs = reader.read(path)
    assert len(docs) == 1
    assert docs[0].content == "42"

  def test_read_file_not_found(self):
    reader = JSONReader()
    docs = reader.read("/nonexistent/file.json")
    assert docs == []

  def test_read_invalid_json(self):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    f.write("{invalid json")
    f.close()
    reader = JSONReader()
    docs = reader.read(f.name)
    assert docs == []

  def test_source_set_to_file_path(self):
    path = write_json({"key": "value"})
    reader = JSONReader()
    docs = reader.read(path)
    assert docs[0].source is not None
    assert str(path) in docs[0].source

  def test_array_items_indexed(self):
    path = write_json([{"a": 1}, {"b": 2}, {"c": 3}])
    reader = JSONReader()
    docs = reader.read(path)
    for i, doc in enumerate(docs):
      assert doc.meta_data["index"] == i


# ---------------------------------------------------------------------------
# read_string
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONReaderString:
  """Parsing JSON from strings."""

  def test_read_string_object(self):
    reader = JSONReader()
    docs = reader.read_string('{"text": "hello"}')
    assert len(docs) == 1
    assert "hello" in docs[0].content

  def test_read_string_array(self):
    reader = JSONReader()
    docs = reader.read_string('[{"a": 1}, {"b": 2}]')
    assert len(docs) == 2

  def test_read_string_invalid(self):
    reader = JSONReader()
    docs = reader.read_string("not json")
    assert docs == []

  def test_read_string_source_default(self):
    reader = JSONReader()
    docs = reader.read_string('{"key": "val"}')
    assert docs[0].source == "string"

  def test_read_string_custom_source(self):
    reader = JSONReader()
    docs = reader.read_string('{"key": "val"}', source="api_response")
    assert docs[0].source == "api_response"


# ---------------------------------------------------------------------------
# content_key
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONReaderContentKey:
  """content_key extracts specific field as content."""

  def test_content_key_extracts_field(self):
    reader = JSONReader(content_key="text")
    docs = reader.read_string('[{"text": "Hello world", "id": 1}]')
    assert len(docs) == 1
    assert docs[0].content == "Hello world"

  def test_content_key_missing_falls_back_to_full_json(self):
    reader = JSONReader(content_key="text")
    docs = reader.read_string('[{"title": "No text field"}]')
    assert len(docs) == 1
    # Falls back to stringifying the whole object
    assert "No text field" in docs[0].content

  def test_content_key_non_string_value_converted(self):
    reader = JSONReader(content_key="count")
    docs = reader.read_string('[{"count": 42}]')
    assert docs[0].content == "42"

  def test_content_key_with_nested_value(self):
    reader = JSONReader(content_key="data")
    docs = reader.read_string('[{"data": {"nested": true}}]')
    assert docs[0].content == "{'nested': True}"  # str() of dict


# ---------------------------------------------------------------------------
# metadata_keys
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONReaderMetadataKeys:
  """metadata_keys extracts specific fields as meta_data."""

  def test_metadata_keys_extracted(self):
    reader = JSONReader(metadata_keys=["author", "date"])
    docs = reader.read_string('[{"text": "content", "author": "Alice", "date": "2024-01-01"}]')
    assert docs[0].meta_data["author"] == "Alice"
    assert docs[0].meta_data["date"] == "2024-01-01"

  def test_metadata_keys_excluded_from_content(self):
    reader = JSONReader(metadata_keys=["author"])
    docs = reader.read_string('[{"text": "content", "author": "Alice"}]')
    # author should NOT appear in content (only in meta_data)
    assert "Alice" not in docs[0].content

  def test_metadata_keys_missing_key_skipped(self):
    reader = JSONReader(metadata_keys=["author", "missing_key"])
    docs = reader.read_string('[{"text": "content", "author": "Alice"}]')
    assert docs[0].meta_data.get("author") == "Alice"
    assert "missing_key" not in docs[0].meta_data

  def test_no_metadata_keys_returns_empty_extra_meta(self):
    reader = JSONReader()
    docs = reader.read_string('[{"text": "content"}]')
    # Only source and index from _parse, no extra metadata
    assert "text" not in docs[0].meta_data

  def test_content_key_with_metadata_keys(self):
    reader = JSONReader(content_key="text", metadata_keys=["author"])
    docs = reader.read_string('[{"text": "Hello", "author": "Bob", "extra": "stuff"}]')
    assert docs[0].content == "Hello"
    assert docs[0].meta_data["author"] == "Bob"


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONReaderFlatten:
  """flatten converts nested objects to dot-separated keys."""

  def test_flatten_nested_object(self):
    reader = JSONReader(flatten=True)
    docs = reader.read_string('[{"user": {"name": "Alice", "age": 30}}]')
    content = docs[0].content
    assert "user.name" in content
    assert "user.age" in content

  def test_flatten_deeply_nested(self):
    reader = JSONReader(flatten=True)
    data = [{"a": {"b": {"c": {"d": "deep"}}}}]
    docs = reader.read_string(json.dumps(data))
    assert "a.b.c.d" in docs[0].content

  def test_flatten_preserves_non_dict_values(self):
    reader = JSONReader(flatten=True)
    docs = reader.read_string('[{"name": "Alice", "tags": [1, 2, 3]}]')
    content = docs[0].content
    assert "name" in content
    assert "tags" in content

  def test_no_flatten_keeps_nested(self):
    reader = JSONReader(flatten=False)
    docs = reader.read_string('[{"user": {"name": "Alice"}}]')
    content = docs[0].content
    # Should contain nested JSON structure
    assert '"user"' in content

  def test_flatten_static_method(self):
    result = JSONReader._flatten_dict({"a": {"b": 1, "c": {"d": 2}}, "e": 3})
    assert result == {"a.b": 1, "a.c.d": 2, "e": 3}

  def test_flatten_empty_dict(self):
    assert JSONReader._flatten_dict({}) == {}

  def test_flatten_with_content_key(self):
    reader = JSONReader(flatten=True, content_key="info.name")
    data = [{"info": {"name": "Alice", "age": 30}}]
    docs = reader.read_string(json.dumps(data))
    # After flatten, key becomes "info.name"
    assert docs[0].content == "Alice"


# ---------------------------------------------------------------------------
# can_read
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONReaderCanRead:
  """can_read checks file extension."""

  def test_json_extension(self):
    reader = JSONReader()
    assert reader.can_read("data.json") is True

  def test_json_uppercase(self):
    reader = JSONReader()
    assert reader.can_read("DATA.JSON") is True

  def test_non_json_extension(self):
    reader = JSONReader()
    assert reader.can_read("data.csv") is False

  def test_no_extension(self):
    reader = JSONReader()
    assert reader.can_read("data") is False

  def test_path_object(self):
    reader = JSONReader()
    assert reader.can_read(Path("dir/file.json")) is True


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONReaderAsync:
  """Async read via thread pool."""

  @pytest.mark.asyncio
  async def test_aread_file(self):
    path = write_json({"key": "async_value"})
    reader = JSONReader()
    docs = await reader.aread(path)
    assert len(docs) == 1
    assert "async_value" in docs[0].content

  @pytest.mark.asyncio
  async def test_aread_file_not_found(self):
    reader = JSONReader()
    docs = await reader.aread("/nonexistent/async.json")
    assert docs == []


# ---------------------------------------------------------------------------
# Document naming
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONReaderNaming:
  """Document names are meaningful."""

  def test_name_from_filename(self):
    path = write_json([{"a": 1}, {"b": 2}])
    reader = JSONReader()
    docs = reader.read(path)
    stem = path.stem
    assert docs[0].name == f"{stem}_0"
    assert docs[1].name == f"{stem}_1"

  def test_name_from_string_source(self):
    reader = JSONReader()
    docs = reader.read_string('[{"a": 1}]', source="api")
    assert docs[0].name == "api_0"

  def test_name_fallback_no_source(self):
    reader = JSONReader()
    docs = reader.read_string('[{"a": 1}]', source="")
    assert docs[0].name is not None
    assert "json_0" in docs[0].name


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONReaderEncoding:
  """Custom encoding support."""

  def test_utf8_default(self):
    reader = JSONReader()
    assert reader.encoding == "utf-8"

  def test_unicode_content(self):
    path = write_json({"text": "Héllo wörld 你好"})
    reader = JSONReader()
    docs = reader.read(path)
    assert "Héllo" in docs[0].content
    assert "你好" in docs[0].content


# ---------------------------------------------------------------------------
# Array filtering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONReaderArrayFiltering:
  """Non-dict items in arrays are filtered out."""

  def test_mixed_array_filters_non_dicts(self):
    reader = JSONReader()
    docs = reader.read_string('[{"a": 1}, "string", 42, {"b": 2}]')
    assert len(docs) == 2  # Only dicts

  def test_all_non_dicts_returns_empty(self):
    reader = JSONReader()
    docs = reader.read_string('[1, 2, "three"]')
    assert len(docs) == 0


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONReaderImports:
  """Importable from convenience paths."""

  def test_import_from_knowledge_reader(self):
    from definable.knowledge.reader import JSONReader as JR

    assert JR is JSONReader
