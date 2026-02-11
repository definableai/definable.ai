"""Tests for TextParser."""

import pytest

from definable.media import File
from definable.readers.base import BaseReader
from definable.readers.models import ReaderConfig


class TestTextParser:
  def test_parse_plain_text(self, text_parser, text_file):
    blocks = text_parser.parse(text_file.content, mime_type="text/plain")
    assert len(blocks) == 1
    assert blocks[0].content == "Hello, world! This is a test file."

  def test_parse_json(self, text_parser, json_file):
    blocks = text_parser.parse(json_file.content, mime_type="application/json")
    assert '"key": "value"' in blocks[0].content

  def test_parse_csv(self, text_parser, csv_file):
    blocks = text_parser.parse(csv_file.content, mime_type="text/csv")
    assert "Alice" in blocks[0].content
    assert "Bob" in blocks[0].content

  def test_parse_python(self, text_parser, python_file):
    blocks = text_parser.parse(python_file.content, mime_type="text/x-python")
    assert "def hello" in blocks[0].content

  def test_handles_encoding_errors(self, text_parser):
    blocks = text_parser.parse(b"\xff\xfe\x00\x01", mime_type="text/plain")
    assert len(blocks) == 1
    assert len(blocks[0].content) > 0

  def test_supported_extensions(self, text_parser):
    exts = text_parser.supported_extensions()
    assert ".txt" in exts
    assert ".md" in exts
    assert ".py" in exts
    assert ".json" in exts
    assert ".csv" in exts

  def test_supported_mime_types(self, text_parser):
    mimes = text_parser.supported_mime_types()
    assert "text/plain" in mimes
    assert "application/json" in mimes

  def test_wildcard_text_mime(self, text_parser):
    assert text_parser.can_parse(mime_type="text/x-custom-type") is True


class TestTextParserViaReader:
  def test_read_plain_text(self, text_file):
    reader = BaseReader()
    result = reader.read(text_file)
    assert result.content == "Hello, world! This is a test file."
    assert result.filename == "test.txt"
    assert result.error is None
    assert result.word_count > 0

  def test_truncation(self):
    config = ReaderConfig(max_content_length=100)
    reader = BaseReader(config=config)
    long_content = b"x" * 200
    file = File(content=long_content, filename="long.txt", mime_type="text/plain")
    result = reader.read(file)
    assert len(result.content) == 100
    assert result.truncated is True

  def test_no_truncation_for_short_content(self):
    config = ReaderConfig(max_content_length=100)
    reader = BaseReader(config=config)
    file = File(content=b"short", filename="short.txt", mime_type="text/plain")
    result = reader.read(file)
    assert result.content == "short"
    assert result.truncated is False


class TestTextParserAsync:
  @pytest.mark.asyncio
  async def test_aread_plain_text(self, text_file):
    reader = BaseReader()
    result = await reader.aread(text_file)
    assert result.content == "Hello, world! This is a test file."
    assert result.error is None

  @pytest.mark.asyncio
  async def test_aread_with_truncation(self):
    config = ReaderConfig(max_content_length=100)
    reader = BaseReader(config=config)
    long_content = b"y" * 200
    file = File(content=long_content, filename="long.txt", mime_type="text/plain")
    result = await reader.aread(file)
    assert len(result.content) == 100
    assert result.truncated is True

  @pytest.mark.asyncio
  async def test_file_size_limit(self):
    config = ReaderConfig(max_file_size=10)
    reader = BaseReader(config=config)
    file = File(content=b"x" * 100, filename="big.txt", mime_type="text/plain")
    result = await reader.aread(file)
    assert result.error is not None
    assert "max_file_size" in result.error
