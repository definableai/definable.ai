"""Tests for BaseReader orchestrator and ReaderConfig."""

import pytest

from definable.media import File
from definable.readers.base import BaseReader
from definable.readers.models import ReaderConfig


class TestReaderConfig:
  def test_defaults(self):
    config = ReaderConfig()
    assert config.max_file_size is None
    assert config.max_content_length is None
    assert config.encoding == "utf-8"
    assert config.timeout == 30.0

  def test_custom(self):
    config = ReaderConfig(
      max_file_size=1024,
      max_content_length=500,
      encoding="latin-1",
      timeout=10.0,
    )
    assert config.max_file_size == 1024
    assert config.encoding == "latin-1"


class TestBaseReader:
  def test_read_text_file(self):
    reader = BaseReader()
    file = File(content=b"Hello, world!", filename="test.txt", mime_type="text/plain")
    result = reader.read(file)
    assert result.content == "Hello, world!"
    assert result.error is None
    assert result.word_count > 0

  def test_read_unsupported_returns_error(self, unknown_file):
    reader = BaseReader()
    result = reader.read(unknown_file)
    assert result.error is not None
    assert result.content == ""

  def test_file_size_limit(self):
    config = ReaderConfig(max_file_size=10)
    reader = BaseReader(config=config)
    file = File(content=b"x" * 100, filename="big.txt", mime_type="text/plain")
    result = reader.read(file)
    assert result.error is not None
    assert "max_file_size" in result.error

  def test_content_truncation(self):
    config = ReaderConfig(max_content_length=5)
    reader = BaseReader(config=config)
    file = File(content=b"Hello, world!", filename="test.txt", mime_type="text/plain")
    result = reader.read(file)
    assert result.truncated is True
    assert len(result.content) == 5

  def test_get_parser(self):
    reader = BaseReader()
    file = File(content=b"test", filename="test.txt", mime_type="text/plain")
    parser = reader.get_parser(file)
    assert parser is not None

  def test_get_parser_unknown(self):
    reader = BaseReader()
    file = File(content=b"\x00", filename="data.bin")
    parser = reader.get_parser(file)
    assert parser is None

  def test_get_reader_alias(self):
    """get_reader() is a backwards-compat alias for get_parser()."""
    reader = BaseReader()
    file = File(content=b"test", filename="test.txt", mime_type="text/plain")
    assert reader.get_reader(file) is not None

  def test_register_returns_self(self):
    reader = BaseReader()
    from definable.readers.parsers.text import TextParser

    result = reader.register(TextParser())
    assert result is reader

  def test_can_read_by_mime_type(self):
    reader = BaseReader()
    file = File(content=b"hello", filename="test.txt", mime_type="text/plain")
    parser = reader.get_parser(file)
    assert parser is not None

  def test_can_read_by_extension(self):
    reader = BaseReader()
    file = File(content=b"hello", filename="script.py")
    parser = reader.get_parser(file)
    assert parser is not None

  def test_cannot_read_unknown_type(self):
    reader = BaseReader()
    file = File(content=b"\x00", filename="data.bin")
    parser = reader.get_parser(file)
    assert parser is None


class TestBaseReaderAsync:
  @pytest.mark.asyncio
  async def test_aread(self, default_reader, text_file):
    result = await default_reader.aread(text_file)
    assert result.content == "Hello, world! This is a test file."

  @pytest.mark.asyncio
  async def test_aread_unsupported(self, default_reader, unknown_file):
    result = await default_reader.aread(unknown_file)
    assert result.error is not None

  @pytest.mark.asyncio
  async def test_aread_all(self, default_reader, text_file, json_file, csv_file):
    results = await default_reader.aread_all([text_file, json_file, csv_file])
    assert len(results) == 3
    assert all(r.error is None for r in results)
    assert "Hello" in results[0].content
    assert "key" in results[1].content
    assert "Alice" in results[2].content

  @pytest.mark.asyncio
  async def test_aread_all_mixed(self, default_reader, text_file, unknown_file):
    results = await default_reader.aread_all([text_file, unknown_file])
    assert len(results) == 2
    assert results[0].error is None
    assert results[1].error is not None
