"""Shared fixtures for readers tests."""

import pytest

from definable.media import File
from definable.readers.base import BaseReader
from definable.readers.models import ReaderConfig
from definable.readers.parsers.text import TextParser
from definable.readers.registry import ParserRegistry


@pytest.fixture
def text_parser():
  return TextParser()


@pytest.fixture
def text_parser_with_config():
  return ReaderConfig(max_content_length=100)


@pytest.fixture
def default_reader():
  """Reader with only guaranteed-available parsers (TextParser)."""
  registry = ParserRegistry(include_defaults=False)
  registry.register(TextParser())
  return BaseReader(registry=registry)


@pytest.fixture
def text_file():
  return File(
    content=b"Hello, world! This is a test file.",
    filename="test.txt",
    mime_type="text/plain",
  )


@pytest.fixture
def json_file():
  return File(
    content=b'{"key": "value", "number": 42}',
    filename="data.json",
    mime_type="application/json",
  )


@pytest.fixture
def csv_file():
  return File(
    content=b"name,age,city\nAlice,30,NYC\nBob,25,LA",
    filename="data.csv",
    mime_type="text/csv",
  )


@pytest.fixture
def python_file():
  return File(
    content=b'def hello():\n    return "Hello, world!"',
    filename="script.py",
    mime_type="text/x-python",
  )


@pytest.fixture
def unknown_file():
  """A file with no mime_type and an unrecognized extension."""
  return File(
    content=b"\x00\x01\x02\x03",
    filename="data.bin",
  )
