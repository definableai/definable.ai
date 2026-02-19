"""
Unit tests for ParserRegistry.

Tests: construction, register(), get_parser(), priority ordering,
default parser registration. No file I/O, no API calls.

Covers:
  - ParserRegistry() with defaults registers built-in parsers
  - ParserRegistry(include_defaults=False) starts empty
  - register() adds a parser and returns self
  - register() sorts by priority descending
  - get_parser() returns highest-priority match
  - get_parser() returns None when no parser matches
  - parsers property returns parsers in priority order
"""

from unittest.mock import MagicMock

import pytest

from definable.reader.registry import ParserRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parser(name="mock", can_parse_result=True):
  """Build a mock BaseParser."""
  parser = MagicMock()
  parser.can_parse = MagicMock(return_value=can_parse_result)
  parser.__repr__ = lambda self: f"<MockParser {name}>"
  return parser


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParserRegistryConstruction:
  """ParserRegistry construction with and without defaults."""

  def test_with_defaults_has_parsers(self):
    registry = ParserRegistry(include_defaults=True)
    assert len(registry.parsers) > 0

  def test_without_defaults_is_empty(self):
    registry = ParserRegistry(include_defaults=False)
    assert len(registry.parsers) == 0

  def test_default_includes_text_parser(self):
    registry = ParserRegistry(include_defaults=True)
    parser_types = [type(p).__name__ for p in registry.parsers]
    assert "TextParser" in parser_types

  def test_default_includes_html_parser(self):
    registry = ParserRegistry(include_defaults=True)
    parser_types = [type(p).__name__ for p in registry.parsers]
    assert "HTMLParser" in parser_types

  def test_default_includes_image_parser(self):
    registry = ParserRegistry(include_defaults=True)
    parser_types = [type(p).__name__ for p in registry.parsers]
    assert "ImageParser" in parser_types


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParserRegistryRegister:
  """ParserRegistry.register() adds parsers with priority."""

  def test_register_returns_self(self):
    registry = ParserRegistry(include_defaults=False)
    parser = _make_parser()
    result = registry.register(parser)
    assert result is registry

  def test_register_adds_parser(self):
    registry = ParserRegistry(include_defaults=False)
    parser = _make_parser()
    registry.register(parser)
    assert parser in registry.parsers

  def test_register_default_priority_100(self):
    registry = ParserRegistry(include_defaults=False)
    parser = _make_parser()
    registry.register(parser)
    assert len(registry._entries) == 1
    assert registry._entries[0][1] == 100

  def test_register_custom_priority(self):
    registry = ParserRegistry(include_defaults=False)
    parser = _make_parser()
    registry.register(parser, priority=50)
    assert registry._entries[0][1] == 50

  def test_register_sorts_by_priority_descending(self):
    registry = ParserRegistry(include_defaults=False)
    low = _make_parser("low")
    high = _make_parser("high")
    registry.register(low, priority=10)
    registry.register(high, priority=200)
    assert registry.parsers[0] is high
    assert registry.parsers[1] is low

  def test_method_chaining(self):
    registry = ParserRegistry(include_defaults=False)
    p1 = _make_parser("p1")
    p2 = _make_parser("p2")
    registry.register(p1, priority=10).register(p2, priority=20)
    assert len(registry.parsers) == 2


# ---------------------------------------------------------------------------
# get_parser()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParserRegistryGetParser:
  """ParserRegistry.get_parser() returns the highest-priority match."""

  def test_returns_matching_parser(self):
    registry = ParserRegistry(include_defaults=False)
    parser = _make_parser(can_parse_result=True)
    registry.register(parser)
    assert registry.get_parser(mime_type="text/plain") is parser

  def test_returns_none_when_no_match(self):
    registry = ParserRegistry(include_defaults=False)
    parser = _make_parser(can_parse_result=False)
    registry.register(parser)
    assert registry.get_parser(mime_type="text/plain") is None

  def test_returns_none_when_empty(self):
    registry = ParserRegistry(include_defaults=False)
    assert registry.get_parser(mime_type="text/plain") is None

  def test_returns_highest_priority_match(self):
    registry = ParserRegistry(include_defaults=False)
    low = _make_parser("low", can_parse_result=True)
    high = _make_parser("high", can_parse_result=True)
    registry.register(low, priority=10)
    registry.register(high, priority=200)
    assert registry.get_parser(mime_type="text/plain") is high

  def test_passes_mime_type_and_extension(self):
    registry = ParserRegistry(include_defaults=False)
    parser = _make_parser(can_parse_result=True)
    registry.register(parser)
    registry.get_parser(mime_type="text/plain", extension=".txt")
    parser.can_parse.assert_called_with("text/plain", ".txt")

  def test_skips_non_matching_returns_first_match(self):
    registry = ParserRegistry(include_defaults=False)
    no_match = _make_parser("no", can_parse_result=False)
    match = _make_parser("yes", can_parse_result=True)
    registry.register(no_match, priority=200)
    registry.register(match, priority=100)
    assert registry.get_parser(mime_type="text/plain") is match


# ---------------------------------------------------------------------------
# parsers property
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParserRegistryParsersProperty:
  """ParserRegistry.parsers returns parsers in priority order."""

  def test_returns_list(self):
    registry = ParserRegistry(include_defaults=False)
    assert isinstance(registry.parsers, list)

  def test_preserves_priority_order(self):
    registry = ParserRegistry(include_defaults=False)
    p1 = _make_parser("p1")
    p2 = _make_parser("p2")
    p3 = _make_parser("p3")
    registry.register(p1, priority=50)
    registry.register(p2, priority=200)
    registry.register(p3, priority=100)
    assert registry.parsers == [p2, p3, p1]
