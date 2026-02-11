"""Tests for ParserRegistry."""

from typing import List, Set

import pytest

from definable.media import File
from definable.readers.base import BaseReader
from definable.readers.models import ContentBlock, ReaderConfig
from definable.readers.parsers.base_parser import BaseParser
from definable.readers.parsers.text import TextParser
from definable.readers.registry import ParserRegistry


class MockPDFParser(BaseParser):
  """Fake PDF parser for testing priority overrides."""

  def __init__(self, provider_name: str = "mock-local") -> None:
    self.provider_name = provider_name

  def supported_mime_types(self) -> List[str]:
    return ["application/pdf"]

  def supported_extensions(self) -> Set[str]:
    return {".pdf"}

  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    return [
      ContentBlock(
        content_type="text",
        content=f"[{self.provider_name}] extracted text",
        mime_type="application/pdf",
      )
    ]


class MultiFormatParser(BaseParser):
  """Parser that handles multiple formats (simulates API-based provider)."""

  def supported_mime_types(self) -> List[str]:
    return ["application/pdf", "image/png", "image/jpeg"]

  def supported_extensions(self) -> Set[str]:
    return {".pdf", ".png", ".jpg", ".jpeg"}

  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    return [
      ContentBlock(
        content_type="text",
        content="[api-provider] OCR text",
        mime_type=mime_type,
      )
    ]


class TestRegistryAutoDetection:
  def test_text_file_auto_detected(self, default_reader, text_file):
    parser = default_reader.get_parser(text_file)
    assert parser is not None
    assert isinstance(parser, TextParser)

  def test_unknown_format_returns_none(self, default_reader, unknown_file):
    parser = default_reader.get_parser(unknown_file)
    assert parser is None

  def test_read_unsupported_returns_error_result(self, default_reader, unknown_file):
    result = default_reader.read(unknown_file)
    assert result.error is not None
    assert result.content == ""


class TestRegistryPriority:
  def test_higher_priority_wins(self):
    registry = ParserRegistry(include_defaults=False)
    local_parser = MockPDFParser(provider_name="local-pypdf")
    api_parser = MockPDFParser(provider_name="api-ocr")

    registry.register(local_parser, priority=50)
    registry.register(api_parser, priority=200)  # Higher priority wins

    reader = BaseReader(registry=registry)
    pdf_file = File(content=b"%PDF", filename="doc.pdf", mime_type="application/pdf")
    result = reader.read(pdf_file)
    assert "[api-ocr]" in result.content

  def test_override_builtin_parser(self):
    registry = ParserRegistry(include_defaults=False)
    registry.register(TextParser(), priority=0)
    registry.register(MockPDFParser(provider_name="builtin"), priority=0)

    # Now override with a higher-priority parser
    registry.register(MockPDFParser(provider_name="custom"), priority=200)

    reader = BaseReader(registry=registry)
    pdf_file = File(content=b"%PDF", filename="doc.pdf", mime_type="application/pdf")
    result = reader.read(pdf_file)
    assert "[custom]" in result.content


class TestMultiFormatParser:
  def test_single_parser_handles_multiple_formats(self):
    registry = ParserRegistry(include_defaults=False)
    registry.register(MultiFormatParser())

    reader = BaseReader(registry=registry)

    pdf_file = File(content=b"%PDF", filename="doc.pdf", mime_type="application/pdf")
    png_file = File(content=b"\x89PNG", filename="img.png")

    assert reader.get_parser(pdf_file) is not None
    assert reader.get_parser(png_file) is not None

    pdf_result = reader.read(pdf_file)
    png_result = reader.read(png_file)
    assert "[api-provider]" in pdf_result.content
    assert "[api-provider]" in png_result.content


class TestRegistryAsync:
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
  async def test_aread_all_mixed_supported_unsupported(self, default_reader, text_file, unknown_file):
    results = await default_reader.aread_all([text_file, unknown_file])
    assert len(results) == 2
    assert results[0].error is None
    assert results[1].error is not None


class TestMethodChaining:
  def test_register_returns_self(self):
    registry = ParserRegistry(include_defaults=False)
    result = registry.register(TextParser())
    assert result is registry

  def test_chaining(self):
    registry = ParserRegistry(include_defaults=False)
    registry.register(TextParser()).register(MockPDFParser())

    assert registry.get_parser("text/plain") is not None
    assert registry.get_parser("application/pdf") is not None


class TestReadersTrueConvenience:
  def test_readers_true_creates_default_reader(self):
    """Test that readers=True on Agent creates a BaseReader."""
    from definable.agents.agent import Agent
    from definable.agents.testing import MockModel

    agent = Agent(model=MockModel(), readers=True)
    assert agent.readers is not None
    assert isinstance(agent.readers, BaseReader)

  def test_readers_none_default(self):
    from definable.agents.agent import Agent
    from definable.agents.testing import MockModel

    agent = Agent(model=MockModel())
    assert agent.readers is None

  def test_readers_single_parser(self):
    from definable.agents.agent import Agent
    from definable.agents.testing import MockModel

    agent = Agent(model=MockModel(), readers=TextParser())
    assert agent.readers is not None
    assert isinstance(agent.readers, BaseReader)
