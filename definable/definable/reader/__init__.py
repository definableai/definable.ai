"""File readers for extracting content from files.

Provides a pre-processing layer that converts files (PDF, DOCX, XLSX,
CSV, audio, images) to structured content before they reach the LLM.

Architecture:
- **Parsers** — stateless, bytes → ContentBlock list. Never do I/O.
- **BaseReader** — orchestrator: resolves File → bytes, detects format,
  dispatches to parser, returns ReaderOutput.
- **Providers** — AI-backed readers (Mistral, etc.) that handle their
  own API I/O.

Quick Start:
    from definable.reader import BaseReader

    # Auto-detect and read files
    reader = BaseReader()
    result = reader.read(file)
    print(result.content)

With Agent:
    from definable.agent import Agent

    # Use defaults
    agent = Agent(model=model, readers=True)

    # Or customize
    from definable.reader import BaseReader
    reader = BaseReader()
    agent = Agent(model=model, readers=reader)
"""

from definable.reader.base import BaseReader
from definable.reader.models import ContentBlock, ReaderConfig, ReaderOutput
from definable.reader.registry import ParserRegistry

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.reader.audio import OPENAI_INPUT_AUDIO_FORMATS, AudioTranscriber, OpenAITranscriber, normalize_audio_format
  from definable.reader.parsers.audio import AudioParser
  from definable.reader.parsers.base_parser import BaseParser
  from definable.reader.parsers.docx import DocxParser
  from definable.reader.parsers.html import HTMLParser
  from definable.reader.parsers.image import ImageParser
  from definable.reader.parsers.ods import OdsParser
  from definable.reader.parsers.pdf import PDFParser
  from definable.reader.parsers.pptx import PptxParser
  from definable.reader.parsers.rtf import RtfParser
  from definable.reader.parsers.text import TextParser
  from definable.reader.parsers.xlsx import XlsxParser
  from definable.reader.providers.anthropic import AnthropicReader
  from definable.reader.providers.google import GoogleReader
  from definable.reader.providers.mistral import MistralOCRReader, MistralReader
  from definable.reader.providers.openai import OpenAIReader

__all__ = [
  # Core API
  "BaseReader",
  "ReaderOutput",
  "ContentBlock",
  "ReaderConfig",
  "ParserRegistry",
  # Parsers (lazy-loaded)
  "AudioParser",
  "BaseParser",
  "DocxParser",
  "HTMLParser",
  "ImageParser",
  "OdsParser",
  "PDFParser",
  "PptxParser",
  "RtfParser",
  "TextParser",
  "XlsxParser",
  # Providers (lazy-loaded)
  "AnthropicReader",
  "GoogleReader",
  "MistralOCRReader",
  "MistralReader",
  "OpenAIReader",
  # Audio (lazy-loaded)
  "AudioTranscriber",
  "OpenAITranscriber",
  "normalize_audio_format",
  "OPENAI_INPUT_AUDIO_FORMATS",
]

_LAZY_IMPORTS = {
  # Parsers
  "BaseParser": ("definable.reader.parsers.base_parser", "BaseParser"),
  "TextParser": ("definable.reader.parsers.text", "TextParser"),
  "PDFParser": ("definable.reader.parsers.pdf", "PDFParser"),
  "DocxParser": ("definable.reader.parsers.docx", "DocxParser"),
  "PptxParser": ("definable.reader.parsers.pptx", "PptxParser"),
  "XlsxParser": ("definable.reader.parsers.xlsx", "XlsxParser"),
  "OdsParser": ("definable.reader.parsers.ods", "OdsParser"),
  "RtfParser": ("definable.reader.parsers.rtf", "RtfParser"),
  "HTMLParser": ("definable.reader.parsers.html", "HTMLParser"),
  "ImageParser": ("definable.reader.parsers.image", "ImageParser"),
  "AudioParser": ("definable.reader.parsers.audio", "AudioParser"),
  # Providers
  "MistralReader": ("definable.reader.providers.mistral", "MistralReader"),
  "MistralOCRReader": ("definable.reader.providers.mistral", "MistralOCRReader"),
  "OpenAIReader": ("definable.reader.providers.openai", "OpenAIReader"),
  "AnthropicReader": ("definable.reader.providers.anthropic", "AnthropicReader"),
  "GoogleReader": ("definable.reader.providers.google", "GoogleReader"),
  # Audio
  "AudioTranscriber": ("definable.reader.audio", "AudioTranscriber"),
  "OpenAITranscriber": ("definable.reader.audio", "OpenAITranscriber"),
  "normalize_audio_format": ("definable.reader.audio", "normalize_audio_format"),
  "OPENAI_INPUT_AUDIO_FORMATS": ("definable.reader.audio", "OPENAI_INPUT_AUDIO_FORMATS"),
}


def __getattr__(name: str):
  if name in _LAZY_IMPORTS:
    import importlib

    module_path, class_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
