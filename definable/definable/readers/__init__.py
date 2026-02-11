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
    from definable.readers import BaseReader

    # Auto-detect and read files
    reader = BaseReader()
    result = reader.read(file)
    print(result.content)

With Agent:
    from definable.agents import Agent

    # Use defaults
    agent = Agent(model=model, readers=True)

    # Or customize
    from definable.readers import BaseReader
    reader = BaseReader()
    agent = Agent(model=model, readers=reader)
"""

from definable.readers.base import BaseReader
from definable.readers.models import ContentBlock, ReaderConfig, ReaderOutput
from definable.readers.registry import ParserRegistry

# Backwards-compatible aliases
FileReader = BaseReader
FileReaderConfig = ReaderConfig
ReaderResult = ReaderOutput
FileReaderRegistry = BaseReader

__all__ = [
  # New API
  "BaseReader",
  "ReaderOutput",
  "ContentBlock",
  "ReaderConfig",
  "ParserRegistry",
  # Backwards-compat aliases
  "FileReader",
  "FileReaderConfig",
  "ReaderResult",
  "FileReaderRegistry",
]

_LAZY_IMPORTS = {
  # Parsers
  "BaseParser": ("definable.readers.parsers.base_parser", "BaseParser"),
  "TextParser": ("definable.readers.parsers.text", "TextParser"),
  "PDFParser": ("definable.readers.parsers.pdf", "PDFParser"),
  "DocxParser": ("definable.readers.parsers.docx", "DocxParser"),
  "PptxParser": ("definable.readers.parsers.pptx", "PptxParser"),
  "XlsxParser": ("definable.readers.parsers.xlsx", "XlsxParser"),
  "OdsParser": ("definable.readers.parsers.ods", "OdsParser"),
  "RtfParser": ("definable.readers.parsers.rtf", "RtfParser"),
  "HTMLParser": ("definable.readers.parsers.html", "HTMLParser"),
  "ImageParser": ("definable.readers.parsers.image", "ImageParser"),
  "AudioParser": ("definable.readers.parsers.audio", "AudioParser"),
  # Providers
  "MistralReader": ("definable.readers.providers.mistral", "MistralReader"),
  "OpenAIReader": ("definable.readers.providers.openai", "OpenAIReader"),
  "AnthropicReader": ("definable.readers.providers.anthropic", "AnthropicReader"),
  "GoogleReader": ("definable.readers.providers.google", "GoogleReader"),
  # Old names → new locations
  "MistralOCRReader": ("definable.readers.providers.mistral", "MistralOCRReader"),
  "ImageFormatConverter": ("definable.readers.mistral.preprocessor", "ImageFormatConverter"),
  "FilePreprocessor": ("definable.readers.mistral.preprocessor", "FilePreprocessor"),
  # Old reader names → parsers (for import compat)
  "TextFileReader": ("definable.readers.parsers.text", "TextParser"),
  "PDFFileReader": ("definable.readers.parsers.pdf", "PDFParser"),
  "DocxFileReader": ("definable.readers.parsers.docx", "DocxParser"),
  "XlsxFileReader": ("definable.readers.parsers.xlsx", "XlsxParser"),
  "OdsFileReader": ("definable.readers.parsers.ods", "OdsParser"),
  "RtfFileReader": ("definable.readers.parsers.rtf", "RtfParser"),
  "AudioFileReader": ("definable.readers.parsers.audio", "AudioParser"),
  "AudioTranscriber": ("definable.readers.audio", "AudioTranscriber"),
  "OpenAITranscriber": ("definable.readers.audio", "OpenAITranscriber"),
}


def __getattr__(name: str):
  if name in _LAZY_IMPORTS:
    import importlib

    module_path, class_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
