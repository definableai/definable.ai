# readers

File content extraction — parse files into structured, multimodal content blocks.

## Installation

```bash
pip install 'definable[readers]'       # All built-in parsers (pypdf, python-docx, etc.)
pip install 'definable[mistral-ocr]'   # Mistral cloud OCR provider
```

Individual parser dependencies can also be installed separately (see Parsers table below).

## Quick Start

```python
from definable.agent import Agent
from definable.reader import BaseReader

agent = Agent(
  model=model,
  readers=True,  # Uses BaseReader with default parsers
)

# Or with a custom reader:
from definable.reader import BaseReader, ReaderConfig

reader = BaseReader(config=ReaderConfig(max_file_size=10_000_000))
agent = Agent(model=model, readers=reader)
```

## Module Structure

```
readers/
├── __init__.py        # Public API (parsers/providers lazy-loaded)
├── base.py            # BaseReader orchestrator
├── registry.py        # ParserRegistry — priority-based format-to-parser mapping
├── models.py          # ContentBlock, ReaderOutput, ReaderConfig
├── detector.py        # Pure-Python MIME detection (magic bytes + extensions)
├── utils.py           # I/O helpers (bytes fetching, filename extraction)
├── audio.py           # AudioTranscriber stubs
├── parsers/
│   ├── base_parser.py # BaseParser ABC
│   ├── text.py        # TextParser (40+ text/code file types)
│   ├── pdf.py         # PDFParser
│   ├── docx.py        # DocxParser
│   ├── pptx.py        # PptxParser
│   ├── xlsx.py        # XlsxParser
│   ├── ods.py         # OdsParser
│   ├── rtf.py         # RtfParser
│   ├── html.py        # HTMLParser
│   ├── image.py       # ImageParser (passthrough)
│   └── audio.py       # AudioParser (passthrough)
└── providers/
    ├── __init__.py    # ProviderReader Protocol
    ├── mistral.py     # MistralReader (cloud OCR)
    ├── openai.py      # OpenAIReader (stub)
    ├── anthropic.py   # AnthropicReader (stub)
    └── google.py      # GoogleReader (stub)
```

## API Reference

### BaseReader

```python
from definable.reader import BaseReader
```

The main orchestrator: File -> bytes -> detect format -> parse -> ReaderOutput.

```python
reader = BaseReader(
  config=ReaderConfig(max_file_size=None, encoding="utf-8"),
  registry=ParserRegistry(),
)
```

| Method | Description |
|--------|-------------|
| `read(file)` | Parse a single file (sync) |
| `aread(file)` | Parse a single file (async) |
| `aread_all(files)` | Parse multiple files concurrently |
| `register(parser, priority=100)` | Add a custom parser |
| `get_parser(file)` | Get the parser that would handle a file |

### ContentBlock

```python
from definable.reader import ContentBlock
```

A single block of extracted content.

| Field | Type | Description |
|-------|------|-------------|
| `content_type` | `str` | `"text"`, `"image"`, `"table"`, `"audio"`, `"raw"` |
| `content` | `str \| bytes` | Extracted content |
| `mime_type` | `Optional[str]` | MIME type |
| `page_number` | `Optional[int]` | Source page number |
| `metadata` | `Dict` | Additional metadata |

### ReaderOutput

```python
from definable.reader import ReaderOutput
```

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `str` | Source filename |
| `blocks` | `List[ContentBlock]` | Extracted content blocks |
| `mime_type` | `Optional[str]` | Detected MIME type |
| `page_count` | `Optional[int]` | Number of pages |
| `word_count` | `Optional[int]` | Word count |
| `truncated` | `bool` | Whether content was truncated |
| `error` | `Optional[str]` | Error message if parsing failed |

Methods: `as_text(separator="\n\n")`, `as_messages()`.

### ReaderConfig

```python
from definable.reader import ReaderConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_file_size` | `Optional[int]` | `None` | Max file size in bytes |
| `max_content_length` | `Optional[int]` | `None` | Max output content length |
| `encoding` | `str` | `"utf-8"` | Text encoding |
| `timeout` | `Optional[float]` | `30.0` | I/O timeout in seconds |

### ParserRegistry

```python
from definable.reader import ParserRegistry
```

Priority-based format-to-parser mapping. Built-in parsers register at priority 0; user parsers default to priority 100 (higher priority wins).

| Method | Description |
|--------|-------------|
| `register(parser, priority=100)` | Register a parser |
| `get_parser(mime_type=, extension=)` | Find the best parser for a format |

### Parsers

```python
from definable.reader import BaseParser  # ABC
```

All parsers implement `parse(data, *, mime_type=, config=) -> List[ContentBlock]` and are stateless (bytes in, blocks out).

| Parser | Formats | Dependencies |
|--------|---------|--------------|
| `TextParser` | `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.py`, `.js`, `.ts`, `.java`, + 30 more | None |
| `PDFParser` | `.pdf` | `pypdf>=4.0.0` |
| `DocxParser` | `.docx` | `python-docx>=1.0.0` |
| `PptxParser` | `.pptx` | `python-pptx>=1.0.0` |
| `XlsxParser` | `.xlsx` | `openpyxl>=3.1.0` |
| `OdsParser` | `.ods` | `odfpy>=1.4.0` |
| `RtfParser` | `.rtf` | `striprtf>=0.0.26` |
| `HTMLParser` | `.html`, `.htm`, `.xhtml` | None (stdlib) |
| `ImageParser` | `.png`, `.jpg`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.avif`, `.heic`, `.svg` | None (passthrough) |
| `AudioParser` | `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.webm` | None (passthrough) |

### AI Providers

```python
from definable.reader import MistralReader
```

| Provider | Status | Description |
|----------|--------|-------------|
| `MistralReader` | Implemented | Cloud OCR via Mistral API; supports PDFs, DOCX, PPTX, images; local fallback for unsupported formats |
| `OpenAIReader` | Stub | Not yet implemented |
| `AnthropicReader` | Stub | Not yet implemented |
| `GoogleReader` | Stub | Not yet implemented |

### Backwards-Compatible Aliases

| Alias | Target |
|-------|--------|
| `FileReader` | `BaseReader` |
| `FileReaderRegistry` | `BaseReader` |
| `FileReaderConfig` | `ReaderConfig` |
| `ReaderResult` | `ReaderOutput` |

## Usage with Agent

```python
# Simple: use default parsers
agent = Agent(model=model, readers=True)

# Custom parser registry
from definable.reader import BaseReader, ParserRegistry

registry = ParserRegistry()
registry.register(MyCustomParser(), priority=200)
agent = Agent(model=model, readers=BaseReader(registry=registry))

# AI provider reader
from definable.reader import MistralReader

reader = MistralReader(api_key="...")
agent = Agent(model=model, readers=reader)
```

## See Also

- `agents/` — Agent integration via `readers=` parameter
- `knowledge/readers/` — Simpler readers for the RAG pipeline (separate module)
