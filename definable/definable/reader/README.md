# readers

File content extraction — parse files into structured, multimodal content blocks. Also provides the audio transcription protocol used by the agent pipeline.

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
├── audio.py           # AudioTranscriber protocol, OpenAITranscriber, normalize_audio_format
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
reader = BaseReader()  # no args needed
reader = BaseReader(config=ReaderConfig(max_file_size=10_000_000))
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

```python
block = ContentBlock(content_type="text", content="Hello world")
```

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

```python
output = ReaderOutput(filename="test.txt", blocks=[block])
output.as_text()  # "Hello world"
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

## Audio Transcription

The `reader/audio.py` module provides the `AudioTranscriber` protocol, the `OpenAITranscriber` implementation (Whisper), and `normalize_audio_format` for converting audio to formats accepted by model APIs.

```python
from definable.reader import (
  AudioTranscriber,
  OpenAITranscriber,
  normalize_audio_format,
  OPENAI_INPUT_AUDIO_FORMATS,
)
```

### AudioTranscriber Protocol

A `@runtime_checkable` protocol any transcriber must satisfy:

```python
@runtime_checkable
class AudioTranscriber(Protocol):
  def transcribe(self, audio_bytes: bytes, mime_type: str, **kwargs) -> str: ...
  async def atranscribe(self, audio_bytes: bytes, mime_type: str, **kwargs) -> str: ...
```

Both sync and async paths are required. Pass raw audio bytes and a MIME type string (e.g. `"audio/wav"`, `"audio/mpeg"`). Returns the transcript as a plain string.

### OpenAITranscriber

Wraps OpenAI's Whisper API (`whisper-1`).

```python
from definable.reader import OpenAITranscriber

# Defaults — model "whisper-1", language auto-detected
transcriber = OpenAITranscriber()

# Explicit language hint (ISO 639-1 code — improves accuracy and latency)
transcriber = OpenAITranscriber(model="whisper-1", language="en")

# Custom API key (defaults to OPENAI_API_KEY env var)
transcriber = OpenAITranscriber(api_key="sk-...")
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"whisper-1"` | Whisper model to use |
| `language` | `Optional[str]` | `None` | ISO 639-1 language code; omit for auto-detect |
| `api_key` | `Optional[str]` | `None` | Falls back to `OPENAI_API_KEY` env var |

```python
text = transcriber.transcribe(audio_bytes, "audio/wav")
text = await transcriber.atranscribe(audio_bytes, "audio/wav")
```

### normalize_audio_format

Converts audio bytes to a format accepted by model APIs. By default targets `OPENAI_INPUT_AUDIO_FORMATS = {"wav", "mp3"}`.

```python
from definable.reader import normalize_audio_format, OPENAI_INPUT_AUDIO_FORMATS

# OPENAI_INPUT_AUDIO_FORMATS = {"wav", "mp3"}

normalized_bytes, output_format = normalize_audio_format(
  audio_bytes,
  source_format,             # e.g. "ogg", "oga", "flac", "m4a"
  target_formats=None,       # defaults to OPENAI_INPUT_AUDIO_FORMATS
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_bytes` | `bytes` | Raw audio data |
| `source_format` | `str` | Source format string (e.g. `"ogg"`, `"oga"`, `"mp3"`) |
| `target_formats` | `Optional[set[str]]` | Accepted output formats; defaults to `OPENAI_INPUT_AUDIO_FORMATS` |

Returns a `Tuple[bytes, str]` — the (possibly re-encoded) audio bytes and the output format string.

- If `source_format` is already in `target_formats`, the input bytes are returned unchanged.
- Otherwise, `ffmpeg` is invoked to transcode. Raises `RuntimeError` if `ffmpeg` is not available.

### Format Handling — Telegram OGA / OGG

Telegram voice notes arrive as `.oga` files (Opus codec in an OGG container). OpenAI's `input_audio` API only accepts `wav` and `mp3`. Use `normalize_audio_format` or set `audio_transcriber=True` on the agent to handle this automatically.

```python
# Manual conversion
from definable.reader import normalize_audio_format

wav_bytes, fmt = normalize_audio_format(oga_bytes, "oga")
# fmt == "wav"

# Or let the agent handle it transparently
from definable.agent import Agent

agent = Agent(model=model, audio_transcriber=True)
```

When `audio_transcriber=True` (or a transcriber instance) is set on an agent, voice messages are transcribed before the model is called, and `msg.audio` is set to `None` afterward so non-audio models never receive raw `input_audio` blocks.

### Agent Integration

```python
from definable.agent import Agent
from definable.reader import OpenAITranscriber

# Shorthand — creates OpenAITranscriber(model="whisper-1") automatically
agent = Agent(model=model, audio_transcriber=True)

# Custom transcriber — e.g. force English, reduce latency
agent = Agent(
  model=model,
  audio_transcriber=OpenAITranscriber(model="whisper-1", language="en"),
)
```

Any object satisfying the `AudioTranscriber` protocol can be passed — implement your own to use Deepgram, AssemblyAI, or a local Whisper model.

```python
class MyTranscriber:
  def transcribe(self, audio_bytes: bytes, mime_type: str, **kwargs) -> str:
    ...

  async def atranscribe(self, audio_bytes: bytes, mime_type: str, **kwargs) -> str:
    ...

agent = Agent(model=model, audio_transcriber=MyTranscriber())
```

### Gotchas

| Trap | Truth |
|------|-------|
| Telegram sends `.oga` | Opus in OGG container. OpenAI Whisper only accepts `wav`/`mp3`. Use `normalize_audio_format()` or `audio_transcriber=True`. |
| `normalize_audio_format` requires ffmpeg | Raises `RuntimeError` if transcoding is needed and `ffmpeg` is not on `PATH`. Passthrough (already-compatible format) works without ffmpeg. |
| `audio_transcriber=True` clears `msg.audio` | After transcription the raw audio bytes are removed from the message so non-audio models never receive `input_audio` blocks. |
| `OPENAI_INPUT_AUDIO_FORMATS` is `{"wav", "mp3"}` | Not `ogg`, `flac`, or `m4a`. Always normalize before passing audio bytes to OpenAI's Whisper or `input_audio` API. |

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

# Reader + audio transcription together
from definable.reader import OpenAITranscriber

agent = Agent(
  model=model,
  readers=True,
  audio_transcriber=OpenAITranscriber(language="en"),
)
```

## See Also

- `agents/` — Agent integration via `readers=` and `audio_transcriber=` parameters
- `knowledge/readers/` — Simpler readers for the RAG pipeline (separate module)
