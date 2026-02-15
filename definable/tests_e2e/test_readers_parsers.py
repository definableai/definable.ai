"""E2E tests — File Parsing (No API Key).

Scenario: "I want to extract text from various file formats."

All tests run locally with no API keys. Optional-dep parsers use
pytest.importorskip().
"""

import pytest

from definable.media import File
from definable.readers.base import BaseReader
from definable.readers.models import ContentBlock, ReaderConfig, ReaderOutput
from definable.readers.parsers.text import TextParser


# ---------------------------------------------------------------------------
# Text-based format tests
# ---------------------------------------------------------------------------


class TestTextFormats:
  """TextParser handles plain text, CSV, JSON, Python, YAML, etc."""

  def test_plain_text_parsing(self):
    """TextParser extracts plain text content."""
    parser = TextParser()
    blocks = parser.parse(b"Hello, world!", mime_type="text/plain")
    assert len(blocks) == 1
    assert blocks[0].content_type == "text"
    assert blocks[0].content == "Hello, world!"

  def test_csv_parsing(self):
    """TextParser preserves CSV structure."""
    parser = TextParser()
    csv_data = b"name,age,city\nAlice,30,NYC\nBob,25,LA"
    blocks = parser.parse(csv_data, mime_type="text/csv")
    assert len(blocks) == 1
    assert "Alice" in blocks[0].content
    assert "Bob" in blocks[0].content

  def test_json_parsing(self):
    """TextParser preserves JSON content as text."""
    parser = TextParser()
    json_data = b'{"key": "value", "number": 42}'
    blocks = parser.parse(json_data, mime_type="application/json")
    assert len(blocks) == 1
    assert '"key": "value"' in blocks[0].content

  def test_html_parsing(self):
    """HTMLParser strips tags and extracts visible text."""
    from definable.readers.parsers.html import HTMLParser

    parser = HTMLParser()
    html = b"""<html>
    <head><title>Test</title><style>body{color:red}</style></head>
    <body>
      <h1>Hello</h1>
      <p>World</p>
      <script>alert('hidden')</script>
    </body>
    </html>"""
    blocks = parser.parse(html, mime_type="text/html")
    assert len(blocks) >= 1
    text = blocks[0].content
    assert "Hello" in text
    assert "World" in text
    # Script and style content should be stripped
    assert "alert" not in text
    assert "color:red" not in text

  def test_rtf_parsing(self):
    """RtfParser strips RTF markup to plain text."""
    pytest.importorskip("striprtf")
    from definable.readers.parsers.rtf import RtfParser

    parser = RtfParser()
    rtf_data = rb"{\rtf1\ansi Hello, RTF world!}"
    blocks = parser.parse(rtf_data, mime_type="application/rtf")
    assert len(blocks) >= 1
    assert "Hello" in blocks[0].content

  def test_python_code_parsing(self):
    """TextParser preserves Python source code."""
    parser = TextParser()
    code = b'def hello():\n    return "Hello, world!"'
    blocks = parser.parse(code, mime_type="text/x-python")
    assert len(blocks) == 1
    assert "def hello" in blocks[0].content

  def test_yaml_parsing(self):
    """TextParser preserves YAML content."""
    parser = TextParser()
    yaml_data = b"name: test\nversion: 1.0\nitems:\n  - alpha\n  - beta\n"
    blocks = parser.parse(yaml_data, mime_type="text/yaml")
    assert len(blocks) == 1
    assert "name: test" in blocks[0].content

  def test_handles_encoding_errors(self):
    """TextParser handles invalid UTF-8 bytes gracefully."""
    parser = TextParser()
    blocks = parser.parse(b"\xff\xfe\x00\x01", mime_type="text/plain")
    assert len(blocks) == 1
    assert len(blocks[0].content) > 0

  def test_supported_extensions(self):
    """TextParser reports expected extensions."""
    parser = TextParser()
    exts = parser.supported_extensions()
    assert ".txt" in exts
    assert ".md" in exts
    assert ".py" in exts
    assert ".json" in exts
    assert ".csv" in exts


# ---------------------------------------------------------------------------
# Spreadsheet format tests
# ---------------------------------------------------------------------------


class TestSpreadsheetFormats:
  """XlsxParser and OdsParser tests (require optional deps)."""

  def test_xlsx_single_sheet(self):
    """XlsxParser extracts a single sheet as table ContentBlock."""
    openpyxl = pytest.importorskip("openpyxl")
    from definable.readers.parsers.xlsx import XlsxParser

    import io

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Product", "Units", "Revenue"])
    ws.append(["Widget A", 150, 4500])
    ws.append(["Widget B", 200, 8000])
    buf = io.BytesIO()
    wb.save(buf)
    xlsx_bytes = buf.getvalue()

    parser = XlsxParser()
    blocks = parser.parse(xlsx_bytes, mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    assert len(blocks) >= 1
    assert blocks[0].content_type == "table"
    assert "Widget A" in blocks[0].content

  def test_xlsx_multiple_sheets(self):
    """XlsxParser extracts multiple sheets as separate ContentBlocks."""
    openpyxl = pytest.importorskip("openpyxl")
    from definable.readers.parsers.xlsx import XlsxParser

    import io

    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1.append(["A", "B"])
    ws1.append([1, 2])

    ws2 = wb.create_sheet("Sheet2")
    ws2.append(["X", "Y"])
    ws2.append([10, 20])

    buf = io.BytesIO()
    wb.save(buf)

    parser = XlsxParser()
    blocks = parser.parse(buf.getvalue(), mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    assert len(blocks) == 2
    # Each block should have sheet name in metadata
    sheet_names = {b.metadata.get("sheet_name") for b in blocks}
    assert "Sheet1" in sheet_names
    assert "Sheet2" in sheet_names

  def test_xlsx_max_rows_limit(self):
    """XlsxParser(max_rows=5) limits extracted rows."""
    openpyxl = pytest.importorskip("openpyxl")
    from definable.readers.parsers.xlsx import XlsxParser

    import io

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["ID", "Value"])
    for i in range(100):
      ws.append([i, f"row_{i}"])
    buf = io.BytesIO()
    wb.save(buf)

    parser = XlsxParser(max_rows=5)
    blocks = parser.parse(buf.getvalue(), mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    assert len(blocks) >= 1
    # Count data rows in content (tab-separated lines)
    lines = blocks[0].content.strip().split("\n")
    assert len(lines) <= 6  # header + 5 data rows

  def test_ods_single_sheet(self):
    """OdsParser extracts an ODS sheet."""
    pytest.importorskip("odf")
    from definable.readers.parsers.ods import OdsParser

    from odf.opendocument import OpenDocumentSpreadsheet
    from odf.table import Table, TableCell, TableRow
    from odf.text import P
    import io

    doc = OpenDocumentSpreadsheet()
    table = Table(name="Data")
    for row_data in [["Name", "Score"], ["Alice", "95"], ["Bob", "87"]]:
      tr = TableRow()
      for cell_val in row_data:
        tc = TableCell()
        tc.addElement(P(text=str(cell_val)))
        tr.addElement(tc)
      table.addElement(tr)
    doc.spreadsheet.addElement(table)

    buf = io.BytesIO()
    doc.save(buf)

    parser = OdsParser()
    blocks = parser.parse(buf.getvalue(), mime_type="application/vnd.oasis.opendocument.spreadsheet")
    assert len(blocks) >= 1
    assert "Alice" in blocks[0].content


# ---------------------------------------------------------------------------
# Binary format tests (image, audio passthrough)
# ---------------------------------------------------------------------------


class TestBinaryFormats:
  """ImageParser and AudioParser passthrough raw bytes."""

  def test_image_passthrough(self, sample_png_bytes):
    """ImageParser wraps PNG bytes in an image ContentBlock."""
    from definable.readers.parsers.image import ImageParser

    parser = ImageParser()
    blocks = parser.parse(sample_png_bytes, mime_type="image/png")
    assert len(blocks) == 1
    assert blocks[0].content_type == "image"
    assert blocks[0].content == sample_png_bytes

  def test_audio_passthrough(self, sample_wav_bytes):
    """AudioParser wraps WAV bytes in an audio ContentBlock."""
    from definable.readers.parsers.audio import AudioParser

    parser = AudioParser()
    blocks = parser.parse(sample_wav_bytes, mime_type="audio/wav")
    assert len(blocks) == 1
    assert blocks[0].content_type == "audio"
    assert blocks[0].content == sample_wav_bytes

  def test_image_as_text_returns_placeholder(self, sample_png_bytes):
    """Image ContentBlock.as_text() returns '[image: image/png]'."""
    block = ContentBlock(content_type="image", content=sample_png_bytes, mime_type="image/png")
    assert block.as_text() == "[image: image/png]"

  def test_audio_as_text_returns_placeholder(self, sample_wav_bytes):
    """Audio ContentBlock.as_text() returns '[audio: audio/wav]'."""
    block = ContentBlock(content_type="audio", content=sample_wav_bytes, mime_type="audio/wav")
    assert block.as_text() == "[audio: audio/wav]"

  def test_image_as_message_content(self, sample_png_bytes):
    """Image block produces OpenAI-format image_url message part."""
    block = ContentBlock(content_type="image", content=sample_png_bytes, mime_type="image/png")
    msg = block.as_message_content()
    assert msg["type"] == "image_url"
    assert msg["image_url"]["url"].startswith("data:image/png;base64,")

  def test_audio_as_message_content(self, sample_wav_bytes):
    """Audio block produces OpenAI-format input_audio message part."""
    block = ContentBlock(content_type="audio", content=sample_wav_bytes, mime_type="audio/wav")
    msg = block.as_message_content()
    assert msg["type"] == "input_audio"
    assert "data" in msg["input_audio"]


# ---------------------------------------------------------------------------
# BaseReader orchestration tests
# ---------------------------------------------------------------------------


class TestBaseReaderOrchestration:
  """BaseReader auto-detection, config enforcement, and concurrent reads."""

  def test_reader_auto_detects_format(self):
    """BaseReader selects correct parser based on filename extension."""
    reader = BaseReader()
    file = File(content=b"name,age\nAlice,30", filename="data.csv")
    result = reader.read(file)
    assert result.error is None
    assert "Alice" in result.content

  def test_reader_with_explicit_mime_type(self):
    """File with explicit mime_type overrides extension-based detection."""
    reader = BaseReader()
    file = File(content=b"hello world", filename="weird.xyz", mime_type="text/plain")
    result = reader.read(file)
    assert result.error is None
    assert result.content == "hello world"

  def test_reader_unknown_format_returns_error(self):
    """Unrecognized binary file returns ReaderOutput with error."""
    reader = BaseReader()
    file = File(content=b"\x00\x01\x02\x03", filename="data.bin")
    result = reader.read(file)
    assert result.error is not None

  def test_reader_max_file_size_enforced(self):
    """ReaderConfig.max_file_size rejects oversized files."""
    config = ReaderConfig(max_file_size=10)
    reader = BaseReader(config=config)
    file = File(content=b"x" * 100, filename="big.txt", mime_type="text/plain")
    result = reader.read(file)
    assert result.error is not None
    assert "max_file_size" in result.error

  def test_reader_content_truncation(self):
    """ReaderConfig.max_content_length truncates long content."""
    config = ReaderConfig(max_content_length=20)
    reader = BaseReader(config=config)
    file = File(content=b"x" * 200, filename="long.txt", mime_type="text/plain")
    result = reader.read(file)
    assert result.truncated is True
    assert len(result.content) == 20

  @pytest.mark.asyncio
  async def test_reader_concurrent_aread_all(self):
    """BaseReader.aread_all reads multiple files concurrently."""
    reader = BaseReader()
    files = [File(content=f"Content {i}".encode(), filename=f"file{i}.txt", mime_type="text/plain") for i in range(5)]
    results = await reader.aread_all(files)
    assert len(results) == 5
    for i, result in enumerate(results):
      assert result.error is None
      assert f"Content {i}" in result.content

  def test_reader_output_metadata(self):
    """ReaderOutput has word_count and mime_type populated."""
    reader = BaseReader()
    file = File(
      content=b"Hello world this is a test",
      filename="test.txt",
      mime_type="text/plain",
    )
    result = reader.read(file)
    assert result.word_count is not None
    assert result.word_count > 0
    assert result.filename == "test.txt"


# ---------------------------------------------------------------------------
# ContentBlock and ReaderOutput conversion tests
# ---------------------------------------------------------------------------


class TestContentBlockConversions:
  """ContentBlock and ReaderOutput helper methods."""

  def test_text_block_as_text(self):
    block = ContentBlock(content_type="text", content="Hello world")
    assert block.as_text() == "Hello world"

  def test_text_block_as_message_content(self):
    block = ContentBlock(content_type="text", content="Hello")
    msg = block.as_message_content()
    assert msg == {"type": "text", "text": "Hello"}

  def test_table_block_as_message_content(self):
    block = ContentBlock(content_type="table", content="a\tb\n1\t2")
    msg = block.as_message_content()
    assert msg["type"] == "text"
    assert "a\tb" in msg["text"]

  def test_reader_output_as_text_joins_blocks(self):
    output = ReaderOutput(
      filename="test.txt",
      blocks=[
        ContentBlock(content_type="text", content="First"),
        ContentBlock(content_type="text", content="Second"),
      ],
    )
    assert output.as_text() == "First\n\nSecond"
    assert output.as_text(separator=" | ") == "First | Second"

  def test_reader_output_as_messages_returns_list(self):
    output = ReaderOutput(
      filename="test.txt",
      blocks=[
        ContentBlock(content_type="text", content="Hello"),
        ContentBlock(content_type="text", content="World"),
      ],
    )
    msgs = output.as_messages()
    assert len(msgs) == 2
    assert msgs[0]["type"] == "text"

  def test_reader_output_content_property_backwards_compat(self):
    """ReaderOutput.content property is backwards-compatible alias for as_text()."""
    output = ReaderOutput(
      filename="test.txt",
      blocks=[ContentBlock(content_type="text", content="Hello")],
    )
    assert output.content == "Hello"

  def test_reader_output_empty_blocks(self):
    output = ReaderOutput(filename="empty.txt")
    assert output.content == ""
    assert output.as_messages() == []

  def test_reader_output_error(self):
    output = ReaderOutput(filename="bad.pdf", error="Failed to parse")
    assert output.error == "Failed to parse"
    assert output.content == ""

  def test_raw_block_as_text(self):
    block = ContentBlock(content_type="raw", content=b"\x00\x01\x02")
    assert "[binary: 3 bytes]" in block.as_text()

  def test_block_page_number(self):
    block = ContentBlock(content_type="text", content="Page 1", page_number=1)
    assert block.page_number == 1

  def test_block_metadata(self):
    block = ContentBlock(content_type="text", content="test", metadata={"key": "val"})
    assert block.metadata["key"] == "val"
