"""
Unit tests for reader models: ContentBlock, ReaderOutput, ReaderConfig.

Tests pure dataclass behavior: construction, defaults, as_text(), as_messages(),
as_message_content(), content property. No file I/O, no API calls.

Covers:
  - ContentBlock construction with all content types
  - ContentBlock.as_text() for text, image, audio, binary
  - ContentBlock.as_message_content() for text, image, audio, table
  - ReaderOutput construction and defaults
  - ReaderOutput.as_text() joins blocks
  - ReaderOutput.as_messages() returns OpenAI format
  - ReaderOutput.content property (backwards compat)
  - ReaderConfig construction and defaults
"""

import base64

import pytest

from definable.reader.models import ContentBlock, ReaderConfig, ReaderOutput


# ---------------------------------------------------------------------------
# ContentBlock construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContentBlockConstruction:
  """ContentBlock dataclass construction and defaults."""

  def test_text_block(self):
    block = ContentBlock(content_type="text", content="Hello world")
    assert block.content_type == "text"
    assert block.content == "Hello world"

  def test_image_block_with_bytes(self):
    data = b"\x89PNG\r\n"
    block = ContentBlock(content_type="image", content=data, mime_type="image/png")
    assert block.content_type == "image"
    assert block.mime_type == "image/png"
    assert isinstance(block.content, bytes)

  def test_audio_block(self):
    block = ContentBlock(content_type="audio", content=b"\xff\xfb", mime_type="audio/mpeg")
    assert block.content_type == "audio"

  def test_table_block(self):
    block = ContentBlock(content_type="table", content="| a | b |\n| 1 | 2 |")
    assert block.content_type == "table"

  def test_raw_block(self):
    block = ContentBlock(content_type="raw", content=b"\x00\x01\x02")
    assert block.content_type == "raw"

  def test_defaults_metadata_empty_dict(self):
    block = ContentBlock(content_type="text", content="hi")
    assert block.metadata == {}

  def test_defaults_page_number_none(self):
    block = ContentBlock(content_type="text", content="hi")
    assert block.page_number is None

  def test_defaults_mime_type_none(self):
    block = ContentBlock(content_type="text", content="hi")
    assert block.mime_type is None

  def test_custom_metadata(self):
    block = ContentBlock(content_type="text", content="hi", metadata={"source": "test"})
    assert block.metadata["source"] == "test"

  def test_page_number_set(self):
    block = ContentBlock(content_type="text", content="page 3", page_number=3)
    assert block.page_number == 3

  def test_metadata_independent_per_instance(self):
    b1 = ContentBlock(content_type="text", content="a")
    b2 = ContentBlock(content_type="text", content="b")
    b1.metadata["key"] = "val"
    assert "key" not in b2.metadata


# ---------------------------------------------------------------------------
# ContentBlock.as_text()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContentBlockAsText:
  """ContentBlock.as_text() returns appropriate string representation."""

  def test_text_content_returns_string(self):
    block = ContentBlock(content_type="text", content="Hello")
    assert block.as_text() == "Hello"

  def test_table_content_returns_string(self):
    block = ContentBlock(content_type="table", content="| a | b |")
    assert block.as_text() == "| a | b |"

  def test_image_bytes_returns_placeholder(self):
    block = ContentBlock(content_type="image", content=b"\x89PNG", mime_type="image/png")
    assert block.as_text() == "[image: image/png]"

  def test_image_no_mime_returns_unknown(self):
    block = ContentBlock(content_type="image", content=b"\x89PNG")
    assert block.as_text() == "[image: unknown]"

  def test_audio_bytes_returns_placeholder(self):
    block = ContentBlock(content_type="audio", content=b"\xff\xfb", mime_type="audio/mpeg")
    assert block.as_text() == "[audio: audio/mpeg]"

  def test_audio_no_mime_returns_unknown(self):
    block = ContentBlock(content_type="audio", content=b"\xff\xfb")
    assert block.as_text() == "[audio: unknown]"

  def test_raw_bytes_returns_size(self):
    block = ContentBlock(content_type="raw", content=b"\x00\x01\x02")
    assert block.as_text() == "[binary: 3 bytes]"


# ---------------------------------------------------------------------------
# ContentBlock.as_message_content()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContentBlockAsMessageContent:
  """ContentBlock.as_message_content() returns OpenAI-format content parts."""

  def test_text_block_returns_text_type(self):
    block = ContentBlock(content_type="text", content="Hello")
    msg = block.as_message_content()
    assert msg["type"] == "text"
    assert msg["text"] == "Hello"

  def test_table_block_returns_text_type(self):
    block = ContentBlock(content_type="table", content="| a | b |")
    msg = block.as_message_content()
    assert msg["type"] == "text"
    assert msg["text"] == "| a | b |"

  def test_image_block_returns_image_url_type(self):
    data = b"\x89PNG"
    block = ContentBlock(content_type="image", content=data, mime_type="image/png")
    msg = block.as_message_content()
    assert msg["type"] == "image_url"
    assert "image_url" in msg
    b64 = base64.b64encode(data).decode("ascii")
    assert msg["image_url"]["url"] == f"data:image/png;base64,{b64}"

  def test_image_block_defaults_to_png_mime(self):
    data = b"\x89PNG"
    block = ContentBlock(content_type="image", content=data)
    msg = block.as_message_content()
    assert "data:image/png;base64," in msg["image_url"]["url"]

  def test_audio_block_returns_input_audio_type(self):
    data = b"\xff\xfb"
    block = ContentBlock(content_type="audio", content=data, mime_type="audio/wav")
    msg = block.as_message_content()
    assert msg["type"] == "input_audio"
    b64 = base64.b64encode(data).decode("ascii")
    assert msg["input_audio"]["data"] == b64
    assert msg["input_audio"]["format"] == "audio/wav"

  def test_audio_block_defaults_to_mpeg_mime(self):
    data = b"\xff\xfb"
    block = ContentBlock(content_type="audio", content=data)
    msg = block.as_message_content()
    assert msg["input_audio"]["format"] == "audio/mpeg"

  def test_raw_block_fallback_to_text(self):
    block = ContentBlock(content_type="raw", content=b"\x00\x01")
    msg = block.as_message_content()
    assert msg["type"] == "text"
    assert "binary" in msg["text"]

  def test_image_string_content_fallback_to_text(self):
    block = ContentBlock(content_type="image", content="not bytes")
    msg = block.as_message_content()
    assert msg["type"] == "text"


# ---------------------------------------------------------------------------
# ReaderOutput construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReaderOutputConstruction:
  """ReaderOutput dataclass construction and defaults."""

  def test_minimal_construction(self):
    out = ReaderOutput(filename="test.txt")
    assert out.filename == "test.txt"

  def test_defaults_blocks_empty(self):
    out = ReaderOutput(filename="test.txt")
    assert out.blocks == []

  def test_defaults_mime_type_none(self):
    out = ReaderOutput(filename="test.txt")
    assert out.mime_type is None

  def test_defaults_page_count_none(self):
    out = ReaderOutput(filename="test.txt")
    assert out.page_count is None

  def test_defaults_word_count_none(self):
    out = ReaderOutput(filename="test.txt")
    assert out.word_count is None

  def test_defaults_truncated_false(self):
    out = ReaderOutput(filename="test.txt")
    assert out.truncated is False

  def test_defaults_error_none(self):
    out = ReaderOutput(filename="test.txt")
    assert out.error is None

  def test_defaults_metadata_empty(self):
    out = ReaderOutput(filename="test.txt")
    assert out.metadata == {}

  def test_metadata_independent_per_instance(self):
    o1 = ReaderOutput(filename="a.txt")
    o2 = ReaderOutput(filename="b.txt")
    o1.metadata["key"] = "val"
    assert "key" not in o2.metadata


# ---------------------------------------------------------------------------
# ReaderOutput.as_text()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReaderOutputAsText:
  """ReaderOutput.as_text() joins blocks with separator."""

  def test_empty_blocks_returns_empty_string(self):
    out = ReaderOutput(filename="test.txt")
    assert out.as_text() == ""

  def test_single_block(self):
    block = ContentBlock(content_type="text", content="Hello")
    out = ReaderOutput(filename="test.txt", blocks=[block])
    assert out.as_text() == "Hello"

  def test_multiple_blocks_default_separator(self):
    b1 = ContentBlock(content_type="text", content="Hello")
    b2 = ContentBlock(content_type="text", content="World")
    out = ReaderOutput(filename="test.txt", blocks=[b1, b2])
    assert out.as_text() == "Hello\n\nWorld"

  def test_custom_separator(self):
    b1 = ContentBlock(content_type="text", content="A")
    b2 = ContentBlock(content_type="text", content="B")
    out = ReaderOutput(filename="test.txt", blocks=[b1, b2])
    assert out.as_text(separator=" | ") == "A | B"


# ---------------------------------------------------------------------------
# ReaderOutput.as_messages()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReaderOutputAsMessages:
  """ReaderOutput.as_messages() returns OpenAI-format content parts."""

  def test_empty_blocks_returns_empty_list(self):
    out = ReaderOutput(filename="test.txt")
    assert out.as_messages() == []

  def test_text_blocks_as_messages(self):
    b1 = ContentBlock(content_type="text", content="Hello")
    out = ReaderOutput(filename="test.txt", blocks=[b1])
    msgs = out.as_messages()
    assert len(msgs) == 1
    assert msgs[0]["type"] == "text"
    assert msgs[0]["text"] == "Hello"

  def test_mixed_blocks_as_messages(self):
    b1 = ContentBlock(content_type="text", content="Hello")
    b2 = ContentBlock(content_type="image", content=b"\x89PNG", mime_type="image/png")
    out = ReaderOutput(filename="test.txt", blocks=[b1, b2])
    msgs = out.as_messages()
    assert len(msgs) == 2
    assert msgs[0]["type"] == "text"
    assert msgs[1]["type"] == "image_url"


# ---------------------------------------------------------------------------
# ReaderOutput.content property
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReaderOutputContentProperty:
  """ReaderOutput.content backwards-compatible property."""

  def test_returns_as_text(self):
    b1 = ContentBlock(content_type="text", content="Hello")
    out = ReaderOutput(filename="test.txt", blocks=[b1])
    assert out.content == "Hello"

  def test_empty_blocks_returns_empty_string(self):
    out = ReaderOutput(filename="test.txt")
    assert out.content == ""


# ---------------------------------------------------------------------------
# ReaderConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReaderConfig:
  """ReaderConfig dataclass defaults and custom values."""

  def test_defaults_max_file_size_none(self):
    cfg = ReaderConfig()
    assert cfg.max_file_size is None

  def test_defaults_max_content_length_none(self):
    cfg = ReaderConfig()
    assert cfg.max_content_length is None

  def test_defaults_encoding_utf8(self):
    cfg = ReaderConfig()
    assert cfg.encoding == "utf-8"

  def test_defaults_timeout_30(self):
    cfg = ReaderConfig()
    assert cfg.timeout == 30.0

  def test_custom_values(self):
    cfg = ReaderConfig(max_file_size=1024, max_content_length=500, encoding="latin-1", timeout=60.0)
    assert cfg.max_file_size == 1024
    assert cfg.max_content_length == 500
    assert cfg.encoding == "latin-1"
    assert cfg.timeout == 60.0
