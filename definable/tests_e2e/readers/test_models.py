"""Tests for ContentBlock, ReaderOutput, and ReaderConfig."""

from definable.readers.models import ContentBlock, ReaderConfig, ReaderOutput


class TestContentBlock:
  def test_text_block(self):
    block = ContentBlock(content_type="text", content="Hello world")
    assert block.as_text() == "Hello world"

  def test_image_block_as_text(self):
    block = ContentBlock(content_type="image", content=b"\x89PNG", mime_type="image/png")
    assert block.as_text() == "[image: image/png]"

  def test_audio_block_as_text(self):
    block = ContentBlock(content_type="audio", content=b"\xff\xfb", mime_type="audio/mpeg")
    assert block.as_text() == "[audio: audio/mpeg]"

  def test_raw_block_as_text(self):
    block = ContentBlock(content_type="raw", content=b"\x00\x01\x02")
    assert "[binary: 3 bytes]" in block.as_text()

  def test_text_as_message_content(self):
    block = ContentBlock(content_type="text", content="Hello")
    msg = block.as_message_content()
    assert msg == {"type": "text", "text": "Hello"}

  def test_table_as_message_content(self):
    block = ContentBlock(content_type="table", content="a\tb\n1\t2")
    msg = block.as_message_content()
    assert msg["type"] == "text"
    assert "a\tb" in msg["text"]

  def test_image_as_message_content(self):
    block = ContentBlock(content_type="image", content=b"\x89PNG", mime_type="image/png")
    msg = block.as_message_content()
    assert msg["type"] == "image_url"
    assert msg["image_url"]["url"].startswith("data:image/png;base64,")

  def test_audio_as_message_content(self):
    block = ContentBlock(content_type="audio", content=b"\xff\xfb", mime_type="audio/mpeg")
    msg = block.as_message_content()
    assert msg["type"] == "input_audio"

  def test_page_number(self):
    block = ContentBlock(content_type="text", content="Page 1", page_number=1)
    assert block.page_number == 1

  def test_metadata(self):
    block = ContentBlock(content_type="text", content="test", metadata={"key": "value"})
    assert block.metadata["key"] == "value"


class TestReaderOutput:
  def test_basic_output(self):
    output = ReaderOutput(
      filename="test.txt",
      blocks=[ContentBlock(content_type="text", content="Hello")],
    )
    assert output.filename == "test.txt"
    assert output.content == "Hello"
    assert output.error is None

  def test_as_text_joins_blocks(self):
    output = ReaderOutput(
      filename="test.txt",
      blocks=[
        ContentBlock(content_type="text", content="First"),
        ContentBlock(content_type="text", content="Second"),
      ],
    )
    assert output.as_text() == "First\n\nSecond"
    assert output.as_text(separator=" | ") == "First | Second"

  def test_as_messages(self):
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

  def test_content_property(self):
    output = ReaderOutput(
      filename="test.txt",
      blocks=[ContentBlock(content_type="text", content="Hello")],
    )
    # Backwards-compatible .content property
    assert output.content == "Hello"

  def test_error_output(self):
    output = ReaderOutput(filename="bad.pdf", error="Failed to read")
    assert output.error == "Failed to read"
    assert output.content == ""

  def test_metadata(self):
    output = ReaderOutput(
      filename="doc.pdf",
      blocks=[ContentBlock(content_type="text", content="text")],
      page_count=5,
      word_count=100,
      metadata={"author": "Test"},
    )
    assert output.page_count == 5
    assert output.word_count == 100
    assert output.metadata["author"] == "Test"

  def test_empty_blocks(self):
    output = ReaderOutput(filename="empty.txt")
    assert output.content == ""
    assert output.as_messages() == []


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
