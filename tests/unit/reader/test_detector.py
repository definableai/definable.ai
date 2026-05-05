"""Unit tests for the reader format detector module.

Covers magic byte detection, extension detection, combined detection,
and ZIP/RIFF subtype handling.
"""

import pytest

from definable.reader.detector import (
  detect,
  detect_from_bytes,
  detect_from_extension,
  extract_extension,
)


# ===========================================================================
# extract_extension
# ===========================================================================


@pytest.mark.unit
class TestExtractExtension:
  """Tests for extract_extension()."""

  def test_from_filename(self):
    assert extract_extension(filename="report.pdf") == ".pdf"

  def test_from_filename_uppercase(self):
    assert extract_extension(filename="PHOTO.JPG") == ".jpg"

  def test_from_filepath(self):
    assert extract_extension(filepath="/tmp/data.csv") == ".csv"

  def test_from_url(self):
    assert extract_extension(url="https://example.com/doc.docx") == ".docx"

  def test_url_with_query_params(self):
    # os.path.splitext + urlparse.path should handle this
    assert extract_extension(url="https://example.com/file.txt?v=1") == ".txt"

  def test_no_extension(self):
    assert extract_extension(filename="Makefile") is None

  def test_all_none(self):
    assert extract_extension() is None

  def test_filename_takes_priority(self):
    assert extract_extension(filename="a.py", filepath="/tmp/b.js") == ".py"

  def test_filepath_fallback(self):
    assert extract_extension(filepath="/tmp/b.go") == ".go"


# ===========================================================================
# detect_from_extension
# ===========================================================================


@pytest.mark.unit
class TestDetectFromExtension:
  """Tests for detect_from_extension()."""

  def test_pdf(self):
    assert detect_from_extension(filename="file.pdf") == "application/pdf"

  def test_docx(self):
    result = detect_from_extension(filename="doc.docx")
    assert result is not None
    assert "wordprocessingml" in result

  def test_xlsx(self):
    result = detect_from_extension(filename="sheet.xlsx")
    assert result is not None
    assert "spreadsheetml" in result

  def test_pptx(self):
    result = detect_from_extension(filename="slides.pptx")
    assert result is not None
    assert "presentationml" in result

  def test_png(self):
    assert detect_from_extension(filename="img.png") == "image/png"

  def test_jpeg(self):
    assert detect_from_extension(filename="photo.jpg") == "image/jpeg"

  def test_mp3(self):
    assert detect_from_extension(filename="song.mp3") == "audio/mpeg"

  def test_python_file(self):
    assert detect_from_extension(filename="script.py") == "text/x-python"

  def test_markdown(self):
    assert detect_from_extension(filename="README.md") == "text/markdown"

  def test_json(self):
    assert detect_from_extension(filename="config.json") == "application/json"

  def test_unknown_extension(self):
    assert detect_from_extension(filename="data.xyz123") is None

  def test_no_inputs(self):
    assert detect_from_extension() is None


# ===========================================================================
# detect_from_bytes
# ===========================================================================


@pytest.mark.unit
class TestDetectFromBytes:
  """Tests for detect_from_bytes()."""

  def test_pdf_magic(self):
    assert detect_from_bytes(b"%PDF-1.5 ...") == "application/pdf"

  def test_png_magic(self):
    assert detect_from_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50) == "image/png"

  def test_jpeg_magic(self):
    assert detect_from_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50) == "image/jpeg"

  def test_gif87a_magic(self):
    assert detect_from_bytes(b"GIF87a" + b"\x00" * 50) == "image/gif"

  def test_gif89a_magic(self):
    assert detect_from_bytes(b"GIF89a" + b"\x00" * 50) == "image/gif"

  def test_rtf_magic(self):
    assert detect_from_bytes(b"{\\rtf1" + b"\x00" * 50) == "text/rtf"

  def test_mp3_frame_sync(self):
    assert detect_from_bytes(b"\xff\xfb" + b"\x00" * 50) == "audio/mpeg"

  def test_mp3_id3_tag(self):
    assert detect_from_bytes(b"ID3" + b"\x00" * 50) == "audio/mpeg"

  def test_ogg_magic(self):
    assert detect_from_bytes(b"OggS" + b"\x00" * 50) == "audio/ogg"

  def test_flac_magic(self):
    assert detect_from_bytes(b"fLaC" + b"\x00" * 50) == "audio/flac"

  def test_tiff_little_endian(self):
    assert detect_from_bytes(b"\x49\x49\x2a\x00" + b"\x00" * 50) == "image/tiff"

  def test_tiff_big_endian(self):
    assert detect_from_bytes(b"\x4d\x4d\x00\x2a" + b"\x00" * 50) == "image/tiff"

  def test_bmp_magic(self):
    assert detect_from_bytes(b"BM" + b"\x00" * 50) == "image/bmp"

  def test_too_short_data(self):
    assert detect_from_bytes(b"ab") is None

  def test_unrecognized_data(self):
    assert detect_from_bytes(b"\x00\x01\x02\x03" + b"\x00" * 50) is None

  def test_riff_wav_subtype(self):
    # RIFF header: 'RIFF' + 4 bytes size + 'WAVE'
    data = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 50
    assert detect_from_bytes(data) == "audio/wav"

  def test_riff_webp_subtype(self):
    data = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 50
    assert detect_from_bytes(data) == "image/webp"

  def test_riff_too_short(self):
    # RIFF magic but too short for subtype detection
    assert detect_from_bytes(b"RIFF\x00\x00\x00\x00") is None


# ===========================================================================
# detect (combined)
# ===========================================================================


@pytest.mark.unit
class TestDetectCombined:
  """Tests for the combined detect() function."""

  def test_explicit_mime_takes_priority(self):
    # Even with valid PNG bytes, explicit mime wins
    assert detect(data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, mime_type="image/gif") == "image/gif"

  def test_bytes_over_extension(self):
    # Magic bytes should be preferred over extension
    result = detect(data=b"%PDF-1.5" + b"\x00" * 50, filename="file.txt")
    assert result == "application/pdf"

  def test_falls_back_to_extension(self):
    # Unrecognized bytes → extension detection
    result = detect(data=b"\x00\x01\x02\x03" * 10, filename="doc.py")
    assert result == "text/x-python"

  def test_no_data_uses_extension(self):
    result = detect(filename="report.xlsx")
    assert result is not None
    assert "spreadsheetml" in result

  def test_all_none_returns_none(self):
    assert detect() is None


# ===========================================================================
# Parsers
# ===========================================================================


@pytest.mark.unit
class TestTextParser:
  """Tests for TextParser."""

  def test_parse_text_file(self):
    from definable.reader.parsers.text import TextParser

    parser = TextParser()
    blocks = parser.parse(b"Hello, world!", mime_type="text/plain")
    assert len(blocks) == 1
    assert blocks[0].content == "Hello, world!"
    assert blocks[0].content_type == "text"

  def test_parse_utf8(self):
    from definable.reader.parsers.text import TextParser

    parser = TextParser()
    blocks = parser.parse("Caf\u00e9 \u2603".encode("utf-8"))
    assert "Caf\u00e9" in blocks[0].content

  def test_can_parse_text_mime(self):
    from definable.reader.parsers.text import TextParser

    parser = TextParser()
    assert parser.can_parse(mime_type="text/plain") is True
    assert parser.can_parse(mime_type="text/x-python") is True
    assert parser.can_parse(mime_type="text/x-unknown-subtype") is True

  def test_can_parse_json_mime(self):
    from definable.reader.parsers.text import TextParser

    parser = TextParser()
    assert parser.can_parse(mime_type="application/json") is True

  def test_can_parse_extension(self):
    from definable.reader.parsers.text import TextParser

    parser = TextParser()
    assert parser.can_parse(extension=".py") is True
    assert parser.can_parse(extension=".md") is True
    assert parser.can_parse(extension=".pdf") is False

  def test_supported_mime_types(self):
    from definable.reader.parsers.text import TextParser

    parser = TextParser()
    mimes = parser.supported_mime_types()
    assert "text/plain" in mimes
    assert "application/json" in mimes

  def test_supported_extensions(self):
    from definable.reader.parsers.text import TextParser

    parser = TextParser()
    exts = parser.supported_extensions()
    assert ".py" in exts
    assert ".txt" in exts
    assert ".md" in exts


@pytest.mark.unit
class TestHTMLParser:
  """Tests for HTMLParser."""

  def test_extracts_visible_text(self):
    from definable.reader.parsers.html import HTMLParser

    parser = HTMLParser()
    html = b"<html><body><p>Hello</p><p>World</p></body></html>"
    blocks = parser.parse(html)
    assert len(blocks) == 1
    assert "Hello" in blocks[0].content
    assert "World" in blocks[0].content

  def test_strips_script_tags(self):
    from definable.reader.parsers.html import HTMLParser

    parser = HTMLParser()
    html = b"<html><body><p>Visible</p><script>alert('hidden')</script></body></html>"
    blocks = parser.parse(html)
    assert "alert" not in blocks[0].content
    assert "Visible" in blocks[0].content

  def test_strips_style_tags(self):
    from definable.reader.parsers.html import HTMLParser

    parser = HTMLParser()
    html = b"<html><body><style>.hidden{display:none}</style><p>Text</p></body></html>"
    blocks = parser.parse(html)
    assert ".hidden" not in blocks[0].content
    assert "Text" in blocks[0].content

  def test_strips_head_tag(self):
    from definable.reader.parsers.html import HTMLParser

    parser = HTMLParser()
    html = b"<html><head><title>Secret</title></head><body>Public</body></html>"
    blocks = parser.parse(html)
    assert "Secret" not in blocks[0].content
    assert "Public" in blocks[0].content

  def test_can_parse_html_mime(self):
    from definable.reader.parsers.html import HTMLParser

    parser = HTMLParser()
    assert parser.can_parse(mime_type="text/html") is True
    assert parser.can_parse(mime_type="application/xhtml+xml") is True

  def test_can_parse_extension(self):
    from definable.reader.parsers.html import HTMLParser

    parser = HTMLParser()
    assert parser.can_parse(extension=".html") is True
    assert parser.can_parse(extension=".htm") is True
    assert parser.can_parse(extension=".xhtml") is True

  def test_empty_html(self):
    from definable.reader.parsers.html import HTMLParser

    parser = HTMLParser()
    blocks = parser.parse(b"<html><body></body></html>")
    assert blocks[0].content == ""


@pytest.mark.unit
class TestImageParser:
  """Tests for ImageParser."""

  def test_passthrough_bytes(self):
    from definable.reader.parsers.image import ImageParser

    parser = ImageParser()
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    blocks = parser.parse(raw, mime_type="image/png")
    assert len(blocks) == 1
    assert blocks[0].content_type == "image"
    assert blocks[0].content is raw

  def test_default_mime_type(self):
    from definable.reader.parsers.image import ImageParser

    parser = ImageParser()
    blocks = parser.parse(b"fake image data")
    assert blocks[0].mime_type == "image/png"

  def test_can_parse_image_mimes(self):
    from definable.reader.parsers.image import ImageParser

    parser = ImageParser()
    assert parser.can_parse(mime_type="image/png") is True
    assert parser.can_parse(mime_type="image/jpeg") is True
    assert parser.can_parse(mime_type="image/gif") is True
    assert parser.can_parse(mime_type="image/webp") is True
    assert parser.can_parse(mime_type="image/svg+xml") is True

  def test_can_parse_extensions(self):
    from definable.reader.parsers.image import ImageParser

    parser = ImageParser()
    assert parser.can_parse(extension=".png") is True
    assert parser.can_parse(extension=".jpg") is True
    assert parser.can_parse(extension=".svg") is True

  def test_rejects_non_image(self):
    from definable.reader.parsers.image import ImageParser

    parser = ImageParser()
    assert parser.can_parse(mime_type="text/plain") is False
    assert parser.can_parse(extension=".txt") is False


@pytest.mark.unit
class TestAudioParser:
  """Tests for AudioParser."""

  def test_passthrough_bytes(self):
    from definable.reader.parsers.audio import AudioParser

    parser = AudioParser()
    raw = b"fake audio data"
    blocks = parser.parse(raw, mime_type="audio/mpeg")
    assert len(blocks) == 1
    assert blocks[0].content_type == "audio"
    assert blocks[0].content is raw

  def test_default_mime_type(self):
    from definable.reader.parsers.audio import AudioParser

    parser = AudioParser()
    blocks = parser.parse(b"data")
    assert blocks[0].mime_type == "audio/mpeg"

  def test_can_parse_audio_mimes(self):
    from definable.reader.parsers.audio import AudioParser

    parser = AudioParser()
    assert parser.can_parse(mime_type="audio/mpeg") is True
    assert parser.can_parse(mime_type="audio/wav") is True
    assert parser.can_parse(mime_type="audio/ogg") is True
    assert parser.can_parse(mime_type="audio/flac") is True

  def test_can_parse_extensions(self):
    from definable.reader.parsers.audio import AudioParser

    parser = AudioParser()
    assert parser.can_parse(extension=".mp3") is True
    assert parser.can_parse(extension=".wav") is True
    assert parser.can_parse(extension=".ogg") is True

  def test_rejects_non_audio(self):
    from definable.reader.parsers.audio import AudioParser

    parser = AudioParser()
    assert parser.can_parse(mime_type="image/png") is False


# ===========================================================================
# BaseReader
# ===========================================================================


@pytest.mark.unit
class TestBaseReader:
  """Tests for BaseReader orchestration."""

  def test_default_construction(self):
    from definable.reader.base import BaseReader

    reader = BaseReader()
    assert reader.config is None
    assert reader.registry is not None

  def test_register_returns_self(self):
    from definable.reader.base import BaseReader
    from definable.reader.parsers.text import TextParser

    reader = BaseReader()
    result = reader.register(TextParser(), priority=200)
    assert result is reader

  def test_read_text_file(self):
    from definable.media import File
    from definable.reader.base import BaseReader

    reader = BaseReader()
    f = File(name="test.txt", content=b"Hello, world!")
    output = reader.read(f)
    assert output.error is None
    assert output.word_count is not None and output.word_count > 0
    assert "Hello" in output.as_text()

  def test_read_html_file(self):
    from definable.media import File
    from definable.reader.base import BaseReader

    reader = BaseReader()
    f = File(name="page.html", content=b"<html><body><p>Test</p></body></html>")
    output = reader.read(f)
    assert output.error is None
    assert "Test" in output.as_text()

  def test_read_unknown_format(self):
    from definable.media import File
    from definable.reader.base import BaseReader

    reader = BaseReader()
    f = File(name="data.unknownformat", content=b"\x00\x01\x02\x03")
    output = reader.read(f)
    assert output.error is not None
    assert "No parser" in output.error

  def test_max_file_size_enforced(self):
    from definable.media import File
    from definable.reader.base import BaseReader
    from definable.reader.models import ReaderConfig

    reader = BaseReader(config=ReaderConfig(max_file_size=10))
    f = File(name="big.txt", content=b"x" * 100)
    output = reader.read(f)
    assert output.error is not None
    assert "exceeds" in output.error

  def test_truncation(self):
    from definable.media import File
    from definable.reader.base import BaseReader
    from definable.reader.models import ReaderConfig

    reader = BaseReader(config=ReaderConfig(max_content_length=5))
    f = File(name="long.txt", content=b"Hello, World! This is a long text.")
    output = reader.read(f)
    assert output.truncated is True

  def test_get_parser_for_text(self):
    from definable.media import File
    from definable.reader.base import BaseReader

    reader = BaseReader()
    f = File(name="script.py", content=b"print('hi')")
    parser = reader.get_parser(f)
    assert parser is not None

  def test_get_parser_unknown(self):
    from definable.media import File
    from definable.reader.base import BaseReader

    reader = BaseReader()
    f = File(name="data.unknownformat", content=b"\x00\x01")
    parser = reader.get_parser(f)
    assert parser is None
