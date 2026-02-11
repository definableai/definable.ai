"""Tests for MIME type detection."""

from definable.readers.detector import (
  detect,
  detect_from_bytes,
  detect_from_extension,
  extract_extension,
)


class TestDetectFromBytes:
  def test_pdf(self):
    assert detect_from_bytes(b"%PDF-1.4 content") == "application/pdf"

  def test_png(self):
    assert detect_from_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10) == "image/png"

  def test_jpeg(self):
    assert detect_from_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10) == "image/jpeg"

  def test_gif87(self):
    assert detect_from_bytes(b"GIF87a" + b"\x00" * 10) == "image/gif"

  def test_gif89(self):
    assert detect_from_bytes(b"GIF89a" + b"\x00" * 10) == "image/gif"

  def test_rtf(self):
    assert detect_from_bytes(b"{\\rtf1" + b"\x00" * 10) == "text/rtf"

  def test_tiff_le(self):
    assert detect_from_bytes(b"\x49\x49\x2a\x00" + b"\x00" * 10) == "image/tiff"

  def test_tiff_be(self):
    assert detect_from_bytes(b"\x4d\x4d\x00\x2a" + b"\x00" * 10) == "image/tiff"

  def test_bmp(self):
    assert detect_from_bytes(b"BM" + b"\x00" * 20) == "image/bmp"

  def test_mp3_with_id3(self):
    assert detect_from_bytes(b"ID3" + b"\x00" * 10) == "audio/mpeg"

  def test_mp3_frame_sync(self):
    assert detect_from_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 10) == "audio/mpeg"

  def test_ogg(self):
    assert detect_from_bytes(b"OggS" + b"\x00" * 10) == "audio/ogg"

  def test_flac(self):
    assert detect_from_bytes(b"fLaC" + b"\x00" * 10) == "audio/flac"

  def test_wav(self):
    data = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 10
    assert detect_from_bytes(data) == "audio/wav"

  def test_webp(self):
    data = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 10
    assert detect_from_bytes(data) == "image/webp"

  def test_unknown(self):
    assert detect_from_bytes(b"\x00\x01\x02\x03") is None

  def test_too_short(self):
    assert detect_from_bytes(b"\x00") is None


class TestDetectFromExtension:
  def test_pdf(self):
    assert detect_from_extension(filename="doc.pdf") == "application/pdf"

  def test_docx(self):
    assert detect_from_extension(filename="doc.docx") == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

  def test_python(self):
    assert detect_from_extension(filename="script.py") == "text/x-python"

  def test_from_filepath(self):
    assert detect_from_extension(filepath="/tmp/notes.md") == "text/markdown"

  def test_from_url(self):
    assert detect_from_extension(url="https://example.com/data.json") == "application/json"

  def test_unknown_extension(self):
    assert detect_from_extension(filename="data.xyz") is None

  def test_no_extension(self):
    assert detect_from_extension(filename="README") is None


class TestDetectCombined:
  def test_explicit_mime_takes_priority(self):
    result = detect(
      data=b"%PDF-1.4",
      filename="doc.txt",
      mime_type="text/plain",
    )
    assert result == "text/plain"

  def test_bytes_over_extension(self):
    result = detect(data=b"%PDF-1.4", filename="doc.txt")
    assert result == "application/pdf"

  def test_extension_fallback(self):
    result = detect(data=b"not-magic", filename="doc.pdf")
    assert result == "application/pdf"

  def test_no_detection(self):
    result = detect(data=b"\x00\x01", filename="data.xyz")
    assert result is None


class TestExtractExtension:
  def test_from_filename(self):
    assert extract_extension(filename="report.pdf") == ".pdf"

  def test_from_filepath(self):
    assert extract_extension(filepath="/tmp/doc.docx") == ".docx"

  def test_from_url(self):
    assert extract_extension(url="https://example.com/image.png") == ".png"

  def test_uppercase(self):
    assert extract_extension(filename="DOC.PDF") == ".pdf"

  def test_no_extension(self):
    assert extract_extension(filename="README") is None

  def test_none_inputs(self):
    assert extract_extension() is None
