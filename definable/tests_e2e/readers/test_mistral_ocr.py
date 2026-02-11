"""Tests for MistralReader — all mocked, no API key needed."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Tuple
from unittest.mock import MagicMock, patch

import pytest

from definable.media import File
from definable.readers.models import ReaderConfig
from definable.readers.providers.mistral import (
  MistralReader,
  _MISTRAL_MAX_FILE_SIZE,
  _NATIVE_EXTENSIONS,
)
from definable.readers.utils import extract_extension


# ── Mock helpers ──────────────────────────────────────────────────────


class _MockPage:
  def __init__(self, markdown: str):
    self.markdown = markdown


class _MockOCRResponse:
  def __init__(self, pages: list[_MockPage]):
    self.pages = pages


class _MockSignedURL:
  def __init__(self, url: str = "https://signed.example.com/doc.pdf"):
    self.url = url


class _MockUploadedFile:
  def __init__(self, file_id: str = "file-abc123"):
    self.id = file_id


def _make_ocr_response(page_texts: list[str]) -> _MockOCRResponse:
  return _MockOCRResponse(pages=[_MockPage(t) for t in page_texts])


def _make_mock_client(
  ocr_response: Optional[_MockOCRResponse] = None,
) -> MagicMock:
  """Create a mock Mistral client."""
  if ocr_response is None:
    ocr_response = _make_ocr_response(["# Page 1\n\nHello world"])

  client = MagicMock()
  client.ocr.process.return_value = ocr_response

  async def _async_process(**kwargs):
    return ocr_response

  client.ocr.process_async = MagicMock(side_effect=_async_process)

  uploaded = _MockUploadedFile()
  client.files.upload.return_value = uploaded
  client.files.get_signed_url.return_value = _MockSignedURL()
  client.files.delete.return_value = None

  return client


@dataclass
class _MockPreprocessor:
  """Fake preprocessor for testing."""

  extensions: Set[str] = field(default_factory=lambda: {".tiff", ".tif"})
  images: List[Tuple[bytes, str]] = field(default_factory=lambda: [(b"\xff\xd8fake-jpeg", "image/jpeg")])

  def can_process(self, file: File) -> bool:
    ext = extract_extension(file)
    return ext is not None and ext in self.extensions

  def to_images(self, raw: bytes, file: File) -> List[Tuple[bytes, str]]:
    return self.images

  async def ato_images(self, raw: bytes, file: File) -> List[Tuple[bytes, str]]:
    return self.images


def _make_reader(
  preprocessor: Any = False,
  **kwargs: Any,
) -> MistralReader:
  """Create a MistralReader with mocked mistralai import."""
  with patch("definable.readers.providers.mistral._import_mistralai"):
    return MistralReader(
      api_key="test-key",
      preprocessor=preprocessor,
      **kwargs,
    )


# ── Test: Classification ──────────────────────────────────────────────


class TestClassification:
  def test_url_native_pdf(self):
    reader = _make_reader()
    file = File(url="https://example.com/doc.pdf", filename="doc.pdf")
    assert reader._classify(file) == "url_native"

  def test_url_native_image(self):
    reader = _make_reader()
    file = File(url="https://example.com/photo.png", filename="photo.png")
    assert reader._classify(file) == "url_native"

  def test_upload_document_pdf(self):
    reader = _make_reader()
    file = File(content=b"%PDF-1.4", filename="doc.pdf")
    assert reader._classify(file) == "upload_document"

  def test_upload_document_pptx(self):
    reader = _make_reader()
    file = File(content=b"PK\x03\x04", filename="slides.pptx")
    assert reader._classify(file) == "upload_document"

  def test_base64_image_png(self):
    reader = _make_reader()
    file = File(content=b"\x89PNG", filename="image.png")
    assert reader._classify(file) == "base64_image"

  def test_preprocess_tiff(self):
    reader = _make_reader(preprocessor=_MockPreprocessor())
    file = File(content=b"II*\x00", filename="scan.tiff")
    assert reader._classify(file) == "preprocess"

  def test_unsupported_no_preprocessor(self):
    reader = _make_reader(preprocessor=False)
    file = File(content=b"II*\x00", filename="scan.tiff")
    assert reader._classify(file) == "unsupported"

  def test_upload_document_docx(self):
    reader = _make_reader()
    file = File(content=b"PK\x03\x04", filename="report.docx")
    assert reader._classify(file) == "upload_document"


# ── Test: Supported formats ───────────────────────────────────────────


class TestSupportedFormats:
  def test_native_extensions(self):
    reader = _make_reader(preprocessor=False)
    exts = reader.supported_extensions()
    assert exts == _NATIVE_EXTENSIONS

  def test_extensions_with_preprocessor(self):
    pp = _MockPreprocessor(extensions={".tiff", ".tif"})
    reader = _make_reader(preprocessor=pp)
    exts = reader.supported_extensions()
    assert ".tiff" in exts
    assert ".tif" in exts
    assert ".pdf" in exts

  def test_extensions_without_preprocessor(self):
    reader = _make_reader(preprocessor=False)
    exts = reader.supported_extensions()
    assert ".tiff" not in exts

  def test_mime_types_with_preprocessor(self):
    pp = _MockPreprocessor()
    reader = _make_reader(preprocessor=pp)
    mimes = reader.supported_mime_types()
    assert "image/tiff" in mimes
    assert "application/pdf" in mimes

  def test_mime_types_without_preprocessor(self):
    reader = _make_reader(preprocessor=False)
    mimes = reader.supported_mime_types()
    assert "image/tiff" not in mimes
    assert "application/pdf" in mimes


# ── Test: URL shortcut ────────────────────────────────────────────────


class TestURLShortcut:
  def test_url_pdf_no_upload(self):
    reader = _make_reader()
    client = _make_mock_client()
    file = File(url="https://example.com/doc.pdf", filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=client):
      result = reader.read(file)

    assert result.error is None
    assert "Hello world" in result.content
    client.files.upload.assert_not_called()
    client.ocr.process.assert_called_once()
    call_kwargs = client.ocr.process.call_args[1]
    assert call_kwargs["document"]["type"] == "document_url"

  def test_url_image_no_upload(self):
    reader = _make_reader()
    client = _make_mock_client()
    file = File(url="https://example.com/photo.jpeg", filename="photo.jpeg")

    with patch.object(reader, "_create_client", return_value=client):
      result = reader.read(file)

    assert result.error is None
    client.files.upload.assert_not_called()
    call_kwargs = client.ocr.process.call_args[1]
    assert call_kwargs["document"]["type"] == "image_url"

  @pytest.mark.asyncio
  async def test_async_url_pdf(self):
    reader = _make_reader()
    client = _make_mock_client()
    file = File(url="https://example.com/doc.pdf", filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=client):
      result = await reader.aread(file)

    assert result.error is None
    assert "Hello world" in result.content
    client.files.upload.assert_not_called()
    client.ocr.process_async.assert_called_once()


# ── Test: Upload workflow ─────────────────────────────────────────────


class TestUploadWorkflow:
  def test_upload_and_cleanup_on_success(self):
    reader = _make_reader()
    client = _make_mock_client()
    file = File(content=b"%PDF-1.4 content here", filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=client):
      result = reader.read(file)

    assert result.error is None
    client.files.upload.assert_called_once()
    client.files.get_signed_url.assert_called_once()
    client.files.delete.assert_called_once()

  def test_cleanup_on_ocr_error(self):
    reader = _make_reader()
    client = _make_mock_client()
    client.ocr.process.side_effect = RuntimeError("OCR failed")
    file = File(content=b"%PDF-1.4 content here", filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=client):
      result = reader.read(file)

    assert result.error is not None
    assert "OCR failed" in result.error
    client.files.delete.assert_called_once()


# ── Test: Preprocessor integration ────────────────────────────────────


class TestPreprocessorIntegration:
  def test_single_image_preprocess(self):
    pp = _MockPreprocessor(images=[(b"\xff\xd8fake", "image/jpeg")])
    reader = _make_reader(preprocessor=pp)
    client = _make_mock_client()
    file = File(content=b"TIFF-data", filename="scan.tiff")

    with patch.object(reader, "_create_client", return_value=client):
      result = reader.read(file)

    assert result.error is None
    assert result.metadata.get("preprocessed") is True
    client.ocr.process.assert_called_once()

  def test_multi_frame_tiff(self):
    """Multi-frame TIFF -> multiple OCR calls."""
    pp = _MockPreprocessor(
      images=[
        (b"\xff\xd8frame1", "image/jpeg"),
        (b"\xff\xd8frame2", "image/jpeg"),
        (b"\xff\xd8frame3", "image/jpeg"),
      ]
    )
    reader = _make_reader(preprocessor=pp)
    client = _make_mock_client()
    file = File(content=b"TIFF-multiframe", filename="scan.tiff")

    with patch.object(reader, "_create_client", return_value=client):
      result = reader.read(file)

    assert result.error is None
    assert client.ocr.process.call_count == 3

  def test_preprocessor_disabled(self):
    reader = _make_reader(preprocessor=False, local_fallback=False)
    file = File(content=b"TIFF-data", filename="scan.tiff")

    with patch.object(reader, "_create_client", return_value=_make_mock_client()):
      result = reader.read(file)

    assert result.error is not None
    assert "Unsupported" in result.error

  @pytest.mark.asyncio
  async def test_async_preprocess(self):
    pp = _MockPreprocessor(images=[(b"\xff\xd8fake", "image/jpeg")])
    reader = _make_reader(preprocessor=pp)
    client = _make_mock_client()
    file = File(content=b"TIFF-data", filename="scan.tiff")

    with patch.object(reader, "_create_client", return_value=client):
      result = await reader.aread(file)

    assert result.error is None
    assert result.metadata.get("preprocessed") is True


# ── Test: Error handling ──────────────────────────────────────────────


class TestErrorHandling:
  def test_api_error_returns_error_result(self):
    reader = _make_reader()
    client = _make_mock_client()
    client.ocr.process.side_effect = RuntimeError("API unavailable")
    file = File(url="https://example.com/doc.pdf", filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=client):
      result = reader.read(file)

    assert result.error is not None
    assert "API unavailable" in result.error
    assert result.content == ""

  def test_file_too_large(self):
    reader = _make_reader()
    oversized = b"x" * (_MISTRAL_MAX_FILE_SIZE + 1)
    file = File(content=oversized, filename="huge.pdf")

    with patch.object(reader, "_create_client", return_value=_make_mock_client()):
      result = reader.read(file)

    assert result.error is not None
    assert "50 MB" in result.error

  def test_user_configured_max_file_size(self):
    reader = _make_reader(config=ReaderConfig(max_file_size=100))
    file = File(content=b"x" * 200, filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=_make_mock_client()):
      result = reader.read(file)

    assert result.error is not None
    assert "max_file_size" in result.error

  @pytest.mark.asyncio
  async def test_async_api_error(self):
    reader = _make_reader()
    client = _make_mock_client()

    async def _fail(**kwargs):
      raise RuntimeError("Async API error")

    client.ocr.process_async = MagicMock(side_effect=_fail)
    file = File(url="https://example.com/doc.pdf", filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=client):
      result = await reader.aread(file)

    assert result.error is not None
    assert "Async API error" in result.error

  def test_no_api_key_raises(self):
    reader = _make_reader()
    reader.api_key = None
    file = File(content=b"%PDF", filename="doc.pdf")
    with patch.dict("os.environ", {}, clear=True):
      result = reader.read(file)
    assert result.error is not None
    assert "API key" in result.error


# ── Test: can_read ────────────────────────────────────────────────────


class TestCanRead:
  def test_pdf(self):
    reader = _make_reader()
    file = File(content=b"%PDF", filename="doc.pdf", mime_type="application/pdf")
    assert reader.can_read(file) is True

  def test_png(self):
    reader = _make_reader()
    file = File(content=b"\x89PNG", filename="img.png", mime_type="image/png")
    assert reader.can_read(file) is True

  def test_pptx(self):
    reader = _make_reader()
    file = File(content=b"PK\x03\x04", filename="slides.pptx")
    assert reader.can_read(file) is True

  def test_unsupported_no_fallback(self):
    reader = _make_reader(preprocessor=False, local_fallback=False)
    file = File(content=b"\x00\x01", filename="data.bin")
    assert reader.can_read(file) is False

  def test_tiff_with_preprocessor(self):
    pp = _MockPreprocessor()
    reader = _make_reader(preprocessor=pp)
    file = File(content=b"tiff", filename="scan.tiff", mime_type="image/tiff")
    assert reader.can_read(file) is True

  def test_tiff_without_preprocessor(self):
    reader = _make_reader(preprocessor=False, local_fallback=False)
    file = File(content=b"tiff", filename="scan.tiff", mime_type="image/tiff")
    assert reader.can_read(file) is False


# ── Test: Result metadata ────────────────────────────────────────────


class TestResultMetadata:
  def test_result_has_provider_metadata(self):
    reader = _make_reader()
    client = _make_mock_client()
    file = File(url="https://example.com/doc.pdf", filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=client):
      result = reader.read(file)

    assert result.metadata["provider"] == "mistral-ocr"
    assert result.metadata["preprocessed"] is False

  def test_result_page_and_word_count(self):
    response = _make_ocr_response(["Page one content", "Page two content"])
    reader = _make_reader()
    client = _make_mock_client(ocr_response=response)
    file = File(url="https://example.com/doc.pdf", filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=client):
      result = reader.read(file)

    assert result.page_count == 2
    assert result.word_count is not None
    assert result.word_count > 0

  def test_empty_pages_skipped(self):
    response = _make_ocr_response(["Content here", "", "  ", "More content"])
    reader = _make_reader()
    client = _make_mock_client(ocr_response=response)
    file = File(url="https://example.com/doc.pdf", filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=client):
      result = reader.read(file)

    # Only non-empty pages counted
    assert result.page_count == 2


# ── Test: Backwards-compat aliases ────────────────────────────────────


class TestBackwardsCompat:
  def test_mistral_ocr_reader_alias(self):
    from definable.readers.providers.mistral import MistralOCRReader

    assert MistralOCRReader is MistralReader

  def test_read_file_alias(self):
    reader = _make_reader()
    client = _make_mock_client()
    file = File(url="https://example.com/doc.pdf", filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=client):
      result = reader.read_file(file)  # Old name

    assert result.error is None

  @pytest.mark.asyncio
  async def test_aread_file_alias(self):
    reader = _make_reader()
    client = _make_mock_client()
    file = File(url="https://example.com/doc.pdf", filename="doc.pdf")

    with patch.object(reader, "_create_client", return_value=client):
      result = await reader.aread_file(file)  # Old name

    assert result.error is None


# ── Test: extract_extension helper ───────────────────────────────────


class TestExtractExtension:
  def test_from_filename(self):
    file = File(content=b"data", filename="report.pdf")
    assert extract_extension(file) == ".pdf"

  def test_from_filepath(self):
    file = File(content=b"data", filepath="/tmp/doc.docx")
    assert extract_extension(file) == ".docx"

  def test_from_url(self):
    file = File(url="https://example.com/path/image.png")
    assert extract_extension(file) == ".png"

  def test_no_extension(self):
    file = File(content=b"data", filename="README")
    assert extract_extension(file) is None

  def test_uppercase_normalised(self):
    file = File(content=b"data", filename="DOC.PDF")
    assert extract_extension(file) == ".pdf"
