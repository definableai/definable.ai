"""MistralReader — multi-format cloud OCR using the Mistral API.

Full port of the old MistralOCRReader. Supports PDFs, DOCX, PPTX, and
images (PNG, JPEG, AVIF) natively. Non-native image formats (TIFF, BMP,
HEIC, WebP) are converted to JPEG via a pluggable FilePreprocessor.

Requires optional dependency: ``mistralai>=1.0.0``
"""

import asyncio
import base64
import os
from typing import Any, List, Optional, Set

from definable.media import File
from definable.reader.models import ContentBlock, ReaderConfig, ReaderOutput
from definable.reader.utils import (
  extract_extension,
  get_file_bytes_async,
  get_file_bytes_sync,
  get_filename,
)
from definable.utils.log import log_debug, log_warning

# Mistral's hard limit for file uploads
_MISTRAL_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Formats the Mistral OCR API handles directly
_NATIVE_DOCUMENT_EXTENSIONS: Set[str] = {".pdf", ".docx", ".pptx"}
_NATIVE_IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".avif"}
_NATIVE_EXTENSIONS: Set[str] = _NATIVE_DOCUMENT_EXTENSIONS | _NATIVE_IMAGE_EXTENSIONS

_EXTENSION_TO_MIME = {
  ".pdf": "application/pdf",
  ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".avif": "image/avif",
}


def _import_mistralai():
  try:
    import mistralai

    return mistralai
  except ImportError as e:
    raise ImportError(
      "MistralReader requires the 'mistralai' package. Install it with: pip install 'mistralai>=1.0.0' or: pip install 'definable[mistral-ocr]'"
    ) from e


class MistralReader:
  """Cloud OCR reader using the Mistral OCR API.

  Handles PDFs, DOCX, PPTX, and common image formats. For files
  provided with a URL and a native format, the URL is passed directly
  to the API (no upload/download needed).

  Args:
    config: Optional reader configuration (max_file_size, etc.).
    api_key: Mistral API key. Falls back to ``MISTRAL_API_KEY`` env var.
    model: OCR model name (default: ``mistral-ocr-latest``).
    page_separator: String used to join pages in output.
    include_image_base64: If True, request image base64 data in response.
    preprocessor: Controls non-native format handling:
      - ``None`` (default): auto-create ``ImageFormatConverter`` if
        Pillow is installed, otherwise disable.
      - ``False``: disable preprocessing entirely.
      - A ``FilePreprocessor`` instance: use that preprocessor.
    local_fallback: If True, fall back to local parsers for unsupported formats.
  """

  def __init__(
    self,
    config: Optional[ReaderConfig] = None,
    api_key: Optional[str] = None,
    model: str = "mistral-ocr-latest",
    page_separator: str = "\n\n",
    include_image_base64: bool = False,
    preprocessor: Any = None,
    local_fallback: bool = True,
  ) -> None:
    _import_mistralai()
    self.config = config
    self.api_key = api_key
    self.model = model
    self.page_separator = page_separator
    self.include_image_base64 = include_image_base64
    self.preprocessor = preprocessor
    self.local_fallback = local_fallback
    self._resolved_preprocessor: Any = self._resolve_preprocessor()
    self._fallback_registry: Any = self._build_fallback_registry() if local_fallback else None

  def _resolve_preprocessor(self) -> Any:
    """Resolve the preprocessor setting to an actual instance or None."""
    if self.preprocessor is False:
      return None
    if self.preprocessor is not None:
      return self.preprocessor
    try:
      from PIL import Image as _PilImage  # noqa: F401
      from definable.reader.mistral.preprocessor import ImageFormatConverter

      log_debug("MistralReader: Pillow available, enabling ImageFormatConverter")
      return ImageFormatConverter()
    except ImportError:
      log_debug("MistralReader: Pillow not available, preprocessing disabled")
      return None

  def _build_fallback_registry(self) -> Any:
    """Build a ParserRegistry for unsupported-format fallback."""
    from definable.reader.registry import ParserRegistry

    return ParserRegistry(include_defaults=True)

  # ── Public interface ──────────────────────────────────────────────────

  def can_read(self, file: File) -> bool:
    """Check whether this reader can handle the given file."""
    ext = extract_extension(file)
    if ext in _NATIVE_EXTENSIONS:
      return True
    if self._resolved_preprocessor is not None and self._resolved_preprocessor.can_process(file):
      return True
    if self.local_fallback and self._fallback_registry:
      mime_type = file.mime_type
      return self._fallback_registry.get_parser(mime_type, ext) is not None
    return False

  def supported_mime_types(self) -> List[str]:
    mimes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/vnd.openxmlformats-officedocument.presentationml.presentation",
      "image/png",
      "image/jpeg",
      "image/avif",
    ]
    if self._resolved_preprocessor is not None:
      mimes.extend([
        "image/tiff",
        "image/bmp",
        "image/heic",
        "image/webp",
      ])
    return mimes

  def supported_extensions(self) -> Set[str]:
    exts = set(_NATIVE_EXTENSIONS)
    if self._resolved_preprocessor is not None:
      pp = self._resolved_preprocessor
      if hasattr(pp, "extensions"):
        exts |= pp.extensions
    return exts

  def read(self, file: File) -> ReaderOutput:
    """Synchronously read a file via Mistral OCR."""
    try:
      classification = self._classify(file)
      if classification == "unsupported":
        return self._try_local_fallback_sync(file)
      return self._read_by_classification(file, classification)
    except Exception as e:
      return self._make_error_output(file, str(e))

  # Keep read_file as alias for backwards compat
  def read_file(self, file: File) -> ReaderOutput:
    """Backwards-compatible alias for read()."""
    return self.read(file)

  async def aread(self, file: File) -> ReaderOutput:
    """Asynchronously read a file via Mistral OCR."""
    try:
      classification = self._classify(file)
      if classification == "unsupported":
        return await self._try_local_fallback_async(file)
      return await self._aread_by_classification(file, classification)
    except Exception as e:
      return self._make_error_output(file, str(e))

  # Keep aread_file as alias for backwards compat
  async def aread_file(self, file: File) -> ReaderOutput:
    """Backwards-compatible alias for aread()."""
    return await self.aread(file)

  # ── Classification ───────────────────────────────────────────────────

  def _classify(self, file: File) -> str:
    """Classify a file into a processing strategy."""
    ext = extract_extension(file)

    # URL shortcut: pass URL directly for native formats
    if file.url and ext in _NATIVE_EXTENSIONS:
      return "url_native"

    # For files with bytes/filepath, check native formats
    if ext in _NATIVE_DOCUMENT_EXTENSIONS:
      return "upload_document"

    if ext in _NATIVE_IMAGE_EXTENSIONS:
      return "base64_image"

    # Check preprocessor
    if self._resolved_preprocessor is not None and self._resolved_preprocessor.can_process(file):
      return "preprocess"

    return "unsupported"

  # ── Local fallback ───────────────────────────────────────────────────

  def _try_local_fallback_sync(self, file: File) -> ReaderOutput:
    """Try local parsers for unsupported formats."""
    if not self._fallback_registry:
      return self._make_error_output(file, "Unsupported file format for MistralReader")
    ext = extract_extension(file)
    parser = self._fallback_registry.get_parser(file.mime_type, ext)
    if parser is None:
      return self._make_error_output(file, "Unsupported file format for MistralReader")
    log_debug(f"MistralReader: falling back to {type(parser).__name__} for {get_filename(file)}")
    raw = get_file_bytes_sync(file, encoding=self._encoding, timeout=self._timeout)
    blocks = parser.parse(data=raw, mime_type=file.mime_type, config=self.config)
    return self._build_output_from_blocks(file, blocks)

  async def _try_local_fallback_async(self, file: File) -> ReaderOutput:
    """Try local parsers for unsupported formats (async)."""
    if not self._fallback_registry:
      return self._make_error_output(file, "Unsupported file format for MistralReader")
    ext = extract_extension(file)
    parser = self._fallback_registry.get_parser(file.mime_type, ext)
    if parser is None:
      return self._make_error_output(file, "Unsupported file format for MistralReader")
    log_debug(f"MistralReader: falling back to {type(parser).__name__} for {get_filename(file)}")
    raw = await get_file_bytes_async(file, encoding=self._encoding, timeout=self._timeout)
    blocks = parser.parse(data=raw, mime_type=file.mime_type, config=self.config)
    return self._build_output_from_blocks(file, blocks)

  # ── Client factory ────────────────────────────────────────────────────

  def _create_client(self) -> Any:
    """Create a Mistral client instance."""
    api_key = self._get_api_key()
    from mistralai import Mistral

    return Mistral(api_key=api_key)

  # ── Sync dispatch ────────────────────────────────────────────────────

  def _read_by_classification(self, file: File, classification: str) -> ReaderOutput:
    client = self._create_client()

    if classification == "url_native":
      return self._read_url_native(client, file)
    elif classification == "upload_document":
      raw = get_file_bytes_sync(file, encoding=self._encoding, timeout=self._timeout)
      self._check_mistral_file_size(raw)
      return self._read_upload_document(client, file, raw)
    elif classification == "base64_image":
      raw = get_file_bytes_sync(file, encoding=self._encoding, timeout=self._timeout)
      self._check_mistral_file_size(raw)
      return self._read_base64_image(client, file, raw)
    elif classification == "preprocess":
      raw = get_file_bytes_sync(file, encoding=self._encoding, timeout=self._timeout)
      self._check_mistral_file_size(raw)
      return self._read_preprocess(client, file, raw)
    else:
      return self._make_error_output(file, f"Unknown classification: {classification}")

  def _read_url_native(self, client: Any, file: File) -> ReaderOutput:
    ext = extract_extension(file)
    ocr_kwargs = self._ocr_kwargs(file)
    if ext in _NATIVE_DOCUMENT_EXTENSIONS:
      response = client.ocr.process(
        model=self.model,
        document={"type": "document_url", "document_url": file.url},
        **ocr_kwargs,
      )
    else:
      response = client.ocr.process(
        model=self.model,
        document={"type": "image_url", "image_url": file.url},
        **ocr_kwargs,
      )
    return self._build_result(file, response, preprocessed=False)

  def _read_upload_document(self, client: Any, file: File, raw: bytes) -> ReaderOutput:
    filename = get_filename(file)
    uploaded = client.files.upload(
      file={"file_name": filename, "content": raw},
      purpose="ocr",
    )
    try:
      signed_url = client.files.get_signed_url(file_id=uploaded.id)
      ocr_kwargs = self._ocr_kwargs(file)
      response = client.ocr.process(
        model=self.model,
        document={"type": "document_url", "document_url": signed_url.url},
        **ocr_kwargs,
      )
      return self._build_result(file, response, preprocessed=False)
    finally:
      try:
        client.files.delete(file_id=uploaded.id)
      except Exception as exc:
        log_warning(f"Failed to delete uploaded file {uploaded.id}: {exc}")

  def _read_base64_image(self, client: Any, file: File, raw: bytes) -> ReaderOutput:
    ext = extract_extension(file)
    mime = _EXTENSION_TO_MIME.get(ext or "", "image/png")
    data_uri = self._make_data_uri(raw, mime)
    ocr_kwargs = self._ocr_kwargs(file)
    response = client.ocr.process(
      model=self.model,
      document={"type": "image_url", "image_url": data_uri},
      **ocr_kwargs,
    )
    return self._build_result(file, response, preprocessed=False)

  def _read_preprocess(self, client: Any, file: File, raw: bytes) -> ReaderOutput:
    images = self._resolved_preprocessor.to_images(raw, file)
    ocr_kwargs = self._ocr_kwargs(file)
    all_pages: List[str] = []
    for img_bytes, img_mime in images:
      data_uri = self._make_data_uri(img_bytes, img_mime)
      response = client.ocr.process(
        model=self.model,
        document={"type": "image_url", "image_url": data_uri},
        **ocr_kwargs,
      )
      all_pages.extend(self._extract_pages(response))

    content = self.page_separator.join(all_pages)
    content, truncated = self._truncate(content)
    blocks = [
      ContentBlock(
        content_type="text",
        content=content,
        mime_type=file.mime_type,
      )
    ]
    return ReaderOutput(
      filename=get_filename(file),
      blocks=blocks,
      mime_type=file.mime_type,
      page_count=len(all_pages),
      word_count=len(content.split()),
      truncated=truncated,
      metadata={"provider": "mistral-ocr", "preprocessed": True},
    )

  # ── Async dispatch ───────────────────────────────────────────────────

  async def _aread_by_classification(self, file: File, classification: str) -> ReaderOutput:
    client = self._create_client()

    if classification == "url_native":
      return await self._aread_url_native(client, file)
    elif classification == "upload_document":
      raw = await get_file_bytes_async(file, encoding=self._encoding, timeout=self._timeout)
      self._check_mistral_file_size(raw)
      return await self._aread_upload_document(client, file, raw)
    elif classification == "base64_image":
      raw = await get_file_bytes_async(file, encoding=self._encoding, timeout=self._timeout)
      self._check_mistral_file_size(raw)
      return await self._aread_base64_image(client, file, raw)
    elif classification == "preprocess":
      raw = await get_file_bytes_async(file, encoding=self._encoding, timeout=self._timeout)
      self._check_mistral_file_size(raw)
      return await self._aread_preprocess(client, file, raw)
    else:
      return self._make_error_output(file, f"Unknown classification: {classification}")

  async def _aread_url_native(self, client: Any, file: File) -> ReaderOutput:
    ext = extract_extension(file)
    ocr_kwargs = self._ocr_kwargs(file)
    if ext in _NATIVE_DOCUMENT_EXTENSIONS:
      response = await client.ocr.process_async(
        model=self.model,
        document={"type": "document_url", "document_url": file.url},
        **ocr_kwargs,
      )
    else:
      response = await client.ocr.process_async(
        model=self.model,
        document={"type": "image_url", "image_url": file.url},
        **ocr_kwargs,
      )
    return self._build_result(file, response, preprocessed=False)

  async def _aread_upload_document(self, client: Any, file: File, raw: bytes) -> ReaderOutput:
    filename = get_filename(file)
    loop = asyncio.get_running_loop()
    uploaded = await loop.run_in_executor(
      None,
      lambda: client.files.upload(
        file={"file_name": filename, "content": raw},
        purpose="ocr",
      ),
    )
    try:
      signed_url = await loop.run_in_executor(
        None,
        lambda: client.files.get_signed_url(file_id=uploaded.id),
      )
      ocr_kwargs = self._ocr_kwargs(file)
      response = await client.ocr.process_async(
        model=self.model,
        document={"type": "document_url", "document_url": signed_url.url},
        **ocr_kwargs,
      )
      return self._build_result(file, response, preprocessed=False)
    finally:
      try:
        await loop.run_in_executor(
          None,
          lambda: client.files.delete(file_id=uploaded.id),
        )
      except Exception as exc:
        log_warning(f"Failed to delete uploaded file {uploaded.id}: {exc}")

  async def _aread_base64_image(self, client: Any, file: File, raw: bytes) -> ReaderOutput:
    ext = extract_extension(file)
    mime = _EXTENSION_TO_MIME.get(ext or "", "image/png")
    data_uri = self._make_data_uri(raw, mime)
    ocr_kwargs = self._ocr_kwargs(file)
    response = await client.ocr.process_async(
      model=self.model,
      document={"type": "image_url", "image_url": data_uri},
      **ocr_kwargs,
    )
    return self._build_result(file, response, preprocessed=False)

  async def _aread_preprocess(self, client: Any, file: File, raw: bytes) -> ReaderOutput:
    images = await self._resolved_preprocessor.ato_images(raw, file)
    ocr_kwargs = self._ocr_kwargs(file)
    all_pages: List[str] = []
    for img_bytes, img_mime in images:
      data_uri = self._make_data_uri(img_bytes, img_mime)
      response = await client.ocr.process_async(
        model=self.model,
        document={"type": "image_url", "image_url": data_uri},
        **ocr_kwargs,
      )
      all_pages.extend(self._extract_pages(response))

    content = self.page_separator.join(all_pages)
    content, truncated = self._truncate(content)
    blocks = [
      ContentBlock(
        content_type="text",
        content=content,
        mime_type=file.mime_type,
      )
    ]
    return ReaderOutput(
      filename=get_filename(file),
      blocks=blocks,
      mime_type=file.mime_type,
      page_count=len(all_pages),
      word_count=len(content.split()),
      truncated=truncated,
      metadata={"provider": "mistral-ocr", "preprocessed": True},
    )

  # ── Helpers ──────────────────────────────────────────────────────────

  def _ocr_kwargs(self, file: File) -> dict:
    """Build extra kwargs for ``ocr.process`` / ``ocr.process_async``."""
    ext = extract_extension(file)
    needs_base64_images = ext in _NATIVE_DOCUMENT_EXTENSIONS and ext != ".pdf"

    if self.include_image_base64:
      return {"include_image_base64": True}
    if needs_base64_images:
      return {"image_limit": 0}
    return {"include_image_base64": False}

  def _get_api_key(self) -> str:
    key = self.api_key or os.environ.get("MISTRAL_API_KEY")
    if not key:
      raise ValueError("Mistral API key is required. Pass api_key= or set MISTRAL_API_KEY env var.")
    return key

  def _check_mistral_file_size(self, raw: bytes) -> None:
    """Enforce both user-configured and Mistral's hard file-size limits."""
    max_size = self.config.max_file_size if self.config else None
    if max_size and len(raw) > max_size:
      raise ValueError(f"File size {len(raw)} exceeds max_file_size {max_size}")
    if len(raw) > _MISTRAL_MAX_FILE_SIZE:
      raise ValueError(f"File size {len(raw)} bytes exceeds Mistral's 50 MB limit ({_MISTRAL_MAX_FILE_SIZE} bytes)")

  @staticmethod
  def _make_data_uri(raw: bytes, mime: str) -> str:
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"

  def _build_result(self, file: File, response: Any, *, preprocessed: bool) -> ReaderOutput:
    """Convert a Mistral OCR response to a ReaderOutput."""
    pages = self._extract_pages(response)
    content = self.page_separator.join(pages)
    content, truncated = self._truncate(content)
    blocks = [
      ContentBlock(
        content_type="text",
        content=content,
        mime_type=file.mime_type,
      )
    ]
    return ReaderOutput(
      filename=get_filename(file),
      blocks=blocks,
      mime_type=file.mime_type,
      page_count=len(pages),
      word_count=len(content.split()),
      truncated=truncated,
      metadata={"provider": "mistral-ocr", "preprocessed": preprocessed},
    )

  def _build_output_from_blocks(self, file: File, blocks: List[ContentBlock]) -> ReaderOutput:
    """Build a ReaderOutput from parsed blocks."""
    text = "\n\n".join(b.as_text() for b in blocks)
    return ReaderOutput(
      filename=get_filename(file),
      blocks=blocks,
      mime_type=file.mime_type,
      word_count=len(text.split()) if text else 0,
    )

  @staticmethod
  def _extract_pages(response: Any) -> List[str]:
    """Pull markdown text from each page in a Mistral OCR response."""
    pages: List[str] = []
    if hasattr(response, "pages"):
      for page in response.pages:
        text = getattr(page, "markdown", "") or ""
        if text.strip():
          pages.append(text)
    return pages

  def _truncate(self, content: str) -> tuple[str, bool]:
    """Truncate content if max_content_length is set."""
    max_len = self.config.max_content_length if self.config else None
    if max_len and len(content) > max_len:
      return content[:max_len], True
    return content, False

  def _make_error_output(self, file: File, error: str) -> ReaderOutput:
    """Create an error ReaderOutput."""
    filename = get_filename(file)
    log_warning(f"MistralReader error for {filename}: {error}")
    return ReaderOutput(
      filename=filename,
      mime_type=file.mime_type,
      error=error,
    )

  @property
  def _encoding(self) -> str:
    return self.config.encoding if self.config else "utf-8"

  @property
  def _timeout(self) -> float | None:
    return self.config.timeout if self.config else 30.0


# Backwards-compatible alias
MistralOCRReader = MistralReader
