"""Pure-Python MIME type detection using magic byte signatures and file extensions.

No external dependencies — uses only stdlib.
"""

import os
import zipfile
from io import BytesIO
from urllib.parse import urlparse

# Magic byte signatures: (prefix_bytes, mime_type)
_MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
  (b"%PDF", "application/pdf"),
  (b"\x89PNG\r\n\x1a\n", "image/png"),
  (b"\xff\xd8\xff", "image/jpeg"),
  (b"GIF87a", "image/gif"),
  (b"GIF89a", "image/gif"),
  (b"{\\rtf", "text/rtf"),
  (b"RIFF", "_riff"),  # Needs subtype check (WAV, WEBP)
  (b"\x49\x49\x2a\x00", "image/tiff"),  # TIFF little-endian
  (b"\x4d\x4d\x00\x2a", "image/tiff"),  # TIFF big-endian
  (b"BM", "image/bmp"),
  (b"\xff\xfb", "audio/mpeg"),  # MP3 frame sync
  (b"\xff\xf3", "audio/mpeg"),  # MP3 frame sync
  (b"\xff\xf2", "audio/mpeg"),  # MP3 frame sync
  (b"ID3", "audio/mpeg"),  # MP3 with ID3 tag
  (b"OggS", "audio/ogg"),
  (b"fLaC", "audio/flac"),
  (b"PK\x03\x04", "_zip"),  # Needs subtype check (DOCX, XLSX, PPTX, ODS)
]

# Extension-to-MIME mapping
_EXTENSION_TO_MIME: dict[str, str] = {
  # Documents
  ".pdf": "application/pdf",
  ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ".ods": "application/vnd.oasis.opendocument.spreadsheet",
  ".rtf": "text/rtf",
  # Text
  ".txt": "text/plain",
  ".md": "text/markdown",
  ".csv": "text/csv",
  ".json": "application/json",
  ".xml": "application/xml",
  ".html": "text/html",
  ".htm": "text/html",
  ".yaml": "application/x-yaml",
  ".yml": "application/x-yaml",
  ".toml": "application/toml",
  ".ini": "text/plain",
  ".cfg": "text/plain",
  ".conf": "text/plain",
  ".rst": "text/x-rst",
  ".log": "text/plain",
  # Code
  ".py": "text/x-python",
  ".js": "text/javascript",
  ".ts": "text/javascript",
  ".jsx": "text/javascript",
  ".tsx": "text/javascript",
  ".java": "text/x-java",
  ".c": "text/x-c",
  ".cpp": "text/x-c++",
  ".h": "text/x-c",
  ".hpp": "text/x-c++",
  ".cs": "text/x-csharp",
  ".go": "text/x-go",
  ".rs": "text/x-rust",
  ".rb": "text/x-ruby",
  ".php": "text/x-php",
  ".swift": "text/x-swift",
  ".kt": "text/x-kotlin",
  ".scala": "text/x-scala",
  ".r": "text/x-r",
  ".sql": "text/x-sql",
  ".sh": "text/x-sh",
  ".bash": "text/x-sh",
  ".zsh": "text/x-sh",
  ".ps1": "text/x-powershell",
  ".bat": "text/x-bat",
  ".cmd": "text/x-bat",
  ".lua": "text/x-lua",
  ".perl": "text/x-perl",
  ".pl": "text/x-perl",
  ".css": "text/css",
  ".scss": "text/css",
  ".sass": "text/css",
  ".less": "text/css",
  ".vue": "text/html",
  ".svelte": "text/html",
  # Images
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".bmp": "image/bmp",
  ".tiff": "image/tiff",
  ".tif": "image/tiff",
  ".webp": "image/webp",
  ".avif": "image/avif",
  ".heic": "image/heic",
  ".svg": "image/svg+xml",
  # Audio
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".ogg": "audio/ogg",
  ".flac": "audio/flac",
  ".m4a": "audio/mp4",
  ".webm": "audio/webm",
}

# ZIP internal markers for distinguishing Office formats
_ZIP_CONTENT_TYPES = {
  "word/": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "ppt/": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "xl/": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def detect_from_bytes(data: bytes) -> str | None:
  """Detect MIME type from magic byte signatures."""
  if len(data) < 4:
    return None

  for sig, mime in _MAGIC_SIGNATURES:
    if data[: len(sig)] == sig:
      if mime == "_riff":
        return _detect_riff_subtype(data)
      if mime == "_zip":
        return _detect_zip_subtype(data)
      return mime

  return None


def detect_from_extension(
  filename: str | None = None,
  filepath: str | None = None,
  url: str | None = None,
) -> str | None:
  """Detect MIME type from file extension."""
  ext = extract_extension(filename, filepath, url)
  if ext:
    return _EXTENSION_TO_MIME.get(ext)
  return None


def detect(
  data: bytes | None = None,
  filename: str | None = None,
  filepath: str | None = None,
  url: str | None = None,
  mime_type: str | None = None,
) -> str | None:
  """Combined detection: explicit > bytes > extension.

  Returns the most reliable MIME type available.
  """
  # Explicit MIME type takes priority
  if mime_type:
    return mime_type

  # Try magic bytes
  if data:
    result = detect_from_bytes(data)
    if result:
      return result

  # Fall back to extension
  return detect_from_extension(filename, filepath, url)


def extract_extension(
  filename: str | None = None,
  filepath: str | None = None,
  url: str | None = None,
) -> str | None:
  """Extract the lowercase file extension from available sources."""
  if filename:
    _, ext = os.path.splitext(filename)
    if ext:
      return ext.lower()

  if filepath:
    _, ext = os.path.splitext(str(filepath))
    if ext:
      return ext.lower()

  if url:
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    if ext:
      return ext.lower()

  return None


def _detect_riff_subtype(data: bytes) -> str | None:
  """Detect RIFF subtypes: WAV, WEBP, AVI."""
  if len(data) < 12:
    return None
  subtype = data[8:12]
  if subtype == b"WAVE":
    return "audio/wav"
  if subtype == b"WEBP":
    return "image/webp"
  if subtype == b"AVI ":
    return "video/avi"
  return "application/octet-stream"


def _detect_zip_subtype(data: bytes) -> str:
  """Inspect ZIP contents to distinguish DOCX/XLSX/PPTX/ODS."""
  try:
    with zipfile.ZipFile(BytesIO(data)) as zf:
      names = zf.namelist()
      # Check for ODS
      if "content.xml" in names and "META-INF/manifest.xml" in names:
        return "application/vnd.oasis.opendocument.spreadsheet"
      # Check for Office Open XML
      for name in names:
        for prefix, mime in _ZIP_CONTENT_TYPES.items():
          if name.startswith(prefix):
            return mime
  except (zipfile.BadZipFile, OSError):
    pass
  return "application/zip"
