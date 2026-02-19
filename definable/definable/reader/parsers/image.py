"""Image parser — passthrough for image files.

Packages raw image bytes as a ContentBlock for downstream consumers
(vision models, image processing pipelines). No parsing is performed.
"""

from typing import List, Set

from definable.reader.models import ContentBlock, ReaderConfig
from definable.reader.parsers.base_parser import BaseParser


class ImageParser(BaseParser):
  """Passthrough parser for image files.

  Wraps raw image bytes in an image ContentBlock. Downstream consumers
  (vision models, etc.) handle the actual image processing.
  """

  def supported_mime_types(self) -> List[str]:
    return [
      "image/png",
      "image/jpeg",
      "image/gif",
      "image/bmp",
      "image/tiff",
      "image/webp",
      "image/avif",
      "image/heic",
      "image/svg+xml",
    ]

  def supported_extensions(self) -> Set[str]:
    return {
      ".png",
      ".jpg",
      ".jpeg",
      ".gif",
      ".bmp",
      ".tiff",
      ".tif",
      ".webp",
      ".avif",
      ".heic",
      ".svg",
    }

  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    return [
      ContentBlock(
        content_type="image",
        content=data,
        mime_type=mime_type or "image/png",
      )
    ]
