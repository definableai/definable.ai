"""RTF parser — extracts text from Rich Text Format files using striprtf.

Requires optional dependency: ``striprtf>=0.0.26``
"""

from typing import List, Set

from definable.readers.models import ContentBlock, ReaderConfig
from definable.readers.parsers.base_parser import BaseParser


def _import_striprtf():
  try:
    from striprtf.striprtf import rtf_to_text

    return rtf_to_text
  except ImportError as e:
    raise ImportError(
      "RtfParser requires the 'striprtf' package. Install it with: pip install 'striprtf>=0.0.26' or: pip install 'definable-ai[readers]'"
    ) from e


class RtfParser(BaseParser):
  """Parses RTF files using striprtf (local, no API calls).

  Decodes raw bytes and strips RTF markup to produce clean plain text.
  """

  def __init__(self) -> None:
    _import_striprtf()

  def supported_mime_types(self) -> List[str]:
    return ["text/rtf", "application/rtf"]

  def supported_extensions(self) -> Set[str]:
    return {".rtf"}

  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    rtf_to_text = _import_striprtf()
    encoding = config.encoding if config else "utf-8"
    rtf_content = data.decode(encoding, errors="replace")
    text = rtf_to_text(rtf_content)
    return [
      ContentBlock(
        content_type="text",
        content=text,
        mime_type="text/rtf",
      )
    ]
