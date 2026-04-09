from typing import TYPE_CHECKING

from definable.knowledge.base import Knowledge
from definable.knowledge.document import Document
from definable.knowledge.reader import Reader, ReaderConfig

if TYPE_CHECKING:
  from definable.knowledge.embedder.fallback import FallbackEmbedder
  from definable.knowledge.fts.hybrid import HybridSearchConfig
  from definable.knowledge.fts.index import FTSIndex
  from definable.knowledge.scoring.mmr import MMRConfig
  from definable.knowledge.scoring.temporal import TemporalDecay

__all__ = [
  # Core — these belong in knowledge
  "Knowledge",
  "Document",
  "Reader",
  "ReaderConfig",
  # Scoring & diversity
  "TemporalDecay",  # noqa: F822
  "MMRConfig",  # noqa: F822
  # Hybrid search
  "FTSIndex",  # noqa: F822
  "HybridSearchConfig",  # noqa: F822
  # Embedder fallback
  "FallbackEmbedder",  # noqa: F822
]


def __getattr__(name: str):
  # --- Scoring & diversity ---
  if name == "TemporalDecay":
    from definable.knowledge.scoring.temporal import TemporalDecay

    return TemporalDecay
  if name == "MMRConfig":
    from definable.knowledge.scoring.mmr import MMRConfig

    return MMRConfig

  # --- Hybrid search ---
  if name == "FTSIndex":
    from definable.knowledge.fts.index import FTSIndex

    return FTSIndex
  if name == "HybridSearchConfig":
    from definable.knowledge.fts.hybrid import HybridSearchConfig

    return HybridSearchConfig

  # --- Embedder fallback ---
  if name == "FallbackEmbedder":
    from definable.knowledge.embedder.fallback import FallbackEmbedder

    return FallbackEmbedder

  # --- Readers ---
  if name == "TextReader":
    from definable.knowledge.reader.text import TextReader

    return TextReader
  if name == "PDFReader":
    from definable.knowledge.reader.pdf import PDFReader

    return PDFReader
  if name == "URLReader":
    from definable.knowledge.reader.url import URLReader

    return URLReader

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
