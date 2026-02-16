from definable.knowledge.base import Knowledge
from definable.knowledge.chunkers import Chunker
from definable.knowledge.document import Document
from definable.knowledge.embedders import Embedder
from definable.knowledge.readers import Reader, ReaderConfig
from definable.knowledge.rerankers import Reranker
from definable.knowledge.vector_dbs import VectorDB

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.knowledge.chunkers.recursive import RecursiveChunker
  from definable.knowledge.chunkers.text import TextChunker
  from definable.knowledge.embedders.voyageai import VoyageAIEmbedder
  from definable.knowledge.readers.pdf import PDFReader
  from definable.knowledge.readers.text import TextReader
  from definable.knowledge.readers.url import URLReader
  from definable.knowledge.rerankers.cohere import CohereReranker
  from definable.knowledge.vector_dbs.memory import InMemoryVectorDB
  from definable.knowledge.vector_dbs.pgvector import PgVectorDB

__all__ = [
  # Core
  "Knowledge",
  "Document",
  # Base classes
  "Embedder",
  "Reranker",
  "Reader",
  "ReaderConfig",
  "Chunker",
  "VectorDB",
  # Implementations (lazy-loaded)
  "CohereReranker",
  "InMemoryVectorDB",
  "PDFReader",
  "PgVectorDB",
  "RecursiveChunker",
  "TextChunker",
  "TextReader",
  "URLReader",
  "VoyageAIEmbedder",
]


def __getattr__(name: str):
  # Embedder implementations
  if name == "VoyageAIEmbedder":
    from definable.knowledge.embedders.voyageai import VoyageAIEmbedder

    return VoyageAIEmbedder

  # Reranker implementations
  if name == "CohereReranker":
    from definable.knowledge.rerankers.cohere import CohereReranker

    return CohereReranker

  # Reader implementations
  if name == "TextReader":
    from definable.knowledge.readers.text import TextReader

    return TextReader
  if name == "PDFReader":
    from definable.knowledge.readers.pdf import PDFReader

    return PDFReader
  if name == "URLReader":
    from definable.knowledge.readers.url import URLReader

    return URLReader

  # Chunker implementations
  if name == "TextChunker":
    from definable.knowledge.chunkers.text import TextChunker

    return TextChunker
  if name == "RecursiveChunker":
    from definable.knowledge.chunkers.recursive import RecursiveChunker

    return RecursiveChunker

  # VectorDB implementations
  if name == "InMemoryVectorDB":
    from definable.knowledge.vector_dbs.memory import InMemoryVectorDB

    return InMemoryVectorDB
  if name == "PgVectorDB":
    from definable.knowledge.vector_dbs.pgvector import PgVectorDB

    return PgVectorDB

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
