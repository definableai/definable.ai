from typing import TYPE_CHECKING

from definable.knowledge.base import Knowledge
from definable.knowledge.document import Document
from definable.knowledge.reader import Reader, ReaderConfig

if TYPE_CHECKING:
  from definable.knowledge.chunker import Chunker
  from definable.knowledge.chunker.recursive import RecursiveChunker
  from definable.knowledge.chunker.text import TextChunker
  from definable.knowledge.embedder import Embedder
  from definable.knowledge.embedder.openai import OpenAIEmbedder
  from definable.knowledge.embedder.voyageai import VoyageAIEmbedder
  from definable.knowledge.reader.pdf import PDFReader
  from definable.knowledge.reader.text import TextReader
  from definable.knowledge.reader.url import URLReader
  from definable.knowledge.reranker import Reranker
  from definable.knowledge.reranker.cohere import CohereReranker
  from definable.vectordb import VectorDB

__all__ = [
  # Core — these belong in knowledge
  "Knowledge",
  "Document",
  "Reader",
  "ReaderConfig",
  # Deprecated re-exports — use definable.embedder, definable.chunker, definable.reranker, definable.vectordb
  "Embedder",
  "OpenAIEmbedder",
  "VoyageAIEmbedder",
  "Chunker",
  "RecursiveChunker",
  "TextChunker",
  "Reranker",
  "CohereReranker",
  "VectorDB",
  "PDFReader",
  "TextReader",
  "URLReader",
]


def __getattr__(name: str):
  import warnings

  # --- Embedders (deprecated — use definable.embedder) ---
  if name == "Embedder":
    warnings.warn(
      "Importing Embedder from definable.knowledge is deprecated. Use 'from definable.embedder import Embedder'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.knowledge.embedder import Embedder

    return Embedder
  if name == "OpenAIEmbedder":
    warnings.warn(
      "Importing OpenAIEmbedder from definable.knowledge is deprecated. Use 'from definable.embedder import OpenAIEmbedder'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.knowledge.embedder.openai import OpenAIEmbedder

    return OpenAIEmbedder
  if name == "VoyageAIEmbedder":
    warnings.warn(
      "Importing VoyageAIEmbedder from definable.knowledge is deprecated. Use 'from definable.embedder import VoyageAIEmbedder'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.knowledge.embedder.voyageai import VoyageAIEmbedder

    return VoyageAIEmbedder

  # --- Chunkers (deprecated — use definable.chunker) ---
  if name == "Chunker":
    warnings.warn(
      "Importing Chunker from definable.knowledge is deprecated. Use 'from definable.chunker import Chunker'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.knowledge.chunker import Chunker

    return Chunker
  if name == "TextChunker":
    warnings.warn(
      "Importing TextChunker from definable.knowledge is deprecated. Use 'from definable.chunker import TextChunker'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.knowledge.chunker.text import TextChunker

    return TextChunker
  if name == "RecursiveChunker":
    warnings.warn(
      "Importing RecursiveChunker from definable.knowledge is deprecated. Use 'from definable.chunker import RecursiveChunker'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.knowledge.chunker.recursive import RecursiveChunker

    return RecursiveChunker

  # --- Rerankers (deprecated — use definable.reranker) ---
  if name == "Reranker":
    warnings.warn(
      "Importing Reranker from definable.knowledge is deprecated. Use 'from definable.reranker import Reranker'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.knowledge.reranker import Reranker

    return Reranker
  if name == "CohereReranker":
    warnings.warn(
      "Importing CohereReranker from definable.knowledge is deprecated. Use 'from definable.reranker import CohereReranker'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.knowledge.reranker.cohere import CohereReranker

    return CohereReranker

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

  # --- VectorDBs (deprecated — use definable.vectordb) ---
  if name == "VectorDB":
    warnings.warn(
      "Importing VectorDB from definable.knowledge is deprecated. Use 'from definable.vectordb import VectorDB'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.vectordb.base import VectorDB

    return VectorDB
  if name == "InMemoryVectorDB":
    warnings.warn(
      "Importing InMemoryVectorDB from definable.knowledge is deprecated. Use 'from definable.vectordb import InMemoryVectorDB'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.vectordb.memory import InMemoryVectorDB

    return InMemoryVectorDB
  if name == "PgVectorDB":
    warnings.warn(
      "Importing PgVectorDB from definable.knowledge is deprecated. Use 'from definable.vectordb import PgVector'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.vectordb.pgvector import PgVector

    return PgVector
  if name == "QdrantVectorDB":
    warnings.warn(
      "Importing QdrantVectorDB from definable.knowledge is deprecated. Use 'from definable.vectordb import Qdrant'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.vectordb.qdrant import Qdrant

    return Qdrant
  if name == "ChromaVectorDB":
    warnings.warn(
      "Importing ChromaVectorDB from definable.knowledge is deprecated. Use 'from definable.vectordb import ChromaDb'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.vectordb.chroma import ChromaDb

    return ChromaDb
  if name == "PineconeVectorDB":
    warnings.warn(
      "Importing PineconeVectorDB from definable.knowledge is deprecated. Use 'from definable.vectordb import PineconeDb'.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.vectordb.pineconedb import PineconeDb

    return PineconeDb

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
