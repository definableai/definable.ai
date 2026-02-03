"""Knowledge base for agent memory and retrieval."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from definable.knowledge.document import Document
from definable.knowledge.embedders import Embedder
from definable.knowledge.rerankers import Reranker

if TYPE_CHECKING:
  from definable.knowledge.chunkers.base import Chunker
  from definable.knowledge.readers.base import Reader
  from definable.knowledge.vector_dbs.base import VectorDB


@dataclass
class Knowledge:
  """
  Central knowledge base orchestrating the RAG pipeline.

  Pipeline: Source → Reader → Chunker → Embedder → VectorDB
  Retrieval: Query → Embedder → VectorDB → Reranker → Results

  Example:
    from definable.knowledge import Knowledge, Document
    from definable.knowledge.vector_dbs import InMemoryVectorDB
    from definable.knowledge.embedders import VoyageAIEmbedder
    from definable.knowledge.chunkers import TextChunker

    kb = Knowledge(
      vector_db=InMemoryVectorDB(),
      embedder=VoyageAIEmbedder(api_key="..."),
      chunker=TextChunker(chunk_size=500),
    )

    # Add from file
    kb.add("./docs/readme.txt")

    # Add direct document
    kb.add(Document(content="Direct text content"))

    # Search
    results = kb.search("What is the topic?", top_k=5)
  """

  # Components
  vector_db: Optional["VectorDB"] = None
  embedder: Optional[Embedder] = None
  reranker: Optional[Reranker] = None
  chunker: Optional["Chunker"] = None
  readers: List["Reader"] = field(default_factory=list)

  # Configuration
  auto_detect_reader: bool = True

  def __post_init__(self) -> None:
    # Default to in-memory if no vector_db provided
    if self.vector_db is None:
      from definable.knowledge.vector_dbs.memory import InMemoryVectorDB
      self.vector_db = InMemoryVectorDB()

  def _require_vector_db(self) -> "VectorDB":
    if self.vector_db is None:
      raise ValueError("VectorDB required")
    return self.vector_db

  def add(
    self,
    source: Union[str, Path, Document, List[Document]],
    reader: Optional["Reader"] = None,
    chunk: bool = True,
  ) -> List[str]:
    """
    Add content to the knowledge base.

    Args:
      source: File path, URL, Document, or list of Documents
      reader: Optional reader (auto-detected if not provided)
      chunk: Whether to chunk the content

    Returns:
      List of document IDs added
    """
    # 1. Read source into documents
    documents = self._read_source(source, reader)

    # 2. Chunk if requested
    if chunk and self.chunker:
      documents = self.chunker.chunk_many(documents)

    # 3. Embed documents
    if self.embedder:
      for doc in documents:
        if doc.embedding is None:
          doc.embed(self.embedder)

    # 4. Store in vector DB
    vector_db = self._require_vector_db()
    return vector_db.add(documents)

  async def aadd(
    self,
    source: Union[str, Path, Document, List[Document]],
    reader: Optional["Reader"] = None,
    chunk: bool = True,
  ) -> List[str]:
    """Async version of add."""
    # 1. Read source into documents
    documents = await self._aread_source(source, reader)

    # 2. Chunk if requested
    if chunk and self.chunker:
      documents = self.chunker.chunk_many(documents)

    # 3. Embed documents
    if self.embedder:
      for doc in documents:
        if doc.embedding is None:
          await doc.async_embed(self.embedder)

    # 4. Store in vector DB
    vector_db = self._require_vector_db()
    return await vector_db.aadd(documents)

  def search(
    self,
    query: str,
    top_k: int = 10,
    rerank: bool = True,
    filter: Optional[Dict[str, Any]] = None,  # noqa: A002
  ) -> List[Document]:
    """
    Search the knowledge base.

    Args:
      query: Search query text
      top_k: Number of results to return
      rerank: Whether to rerank results
      filter: Optional metadata filter

    Returns:
      List of relevant documents
    """
    # 1. Embed query
    if self.embedder is None:
      raise ValueError("Embedder required for search")

    query_embedding = self.embedder.get_embedding(query)

    # 2. Vector search (fetch more if reranking)
    fetch_k = top_k * 2 if rerank and self.reranker else top_k
    vector_db = self._require_vector_db()
    results = vector_db.search(query_embedding, top_k=fetch_k, filter=filter)

    # 3. Rerank if requested
    if rerank and self.reranker and results:
      results = self.reranker.rerank(query, results)

    return results[:top_k]

  async def asearch(
    self,
    query: str,
    top_k: int = 10,
    rerank: bool = True,
    filter: Optional[Dict[str, Any]] = None,  # noqa: A002
  ) -> List[Document]:
    """Async version of search."""
    # 1. Embed query
    if self.embedder is None:
      raise ValueError("Embedder required for search")

    query_embedding = await self.embedder.async_get_embedding(query)

    # 2. Vector search
    fetch_k = top_k * 2 if rerank and self.reranker else top_k
    vector_db = self._require_vector_db()
    results = await vector_db.asearch(query_embedding, top_k=fetch_k, filter=filter)

    # 3. Rerank if requested
    if rerank and self.reranker and results:
      results = await self.reranker.arerank(query, results)

    return results[:top_k]

  def remove(self, ids: Union[str, List[str]]) -> None:
    """Remove documents by ID."""
    if isinstance(ids, str):
      ids = [ids]
    vector_db = self._require_vector_db()
    vector_db.delete(ids)

  async def aremove(self, ids: Union[str, List[str]]) -> None:
    """Async remove documents."""
    if isinstance(ids, str):
      ids = [ids]
    vector_db = self._require_vector_db()
    await vector_db.adelete(ids)

  def clear(self) -> None:
    """Clear all documents."""
    vector_db = self._require_vector_db()
    vector_db.clear()

  def __len__(self) -> int:
    vector_db = self._require_vector_db()
    return vector_db.count()

  def _read_source(
    self,
    source: Union[str, Path, Document, List[Document]],
    reader: Optional["Reader"],
  ) -> List[Document]:
    """Read source into documents."""
    if isinstance(source, Document):
      return [source]
    if isinstance(source, list) and all(isinstance(d, Document) for d in source):
      return source

    # Use provided reader
    if reader:
      return reader.read(source)

    # Auto-detect reader
    if self.auto_detect_reader:
      for r in self.readers:
        if r.can_read(source):
          return r.read(source)

    # Default: treat as text content
    return [Document(content=str(source), source_type="text")]

  async def _aread_source(
    self,
    source: Union[str, Path, Document, List[Document]],
    reader: Optional["Reader"],
  ) -> List[Document]:
    """Async read source into documents."""
    if isinstance(source, Document):
      return [source]
    if isinstance(source, list) and all(isinstance(d, Document) for d in source):
      return source

    # Use provided reader
    if reader:
      return await reader.aread(source)

    # Auto-detect reader
    if self.auto_detect_reader:
      for r in self.readers:
        if r.can_read(source):
          return await r.aread(source)

    # Default: treat as text content
    return [Document(content=str(source), source_type="text")]
