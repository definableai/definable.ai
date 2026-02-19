"""Knowledge base for agent memory and retrieval."""

import json
from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, cast

from definable.knowledge.document import Document
from definable.knowledge.embedder import Embedder
from definable.knowledge.reranker import Reranker

if TYPE_CHECKING:
  from definable.knowledge.chunker.base import Chunker
  from definable.knowledge.reader.base import Reader
  from definable.model.message import Message
  from definable.vectordb.base import VectorDB


@dataclass
class Knowledge:
  """
  Central knowledge base orchestrating the RAG pipeline.

  Knowledge is a composable lego block — it works standalone or snaps
  directly into an Agent without a config wrapper.

  Pipeline: Source -> Reader -> Chunker -> Embed -> VectorDB.insert (stores)
  Retrieval: Query -> VectorDB.search (embeds query + searches) -> Reranker -> Results

  Knowledge embeds documents before insert using the embedder (its own or
  the VectorDB's). The VectorDB's embedder is used only for search query
  embedding. For backward compatibility, if an ``embedder`` is passed to
  Knowledge, it takes priority over the VectorDB's embedder.

  Example:
    from definable.knowledge import Knowledge, Document
    from definable.vectordb import InMemoryVectorDB
    from definable.knowledge.embedder.voyageai import VoyageAIEmbedder

    kb = Knowledge(
      vector_db=InMemoryVectorDB(embedder=VoyageAIEmbedder(api_key="...")),
      chunker=TextChunker(chunk_size=500),
      top_k=5,
      trigger="always",
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

  # Agent integration fields (absorbed from KnowledgeConfig)
  top_k: int = 5
  rerank: bool = True
  min_score: Optional[float] = None
  context_format: Literal["xml", "markdown", "json"] = "xml"
  context_position: Literal["system", "before_user"] = "system"
  query_from: Literal["last_user", "full_conversation"] = "last_user"
  max_query_length: int = 500
  enabled: bool = True
  trigger: Literal["always", "auto", "never"] = "always"
  decision_prompt: Optional[str] = None
  description: Optional[str] = None
  routing_model: Optional[Any] = None

  def __post_init__(self) -> None:
    # Default to in-memory if no vector_db provided
    if self.vector_db is None:
      from definable.vectordb.memory import InMemoryVectorDB

      self.vector_db = InMemoryVectorDB()

    # Backward compat: pass embedder to vector_db if provided
    if self.embedder is not None and self.vector_db.embedder is None:
      self.vector_db.embedder = self.embedder

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

    # 3. Embed documents that don't already have embeddings
    vector_db = self._require_vector_db()
    embedder = self.embedder or (vector_db.embedder if vector_db else None)
    if embedder:
      for doc in documents:
        if doc.embedding is None:
          doc.embed(embedder)

    # 4. Generate content hash from combined document content
    combined = "".join(doc.content for doc in documents)
    content_hash = md5(combined.encode()).hexdigest()

    # 5. Ensure table/collection exists, then insert (skip duplicates)
    vector_db.create()
    if vector_db.content_hash_exists(content_hash):
      return [doc.id for doc in documents if doc.id is not None]
    if vector_db.upsert_available():
      vector_db.upsert(content_hash, documents)
    else:
      vector_db.insert(content_hash, documents)

    # Return document IDs for backward compat
    return [doc.id for doc in documents if doc.id is not None]

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

    # 3. Embed documents that don't already have embeddings
    vector_db = self._require_vector_db()
    embedder = self.embedder or (vector_db.embedder if vector_db else None)
    if embedder:
      for doc in documents:
        if doc.embedding is None:
          await doc.async_embed(embedder)

    # 4. Generate content hash from combined document content
    combined = "".join(doc.content for doc in documents)
    content_hash = md5(combined.encode()).hexdigest()

    # 5. Ensure table/collection exists, then insert (skip duplicates)
    await vector_db.async_create()
    if vector_db.content_hash_exists(content_hash):
      return [doc.id for doc in documents if doc.id is not None]
    if vector_db.upsert_available():
      await vector_db.async_upsert(content_hash, documents)
    else:
      await vector_db.ainsert(content_hash, documents)

    return [doc.id for doc in documents if doc.id is not None]

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
    # 1. VectorDB handles embedding + search
    fetch_k = top_k * 2 if rerank and self.reranker else top_k
    vector_db = self._require_vector_db()
    results = vector_db.search(query, limit=fetch_k, filters=filter)

    # 2. Rerank if requested
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
    # 1. VectorDB handles embedding + search
    fetch_k = top_k * 2 if rerank and self.reranker else top_k
    vector_db = self._require_vector_db()
    results = await vector_db.asearch(query, limit=fetch_k, filters=filter)

    # 2. Rerank if requested
    if rerank and self.reranker and results:
      results = await self.reranker.arerank(query, results)

    return results[:top_k]

  def remove(self, ids: Union[str, List[str]]) -> None:
    """Remove documents by ID."""
    if isinstance(ids, str):
      ids = [ids]
    vector_db = self._require_vector_db()
    for _id in ids:
      vector_db.delete_by_id(_id)

  async def aremove(self, ids: Union[str, List[str]]) -> None:
    """Async remove documents."""
    if isinstance(ids, str):
      ids = [ids]
    vector_db = self._require_vector_db()
    for _id in ids:
      vector_db.delete_by_id(_id)

  def clear(self) -> None:
    """Clear all documents."""
    vector_db = self._require_vector_db()
    vector_db.delete()

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

    # At this point, source is str | Path (Document and list[Document] handled above)
    src = cast(Union[str, Path], source)

    # Use provided reader
    if reader:
      return reader.read(src)

    # Auto-detect reader
    if self.auto_detect_reader:
      for r in self.readers:
        if r.can_read(src):
          return r.read(src)

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

    # At this point, source is str | Path (Document and list[Document] handled above)
    src = cast(Union[str, Path], source)

    # Use provided reader
    if reader:
      return await reader.aread(src)

    # Auto-detect reader
    if self.auto_detect_reader:
      for r in self.readers:
        if r.can_read(src):
          return await r.aread(src)

    # Default: treat as text content
    return [Document(content=str(source), source_type="text")]

  # --- Agent Integration Helpers ---

  def extract_query(self, messages: List["Message"]) -> Optional[str]:
    """Extract search query from conversation messages.

    Uses ``query_from`` and ``max_query_length`` settings.

    Args:
      messages: List of conversation messages.

    Returns:
      Query string or None if no user message found.
    """
    if self.query_from == "last_user":
      for msg in reversed(messages):
        if msg.role == "user" and msg.content:
          content = msg.content if isinstance(msg.content, str) else str(msg.content)
          return content[: self.max_query_length]
      return None
    elif self.query_from == "full_conversation":
      user_contents: List[str] = []
      for msg in messages:
        if msg.role == "user" and msg.content:
          content = msg.content if isinstance(msg.content, str) else str(msg.content)
          user_contents.append(content)
      if user_contents:
        full_query = " ".join(user_contents)
        return full_query[: self.max_query_length]
      return None
    return None  # type: ignore[unreachable]

  def format_context(self, documents: List[Document]) -> str:
    """Format retrieved documents as a context string.

    Uses ``context_format`` setting to choose XML, Markdown, or JSON.

    Args:
      documents: List of retrieved documents.

    Returns:
      Formatted context string.
    """
    if self.context_format == "markdown":
      return self._format_markdown(documents)
    elif self.context_format == "json":
      return self._format_json(documents)
    return self._format_xml(documents)

  def _format_xml(self, documents: List[Document]) -> str:
    """Format as XML-style context block."""
    lines = ["<knowledge_context>"]
    for i, doc in enumerate(documents):
      attrs = [f'index="{i + 1}"']
      if doc.reranking_score is not None:
        attrs.append(f'relevance="{doc.reranking_score:.3f}"')
      if doc.source:
        attrs.append(f'source="{doc.source}"')
      if doc.name:
        attrs.append(f'name="{doc.name}"')
      attr_str = " ".join(attrs)
      lines.append(f"  <document {attr_str}>")
      lines.append(f"    {doc.content}")
      lines.append("  </document>")
    lines.append("</knowledge_context>")
    return "\n".join(lines)

  def _format_markdown(self, documents: List[Document]) -> str:
    """Format as Markdown."""
    lines = ["## Retrieved Context\n"]
    for i, doc in enumerate(documents):
      score_str = f" (relevance: {doc.reranking_score:.3f})" if doc.reranking_score else ""
      name_str = f" - {doc.name}" if doc.name else ""
      lines.append(f"### Document {i + 1}{name_str}{score_str}")
      lines.append("")
      lines.append(doc.content)
      if doc.source:
        lines.append(f"\n*Source: {doc.source}*")
      lines.append("")
    return "\n".join(lines)

  def _format_json(self, documents: List[Document]) -> str:
    """Format as JSON."""
    data = []
    for doc in documents:
      entry: Dict[str, Any] = {"content": doc.content}
      if doc.reranking_score is not None:
        entry["relevance"] = doc.reranking_score
      if doc.source:
        entry["source"] = doc.source
      if doc.name:
        entry["name"] = doc.name
      data.append(entry)
    return json.dumps(data, indent=2)
