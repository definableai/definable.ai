from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, List, Optional

from definable.knowledge.embedders import Embedder


def _patch_init_for_metadata_alias(cls: type) -> type:
  """Wrap the dataclass __init__ to accept both ``metadata`` and ``meta_data``.

  If the caller passes ``meta_data=...``, it is transparently forwarded to
  the canonical ``metadata`` parameter so existing code keeps working.
  Passing both raises a ``TypeError``.
  """
  original_init = cls.__init__  # type: ignore[misc]

  @wraps(original_init)
  def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
    if "meta_data" in kwargs:
      if "metadata" in kwargs:
        raise TypeError("Document() got both 'metadata' and 'meta_data'. Use 'metadata' only.")
      # Silently accept the legacy spelling
      kwargs["metadata"] = kwargs.pop("meta_data")
    original_init(self, *args, **kwargs)

  cls.__init__ = __init__  # type: ignore[misc, attr-defined]
  return cls


@_patch_init_for_metadata_alias
@dataclass
class Document:
  """Dataclass for managing a document.

  Args:
    content: The text content of the document.
    id: Optional unique identifier.
    name: Optional human-readable name.
    metadata: Arbitrary metadata dict for filtering and display.
      For backwards compatibility, ``meta_data`` is also accepted as
      an alias for this parameter.
    embedder: Optional embedder instance.
    embedding: Pre-computed embedding vector.
    usage: Token usage information from embedding.
    reranking_score: Score assigned by a reranker.
    content_id: Content identifier for deduplication.
    content_origin: Origin of the content.
    size: Size of the content in characters.
    source: Original source (file path, URL).
    source_type: Type of source ("file", "url", "text").
    chunk_index: Position in chunked sequence.
    chunk_total: Total chunks from same source.
    parent_id: ID of parent document (if chunked).
  """

  content: str
  id: Optional[str] = None
  name: Optional[str] = None
  metadata: Dict[str, Any] = field(default_factory=dict)
  embedder: Optional["Embedder"] = None
  embedding: Optional[List[float]] = None
  usage: Optional[Dict[str, Any]] = None
  reranking_score: Optional[float] = None
  content_id: Optional[str] = None
  content_origin: Optional[str] = None
  size: Optional[int] = None

  # Knowledge pipeline fields
  source: Optional[str] = None  # Original source (file path, URL)
  source_type: Optional[str] = None  # "file", "url", "text"
  chunk_index: Optional[int] = None  # Position in chunked sequence
  chunk_total: Optional[int] = None  # Total chunks from same source
  parent_id: Optional[str] = None  # ID of parent document (if chunked)

  @property
  def meta_data(self) -> Dict[str, Any]:
    """Backwards-compatible alias for :attr:`metadata`."""
    return self.metadata

  @meta_data.setter
  def meta_data(self, value: Dict[str, Any]) -> None:
    """Backwards-compatible setter for :attr:`metadata`."""
    self.metadata = value

  def embed(self, embedder: Optional[Embedder] = None) -> None:
    """Embed the document using the provided embedder."""
    _embedder = embedder or self.embedder
    if _embedder is None:
      raise ValueError("No embedder provided")

    self.embedding, self.usage = _embedder.get_embedding_and_usage(self.content)

  async def async_embed(self, embedder: Optional[Embedder] = None) -> None:
    """Embed the document using the provided embedder."""
    _embedder = embedder or self.embedder
    if _embedder is None:
      raise ValueError("No embedder provided")
    self.embedding, self.usage = await _embedder.async_get_embedding_and_usage(self.content)

  def to_dict(self) -> Dict[str, Any]:
    """Returns a dictionary representation of the document."""
    fields = {"name", "metadata", "content"}
    return {
      f: getattr(self, f)
      for f in fields
      if getattr(self, f) is not None or f == "content"  # content is always included
    }

  @classmethod
  def from_dict(cls, document: Dict[str, Any]) -> "Document":
    """Returns a Document object from a dictionary representation.

    Accepts both ``metadata`` and ``meta_data`` keys.
    """
    return cls(**document)

  @classmethod
  def from_json(cls, document: str) -> "Document":
    """Returns a Document object from a json string representation."""
    import json

    return cls(**json.loads(document))
