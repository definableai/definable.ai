"""Cohere reranker implementation."""

import asyncio
from typing import Any, Dict, List, Optional

from definable.knowledge.document import Document
from definable.knowledge.rerankers.base import Reranker
from definable.utils.log import logger

try:
  from cohere import Client as CohereClient
except ImportError:
  raise ImportError("cohere not installed, please run pip install cohere")


class CohereReranker(Reranker):
  """Reranker implementation using Cohere's rerank API."""

  model: str = "rerank-multilingual-v3.0"
  api_key: Optional[str] = None
  cohere_client: Optional[CohereClient] = None
  top_n: Optional[int] = None

  @property
  def client(self) -> CohereClient:
    """Get or create the Cohere client (cached)."""
    if self.cohere_client is not None:
      return self.cohere_client

    _client_params: Dict[str, Any] = {}
    if self.api_key:
      _client_params["api_key"] = self.api_key

    # Cache the client using object.__setattr__ for Pydantic frozen models
    client = CohereClient(**_client_params)
    object.__setattr__(self, "cohere_client", client)
    return client

  def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
    """Internal rerank implementation."""
    # Validate input documents
    if not documents:
      return []

    top_n = self.top_n
    if top_n is not None and top_n <= 0:
      logger.warning(f"top_n should be a positive integer, got {self.top_n}, setting top_n to None")
      top_n = None

    compressed_docs: List[Document] = []
    _docs = [doc.content for doc in documents]
    response = self.client.rerank(query=query, documents=_docs, model=self.model)

    for r in response.results:
      doc = documents[r.index]
      doc.reranking_score = r.relevance_score
      compressed_docs.append(doc)

    # Order by relevance score
    compressed_docs.sort(
      key=lambda x: x.reranking_score if x.reranking_score is not None else float("-inf"),
      reverse=True,
    )

    # Limit to top_n if specified
    if top_n:
      compressed_docs = compressed_docs[:top_n]

    return compressed_docs

  def rerank(self, query: str, documents: List[Document]) -> List[Document]:
    """Rerank documents by relevance to query."""
    try:
      return self._rerank(query=query, documents=documents)
    except Exception as e:
      logger.error(f"Error reranking documents: {e}. Returning original documents")
      return documents

  async def arerank(self, query: str, documents: List[Document]) -> List[Document]:
    """Async rerank documents by relevance to query."""
    try:
      loop = asyncio.get_event_loop()
      return await loop.run_in_executor(None, self._rerank, query, documents)
    except Exception as e:
      logger.error(f"Error async reranking documents: {e}. Returning original documents")
      return documents
