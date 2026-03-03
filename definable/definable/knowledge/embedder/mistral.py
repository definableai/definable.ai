"""Mistral AI embedder implementation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from definable.knowledge.embedder.base import Embedder
from definable.utils.log import log_warning


@dataclass
class MistralEmbedder(Embedder):
  """Embedder using Mistral AI's embedding models.

  Args:
      id: Mistral model ID. Default "mistral-embed".
      dimensions: Output embedding dimensions. Default 1024.
      api_key: Mistral API key. Falls back to MISTRAL_API_KEY env var.
      batch_size: Batch size for bulk embedding. Default 100.

  Example::

      from definable.embedder import MistralEmbedder
      embedder = MistralEmbedder(api_key="your-key")
      vector = embedder.get_embedding("Hello world")
  """

  id: str = "mistral-embed"
  dimensions: int = 1024
  api_key: Optional[str] = None
  _client: Optional[Any] = None
  _async_client: Optional[Any] = None

  @property
  def client(self) -> Any:
    if self._client is not None:
      return self._client
    try:
      from mistralai import Mistral
    except ImportError:
      raise ImportError("`mistralai` not installed. Run: pip install mistralai")

    params: Dict[str, Any] = {}
    if self.api_key:
      params["api_key"] = self.api_key
    self._client = Mistral(**params)
    return self._client

  def get_embedding(self, text: str) -> List[float]:
    response = self.client.embeddings.create(model=self.id, inputs=[text])
    return [float(x) for x in response.data[0].embedding]

  def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    response = self.client.embeddings.create(model=self.id, inputs=[text])
    embedding = [float(x) for x in response.data[0].embedding]
    usage = None
    if hasattr(response, "usage") and response.usage:
      usage = {"prompt_tokens": response.usage.prompt_tokens, "total_tokens": response.usage.total_tokens}
    return embedding, usage

  async def async_get_embedding(self, text: str) -> List[float]:
    response = await self.client.embeddings.create_async(model=self.id, inputs=[text])
    return [float(x) for x in response.data[0].embedding]

  async def async_get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    response = await self.client.embeddings.create_async(model=self.id, inputs=[text])
    embedding = [float(x) for x in response.data[0].embedding]
    usage = None
    if hasattr(response, "usage") and response.usage:
      usage = {"prompt_tokens": response.usage.prompt_tokens, "total_tokens": response.usage.total_tokens}
    return embedding, usage
