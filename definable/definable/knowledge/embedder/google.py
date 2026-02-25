"""Google AI (Gemini) embedder implementation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from definable.knowledge.embedder.base import Embedder
from definable.utils.log import log_warning


@dataclass
class GoogleEmbedder(Embedder):
  """Embedder using Google's Gemini embedding models.

  Args:
      id: Model ID. Default "text-embedding-004".
      dimensions: Output dimensions. Default 768.
      api_key: Google AI API key. Falls back to GOOGLE_API_KEY env var.
      task_type: Task type hint for the model. One of:
                 "retrieval_document", "retrieval_query",
                 "semantic_similarity", "classification", "clustering".
      batch_size: Batch size for bulk embedding. Default 100.

  Example::

      from definable.embedder import GoogleEmbedder
      embedder = GoogleEmbedder(api_key="your-key")
      vector = embedder.get_embedding("Hello world")
  """

  id: str = "text-embedding-004"
  dimensions: int = 768
  api_key: Optional[str] = None
  task_type: Optional[str] = None
  _client: Optional[Any] = None

  @property
  def client(self) -> Any:
    if self._client is not None:
      return self._client
    try:
      from google import genai
    except ImportError:
      raise ImportError("`google-genai` not installed. Run: pip install google-genai")

    params: Dict[str, Any] = {}
    if self.api_key:
      params["api_key"] = self.api_key
    self._client = genai.Client(**params)
    return self._client

  def _build_config(self) -> Optional[Any]:
    """Build embedding config with task_type and dimensions."""
    try:
      from google.genai import types
    except ImportError:
      return None

    config: Dict[str, Any] = {}
    if self.task_type:
      config["task_type"] = self.task_type
    if self.dimensions:
      config["output_dimensionality"] = self.dimensions
    return types.EmbedContentConfig(**config) if config else None

  def get_embedding(self, text: str) -> List[float]:
    try:
      config = self._build_config()
      kwargs: Dict[str, Any] = {"model": self.id, "contents": text}
      if config:
        kwargs["config"] = config
      response = self.client.models.embed_content(**kwargs)
      return [float(x) for x in response.embeddings[0].values]
    except Exception as e:
      log_warning(f"GoogleEmbedder error: {e}")
      return []

  def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    try:
      config = self._build_config()
      kwargs: Dict[str, Any] = {"model": self.id, "contents": text}
      if config:
        kwargs["config"] = config
      response = self.client.models.embed_content(**kwargs)
      embedding = [float(x) for x in response.embeddings[0].values]
      # Google doesn't return usage info for embeddings
      return embedding, None
    except Exception as e:
      log_warning(f"GoogleEmbedder error: {e}")
      return [], None

  async def async_get_embedding(self, text: str) -> List[float]:
    try:
      config = self._build_config()
      kwargs: Dict[str, Any] = {"model": self.id, "contents": text}
      if config:
        kwargs["config"] = config
      response = await self.client.aio.models.embed_content(**kwargs)
      return [float(x) for x in response.embeddings[0].values]
    except Exception as e:
      log_warning(f"GoogleEmbedder async error: {e}")
      return []

  async def async_get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    try:
      config = self._build_config()
      kwargs: Dict[str, Any] = {"model": self.id, "contents": text}
      if config:
        kwargs["config"] = config
      response = await self.client.aio.models.embed_content(**kwargs)
      embedding = [float(x) for x in response.embeddings[0].values]
      return embedding, None
    except Exception as e:
      log_warning(f"GoogleEmbedder async error: {e}")
      return [], None
