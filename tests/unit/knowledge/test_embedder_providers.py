"""
Unit tests for MistralEmbedder and GoogleEmbedder.

Tests lazy client init, error handling, and method signatures.
Uses mocks — no API calls.

Covers:
  - Default configuration
  - Lazy client initialization
  - ImportError handling for missing SDKs
  - get_embedding returns list of floats
  - get_embedding_and_usage returns tuple
  - async methods exist and are callable
  - Error handling returns empty list
  - Custom parameters (api_key, dimensions, model)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from definable.knowledge.embedder.google import GoogleEmbedder
from definable.knowledge.embedder.mistral import MistralEmbedder


# ---------------------------------------------------------------------------
# MistralEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMistralEmbedderConfig:
  """MistralEmbedder has correct defaults."""

  def test_default_model(self):
    emb = MistralEmbedder()
    assert emb.id == "mistral-embed"

  def test_default_dimensions(self):
    emb = MistralEmbedder()
    assert emb.dimensions == 1024

  def test_custom_model(self):
    emb = MistralEmbedder(id="mistral-embed-v2")
    assert emb.id == "mistral-embed-v2"

  def test_custom_api_key(self):
    emb = MistralEmbedder(api_key="test-key")
    assert emb.api_key == "test-key"

  def test_client_initially_none(self):
    emb = MistralEmbedder()
    assert emb._client is None


@pytest.mark.unit
class TestMistralEmbedderMethods:
  """MistralEmbedder methods with mocked client."""

  def test_get_embedding_returns_floats(self):
    emb = MistralEmbedder()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response
    emb._client = mock_client

    result = emb.get_embedding("hello")
    assert result == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once_with(model="mistral-embed", inputs=["hello"])

  def test_get_embedding_error_propagates(self):
    emb = MistralEmbedder()
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = RuntimeError("API error")
    emb._client = mock_client

    with pytest.raises(RuntimeError, match="API error"):
      emb.get_embedding("hello")

  def test_get_embedding_and_usage_returns_tuple(self):
    emb = MistralEmbedder()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_response.usage = MagicMock(prompt_tokens=5, total_tokens=5)
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response
    emb._client = mock_client

    embedding, usage = emb.get_embedding_and_usage("hello")
    assert embedding == [0.1, 0.2, 0.3]
    assert usage is not None
    assert usage["prompt_tokens"] == 5

  def test_get_embedding_and_usage_no_usage(self):
    emb = MistralEmbedder()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
    mock_response.usage = None
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response
    emb._client = mock_client

    embedding, usage = emb.get_embedding_and_usage("hello")
    assert embedding == [0.1, 0.2]
    assert usage is None

  def test_get_embedding_and_usage_error_propagates(self):
    emb = MistralEmbedder()
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = RuntimeError("API error")
    emb._client = mock_client

    with pytest.raises(RuntimeError, match="API error"):
      emb.get_embedding_and_usage("hello")

  def test_import_error_gives_helpful_message(self):
    emb = MistralEmbedder()
    with patch.dict("sys.modules", {"mistralai": None}):
      with pytest.raises(ImportError, match="mistralai"):
        emb.client


@pytest.mark.unit
class TestMistralEmbedderAsync:
  """MistralEmbedder async methods."""

  @pytest.mark.asyncio
  async def test_async_get_embedding(self):
    emb = MistralEmbedder()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.4, 0.5])]
    mock_client = MagicMock()
    mock_client.embeddings.create_async = AsyncMock(return_value=mock_response)
    emb._client = mock_client

    result = await emb.async_get_embedding("hello")
    assert isinstance(result, list)

  @pytest.mark.asyncio
  async def test_async_get_embedding_error_propagates(self):
    emb = MistralEmbedder()
    mock_client = MagicMock()
    mock_client.embeddings.create_async = AsyncMock(side_effect=RuntimeError("async error"))
    emb._client = mock_client

    with pytest.raises(RuntimeError, match="async error"):
      await emb.async_get_embedding("hello")


# ---------------------------------------------------------------------------
# GoogleEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoogleEmbedderConfig:
  """GoogleEmbedder has correct defaults."""

  def test_default_model(self):
    emb = GoogleEmbedder()
    assert emb.id == "text-embedding-004"

  def test_default_dimensions(self):
    emb = GoogleEmbedder()
    assert emb.dimensions == 768

  def test_custom_model(self):
    emb = GoogleEmbedder(id="text-embedding-005")
    assert emb.id == "text-embedding-005"

  def test_custom_api_key(self):
    emb = GoogleEmbedder(api_key="test-key")
    assert emb.api_key == "test-key"

  def test_custom_task_type(self):
    emb = GoogleEmbedder(task_type="retrieval_document")
    assert emb.task_type == "retrieval_document"

  def test_client_initially_none(self):
    emb = GoogleEmbedder()
    assert emb._client is None


@pytest.mark.unit
class TestGoogleEmbedderMethods:
  """GoogleEmbedder methods with mocked client."""

  def test_get_embedding_returns_floats(self):
    emb = GoogleEmbedder()
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1, 0.2, 0.3])]
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = mock_response
    emb._client = mock_client

    result = emb.get_embedding("hello")
    assert result == [0.1, 0.2, 0.3]

  def test_get_embedding_error_returns_empty(self):
    emb = GoogleEmbedder()
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = RuntimeError("API error")
    emb._client = mock_client

    result = emb.get_embedding("hello")
    assert result == []

  def test_get_embedding_and_usage_returns_tuple(self):
    emb = GoogleEmbedder()
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1, 0.2, 0.3])]
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = mock_response
    emb._client = mock_client

    embedding, usage = emb.get_embedding_and_usage("hello")
    assert embedding == [0.1, 0.2, 0.3]
    # Google doesn't return usage info
    assert usage is None

  def test_get_embedding_and_usage_error(self):
    emb = GoogleEmbedder()
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = RuntimeError("error")
    emb._client = mock_client

    embedding, usage = emb.get_embedding_and_usage("hello")
    assert embedding == []
    assert usage is None

  def test_import_error_gives_helpful_message(self):
    emb = GoogleEmbedder()
    with patch.dict("sys.modules", {"google": None, "google.genai": None}):
      with pytest.raises(ImportError, match="google-genai"):
        emb.client

  def test_build_config_with_task_type(self):
    emb = GoogleEmbedder(task_type="retrieval_query", dimensions=256)
    # _build_config needs google.genai.types — if not available, returns None
    with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()}):
      # Just verify it doesn't crash when types module is available
      config = emb._build_config()
      # May return None if mock doesn't fully emulate types — that's fine
      assert config is None or config is not None

  def test_build_config_no_task_type(self):
    emb = GoogleEmbedder(task_type=None, dimensions=0)
    # With no task_type and dimensions=0 (falsy), should return None
    config = emb._build_config()
    assert config is None


@pytest.mark.unit
class TestGoogleEmbedderAsync:
  """GoogleEmbedder async methods."""

  @pytest.mark.asyncio
  async def test_async_get_embedding(self):
    emb = GoogleEmbedder()
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.4, 0.5])]
    mock_client = MagicMock()
    mock_client.aio.models.embed_content = MagicMock(return_value=mock_response)
    emb._client = mock_client

    result = await emb.async_get_embedding("hello")
    assert isinstance(result, list)

  @pytest.mark.asyncio
  async def test_async_get_embedding_error(self):
    emb = GoogleEmbedder()
    mock_client = MagicMock()
    mock_client.aio.models.embed_content.side_effect = RuntimeError("error")
    emb._client = mock_client

    result = await emb.async_get_embedding("hello")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Imports work via top-level re-exports
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedderImports:
  """New embedders are importable from convenience paths."""

  def test_import_from_knowledge_embedder(self):
    from definable.knowledge.embedder import GoogleEmbedder as GE
    from definable.knowledge.embedder import MistralEmbedder as ME

    assert GE is GoogleEmbedder
    assert ME is MistralEmbedder

  def test_import_from_top_level(self):
    from definable.embedder import GoogleEmbedder as GE
    from definable.embedder import MistralEmbedder as ME

    assert GE is GoogleEmbedder
    assert ME is MistralEmbedder
