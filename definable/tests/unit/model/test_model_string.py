"""
Unit tests for string model shorthand resolution in Agent.__init__.

When Agent(model="openai/gpt-4o") is called, the string is resolved to an
OpenAIChat instance. These tests verify the resolution logic without
making any API calls.

Covers:
  - "openai/gpt-4o" resolves to OpenAIChat with id="gpt-4o"
  - "deepseek/deepseek-chat" resolves to DeepSeekChat with id="deepseek-chat"
  - "moonshot/kimi-k2" resolves to MoonshotChat with id="kimi-k2"
  - "xai/grok-3" resolves to xAI with id="grok-3"
  - Bare model IDs (no provider) are rejected with clear error
  - Unknown providers are rejected with clear error
  - Empty/malformed strings are rejected
  - Model instance passthrough works unchanged
  - resolve_model_string is accessible from definable.model
"""

from unittest.mock import MagicMock, patch

import pytest

from definable.model.openai.chat import OpenAIChat
from definable.model.utils import resolve_model_string


# ---------------------------------------------------------------------------
# Provider/model-id resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderModelResolution:
  """Agent resolves 'provider/model-id' strings to correct Model instances."""

  @patch("definable.model.openai.chat.OpenAIChat.get_client")
  def test_openai_provider_resolves(self, mock_get_client):
    """'openai/gpt-4o' resolves to OpenAIChat(id='gpt-4o')."""
    mock_get_client.return_value = MagicMock()
    from definable.agent import Agent

    agent = Agent(model="openai/gpt-4o")
    assert isinstance(agent.model, OpenAIChat)
    assert agent.model.id == "gpt-4o"

  @patch("definable.model.openai.chat.OpenAIChat.get_client")
  def test_openai_mini_resolves(self, mock_get_client):
    """'openai/gpt-4o-mini' resolves to OpenAIChat(id='gpt-4o-mini')."""
    mock_get_client.return_value = MagicMock()
    from definable.agent import Agent

    agent = Agent(model="openai/gpt-4o-mini")
    assert isinstance(agent.model, OpenAIChat)
    assert agent.model.id == "gpt-4o-mini"

  @patch("definable.model.openai.chat.OpenAIChat.get_client")
  def test_openai_o3_resolves(self, mock_get_client):
    """'openai/o3-mini' resolves to OpenAIChat(id='o3-mini')."""
    mock_get_client.return_value = MagicMock()
    from definable.agent import Agent

    agent = Agent(model="openai/o3-mini")
    assert isinstance(agent.model, OpenAIChat)
    assert agent.model.id == "o3-mini"

  def test_deepseek_provider_resolves(self):
    """'deepseek/deepseek-chat' resolves to DeepSeekChat."""
    from definable.model.deepseek import DeepSeekChat

    model = resolve_model_string("deepseek/deepseek-chat")
    assert isinstance(model, DeepSeekChat)
    assert model.id == "deepseek-chat"

  def test_moonshot_provider_resolves(self):
    """'moonshot/kimi-k2' resolves to MoonshotChat."""
    from definable.model.moonshot import MoonshotChat

    model = resolve_model_string("moonshot/kimi-k2")
    assert isinstance(model, MoonshotChat)
    assert model.id == "kimi-k2"

  def test_xai_provider_resolves(self):
    """'xai/grok-3' resolves to xAI."""
    from definable.model.xai import xAI

    model = resolve_model_string("xai/grok-3")
    assert isinstance(model, xAI)
    assert model.id == "grok-3"

  def test_provider_is_case_insensitive(self):
    """'OpenAI/gpt-4o' resolves same as 'openai/gpt-4o'."""
    model = resolve_model_string("OpenAI/gpt-4o")
    assert isinstance(model, OpenAIChat)
    assert model.id == "gpt-4o"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStringModelErrors:
  """Invalid model strings produce clear, actionable errors."""

  def test_bare_model_id_rejected(self):
    """'gpt-4o' without provider prefix raises ValueError."""
    with pytest.raises(ValueError, match="provider/model-id"):
      resolve_model_string("gpt-4o")

  def test_unknown_provider_rejected(self):
    """'anthropic/claude-3' raises ValueError listing supported providers."""
    with pytest.raises(ValueError, match="Unknown model provider 'anthropic'"):
      resolve_model_string("anthropic/claude-3")

  def test_empty_string_rejected(self):
    """Empty string raises ValueError."""
    with pytest.raises(ValueError):
      resolve_model_string("")

  def test_empty_model_id_rejected(self):
    """'openai/' with missing model ID raises ValueError."""
    with pytest.raises(ValueError, match="Both provider and model-id are required"):
      resolve_model_string("openai/")

  def test_empty_provider_rejected(self):
    """'/gpt-4o' with missing provider raises ValueError."""
    with pytest.raises(ValueError, match="Both provider and model-id are required"):
      resolve_model_string("/gpt-4o")

  def test_agent_none_model_rejected(self):
    """Agent(model=None) raises TypeError."""
    from definable.agent import Agent

    with pytest.raises(TypeError, match="Agent requires a 'model' argument"):
      Agent(model=None)


# ---------------------------------------------------------------------------
# Model instance passthrough
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelInstancePassthrough:
  """Agent passes Model instances through without wrapping."""

  @patch("definable.model.openai.chat.OpenAIChat.get_client")
  def test_openai_chat_instance_passed_through(self, mock_get_client):
    """An OpenAIChat instance is used directly, not re-wrapped."""
    mock_get_client.return_value = MagicMock()
    model = OpenAIChat(id="gpt-4o")
    from definable.agent import Agent

    agent = Agent(model=model)
    assert agent.model is model
    assert agent.model.id == "gpt-4o"


# ---------------------------------------------------------------------------
# OpenAIChat creation properties (via string resolution)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStringModelProperties:
  """The resolved OpenAIChat has correct default properties."""

  @patch("definable.model.openai.chat.OpenAIChat.get_client")
  def test_resolved_model_has_openai_provider(self, mock_get_client):
    """String-resolved model reports provider as 'OpenAI'."""
    mock_get_client.return_value = MagicMock()
    from definable.agent import Agent

    agent = Agent(model="openai/gpt-4o")
    assert agent.model.provider == "OpenAI"

  @patch("definable.model.openai.chat.OpenAIChat.get_client")
  def test_resolved_model_has_name(self, mock_get_client):
    """String-resolved model has name='OpenAIChat'."""
    mock_get_client.return_value = MagicMock()
    from definable.agent import Agent

    agent = Agent(model="openai/gpt-4o")
    assert agent.model.name == "OpenAIChat"

  @patch("definable.model.openai.chat.OpenAIChat.get_client")
  def test_resolved_model_supports_structured_outputs(self, mock_get_client):
    """String-resolved model supports native structured outputs by default."""
    mock_get_client.return_value = MagicMock()
    from definable.agent import Agent

    agent = Agent(model="openai/gpt-4o")
    assert agent.model.supports_native_structured_outputs is True


# ---------------------------------------------------------------------------
# Public API: resolve_model_string re-exported
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPublicExport:
  """resolve_model_string is accessible from definable.model."""

  def test_resolve_model_string_importable(self):
    """Can import resolve_model_string from definable.model."""
    from definable.model import resolve_model_string as fn

    assert callable(fn)
