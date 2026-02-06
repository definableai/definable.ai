"""Tests for Agent compression integration."""

from typing import List
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from definable.agents import Agent, AgentConfig, CompressionConfig
from definable.agents.testing import MockModel
from definable.compression import CompressionManager
from definable.models.message import Message
from definable.models.metrics import Metrics
from definable.tools.decorator import tool


class TestCompressionConfig:
  """Tests for CompressionConfig dataclass."""

  def test_compression_config_defaults(self):
    """CompressionConfig has correct defaults."""
    config = CompressionConfig()

    assert config.enabled is True
    assert config.model is None
    assert config.tool_results_limit == 3
    assert config.token_limit is None
    assert config.instructions is None

  def test_compression_config_custom_values(self):
    """CompressionConfig accepts custom values."""
    config = CompressionConfig(
      enabled=True,
      model="gpt-4o-mini",
      tool_results_limit=5,
      token_limit=10000,
      instructions="Custom compression instructions",
    )

    assert config.enabled is True
    assert config.model == "gpt-4o-mini"
    assert config.tool_results_limit == 5
    assert config.token_limit == 10000
    assert config.instructions == "Custom compression instructions"

  def test_compression_config_disabled(self):
    """CompressionConfig can be disabled."""
    config = CompressionConfig(enabled=False)
    assert config.enabled is False


class TestAgentCompressionInit:
  """Tests for Agent initialization with compression config."""

  def test_agent_without_compression_config(self):
    """Agent without compression config has no compression manager."""
    agent = Agent(model=MockModel(), tools=[])

    assert agent._compression_manager is None

  def test_agent_with_compression_enabled(self):
    """Agent with compression enabled initializes CompressionManager."""
    config = AgentConfig(
      compression=CompressionConfig(
        enabled=True,
        tool_results_limit=3,
        model="gpt-4o-mini",
      )
    )
    agent = Agent(model=MockModel(), tools=[], config=config)

    assert agent._compression_manager is not None
    assert isinstance(agent._compression_manager, CompressionManager)
    assert agent._compression_manager.compress_tool_results is True
    assert agent._compression_manager.compress_tool_results_limit == 3

  def test_agent_with_compression_disabled(self):
    """Agent with compression disabled has no compression manager."""
    config = AgentConfig(compression=CompressionConfig(enabled=False))
    agent = Agent(model=MockModel(), tools=[], config=config)

    assert agent._compression_manager is None

  def test_agent_compression_manager_settings(self):
    """CompressionManager is initialized with correct settings."""
    agent_model = MockModel()
    config = AgentConfig(
      compression=CompressionConfig(
        enabled=True,
        # String model specs fall back to agent's model
        model="custom-model",
        tool_results_limit=5,
        token_limit=8000,
        instructions="Custom instructions",
      )
    )
    agent = Agent(model=agent_model, tools=[], config=config)

    cm = agent._compression_manager
    assert cm is not None
    # String model falls back to agent's model (string specs not fully supported)
    assert cm.model is agent_model
    assert cm.compress_tool_results_limit == 5
    assert cm.compress_token_limit == 8000
    assert cm.compress_tool_call_instructions == "Custom instructions"

  def test_agent_compression_with_model_instance(self):
    """CompressionManager uses Model instance when passed directly."""
    agent_model = MockModel()
    compression_model = MockModel()  # Different instance
    config = AgentConfig(
      compression=CompressionConfig(
        enabled=True,
        model=compression_model,  # Pass Model instance directly
        tool_results_limit=3,
      )
    )
    agent = Agent(model=agent_model, tools=[], config=config)

    cm = agent._compression_manager
    assert cm is not None
    # Should use the passed Model instance, not the agent's model
    assert cm.model is compression_model
    assert cm.model is not agent_model

  def test_agent_compression_uses_agent_model_when_none(self):
    """CompressionManager uses agent's model when no model specified."""
    agent_model = MockModel()
    config = AgentConfig(
      compression=CompressionConfig(
        enabled=True,
        # model not specified - should use agent's model
        tool_results_limit=3,
      )
    )
    agent = Agent(model=agent_model, tools=[], config=config)

    cm = agent._compression_manager
    assert cm is not None
    assert cm.model is agent_model


class TestAgentConfigWithUpdates:
  """Tests for AgentConfig.with_updates with compression."""

  def test_with_updates_preserves_compression(self):
    """with_updates preserves compression config."""
    original = AgentConfig(
      compression=CompressionConfig(enabled=True, tool_results_limit=3)
    )
    updated = original.with_updates(max_retries=5)

    assert updated.compression is not None
    assert updated.compression.enabled is True
    assert updated.compression.tool_results_limit == 3
    assert updated.max_retries == 5

  def test_with_updates_replaces_compression(self):
    """with_updates can replace compression config."""
    original = AgentConfig(
      compression=CompressionConfig(enabled=True, tool_results_limit=3)
    )
    new_compression = CompressionConfig(enabled=True, tool_results_limit=10)
    updated = original.with_updates(compression=new_compression)

    assert updated.compression.tool_results_limit == 10


class TestCompressionInAgentLoop:
  """Tests for compression integration in agent execution loop."""

  @pytest.fixture
  def simple_tool(self):
    """A simple tool for testing."""

    @tool
    def echo(message: str) -> str:
      """Echo a message back."""
      return f"Echo: {message}"

    return echo

  @pytest.fixture
  def mock_model_with_tool_calls(self):
    """MockModel that makes multiple tool calls before final response."""
    call_count = {"count": 0}

    def side_effect(messages: List, tools: List, **kwargs):
      response = MagicMock()
      response.response_usage = Metrics()
      response.reasoning_content = None
      response.citations = None
      response.images = None
      response.videos = None
      response.audios = None

      # Make 4 tool calls, then final response
      if call_count["count"] < 4 and tools:
        tool_name = tools[0]["function"]["name"] if tools else "echo"
        response.content = None
        response.tool_calls = [
          {
            "id": f"call_{call_count['count']}",
            "type": "function",
            "function": {"name": tool_name, "arguments": '{"message": "test"}'},
          }
        ]
      else:
        response.content = "Final response after tool calls"
        response.tool_calls = []

      call_count["count"] += 1
      return response

    return MockModel(side_effect=side_effect)

  @pytest.mark.asyncio
  async def test_compression_called_when_limit_reached(self, simple_tool, mock_model_with_tool_calls):
    """Compression is triggered when tool_results_limit is reached."""
    config = AgentConfig(
      compression=CompressionConfig(
        enabled=True,
        tool_results_limit=3,
      )
    )
    agent = Agent(
      model=mock_model_with_tool_calls,
      tools=[simple_tool],
      config=config,
    )

    # Mock the compression manager methods
    with patch.object(agent._compression_manager, "ashould_compress", new_callable=AsyncMock) as mock_should:
      with patch.object(agent._compression_manager, "acompress", new_callable=AsyncMock) as mock_compress:
        # First few calls: should_compress returns False
        # After 3 tool results: should return True
        mock_should.side_effect = [False, False, False, True, False]

        await agent.arun("Test with tools")

        # ashould_compress should have been called multiple times
        assert mock_should.call_count >= 3
        # acompress should have been called when should_compress returned True
        assert mock_compress.call_count >= 1

  @pytest.mark.asyncio
  async def test_no_compression_when_disabled(self, simple_tool, mock_model_with_tool_calls):
    """No compression when disabled."""
    config = AgentConfig(compression=CompressionConfig(enabled=False))
    agent = Agent(
      model=mock_model_with_tool_calls,
      tools=[simple_tool],
      config=config,
    )

    # No compression manager should exist
    assert agent._compression_manager is None

    # Should run without errors
    output = await agent.arun("Test without compression")
    assert output.content is not None

  @pytest.mark.asyncio
  async def test_compression_not_called_when_below_limit(self, simple_tool):
    """Compression is not triggered when below tool_results_limit."""
    # Model that makes only 1 tool call
    call_count = {"count": 0}

    def side_effect(messages: List, tools: List, **kwargs):
      response = MagicMock()
      response.response_usage = Metrics()
      response.reasoning_content = None
      response.citations = None
      response.images = None
      response.videos = None
      response.audios = None

      if call_count["count"] == 0 and tools:
        tool_name = tools[0]["function"]["name"]
        response.content = None
        response.tool_calls = [
          {
            "id": "call_0",
            "type": "function",
            "function": {"name": tool_name, "arguments": '{"message": "test"}'},
          }
        ]
      else:
        response.content = "Final response"
        response.tool_calls = []

      call_count["count"] += 1
      return response

    model = MockModel(side_effect=side_effect)

    config = AgentConfig(
      compression=CompressionConfig(
        enabled=True,
        tool_results_limit=5,  # Limit higher than tool calls
      )
    )
    agent = Agent(model=model, tools=[simple_tool], config=config)

    with patch.object(agent._compression_manager, "ashould_compress", new_callable=AsyncMock) as mock_should:
      mock_should.return_value = False  # Never triggers

      with patch.object(agent._compression_manager, "acompress", new_callable=AsyncMock) as mock_compress:
        await agent.arun("Test")

        # acompress should NOT have been called
        assert mock_compress.call_count == 0


class TestCompressionInStreamingLoop:
  """Tests for compression integration in streaming execution loop."""

  @pytest.fixture
  def simple_tool(self):
    """A simple tool for testing."""

    @tool
    def echo(message: str) -> str:
      """Echo a message back."""
      return f"Echo: {message}"

    return echo

  @pytest.mark.asyncio
  async def test_streaming_compression_called(self, simple_tool):
    """Compression is called in streaming loop when limit reached."""
    call_count = {"count": 0}

    # Create async generator for streaming
    async def stream_side_effect(messages: List, tools: List, **kwargs):
      nonlocal call_count

      if call_count["count"] < 3 and tools:
        tool_name = tools[0]["function"]["name"]
        # Yield chunk with tool call
        chunk = MagicMock()
        chunk.content = None
        chunk.tool_calls = [
          {
            "index": 0,
            "id": f"call_{call_count['count']}",
            "type": "function",
            "function": {"name": tool_name, "arguments": '{"message": "test"}'},
          }
        ]
        chunk.response_usage = Metrics()
        yield chunk
      else:
        # Yield content chunks
        for word in ["Final", " ", "response"]:
          chunk = MagicMock()
          chunk.content = word
          chunk.tool_calls = None
          chunk.response_usage = None
          yield chunk
        # Final chunk with usage
        chunk = MagicMock()
        chunk.content = None
        chunk.tool_calls = None
        chunk.response_usage = Metrics()
        yield chunk

      call_count["count"] += 1

    model = MagicMock()
    model.id = "mock-model"
    model.provider = "mock"
    model.ainvoke_stream = stream_side_effect

    config = AgentConfig(
      compression=CompressionConfig(
        enabled=True,
        tool_results_limit=2,
      )
    )
    agent = Agent(model=model, tools=[simple_tool], config=config)

    with patch.object(agent._compression_manager, "ashould_compress", new_callable=AsyncMock) as mock_should:
      with patch.object(agent._compression_manager, "acompress", new_callable=AsyncMock):
        mock_should.side_effect = [False, False, True, False]

        events = []
        async for event in agent.arun_stream("Test streaming"):
          events.append(event)

        # Verify compression was checked
        assert mock_should.call_count >= 2


class TestCompressionManagerIntegration:
  """Integration tests verifying CompressionManager behavior."""

  def test_compression_manager_should_compress_count_based(self):
    """CompressionManager triggers on count-based threshold."""
    cm = CompressionManager(
      compress_tool_results=True,
      compress_tool_results_limit=2,
    )

    # Create messages with uncompressed tool results
    messages = [
      Message(role="user", content="Hello"),
      Message(role="assistant", content="Using tools..."),
      Message(role="tool", content="Result 1", tool_call_id="1", name="tool1"),
      Message(role="tool", content="Result 2", tool_call_id="2", name="tool2"),
    ]

    # Should trigger because 2 uncompressed tool results >= limit of 2
    assert cm.should_compress(messages) is True

  def test_compression_manager_should_not_compress_below_limit(self):
    """CompressionManager doesn't trigger below threshold."""
    cm = CompressionManager(
      compress_tool_results=True,
      compress_tool_results_limit=5,
    )

    messages = [
      Message(role="user", content="Hello"),
      Message(role="tool", content="Result 1", tool_call_id="1", name="tool1"),
      Message(role="tool", content="Result 2", tool_call_id="2", name="tool2"),
    ]

    # Should NOT trigger because 2 < 5
    assert cm.should_compress(messages) is False

  def test_compression_manager_ignores_already_compressed(self):
    """CompressionManager ignores already compressed messages."""
    cm = CompressionManager(
      compress_tool_results=True,
      compress_tool_results_limit=2,
    )

    messages = [
      Message(role="user", content="Hello"),
      # Already compressed
      Message(role="tool", content="Result 1", tool_call_id="1", name="tool1", compressed_content="Compressed 1"),
      Message(role="tool", content="Result 2", tool_call_id="2", name="tool2", compressed_content="Compressed 2"),
      # Only 1 uncompressed
      Message(role="tool", content="Result 3", tool_call_id="3", name="tool3"),
    ]

    # Should NOT trigger because only 1 uncompressed < 2
    assert cm.should_compress(messages) is False
