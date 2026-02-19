"""
Unit tests for Agent constructor and initialization.

Tests the Agent.__init__ method: model resolution, layer wiring,
identity properties, and method existence. No API calls.
"""

import pytest

from definable.agent.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.testing import MockModel


@pytest.mark.unit
class TestAgentModelInit:
  def test_agent_with_mock_model(self):
    """Agent(model=MockModel()) stores the model instance."""
    model = MockModel()
    agent = Agent(model=model)  # type: ignore[arg-type]
    assert agent.model is model

  def test_agent_with_openai_model_object(self):
    """Agent(model=OpenAIChat(...)) stores the OpenAI model directly."""
    from definable.model.openai import OpenAIChat

    model = OpenAIChat(id="gpt-4o-mini")
    agent = Agent(model=model)
    assert agent.model is model

  def test_agent_string_shorthand_resolves_to_openai(self):
    """Agent(model='openai/gpt-4o-mini') resolves to OpenAIChat instance."""
    from definable.model.openai import OpenAIChat

    agent = Agent(model="openai/gpt-4o-mini")
    assert isinstance(agent.model, OpenAIChat)
    assert agent.model.id == "gpt-4o-mini"

  def test_agent_string_shorthand_different_model_id(self):
    """String shorthand preserves the exact model ID."""
    from definable.model.openai import OpenAIChat

    agent = Agent(model="openai/gpt-4o")
    assert isinstance(agent.model, OpenAIChat)
    assert agent.model.id == "gpt-4o"


@pytest.mark.unit
class TestAgentInstructions:
  def test_instructions_stored(self):
    """Agent(instructions='...') stores instructions string."""
    agent = Agent(model=MockModel(), instructions="Be helpful.")  # type: ignore[arg-type]
    assert agent.instructions == "Be helpful."

  def test_instructions_default_is_none(self):
    """When no instructions provided, defaults to None."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert agent.instructions is None


@pytest.mark.unit
class TestAgentTools:
  def test_tools_stored(self):
    """Agent(tools=[...]) stores the tools list."""
    from definable.tool.decorator import tool

    @tool
    def my_tool(x: str) -> str:
      """A test tool."""
      return x

    agent = Agent(model=MockModel(), tools=[my_tool])  # type: ignore[arg-type]
    assert len(agent.tools) == 1

  def test_tools_default_empty(self):
    """When no tools provided, defaults to empty list."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert agent.tools == []


@pytest.mark.unit
class TestAgentMemory:
  def test_memory_true_creates_in_memory_store(self):
    """Agent(memory=True) creates Memory with InMemoryStore."""
    from definable.memory.manager import Memory
    from definable.memory.store.in_memory import InMemoryStore

    agent = Agent(model=MockModel(), memory=True)  # type: ignore[arg-type]
    assert agent.memory is not None
    assert isinstance(agent.memory, Memory)
    assert isinstance(agent.memory.store, InMemoryStore)

  def test_memory_false_is_none(self):
    """Agent(memory=False) results in memory=None."""
    agent = Agent(model=MockModel(), memory=False)  # type: ignore[arg-type]
    assert agent.memory is None

  def test_memory_default_is_none(self):
    """Default memory is None (default param is False)."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert agent.memory is None

  def test_memory_instance_passthrough(self):
    """Agent(memory=Memory(...)) passes the instance through."""
    from definable.memory.manager import Memory
    from definable.memory.store.in_memory import InMemoryStore

    mem = Memory(store=InMemoryStore())
    agent = Agent(model=MockModel(), memory=mem)  # type: ignore[arg-type]
    assert agent.memory is mem


@pytest.mark.unit
class TestAgentThinking:
  def test_thinking_true_creates_default(self):
    """Agent(thinking=True) creates a Thinking instance."""
    from definable.agent.reasoning.thinking import Thinking

    agent = Agent(model=MockModel(), thinking=True)  # type: ignore[arg-type]
    assert agent._thinking is not None
    assert isinstance(agent._thinking, Thinking)

  def test_thinking_none_is_disabled(self):
    """Agent(thinking=None) disables thinking."""
    agent = Agent(model=MockModel(), thinking=None)  # type: ignore[arg-type]
    assert agent._thinking is None

  def test_thinking_default_is_none(self):
    """Default thinking is None."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert agent._thinking is None

  def test_thinking_instance_passthrough(self):
    """Agent(thinking=Thinking(...)) passes the instance through."""
    from definable.agent.reasoning.thinking import Thinking

    t = Thinking(trigger="auto")
    agent = Agent(model=MockModel(), thinking=t)  # type: ignore[arg-type]
    assert agent._thinking is t


@pytest.mark.unit
class TestAgentTracing:
  def test_tracing_true_creates_default(self):
    """Agent(tracing=True) creates a Tracing instance."""
    from definable.agent.tracing.base import Tracing

    agent = Agent(model=MockModel(), tracing=True)  # type: ignore[arg-type]
    assert agent._tracing_config is not None
    assert isinstance(agent._tracing_config, Tracing)

  def test_tracing_false_falls_back_to_config(self):
    """Agent(tracing=False) falls back to config.tracing."""
    from definable.agent.tracing.base import Tracing

    tracing = Tracing(enabled=False)
    config = AgentConfig(tracing=tracing)
    agent = Agent(model=MockModel(), tracing=False, config=config)  # type: ignore[arg-type]
    assert agent._tracing_config is tracing

  def test_tracing_instance_passthrough(self):
    """Agent(tracing=Tracing(...)) passes the instance through."""
    from definable.agent.tracing.base import Tracing

    t = Tracing(enabled=False)
    agent = Agent(model=MockModel(), tracing=t)  # type: ignore[arg-type]
    assert agent._tracing_config is t


@pytest.mark.unit
class TestAgentKnowledge:
  def test_knowledge_true_raises_value_error(self):
    """Agent(knowledge=True) raises ValueError (documented behavior)."""
    with pytest.raises(ValueError, match="knowledge=True is not supported"):
      Agent(model=MockModel(), knowledge=True)  # type: ignore[arg-type]

  def test_knowledge_false_is_none(self):
    """Agent(knowledge=False) results in knowledge=None."""
    agent = Agent(model=MockModel(), knowledge=False)  # type: ignore[arg-type]
    assert agent._knowledge is None

  def test_knowledge_default_is_none(self):
    """Default knowledge is None (default param is False)."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert agent._knowledge is None


@pytest.mark.unit
class TestAgentConfig:
  def test_config_stored(self):
    """Agent(config=AgentConfig(...)) stores config."""
    config = AgentConfig(agent_id="test-id")
    agent = Agent(model=MockModel(), config=config)  # type: ignore[arg-type]
    assert agent.config.agent_id == "test-id"

  def test_config_default(self):
    """Default config is a plain AgentConfig."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert isinstance(agent.config, AgentConfig)


@pytest.mark.unit
class TestAgentIdentity:
  def test_name_from_constructor(self):
    """Agent(name='MyBot') sets agent_name on config."""
    agent = Agent(model=MockModel(), name="MyBot")  # type: ignore[arg-type]
    assert agent.agent_name == "MyBot"

  def test_name_default_is_class_name(self):
    """Default agent_name is the class name 'Agent'."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert agent.agent_name == "Agent"

  def test_session_id_auto_generated(self):
    """Agent generates a session_id when none is provided."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert agent.session_id is not None
    assert len(agent.session_id) > 0

  def test_session_id_custom(self):
    """Agent(session_id='custom') stores the provided session_id."""
    agent = Agent(model=MockModel(), session_id="my-session")  # type: ignore[arg-type]
    assert agent.session_id == "my-session"


@pytest.mark.unit
class TestAgentMethods:
  def test_run_method_exists(self):
    """Agent has a run() method."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert hasattr(agent, "run")
    assert callable(agent.run)

  def test_arun_method_exists(self):
    """Agent has an arun() method."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert hasattr(agent, "arun")
    assert callable(agent.arun)

  def test_use_method_exists(self):
    """Agent has a use() method for middleware."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert hasattr(agent, "use")
    assert callable(agent.use)

  def test_events_property_returns_event_bus(self):
    """Agent.events returns an EventBus instance."""
    from definable.agent.event_bus import EventBus

    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert isinstance(agent.events, EventBus)

  def test_tool_names_property(self):
    """Agent.tool_names returns list of tool name strings."""
    from definable.tool.decorator import tool

    @tool
    def my_tool(x: str) -> str:
      """A test tool."""
      return x

    agent = Agent(model=MockModel(), tools=[my_tool])  # type: ignore[arg-type]
    assert "my_tool" in agent.tool_names
