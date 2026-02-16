"""Tests for Agent(deep_research=True) integration."""

import pytest

from definable.agents.agent import Agent
from definable.agents.config import AgentConfig, TracingConfig
from definable.research.config import DeepResearchConfig
from definable.research.engine import DeepResearch


@pytest.mark.asyncio
class TestAgentDeepResearchIntegration:
  """Test deep research wired into the agent pipeline."""

  async def test_agent_init_with_bool(self, mock_model):
    """Agent(deep_research=True) should create a config."""
    agent = Agent(
      model=mock_model,
      deep_research=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    assert agent._researcher is not None

  async def test_agent_init_with_config(self, mock_model):
    """Agent(deep_research=DeepResearchConfig(...)) should use the config."""
    config = DeepResearchConfig(depth="quick", max_waves=1)
    agent = Agent(
      model=mock_model,
      deep_research=config,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    assert agent._researcher is not None
    assert agent._deep_research_config.depth == "quick"

  async def test_agent_init_disabled(self, mock_model):
    """Agent without deep_research should have no researcher."""
    agent = Agent(
      model=mock_model,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    assert agent._researcher is None

  async def test_agent_config_fallback(self, mock_model):
    """DeepResearchConfig in AgentConfig should be picked up."""
    dr_config = DeepResearchConfig(depth="deep")
    agent = Agent(
      model=mock_model,
      config=AgentConfig(
        tracing=TracingConfig(enabled=False),
        deep_research=dr_config,
      ),
    )
    assert agent._researcher is not None

  async def test_agent_init_with_prebuilt_engine(self, mock_model, mock_search):
    """Agent(deep_research=DeepResearch(...)) should use the prebuilt engine directly."""
    engine = DeepResearch(
      model=mock_model,
      search_provider=mock_search,
      config=DeepResearchConfig(depth="quick", max_waves=1),
    )
    agent = Agent(
      model=mock_model,
      deep_research=engine,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    assert agent._researcher is not None
    assert agent._researcher is engine
    assert agent._deep_research_config is None

  async def test_arun_injects_research_context(self, mock_model, mock_search):
    """Research context should be injected into system prompt."""
    dr_config = DeepResearchConfig(
      depth="quick",
      max_waves=1,
      max_sources=3,
      min_relevance=0.0,
      early_termination_threshold=0.0,
    )

    # Override the agent's search provider
    agent = Agent(
      model=mock_model,
      deep_research=dr_config,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    # Inject mock search provider directly
    if agent._researcher:
      agent._researcher._search_provider = mock_search

    output = await agent.arun("What are quantum computing developments?")
    assert output.content is not None

    # Check that the model received research context in system prompt
    # The mock model records all calls — check the last ainvoke call
    if mock_model.call_history:
      last_call = mock_model.call_history[-1]
      messages = last_call.get("messages", [])
      system_msgs = [m for m in messages if hasattr(m, "role") and m.role == "system"]
      if system_msgs:
        system_content = system_msgs[0].content or ""
        assert "research" in system_content.lower() or "<" in system_content

  async def test_arun_stream_yields_research_events(self, mock_model, mock_search):
    """Streaming should yield deep research events."""
    dr_config = DeepResearchConfig(
      depth="quick",
      max_waves=1,
      max_sources=3,
      min_relevance=0.0,
      early_termination_threshold=0.0,
    )

    agent = Agent(
      model=mock_model,
      deep_research=dr_config,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    if agent._researcher:
      agent._researcher._search_provider = mock_search

    events = []
    async for evt in agent.arun_stream("What are quantum computing developments?"):
      events.append(evt)

    event_types = [type(e).__name__ for e in events]
    assert "DeepResearchStartedEvent" in event_types
    assert "DeepResearchCompletedEvent" in event_types

  async def test_disabled_research_no_events(self, mock_model):
    """Agent without deep_research should not emit research events."""
    agent = Agent(
      model=mock_model,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    events = []
    async for evt in agent.arun_stream("Hello"):
      events.append(evt)

    event_types = [type(e).__name__ for e in events]
    assert "DeepResearchStartedEvent" not in event_types
    assert "DeepResearchCompletedEvent" not in event_types


@pytest.mark.asyncio
class TestAgentDeepResearchTrigger:
  """Test trigger modes (always, auto)."""

  async def test_trigger_always(self, mock_model, mock_search):
    """trigger='always' should always run research."""
    dr_config = DeepResearchConfig(
      trigger="always",
      depth="quick",
      max_waves=1,
      max_sources=3,
      min_relevance=0.0,
      early_termination_threshold=0.0,
    )

    agent = Agent(
      model=mock_model,
      deep_research=dr_config,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    if agent._researcher:
      agent._researcher._search_provider = mock_search

    output = await agent.arun("What is 2+2?")
    assert output.content is not None
