"""Integration tests — Agent wiring, observability composition, re-exports."""

import dataclasses


from definable.agent.observability.config import ObservabilityConfig
from definable.agent.observability.collector import ObservabilityExporter


class TestAgentObservabilityWiring:
  """Tests for Agent(observability=...) parameter."""

  def test_disabled_by_default(self):
    """Observability should be disabled by default."""
    from definable.agent import Agent

    agent = Agent(model="openai/gpt-4o-mini")
    assert agent.observability is None
    assert agent._observability_exporter is None

  def test_enabled_with_bool(self):
    """observability=True should create config and exporter."""
    from definable.agent import Agent

    agent = Agent(model="openai/gpt-4o-mini", observability=True)
    assert agent.observability is not None
    assert agent.observability.enabled is True
    assert agent._observability_exporter is not None
    assert isinstance(agent._observability_exporter, ObservabilityExporter)

  def test_enabled_with_config(self):
    """observability=ObservabilityConfig(...) should use provided config."""
    from definable.agent import Agent

    config = ObservabilityConfig(enabled=True, buffer_size=500, theme="light")
    agent = Agent(model="openai/gpt-4o-mini", observability=config)
    assert agent.observability is config
    assert agent.observability.buffer_size == 500
    assert agent.observability.theme == "light"
    assert agent._observability_exporter is not None

  def test_disabled_config(self):
    """ObservabilityConfig(enabled=False) should not create exporter."""
    from definable.agent import Agent

    config = ObservabilityConfig(enabled=False)
    agent = Agent(model="openai/gpt-4o-mini", observability=config)
    assert agent.observability is config
    assert agent._observability_exporter is None

  def test_composes_with_tracing(self):
    """Observability exporter should be added to tracing exporters."""
    from definable.agent import Agent
    from definable.agent.tracing.base import Tracing

    tracing = Tracing(exporters=[])
    agent = Agent(model="openai/gpt-4o-mini", tracing=tracing, observability=True)

    # The exporter should be added to the tracing pipeline
    assert agent._observability_exporter is not None
    # Check the tracing config has the exporter
    assert agent._tracing_config is not None
    assert agent._tracing_config.exporters is not None
    assert agent._observability_exporter in agent._tracing_config.exporters

  def test_composes_with_debug(self):
    """Should work alongside debug=True."""
    from definable.agent import Agent

    agent = Agent(model="openai/gpt-4o-mini", debug=True, observability=True)
    assert agent.observability is not None
    assert agent._observability_exporter is not None
    # Both debug and observability exporters should be present
    assert agent._tracing_config is not None
    assert agent._tracing_config.exporters is not None
    assert len(agent._tracing_config.exporters) >= 2  # DebugExporter + ObservabilityExporter

  def test_false_means_disabled(self):
    """observability=False should behave same as default (disabled)."""
    from definable.agent import Agent

    agent = Agent(model="openai/gpt-4o-mini", observability=False)
    assert agent.observability is None
    assert agent._observability_exporter is None


class TestObservabilityConfigDefaults:
  """Tests for ObservabilityConfig defaults and serialization."""

  def test_default_values(self):
    """Should have sensible defaults."""
    config = ObservabilityConfig()
    assert config.enabled is False
    assert config.trace_dir == "./traces"
    assert config.buffer_size == 10_000
    assert config.theme == "dark"

  def test_custom_values(self):
    """Should accept custom values."""
    config = ObservabilityConfig(
      enabled=True,
      trace_dir="/tmp/my-traces",
      buffer_size=5000,
      theme="light",
    )
    assert config.enabled is True
    assert config.trace_dir == "/tmp/my-traces"
    assert config.buffer_size == 5000
    assert config.theme == "light"

  def test_is_dataclass(self):
    """Should be a proper dataclass."""
    config = ObservabilityConfig(enabled=True)
    assert dataclasses.is_dataclass(config)
    d = dataclasses.asdict(config)
    assert d["enabled"] is True
    assert d["theme"] == "dark"


class TestReExports:
  """Tests for public API re-exports."""

  def test_import_from_observability_package(self):
    """Should be importable from definable.agent.observability."""
    from definable.agent.observability import ObservabilityConfig, ObservabilityExporter

    assert ObservabilityConfig is not None
    assert ObservabilityExporter is not None

  def test_import_from_agent_package(self):
    """ObservabilityConfig should be importable from definable.agent."""
    from definable.agent import ObservabilityConfig

    assert ObservabilityConfig is not None

  def test_in_agent_all(self):
    """ObservabilityConfig should be in definable.agent.__all__."""
    import definable.agent as agent_mod

    assert "ObservabilityConfig" in agent_mod.__all__

  def test_observability_exporter_satisfies_protocol(self):
    """ObservabilityExporter should satisfy TraceExporter protocol."""
    from definable.agent.tracing.base import TraceExporter

    exporter = ObservabilityExporter(buffer_size=10)
    assert isinstance(exporter, TraceExporter)
