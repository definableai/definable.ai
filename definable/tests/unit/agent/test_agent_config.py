"""
Unit tests for agent configuration dataclasses.

Tests pure Python logic: defaults, immutability, with_updates(), type validation.
No API calls. No external dependencies.

Covers:
  - Thinking defaults and trigger values
  - CompressionConfig defaults
  - Knowledge fields
  - Memory defaults
  - AgentConfig is frozen (immutable)
  - AgentConfig.with_updates() returns new instance without mutating original
  - Tracing defaults

Migrated from tests_e2e/unit/test_config_validation.py — all original tests preserved.
"""

import dataclasses

import pytest

from definable.agent.config import (
  AgentConfig,
  CompressionConfig,
)
from definable.agent.reasoning.thinking import Thinking
from definable.agent.tracing import Tracing
from definable.memory.manager import Memory


@pytest.mark.unit
class TestThinking:
  def test_defaults(self):
    cfg = Thinking()
    assert cfg.enabled is True
    assert cfg.model is None
    assert cfg.instructions is None
    assert cfg.trigger == "always"
    assert cfg.description is None

  def test_all_trigger_values_accepted(self):
    for trigger in ("always", "auto", "never"):
      cfg = Thinking(trigger=trigger)
      assert cfg.trigger == trigger

  def test_custom_model_stored(self):
    sentinel = object()
    cfg = Thinking(model=sentinel)  # type: ignore[arg-type]
    assert cfg.model is sentinel

  def test_custom_instructions_stored(self):
    cfg = Thinking(instructions="Think step by step.")
    assert cfg.instructions == "Think step by step."


@pytest.mark.unit
class TestCompressionConfig:
  def test_defaults(self):
    cfg = CompressionConfig()
    assert cfg.enabled is True
    assert cfg.model is None
    assert cfg.tool_results_limit == 3
    assert cfg.token_limit is None
    assert cfg.instructions is None

  def test_custom_values(self):
    cfg = CompressionConfig(enabled=False, tool_results_limit=5, token_limit=1000)
    assert cfg.enabled is False
    assert cfg.tool_results_limit == 5
    assert cfg.token_limit == 1000


@pytest.mark.unit
class TestMemory:
  def test_defaults(self):
    cfg = Memory()
    assert cfg.store is None
    assert cfg.enabled is True
    assert cfg.max_messages == 100
    assert cfg.pin_count == 2
    assert cfg.recent_count == 5
    assert cfg.description is None

  def test_custom_max_messages(self):
    cfg = Memory(max_messages=50)
    assert cfg.max_messages == 50

  def test_disabled(self):
    cfg = Memory(enabled=False)
    assert cfg.enabled is False


@pytest.mark.unit
class TestTracing:
  def test_defaults(self):
    cfg = Tracing()
    assert cfg.enabled is True
    assert cfg.exporters is None
    assert cfg.event_filter is None
    assert cfg.batch_size == 1
    assert cfg.flush_interval_ms == 5000

  def test_disabled(self):
    cfg = Tracing(enabled=False)
    assert cfg.enabled is False


@pytest.mark.unit
class TestAgentConfig:
  def test_defaults(self):
    cfg = AgentConfig()
    assert cfg.agent_id is None
    assert cfg.agent_name is None
    assert cfg.max_iterations == 10
    assert cfg.max_tokens is None
    assert cfg.retry_transient_errors is True
    assert cfg.max_retries == 3
    assert cfg.validate_tool_args is True
    assert cfg.strict_output_schema is False

  def test_is_frozen(self):
    """AgentConfig must be immutable -- prevents accidental mutation during runs."""
    cfg = AgentConfig(agent_id="test-123")
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
      cfg.agent_id = "changed"  # type: ignore[misc]

  def test_with_updates_returns_new_instance(self):
    original = AgentConfig(agent_id="original", max_retries=2)
    updated = original.with_updates(max_retries=5, agent_name="Updated")
    # New instance
    assert updated is not original
    # Updated fields
    assert updated.max_retries == 5
    assert updated.agent_name == "Updated"
    # Original untouched
    assert original.max_retries == 2
    assert original.agent_name is None

  def test_with_updates_preserves_unspecified_fields(self):
    original = AgentConfig(agent_id="abc", max_iterations=20, max_retries=5)
    updated = original.with_updates(agent_name="NewName")
    assert updated.agent_id == "abc"
    assert updated.max_iterations == 20
    assert updated.max_retries == 5

  def test_with_updates_preserves_non_serializable_fields(self):
    tracing = Tracing(enabled=False)
    original = AgentConfig(tracing=tracing)
    updated = original.with_updates(agent_name="Test")
    assert updated.tracing is tracing  # same object

  def test_custom_values(self):
    cfg = AgentConfig(
      agent_id="my-agent",
      agent_name="Test Agent",
      max_iterations=5,
      max_tokens=2048,
      max_retries=1,
    )
    assert cfg.agent_id == "my-agent"
    assert cfg.agent_name == "Test Agent"
    assert cfg.max_iterations == 5
    assert cfg.max_tokens == 2048
    assert cfg.max_retries == 1

  def test_repr_is_informative(self):
    cfg = AgentConfig(agent_id="repr-test", agent_name="MyAgent")
    r = repr(cfg)
    assert "repr-test" in r
    assert "MyAgent" in r
