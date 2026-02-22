"""Tests for pipeline DX APIs — Agent.hook(), Agent.pipeline, DebugConfig, ToolRetry."""

import pytest

from definable.agent import Agent
from definable.agent.pipeline import DebugConfig, Pipeline, SubAgentPolicy, ToolRetry
from definable.agent.testing import MockModel


def _agent(**kw) -> Agent:
  """Create a test agent with MockModel."""
  defaults = {"model": MockModel(responses=["hello"])}
  defaults.update(kw)
  return Agent(**defaults)


# ═══════════════════════════════════════════════════════════════════════
# Agent.pipeline property
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineProperty:
  def test_pipeline_exists(self):
    agent = _agent()
    assert agent.pipeline is not None
    assert isinstance(agent.pipeline, Pipeline)

  def test_pipeline_has_default_phases(self):
    agent = _agent()
    names = agent.pipeline.phase_names
    assert "prepare" in names
    assert "recall" in names
    assert "think" in names
    assert "guard_input" in names
    assert "compose" in names
    assert "invoke_loop" in names
    assert "guard_output" in names
    assert "store" in names

  def test_pipeline_phase_order(self):
    agent = _agent()
    names = agent.pipeline.phase_names
    assert names.index("prepare") < names.index("recall")
    assert names.index("recall") < names.index("think")
    assert names.index("think") < names.index("compose")
    assert names.index("compose") < names.index("invoke_loop")
    assert names.index("invoke_loop") < names.index("guard_output")
    assert names.index("guard_output") < names.index("store")


# ═══════════════════════════════════════════════════════════════════════
# Agent.hook() method
# ═══════════════════════════════════════════════════════════════════════


class TestHookMethod:
  def test_hook_decorator(self):
    agent = _agent()

    @agent.hook("before:prepare")
    async def my_hook(state):
      return state

    # Hook registered — no error
    assert True

  def test_hook_direct(self):
    agent = _agent()

    async def my_hook(state):
      return state

    agent.hook("before:prepare", my_hook)
    assert True

  def test_hook_invalid_spec_raises(self):
    agent = _agent()
    with pytest.raises(ValueError):
      agent.hook("invalid", lambda s: s)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline customization via Agent
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineCustomization:
  def test_add_custom_phase(self):
    from definable.agent.pipeline.phase import BasePhase

    class MyPhase(BasePhase):
      _name = "my_custom"

      async def execute(self, state):
        yield state, None

    agent = _agent()
    agent.pipeline.add_phase(MyPhase(), after="recall")
    assert "my_custom" in agent.pipeline.phase_names

  def test_remove_phase(self):
    agent = _agent()
    agent.pipeline.remove_phase("think")
    assert "think" not in agent.pipeline.phase_names

  def test_replace_phase(self):
    from definable.agent.pipeline.phase import BasePhase

    class MyCompose(BasePhase):
      _name = "compose"

      async def execute(self, state):
        yield state, None

    agent = _agent()
    agent.pipeline.replace_phase("compose", MyCompose())
    assert "compose" in agent.pipeline.phase_names


# ═══════════════════════════════════════════════════════════════════════
# DebugConfig acceptance
# ═══════════════════════════════════════════════════════════════════════


class TestDebugConfig:
  def test_debug_true_backward_compat(self):
    agent = _agent(debug=True)
    assert agent._tracing_config is not None

  def test_debug_config_sets_debug_config(self):
    dc = DebugConfig(step_mode=True, breakpoints={"before:invoke_loop"})
    agent = _agent(debug=dc)
    assert agent._debug_config is not None
    assert agent._debug_config.step_mode is True
    assert "before:invoke_loop" in agent._debug_config.breakpoints

  def test_debug_config_also_adds_tracing(self):
    dc = DebugConfig()
    agent = _agent(debug=dc)
    assert agent._tracing_config is not None

  def test_debug_false_no_config(self):
    agent = _agent(debug=False)
    assert agent._debug_config is None


# ═══════════════════════════════════════════════════════════════════════
# SubAgentPolicy acceptance
# ═══════════════════════════════════════════════════════════════════════


class TestSubAgentPolicy:
  def test_sub_agents_true(self):
    agent = _agent(sub_agents=True)
    assert agent._sub_agent_policy is not None
    assert agent._sub_agent_policy.max_concurrent == 5

  def test_sub_agents_policy(self):
    policy = SubAgentPolicy(max_concurrent=3, inherit_tools=False)
    agent = _agent(sub_agents=policy)
    assert agent._sub_agent_policy.max_concurrent == 3
    assert agent._sub_agent_policy.inherit_tools is False

  def test_sub_agents_none(self):
    agent = _agent(sub_agents=None)
    assert agent._sub_agent_policy is None

  def test_sub_agents_false(self):
    agent = _agent(sub_agents=False)
    assert agent._sub_agent_policy is None


# ═══════════════════════════════════════════════════════════════════════
# ToolRetry exception
# ═══════════════════════════════════════════════════════════════════════


class TestToolRetryDX:
  def test_raise_and_catch(self):
    with pytest.raises(ToolRetry) as exc_info:
      raise ToolRetry("query too short", max_retries=5)
    assert exc_info.value.message == "query too short"
    assert exc_info.value.max_retries == 5

  def test_import_from_top_level(self):
    from definable import ToolRetry as TR

    assert TR is ToolRetry

  def test_import_from_agent(self):
    from definable.agent import ToolRetry as TR

    assert TR is ToolRetry


# ═══════════════════════════════════════════════════════════════════════
# Top-level re-exports
# ═══════════════════════════════════════════════════════════════════════


class TestReExports:
  def test_pipeline_from_definable(self):
    from definable import Pipeline

    assert Pipeline is not None

  def test_tool_retry_from_definable(self):
    from definable import ToolRetry

    assert ToolRetry is not None

  def test_debug_config_from_definable(self):
    from definable import DebugConfig

    assert DebugConfig is not None

  def test_sub_agent_policy_from_definable(self):
    from definable import SubAgentPolicy

    assert SubAgentPolicy is not None

  def test_pipeline_from_agent(self):
    from definable.agent import Pipeline

    assert Pipeline is not None

  def test_loop_state_from_agent(self):
    from definable.agent import LoopState

    assert LoopState is not None
