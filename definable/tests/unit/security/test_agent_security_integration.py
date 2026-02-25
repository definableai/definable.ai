"""Tests for Agent + SecurityConfig integration."""

import pytest

from definable.agent.security import SecurityConfig, ToolPolicy
from definable.agent.security.content_defense import ContentDefenseConfig
from definable.agent.security.rate_limiter import RateLimitConfig


class TestAgentSecurityInit:
  def test_security_none_by_default(self):
    from definable.agent.testing import create_test_agent

    agent = create_test_agent()
    assert agent.security is None

  def test_security_true_creates_default_config(self):
    from definable.agent.testing import create_test_agent

    agent = create_test_agent(security=True)
    assert agent.security is not None
    assert isinstance(agent.security, SecurityConfig)

  def test_security_config_accepted(self):
    from definable.agent.testing import create_test_agent

    config = SecurityConfig(
      tool_policy=ToolPolicy(mode="allowlist", allowed_tools={"search"}),
      rate_limit=RateLimitConfig(max_requests=5),
    )
    agent = create_test_agent(security=config)
    assert agent.security is config

  def test_tool_policy_auto_injects_guardrail(self):
    from definable.agent.testing import create_test_agent

    agent = create_test_agent(
      security=SecurityConfig(
        tool_policy=ToolPolicy(mode="allowlist", allowed_tools={"search"}),
      ),
    )
    assert agent.guardrails is not None
    assert len(agent.guardrails.tool) >= 1

  def test_content_defense_auto_injects_guardrail(self):
    from definable.agent.testing import create_test_agent

    agent = create_test_agent(
      security=SecurityConfig(
        content_defense=ContentDefenseConfig(),
      ),
    )
    assert agent.guardrails is not None
    assert len(agent.guardrails.input) >= 1

  @pytest.mark.asyncio
  async def test_security_audit_returns_report(self):
    from definable.agent.testing import create_test_agent

    agent = create_test_agent()
    report = await agent.security_audit()
    assert report.score >= 0
    assert report.score <= 100


class TestAgentExports:
  def test_security_config_importable(self):
    from definable.agent import SecurityConfig

    assert SecurityConfig is not None

  def test_tool_policy_importable(self):
    from definable.agent import ToolPolicy

    assert ToolPolicy is not None

  def test_security_report_importable(self):
    from definable.agent import SecurityReport

    assert SecurityReport is not None

  def test_usage_tracker_importable(self):
    from definable.agent import UsageTracker

    assert UsageTracker is not None

  def test_usage_snapshot_importable(self):
    from definable.agent import UsageSnapshot

    assert UsageSnapshot is not None
