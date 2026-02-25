"""Tests for security audit system."""

import pytest
from unittest.mock import MagicMock, PropertyMock

from definable.agent.security.audit import (
  SecurityFinding,
  SecurityReport,
  SecuritySeverity,
  security_audit,
)


# ------------------------------------------------------------------
# SecurityReport
# ------------------------------------------------------------------


class TestSecurityReport:
  def test_empty_report(self):
    report = SecurityReport(agent_name="test")
    assert report.score == 100
    assert report.critical_count == 0
    assert report.warning_count == 0
    assert report.info_count == 0

  def test_counts(self):
    report = SecurityReport(
      findings=[
        SecurityFinding(SecuritySeverity.critical, "cat", "t", "d", "r"),
        SecurityFinding(SecuritySeverity.critical, "cat", "t", "d", "r"),
        SecurityFinding(SecuritySeverity.warning, "cat", "t", "d", "r"),
        SecurityFinding(SecuritySeverity.info, "cat", "t", "d", "r"),
      ]
    )
    assert report.critical_count == 2
    assert report.warning_count == 1
    assert report.info_count == 1

  def test_to_dict(self):
    report = SecurityReport(
      agent_name="agent",
      checked_at="2026-01-01",
      score=85,
      findings=[
        SecurityFinding(SecuritySeverity.warning, "tools", "test", "desc", "fix"),
      ],
    )
    d = report.to_dict()
    assert d["agent_name"] == "agent"
    assert d["score"] == 85
    assert d["summary"]["warning"] == 1
    assert len(d["findings"]) == 1

  def test_str_representation(self):
    report = SecurityReport(
      agent_name="my_agent",
      checked_at="2026-01-01",
      score=90,
      findings=[
        SecurityFinding(SecuritySeverity.warning, "tools", "Dangerous tool", "desc", "fix"),
      ],
    )
    text = str(report)
    assert "my_agent" in text
    assert "90/100" in text
    assert "Dangerous tool" in text


# ------------------------------------------------------------------
# security_audit function
# ------------------------------------------------------------------


class TestSecurityAudit:
  @pytest.mark.asyncio
  async def test_audit_clean_agent(self):
    """Agent with guardrails should have fewer findings."""
    from definable.agent.guardrail.base import Guardrails

    agent = MagicMock()
    agent.config = MagicMock()
    agent.config.agent_name = "clean_agent"
    agent.model = MagicMock()
    agent.model.__class__.__name__ = "OpenAIChat"
    agent.instructions = "You are a helpful assistant."
    agent.tools = []
    agent.toolkits = []
    agent.guardrails = Guardrails(input=[], output=[], tool=[])
    type(agent)._interfaces = PropertyMock(return_value=[])

    report = await security_audit(agent)
    assert isinstance(report, SecurityReport)
    assert report.agent_name == "clean_agent"
    # Clean agent should have some info findings but no critical
    assert report.critical_count == 0

  @pytest.mark.asyncio
  async def test_audit_no_guardrails(self):
    """Agent without guardrails should get a warning."""
    agent = MagicMock()
    agent.config = MagicMock()
    agent.config.agent_name = "bare_agent"
    agent.model = MagicMock()
    agent.model.__class__.__name__ = "OpenAIChat"
    agent.instructions = ""
    agent.tools = []
    agent.toolkits = []
    agent.guardrails = None
    type(agent)._interfaces = PropertyMock(return_value=[])

    report = await security_audit(agent)
    categories = [f.category for f in report.findings]
    assert "guardrails" in categories

  @pytest.mark.asyncio
  async def test_audit_detects_secrets(self):
    """Agent with API keys in instructions should get critical finding."""
    agent = MagicMock()
    agent.config = MagicMock()
    agent.config.agent_name = "leaky_agent"
    agent.model = MagicMock()
    agent.model.__class__.__name__ = "OpenAIChat"
    agent.instructions = "Use this key: sk-aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"
    agent.tools = []
    agent.toolkits = []
    agent.guardrails = None
    type(agent)._interfaces = PropertyMock(return_value=[])

    report = await security_audit(agent)
    secret_findings = [f for f in report.findings if f.category == "secrets"]
    assert len(secret_findings) > 0
    assert secret_findings[0].severity == SecuritySeverity.critical

  @pytest.mark.asyncio
  async def test_score_calculation(self):
    """Score should decrease with findings."""
    agent = MagicMock()
    agent.config = MagicMock()
    agent.config.agent_name = "test"
    agent.model = MagicMock()
    agent.model.__class__.__name__ = "OpenAIChat"
    agent.instructions = 'password = "secret123456789"'
    agent.tools = []
    agent.toolkits = []
    agent.guardrails = None
    type(agent)._interfaces = PropertyMock(return_value=[])

    report = await security_audit(agent)
    assert report.score < 100
