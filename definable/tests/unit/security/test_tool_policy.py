"""Tests for ToolPolicy and ToolPolicyGuardrail."""

import pytest

from definable.agent.security.tool_policy import (
  DEFAULT_DANGEROUS_TOOLS,
  ToolPolicy,
  ToolPolicyGuardrail,
)
from definable.agent.events import RunContext


@pytest.fixture
def context():
  return RunContext(run_id="test-run", session_id="test")


# ------------------------------------------------------------------
# ToolPolicy unit tests
# ------------------------------------------------------------------


class TestToolPolicy:
  def test_default_mode_is_full(self):
    policy = ToolPolicy()
    assert policy.mode == "full"
    assert policy.is_allowed("any_tool")

  def test_deny_mode_blocks_all(self):
    policy = ToolPolicy(mode="deny")
    assert not policy.is_allowed("search")
    assert not policy.is_allowed("shell_command")

  def test_allowlist_mode_permits_only_listed(self):
    policy = ToolPolicy(mode="allowlist", allowed_tools={"search", "calculate"})
    assert policy.is_allowed("search")
    assert policy.is_allowed("calculate")
    assert not policy.is_allowed("delete_file")

  def test_allowlist_empty_blocks_all(self):
    policy = ToolPolicy(mode="allowlist")
    assert not policy.is_allowed("anything")

  def test_full_mode_allows_all_by_default(self):
    policy = ToolPolicy(mode="full")
    assert policy.is_allowed("shell_command")

  def test_full_mode_blocks_dangerous_when_enabled(self):
    policy = ToolPolicy(mode="full", block_dangerous=True)
    assert not policy.is_allowed("shell_command")
    assert policy.is_allowed("search")

  def test_full_mode_allows_dangerous_if_explicitly_listed(self):
    policy = ToolPolicy(mode="full", block_dangerous=True, allowed_tools={"shell_command"})
    assert policy.is_allowed("shell_command")

  def test_is_dangerous(self):
    policy = ToolPolicy()
    assert policy.is_dangerous("shell_command")
    assert policy.is_dangerous("delete_file")
    assert not policy.is_dangerous("search")

  def test_custom_dangerous_tools(self):
    policy = ToolPolicy(dangerous_tools={"custom_danger"})
    assert policy.is_dangerous("custom_danger")
    assert not policy.is_dangerous("shell_command")

  def test_default_dangerous_tools_not_empty(self):
    assert len(DEFAULT_DANGEROUS_TOOLS) > 10


# ------------------------------------------------------------------
# ToolPolicyGuardrail tests
# ------------------------------------------------------------------


class TestToolPolicyGuardrail:
  @pytest.mark.asyncio
  async def test_deny_mode_blocks(self, context):
    guardrail = ToolPolicyGuardrail(ToolPolicy(mode="deny"))
    result = await guardrail.check("search", {}, context)
    assert result.action == "block"

  @pytest.mark.asyncio
  async def test_allowlist_allows_listed(self, context):
    guardrail = ToolPolicyGuardrail(ToolPolicy(mode="allowlist", allowed_tools={"search"}))
    result = await guardrail.check("search", {}, context)
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_allowlist_blocks_unlisted(self, context):
    guardrail = ToolPolicyGuardrail(ToolPolicy(mode="allowlist", allowed_tools={"search"}))
    result = await guardrail.check("delete_file", {}, context)
    assert result.action == "block"
    assert result.message is not None
    assert "not in the allowlist" in result.message

  @pytest.mark.asyncio
  async def test_full_mode_allows(self, context):
    guardrail = ToolPolicyGuardrail(ToolPolicy(mode="full"))
    result = await guardrail.check("anything", {}, context)
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_full_blocks_dangerous_when_configured(self, context):
    guardrail = ToolPolicyGuardrail(ToolPolicy(mode="full", block_dangerous=True))
    result = await guardrail.check("shell_command", {}, context)
    assert result.action == "block"
    assert result.message is not None
    assert "dangerous" in result.message

  @pytest.mark.asyncio
  async def test_guardrail_name(self):
    guardrail = ToolPolicyGuardrail(ToolPolicy())
    assert guardrail.name == "tool_policy"
