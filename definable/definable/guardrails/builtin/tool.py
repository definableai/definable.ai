"""Built-in tool guardrails: tool_allowlist, tool_blocklist."""

from __future__ import annotations

from typing import Any, Dict, Set

from definable.guardrails.base import GuardrailResult
from definable.run.base import RunContext


# ------------------------------------------------------------------
# tool_allowlist
# ------------------------------------------------------------------


class _ToolAllowlistGuardrail:
  """Only allow tools whose names appear in the allowlist."""

  def __init__(self, allowed: Set[str]):
    self.name = "tool_allowlist"
    self._allowed = allowed

  async def check(self, tool_name: str, tool_args: Dict[str, Any], context: RunContext) -> GuardrailResult:
    if tool_name in self._allowed:
      return GuardrailResult.allow()
    return GuardrailResult.block(f"Tool '{tool_name}' is not in the allowlist")


def tool_allowlist(allowed: Set[str]) -> _ToolAllowlistGuardrail:
  """Create a tool guardrail that only allows the named tools."""
  return _ToolAllowlistGuardrail(allowed)


# ------------------------------------------------------------------
# tool_blocklist
# ------------------------------------------------------------------


class _ToolBlocklistGuardrail:
  """Block tools whose names appear in the blocklist."""

  def __init__(self, blocked: Set[str]):
    self.name = "tool_blocklist"
    self._blocked = blocked

  async def check(self, tool_name: str, tool_args: Dict[str, Any], context: RunContext) -> GuardrailResult:
    if tool_name in self._blocked:
      return GuardrailResult.block(f"Tool '{tool_name}' is blocked")
    return GuardrailResult.allow()


def tool_blocklist(blocked: Set[str]) -> _ToolBlocklistGuardrail:
  """Create a tool guardrail that blocks the named tools."""
  return _ToolBlocklistGuardrail(blocked)
