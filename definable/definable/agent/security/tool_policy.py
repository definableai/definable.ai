"""Tool execution security — declarative policy for which tools can run.

Provides ToolPolicy (deny/allowlist/full modes), a dangerous tools registry,
and a ToolGuardrail adapter that plugs into the existing guardrail system.

Usage::

    from definable.agent.security import ToolPolicy, SecurityConfig

    agent = Agent(
        model=model,
        security=SecurityConfig(
            tool_policy=ToolPolicy(mode="allowlist", allowed_tools={"search", "calculate"}),
        ),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Set

from definable.agent.events import RunContext
from definable.agent.guardrail.base import GuardrailResult
from definable.utils.log import log_debug, log_warning


# ------------------------------------------------------------------
# Default dangerous tools — shell execution, file mutation, code eval
# ------------------------------------------------------------------

DEFAULT_DANGEROUS_TOOLS: frozenset[str] = frozenset({
  # Shell / subprocess
  "shell_command",
  "run_shell",
  "execute_command",
  "exec",
  "run_bash",
  # File mutation
  "write_file",
  "delete_file",
  "move_file",
  "remove_file",
  "create_file",
  # Code execution
  "run_python",
  "eval",
  "run_applescript",
  "execute_code",
  # System
  "run_process",
  "kill_process",
})


# ------------------------------------------------------------------
# ToolPolicy
# ------------------------------------------------------------------


@dataclass
class ToolPolicy:
  """Declarative tool execution policy for an agent.

  Attributes:
    mode: ``"deny"`` blocks all tool calls, ``"allowlist"`` only permits
      tools in *allowed_tools*, ``"full"`` allows everything (default).
    allowed_tools: Tool names permitted when mode is ``"allowlist"``.
    dangerous_tools: Set of inherently risky tool names. When a tool in
      this set is invoked and mode is not ``"full"``, it requires either
      explicit inclusion in *allowed_tools* or an approval callback.
      Defaults to :data:`DEFAULT_DANGEROUS_TOOLS`.
    block_dangerous: When True (default), tools in *dangerous_tools* are
      blocked unless explicitly allowed — even in ``"full"`` mode.
      Set to False to disable the dangerous-tools check entirely.
  """

  mode: Literal["deny", "allowlist", "full"] = "full"
  allowed_tools: Optional[Set[str]] = None
  dangerous_tools: Optional[Set[str]] = None
  block_dangerous: bool = False

  def __post_init__(self) -> None:
    if self.allowed_tools is None:
      self.allowed_tools = set()
    if self.dangerous_tools is None:
      self.dangerous_tools = set(DEFAULT_DANGEROUS_TOOLS)

  def is_allowed(self, tool_name: str) -> bool:
    """Check if a tool is allowed under this policy."""
    if self.mode == "deny":
      return False
    if self.mode == "allowlist":
      return tool_name in (self.allowed_tools or set())
    # mode == "full"
    if self.block_dangerous and tool_name in (self.dangerous_tools or set()):
      return tool_name in (self.allowed_tools or set())
    return True

  def is_dangerous(self, tool_name: str) -> bool:
    """Check if a tool is in the dangerous tools registry."""
    return tool_name in (self.dangerous_tools or set())


# ------------------------------------------------------------------
# ToolPolicyGuardrail — bridges ToolPolicy into the guardrail system
# ------------------------------------------------------------------


class ToolPolicyGuardrail:
  """ToolGuardrail implementation that enforces a :class:`ToolPolicy`.

  Conforms to the ``ToolGuardrail`` protocol from
  ``definable.agent.guardrail.base``.
  """

  name: str = "tool_policy"

  def __init__(self, policy: ToolPolicy) -> None:
    self._policy = policy

  async def check(
    self,
    tool_name: str,
    tool_args: Dict[str, Any],
    context: RunContext,
  ) -> GuardrailResult:
    """Evaluate the tool call against the policy."""
    if self._policy.mode == "deny":
      log_debug(f"ToolPolicy(deny): blocking tool '{tool_name}'")
      return GuardrailResult.block(f"Tool execution is disabled by policy (mode='deny'). Tool '{tool_name}' was blocked.")

    if self._policy.mode == "allowlist":
      if tool_name in (self._policy.allowed_tools or set()):
        return GuardrailResult.allow()
      log_debug(f"ToolPolicy(allowlist): tool '{tool_name}' not in allowlist")
      return GuardrailResult.block(
        f"Tool '{tool_name}' is not in the allowlist. Allowed: {', '.join(sorted(self._policy.allowed_tools or set())) or '(none)'}."
      )

    # mode == "full"
    if self._policy.block_dangerous and self._policy.is_dangerous(tool_name):
      if tool_name in (self._policy.allowed_tools or set()):
        log_debug(f"ToolPolicy(full): dangerous tool '{tool_name}' explicitly allowed")
        return GuardrailResult.allow()
      log_warning(f"ToolPolicy: blocking dangerous tool '{tool_name}' (not in allowed_tools)")
      return GuardrailResult.block(
        f"Tool '{tool_name}' is classified as dangerous and requires explicit approval. "
        f"Add it to ToolPolicy(allowed_tools={{'{tool_name}'}}) to permit."
      )

    return GuardrailResult.allow()
