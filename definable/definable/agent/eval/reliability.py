"""ReliabilityEval — verify that the agent called the expected tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
from uuid import uuid4

from definable.agent.eval.base import BaseEval, EvalCase
from definable.agent.eval.result import ReliabilityResult
from definable.utils.log import log_error

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.team.team import Team


@dataclass
class ReliabilityEval(BaseEval):
  """Evaluate whether the agent called the expected tools.

  Runs the agent and checks that the specified tools were invoked.
  Supports both permissive mode (extra tools OK) and strict mode
  (only expected tools allowed).

  Args:
      expected_tools: List of tool function names that must be called.
      strict: If True, fail when tools outside expected_tools are called.

  Example::

      eval = ReliabilityEval(expected_tools=["search_web", "summarize"])
      result = await eval.arun(agent, EvalCase(input="Research quantum computing"))
      print(result.missing_tools)  # tools that should have been called but weren't
  """

  name: str = "reliability"
  expected_tools: List[str] = field(default_factory=list)
  strict: bool = False
  _id: str = field(default_factory=lambda: str(uuid4()))

  async def evaluate(self, agent: "Agent", case: EvalCase) -> ReliabilityResult:
    """Run the agent and verify tool calls."""
    return await self._check(agent, None, case)

  async def evaluate_team(self, team: "Team", case: EvalCase) -> ReliabilityResult:
    """Run the team and verify tool calls."""
    return await self._check(None, team, case)

  async def _check(
    self,
    agent: Optional["Agent"],
    team: Optional["Team"],
    case: EvalCase,
  ) -> ReliabilityResult:
    """Core tool-check logic."""
    # Use case-level expected_tools override if provided
    expected = case.metadata.get("expected_tools", self.expected_tools)
    if not expected:
      return ReliabilityResult(
        eval_name=self.name,
        success=False,
        reason="No expected tools specified.",
        expected_tools=[],
      )

    # Run agent/team
    try:
      if agent:
        output = await agent.arun(case.input)
      elif team:
        output = await team.arun(case.input)
      else:
        return ReliabilityResult(
          eval_name=self.name,
          success=False,
          reason="No agent or team provided.",
        )
    except Exception as e:
      log_error(f"ReliabilityEval: execution failed: {e}")
      return ReliabilityResult(
        eval_name=self.name,
        success=False,
        reason=f"Execution failed: {e}",
        expected_tools=list(expected),
      )

    # Extract actual tool names from output
    actual_tools = self._extract_tool_names(output)

    # Compare
    expected_set = set(expected)
    actual_set = set(actual_tools)

    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)

    # Determine success
    all_expected_called = len(missing) == 0
    no_extra = len(extra) == 0
    success = all_expected_called and (no_extra if self.strict else True)

    reasons: list[str] = []
    if missing:
      reasons.append(f"Missing tools: {missing}")
    if self.strict and extra:
      reasons.append(f"Unexpected tools: {extra}")

    return ReliabilityResult(
      eval_name=self.name,
      score=10.0 if success else 0.0,
      success=success,
      reason="; ".join(reasons) if reasons else "All expected tools were called.",
      expected_tools=list(expected),
      actual_tools=actual_tools,
      missing_tools=missing,
      extra_tools=extra,
      strict=self.strict,
    )

  def _extract_tool_names(self, output: object) -> List[str]:
    """Extract tool function names from a RunOutput."""
    names: List[str] = []

    # RunOutput has .tool_executions or .messages with tool calls
    if hasattr(output, "tool_executions") and output.tool_executions:
      for te in output.tool_executions:
        if hasattr(te, "function_name") and te.function_name:
          names.append(te.function_name)
        elif hasattr(te, "tool_name") and te.tool_name:
          names.append(te.tool_name)

    # Fallback: scan messages for tool_call roles
    if not names and hasattr(output, "messages"):
      for msg in output.messages or []:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
          for tc in msg.tool_calls:
            if isinstance(tc, dict):
              func = tc.get("function", {})
              name = func.get("name", "") if isinstance(func, dict) else ""
              if name:
                names.append(name)
            elif hasattr(tc, "function") and hasattr(tc.function, "name"):
              names.append(tc.function.name)

    return names
