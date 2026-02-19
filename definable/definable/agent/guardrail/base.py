"""Core guardrail types: result, protocols, and container."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from definable.agent.events import RunContext
from definable.utils.log import log_debug, log_warning


@dataclass
class GuardrailResult:
  """Result returned by a guardrail check.

  Attributes:
    action: One of "allow", "block", "modify", "warn".
    message: Human-readable explanation (required for block/modify/warn).
    modified_text: Replacement text when action is "modify".
    metadata: Optional extra data for tracing / debugging.
  """

  action: Literal["allow", "block", "modify", "warn"]
  message: Optional[str] = None
  modified_text: Optional[str] = None
  metadata: Optional[Dict[str, Any]] = None

  # ------------------------------------------------------------------
  # Factory helpers
  # ------------------------------------------------------------------

  @staticmethod
  def allow() -> GuardrailResult:
    return GuardrailResult(action="allow")

  @staticmethod
  def block(reason: str) -> GuardrailResult:
    return GuardrailResult(action="block", message=reason)

  @staticmethod
  def modify(new_text: str, reason: str = "") -> GuardrailResult:
    return GuardrailResult(action="modify", modified_text=new_text, message=reason or None)

  @staticmethod
  def warn(message: str) -> GuardrailResult:
    return GuardrailResult(action="warn", message=message)


# ------------------------------------------------------------------
# Protocols
# ------------------------------------------------------------------


@runtime_checkable
class InputGuardrail(Protocol):
  """Protocol for guardrails that check user input before the LLM call."""

  name: str

  async def check(self, text: str, context: RunContext) -> GuardrailResult: ...


@runtime_checkable
class OutputGuardrail(Protocol):
  """Protocol for guardrails that check LLM output after the call."""

  name: str

  async def check(self, text: str, context: RunContext) -> GuardrailResult: ...


@runtime_checkable
class ToolGuardrail(Protocol):
  """Protocol for guardrails that check tool calls before execution."""

  name: str

  async def check(self, tool_name: str, tool_args: Dict[str, Any], context: RunContext) -> GuardrailResult: ...


# ------------------------------------------------------------------
# Container
# ------------------------------------------------------------------


@dataclass
class Guardrails:
  """Container holding input, output, and tool guardrails.

  Attributes:
    input: Guardrails run on user input before the LLM call.
    output: Guardrails run on LLM output after the call.
    tool: Guardrails run on each tool call before execution.
    mode: ``"fail_fast"`` stops at the first block; ``"run_all"``
      runs every guardrail and collects results.
    on_block: ``"raise"`` raises an exception; ``"return_message"``
      returns a :class:`RunOutput` with ``status=blocked``.
  """

  input: List[InputGuardrail] = field(default_factory=list)
  output: List[OutputGuardrail] = field(default_factory=list)
  tool: List[ToolGuardrail] = field(default_factory=list)
  mode: Literal["fail_fast", "run_all"] = "fail_fast"
  on_block: Literal["raise", "return_message"] = "raise"

  # ------------------------------------------------------------------
  # Runner methods
  # ------------------------------------------------------------------

  async def run_input_checks(self, text: str, context: RunContext) -> List[GuardrailResult]:
    """Run all input guardrails, respecting *mode*."""
    return await self._run_checks(self.input, text, context, "input")

  async def run_output_checks(self, text: str, context: RunContext) -> List[GuardrailResult]:
    """Run all output guardrails, respecting *mode*."""
    return await self._run_checks(self.output, text, context, "output")

  async def run_tool_checks(
    self,
    tool_name: str,
    tool_args: Dict[str, Any],
    context: RunContext,
  ) -> List[GuardrailResult]:
    """Run all tool guardrails, respecting *mode*."""
    results: List[GuardrailResult] = []
    for guardrail in self.tool:
      start = time.perf_counter()
      try:
        result = await guardrail.check(tool_name, tool_args, context)
      except Exception as exc:
        log_warning(f"Tool guardrail '{guardrail.name}' raised: {exc}")
        result = GuardrailResult.block(f"Guardrail error: {exc}")
      elapsed = (time.perf_counter() - start) * 1000
      result.metadata = {**(result.metadata or {}), "duration_ms": elapsed, "guardrail_name": guardrail.name}
      results.append(result)
      log_debug(f"Tool guardrail '{guardrail.name}' → {result.action} ({elapsed:.1f}ms)")
      if self.mode == "fail_fast" and result.action == "block":
        break
    return results

  # ------------------------------------------------------------------
  # Shared runner
  # ------------------------------------------------------------------

  async def _run_checks(
    self,
    guardrails: list,
    text: str,
    context: RunContext,
    guardrail_type: str,
  ) -> List[GuardrailResult]:
    results: List[GuardrailResult] = []
    for guardrail in guardrails:
      start = time.perf_counter()
      try:
        result = await guardrail.check(text, context)
      except Exception as exc:
        log_warning(f"{guardrail_type.title()} guardrail '{guardrail.name}' raised: {exc}")
        result = GuardrailResult.block(f"Guardrail error: {exc}")
      elapsed = (time.perf_counter() - start) * 1000
      result.metadata = {**(result.metadata or {}), "duration_ms": elapsed, "guardrail_name": guardrail.name}
      results.append(result)
      log_debug(f"{guardrail_type.title()} guardrail '{guardrail.name}' → {result.action} ({elapsed:.1f}ms)")
      if self.mode == "fail_fast" and result.action == "block":
        break
    return results
