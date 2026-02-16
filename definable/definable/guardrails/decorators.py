"""Decorators for creating guardrails from plain functions.

Usage::

    @input_guardrail
    async def no_profanity(text: str, context: RunContext) -> GuardrailResult:
        if "badword" in text.lower():
            return GuardrailResult.block("Profanity detected")
        return GuardrailResult.allow()

    @output_guardrail(name="custom_name")
    async def my_output_guard(text: str, context: RunContext) -> GuardrailResult:
        ...

    @tool_guardrail
    async def no_delete(tool_name: str, tool_args: dict, context: RunContext) -> GuardrailResult:
        if tool_name == "delete_all":
            return GuardrailResult.block("delete_all is forbidden")
        return GuardrailResult.allow()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from definable.guardrails.base import GuardrailResult
from definable.run.base import RunContext


# ------------------------------------------------------------------
# Wrapper classes that satisfy the Protocol contracts
# ------------------------------------------------------------------


class _InputGuardrailWrapper:
  """Wraps a function into an InputGuardrail-compliant object."""

  def __init__(self, fn: Callable, name: str):
    self.name = name
    self._fn = fn

  async def check(self, text: str, context: RunContext) -> GuardrailResult:
    return await self._fn(text, context)

  def __repr__(self) -> str:
    return f"InputGuardrail({self.name!r})"


class _OutputGuardrailWrapper:
  """Wraps a function into an OutputGuardrail-compliant object."""

  def __init__(self, fn: Callable, name: str):
    self.name = name
    self._fn = fn

  async def check(self, text: str, context: RunContext) -> GuardrailResult:
    return await self._fn(text, context)

  def __repr__(self) -> str:
    return f"OutputGuardrail({self.name!r})"


class _ToolGuardrailWrapper:
  """Wraps a function into a ToolGuardrail-compliant object."""

  def __init__(self, fn: Callable, name: str):
    self.name = name
    self._fn = fn

  async def check(self, tool_name: str, tool_args: Dict[str, Any], context: RunContext) -> GuardrailResult:
    return await self._fn(tool_name, tool_args, context)

  def __repr__(self) -> str:
    return f"ToolGuardrail({self.name!r})"


# ------------------------------------------------------------------
# Public decorator factories
# ------------------------------------------------------------------


def input_guardrail(fn: Optional[Callable] = None, *, name: Optional[str] = None):
  """Decorator to create an :class:`InputGuardrail` from a function.

  Supports both ``@input_guardrail`` and ``@input_guardrail(name=...)``.
  """
  if fn is not None:
    # Used as @input_guardrail (no parens)
    return _InputGuardrailWrapper(fn, name=name or fn.__name__)

  # Used as @input_guardrail(name=...)
  def decorator(f: Callable) -> _InputGuardrailWrapper:
    return _InputGuardrailWrapper(f, name=name or f.__name__)

  return decorator


def output_guardrail(fn: Optional[Callable] = None, *, name: Optional[str] = None):
  """Decorator to create an :class:`OutputGuardrail` from a function.

  Supports both ``@output_guardrail`` and ``@output_guardrail(name=...)``.
  """
  if fn is not None:
    return _OutputGuardrailWrapper(fn, name=name or fn.__name__)

  def decorator(f: Callable) -> _OutputGuardrailWrapper:
    return _OutputGuardrailWrapper(f, name=name or f.__name__)

  return decorator


def tool_guardrail(fn: Optional[Callable] = None, *, name: Optional[str] = None):
  """Decorator to create a :class:`ToolGuardrail` from a function.

  Supports both ``@tool_guardrail`` and ``@tool_guardrail(name=...)``.
  """
  if fn is not None:
    return _ToolGuardrailWrapper(fn, name=name or fn.__name__)

  def decorator(f: Callable) -> _ToolGuardrailWrapper:
    return _ToolGuardrailWrapper(f, name=name or f.__name__)

  return decorator
