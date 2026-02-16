"""Composable guardrail combinators: ALL, ANY, NOT, when."""

from __future__ import annotations

from typing import Any, Callable

from definable.guardrails.base import GuardrailResult
from definable.run.base import RunContext


class ALL:
  """All child guardrails must allow; if any blocks, the result is block.

  Works with input, output, and tool guardrails — delegates to the
  appropriate ``check()`` signature based on what the children accept.
  """

  def __init__(self, *guardrails: Any, name: str = "ALL"):
    self.name = name
    self._guardrails = guardrails

  async def check(self, *args: Any, **kwargs: Any) -> GuardrailResult:
    for g in self._guardrails:
      result = await g.check(*args, **kwargs)
      if result.action == "block":
        return result
    return GuardrailResult.allow()


class ANY:
  """At least one child guardrail must allow; block only if all block.

  Works with input, output, and tool guardrails.
  """

  def __init__(self, *guardrails: Any, name: str = "ANY"):
    self.name = name
    self._guardrails = guardrails

  async def check(self, *args: Any, **kwargs: Any) -> GuardrailResult:
    last_block: GuardrailResult | None = None
    for g in self._guardrails:
      result = await g.check(*args, **kwargs)
      if result.action != "block":
        return result
      last_block = result
    # All blocked — return the last block result
    return last_block or GuardrailResult.block("All guardrails blocked")


class NOT:
  """Invert a guardrail: allow ↔ block.

  If the child returns ``allow``, NOT returns ``block`` (with a default reason).
  If the child returns ``block``, NOT returns ``allow``.
  Other actions (``modify``, ``warn``) pass through unchanged.
  """

  def __init__(self, guardrail: Any, name: str = "NOT"):
    self.name = name
    self._guardrail = guardrail

  async def check(self, *args: Any, **kwargs: Any) -> GuardrailResult:
    result = await self._guardrail.check(*args, **kwargs)
    if result.action == "allow":
      return GuardrailResult.block(f"NOT({self._guardrail.name}): inverted allow → block")
    if result.action == "block":
      return GuardrailResult.allow()
    return result  # pass through modify/warn


class when:
  """Conditional guardrail: only run the child if *condition* returns True.

  If the condition is not met the guardrail is skipped (returns allow).
  The *condition* receives a :class:`RunContext` and should return ``bool``.
  """

  def __init__(self, condition: Callable[[RunContext], bool], guardrail: Any, name: str = "when"):
    self.name = name
    self._condition = condition
    self._guardrail = guardrail

  async def check(self, *args: Any, **kwargs: Any) -> GuardrailResult:
    # Extract context — it's always the last positional arg
    context: RunContext | None = None
    if args:
      last = args[-1]
      if isinstance(last, RunContext):
        context = last
    if kwargs.get("context"):
      context = kwargs["context"]

    if context is None or not self._condition(context):
      return GuardrailResult.allow()

    return await self._guardrail.check(*args, **kwargs)
