"""Built-in input guardrails: max_tokens, block_topics, regex_filter."""

from __future__ import annotations

import re
from typing import List, Literal

from definable.guardrails.base import GuardrailResult
from definable.run.base import RunContext


# ------------------------------------------------------------------
# max_tokens
# ------------------------------------------------------------------


class _MaxTokensGuardrail:
  """Block input that exceeds a token limit."""

  def __init__(self, n: int, model_id: str = "gpt-4o"):
    self.name = "max_tokens"
    self._limit = n
    self._model_id = model_id

  async def check(self, text: str, context: RunContext) -> GuardrailResult:
    from definable.tokens import count_text_tokens

    count = count_text_tokens(text, self._model_id)
    if count > self._limit:
      return GuardrailResult.block(f"Input exceeds token limit ({count} > {self._limit})")
    return GuardrailResult.allow()


def max_tokens(n: int, model_id: str = "gpt-4o") -> _MaxTokensGuardrail:
  """Create an input guardrail that blocks input exceeding *n* tokens."""
  return _MaxTokensGuardrail(n, model_id)


# ------------------------------------------------------------------
# block_topics
# ------------------------------------------------------------------


class _BlockTopicsGuardrail:
  """Block input containing any of the given topic keywords (case-insensitive)."""

  def __init__(self, topics: List[str]):
    self.name = "block_topics"
    self._topics = [t.lower() for t in topics]

  async def check(self, text: str, context: RunContext) -> GuardrailResult:
    lower = text.lower()
    for topic in self._topics:
      if topic in lower:
        return GuardrailResult.block(f"Blocked topic detected: {topic}")
    return GuardrailResult.allow()


def block_topics(topics: List[str]) -> _BlockTopicsGuardrail:
  """Create an input guardrail that blocks input containing any of *topics*."""
  return _BlockTopicsGuardrail(topics)


# ------------------------------------------------------------------
# regex_filter
# ------------------------------------------------------------------


class _RegexFilterGuardrail:
  """Block or redact input matching any of the given regex patterns."""

  def __init__(self, patterns: List[str], action: Literal["block", "modify"] = "block"):
    self.name = "regex_filter"
    self._patterns = [re.compile(p) for p in patterns]
    self._action = action

  async def check(self, text: str, context: RunContext) -> GuardrailResult:
    for pattern in self._patterns:
      if pattern.search(text):
        if self._action == "block":
          return GuardrailResult.block(f"Input matches blocked pattern: {pattern.pattern}")
        # modify — redact matches
        redacted = text
        for p in self._patterns:
          redacted = p.sub("[REDACTED]", redacted)
        return GuardrailResult.modify(redacted, reason="Regex patterns redacted")
    return GuardrailResult.allow()


def regex_filter(
  patterns: List[str],
  action: Literal["block", "modify"] = "block",
) -> _RegexFilterGuardrail:
  """Create an input guardrail that blocks or redacts text matching *patterns*."""
  return _RegexFilterGuardrail(patterns, action)
