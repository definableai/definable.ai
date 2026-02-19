"""Built-in output guardrails: pii_filter, max_output_tokens."""

from __future__ import annotations

import re
from typing import Literal

from definable.agent.guardrail.base import GuardrailResult
from definable.agent.events import RunContext

# ------------------------------------------------------------------
# PII regex patterns
# ------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

# Order matters: more specific patterns first to avoid partial matches
_PII_PATTERNS = [
  (_CREDIT_CARD_RE, "[CREDIT_CARD]"),
  (_SSN_RE, "[SSN]"),
  (_EMAIL_RE, "[EMAIL]"),
  (_PHONE_RE, "[PHONE]"),
]


# ------------------------------------------------------------------
# pii_filter
# ------------------------------------------------------------------


class _PIIFilterGuardrail:
  """Detect and redact PII in model output."""

  def __init__(self, action: Literal["modify", "block"] = "modify"):
    self.name = "pii_filter"
    self._action = action

  async def check(self, text: str, context: RunContext) -> GuardrailResult:
    has_pii = any(pattern.search(text) for pattern, _ in _PII_PATTERNS)
    if not has_pii:
      return GuardrailResult.allow()

    if self._action == "block":
      return GuardrailResult.block("PII detected in output")

    # Redact PII
    redacted = text
    for pattern, replacement in _PII_PATTERNS:
      redacted = pattern.sub(replacement, redacted)
    return GuardrailResult.modify(redacted, reason="PII redacted")


def pii_filter(action: Literal["modify", "block"] = "modify") -> _PIIFilterGuardrail:
  """Create an output guardrail that detects and redacts PII."""
  return _PIIFilterGuardrail(action)


# ------------------------------------------------------------------
# max_output_tokens
# ------------------------------------------------------------------


class _MaxOutputTokensGuardrail:
  """Block output that exceeds a token limit."""

  def __init__(self, n: int, model_id: str = "gpt-4o"):
    self.name = "max_output_tokens"
    self._limit = n
    self._model_id = model_id

  async def check(self, text: str, context: RunContext) -> GuardrailResult:
    from definable.tokens import count_text_tokens

    count = count_text_tokens(text, self._model_id)
    if count > self._limit:
      return GuardrailResult.block(f"Output exceeds token limit ({count} > {self._limit})")
    return GuardrailResult.allow()


def max_output_tokens(n: int, model_id: str = "gpt-4o") -> _MaxOutputTokensGuardrail:
  """Create an output guardrail that blocks output exceeding *n* tokens."""
  return _MaxOutputTokensGuardrail(n, model_id)
