"""Custom and composable guardrails.

Demonstrates advanced guardrail patterns:
  1. @input_guardrail decorator — create guardrails from plain functions
  2. @output_guardrail decorator — modify output content
  3. Class-based guardrail — implement the protocol directly
  4. ALL combinator — all guardrails must allow
  5. ANY combinator — at least one must allow
  6. NOT combinator — invert a guardrail
  7. when combinator — conditional execution based on RunContext

Prerequisites:
  No API keys needed — uses MockModel.

Usage:
  python definable/examples/guardrails/02_custom_guardrails.py
"""

from definable.agents import Agent, MockModel
from definable.exceptions import InputCheckError
from definable.guardrails import (
  ALL,
  ANY,
  NOT,
  Guardrails,
  GuardrailResult,
  block_topics,
  input_guardrail,
  max_tokens,
  output_guardrail,
  when,
)


# ---------------------------------------------------------------------------
# 1. Decorator-based custom guardrails
# ---------------------------------------------------------------------------
@input_guardrail
async def no_shouting(text: str, context) -> GuardrailResult:
  """Block messages that are entirely uppercase."""
  if text.isupper() and len(text) > 5:
    return GuardrailResult.block("Please don't shout (all caps detected)")
  return GuardrailResult.allow()


@input_guardrail(name="min_length")
async def require_minimum_length(text: str, context) -> GuardrailResult:
  """Require at least 3 characters of input."""
  if len(text.strip()) < 3:
    return GuardrailResult.block("Message too short (minimum 3 characters)")
  return GuardrailResult.allow()


@output_guardrail
async def redact_internal_urls(text: str, context) -> GuardrailResult:
  """Replace internal URLs with [REDACTED]."""
  import re

  cleaned = re.sub(r"https?://internal\.\S+", "[INTERNAL_URL_REDACTED]", text)
  if cleaned != text:
    return GuardrailResult.modify(cleaned, reason="Internal URLs redacted")
  return GuardrailResult.allow()


# ---------------------------------------------------------------------------
# 2. Class-based custom guardrail
# ---------------------------------------------------------------------------
class LanguageGuardrail:
  """Block input that contains non-ASCII characters."""

  name = "ascii_only"

  async def check(self, text: str, context) -> GuardrailResult:
    if not text.isascii():
      return GuardrailResult.block("Only ASCII characters are allowed")
    return GuardrailResult.allow()


# ---------------------------------------------------------------------------
# 3. Demo: decorator + class guardrails
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Custom guardrails (decorator + class-based)")
print("=" * 60)

agent = Agent(
  model=MockModel(responses=["Here is the info: https://internal.corp/secret-doc"]),
  guardrails=Guardrails(
    input=[no_shouting, require_minimum_length, LanguageGuardrail()],
    output=[redact_internal_urls],
    on_block="return_message",
  ),
)

# Allowed
output = agent.run("Show me the document")
print(f"  Output: {output.content}")

# Blocked by no_shouting
output = agent.run("GIVE ME EVERYTHING NOW")
print(f"  Shouting: status={output.status.value}, content={output.content}")

# Blocked by min_length
output = agent.run("Hi")
print(f"  Too short: status={output.status.value}, content={output.content}")

print()

# ---------------------------------------------------------------------------
# 4. ALL combinator — every guardrail must allow
# ---------------------------------------------------------------------------
print("=" * 60)
print("2. ALL combinator — every guardrail must allow")
print("=" * 60)

strict_input = ALL(
  max_tokens(100),
  block_topics(["spam", "scam"]),
  no_shouting,
  name="strict_input",
)

agent_all = Agent(
  model=MockModel(responses=["Got it!"]),
  guardrails=Guardrails(input=[strict_input], on_block="raise"),
)

output = agent_all.run("Please help me with my account")
print(f"  Allowed: {output.content}")

try:
  agent_all.run("This is a scam message")
except InputCheckError as e:
  print(f"  ALL blocked: {e.message}")

print()

# ---------------------------------------------------------------------------
# 5. ANY combinator — at least one must allow
# ---------------------------------------------------------------------------
print("=" * 60)
print("3. ANY combinator — at least one must allow")
print("=" * 60)

# Either a short message OR one that mentions "urgent" is allowed
flexible_check = ANY(
  max_tokens(10),
  block_topics(["urgent"]),  # NOTE: block_topics blocks "urgent", so NOT inverts it
  name="flexible_check",
)

agent_any = Agent(
  model=MockModel(responses=["Processing your request."]),
  guardrails=Guardrails(input=[flexible_check], on_block="return_message"),
)

output = agent_any.run("Short msg")
print(f"  Short message: status={output.status.value}")

print()

# ---------------------------------------------------------------------------
# 6. NOT combinator — invert a guardrail
# ---------------------------------------------------------------------------
print("=" * 60)
print("4. NOT combinator — require a topic instead of blocking it")
print("=" * 60)

# block_topics blocks "support", NOT inverts it → require "support"
must_be_support = NOT(
  block_topics(["support"]),
  name="must_mention_support",
)

agent_not = Agent(
  model=MockModel(responses=["I'll help with your support request."]),
  guardrails=Guardrails(input=[must_be_support], on_block="return_message"),
)

output = agent_not.run("I need support with my billing")
print(f"  Has 'support': status={output.status.value}")

output = agent_not.run("What's the weather today?")
print(f"  No 'support': status={output.status.value}")

print()

# ---------------------------------------------------------------------------
# 7. when combinator — conditional guardrails
# ---------------------------------------------------------------------------
print("=" * 60)
print("5. when combinator — conditional execution")
print("=" * 60)

# Only enforce topic restriction for non-admin users
restricted_for_users = when(
  condition=lambda ctx: getattr(ctx, "user_id", None) != "admin",
  guardrail=block_topics(["billing"]),
  name="billing_restricted_for_non_admins",
)

agent_when = Agent(
  model=MockModel(responses=["Processed."]),
  guardrails=Guardrails(input=[restricted_for_users], on_block="return_message"),
)

# Regular user — topic restriction applies
output = agent_when.run("I have a billing question")
print(f"  Regular user: status={output.status.value}")

# Admin user — topic restriction skipped
output = agent_when.run(
  "I have a billing question",
  user_id="admin",
)
print(f"  Admin user: status={output.status.value}")

print()
print("Done!")
