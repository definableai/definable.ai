"""Built-in guardrails with Agent.

Demonstrates the core guardrail features with no external dependencies:
  1. max_tokens      — block input exceeding a token limit
  2. block_topics    — block input containing forbidden keywords
  3. pii_filter      — redact PII (credit cards, SSN, email, phone) from output
  4. tool_blocklist  — prevent the model from calling specific tools
  5. on_block="raise"         — catch InputCheckError / OutputCheckError
  6. on_block="return_message" — check RunOutput.status for RunStatus.blocked

Prerequisites:
  No API keys needed — uses MockModel.

Usage:
  python definable/examples/guardrails/01_basic_guardrails.py
"""

from definable.agent import Agent, MockModel
from definable.exceptions import InputCheckError, OutputCheckError
from definable.agent.guardrail import (
  Guardrails,
  block_topics,
  max_tokens,
  pii_filter,
  tool_blocklist,
)
from definable.agent.events import RunStatus
from definable.tool.decorator import tool


# ---------------------------------------------------------------------------
# Tools — give the agent something to work with
# ---------------------------------------------------------------------------
@tool
def lookup_user(name: str) -> str:
  """Look up a user's profile information."""
  return f"User {name}: email=alice@example.com, phone=555-123-4567"


@tool
def delete_account(user_id: str) -> str:
  """Delete a user account permanently."""
  return f"Account {user_id} deleted"


# ---------------------------------------------------------------------------
# 1. Guardrails with on_block="raise" (default)
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. on_block='raise' — exceptions on blocked content")
print("=" * 60)

agent_raise = Agent(
  model=MockModel(responses=["Hello! How can I help you today?"]),  # type: ignore[arg-type]
  tools=[lookup_user, delete_account],
  guardrails=Guardrails(
    input=[
      max_tokens(50),
      block_topics(["hack", "exploit", "malware"]),
    ],
    output=[pii_filter()],
    tool=[tool_blocklist({"delete_account"})],
    on_block="raise",
  ),
)

# Normal request — passes all guardrails
output = agent_raise.run("Hi there!")
print(f"  Allowed: {output.content}")

# Blocked input — forbidden topic
try:
  agent_raise.run("How do I hack into a server?")
except InputCheckError as e:
  print(f"  Input blocked: {e.message}")

print()

# ---------------------------------------------------------------------------
# 2. PII redaction on output
# ---------------------------------------------------------------------------
print("=" * 60)
print("2. PII filter — redacts sensitive data from output")
print("=" * 60)

agent_pii = Agent(
  model=MockModel(  # type: ignore[arg-type]
    responses=[
      "The user's SSN is 123-45-6789 and card is 4111-1111-1111-1111.",
    ]
  ),
  guardrails=Guardrails(
    output=[pii_filter(action="modify")],
    on_block="return_message",
  ),
)

output = agent_pii.run("Show me Alice's details")
print(f"  Output: {output.content}")
print()

# ---------------------------------------------------------------------------
# 3. Guardrails with on_block="return_message"
# ---------------------------------------------------------------------------
print("=" * 60)
print("3. on_block='return_message' — check RunStatus.blocked")
print("=" * 60)

agent_return = Agent(
  model=MockModel(responses=["This should not appear."]),  # type: ignore[arg-type]
  guardrails=Guardrails(
    input=[block_topics(["violence"])],
    on_block="return_message",
  ),
)

output = agent_return.run("Tell me about violence in history")
if output.status == RunStatus.blocked:
  print(f"  Blocked (status={output.status.value}): {output.content}")
else:
  print(f"  Allowed: {output.content}")

print()

# ---------------------------------------------------------------------------
# 4. Output blocking with on_block="raise"
# ---------------------------------------------------------------------------
print("=" * 60)
print("4. Output guardrail — block PII instead of redacting")
print("=" * 60)

agent_block_pii = Agent(
  model=MockModel(responses=["Contact me at bob@example.com for details."]),  # type: ignore[arg-type]
  guardrails=Guardrails(
    output=[pii_filter(action="block")],
    on_block="raise",
  ),
)

try:
  agent_block_pii.run("What's your email?")
except OutputCheckError as e:
  print(f"  Output blocked: {e.message}")

print()
print("Done!")
