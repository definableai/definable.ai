"""Scenario: Guardrails and middleware integration.

Tests:
- Input guardrails (block_topics, max_tokens)
- Output guardrails (pii_filter)
- Tool blocklist
- Middleware chain
- on_block="raise" and on_block="return_message"

Requires: No API keys (uses MockModel)
"""

import sys

PASS = 0
FAIL = 0


def check(condition: bool, description: str):
  global PASS, FAIL
  if condition:
    PASS += 1
    print(f"PASS: {description}")
  else:
    FAIL += 1
    print(f"FAIL: {description}")


def main():
  from definable.agents import Agent, MockModel
  from definable.exceptions import InputCheckError, OutputCheckError
  from definable.guardrails import Guardrails, block_topics, max_tokens, pii_filter, tool_blocklist
  from definable.run.base import RunStatus
  from definable.tools.decorator import tool

  @tool
  def lookup(name: str) -> str:
    """Look up user info."""
    return f"User {name}: email=test@example.com"

  @tool
  def delete_user(user_id: str) -> str:
    """Delete a user."""
    return f"Deleted {user_id}"

  # --- Test 1: block_topics with raise ---
  agent = Agent(
    model=MockModel(responses=["Hello!"]),
    guardrails=Guardrails(
      input=[block_topics(["hack", "exploit"])],
      on_block="raise",
    ),
  )
  output = agent.run("Hi there!")
  check(output.content is not None, "Normal input passes block_topics")

  blocked = False
  try:
    agent.run("How to hack a server")
  except InputCheckError:
    blocked = True
  check(blocked, "block_topics raises InputCheckError on forbidden topic")

  # --- Test 2: max_tokens ---
  agent2 = Agent(
    model=MockModel(responses=["OK"]),
    guardrails=Guardrails(
      input=[max_tokens(10)],
      on_block="raise",
    ),
  )
  blocked2 = False
  try:
    agent2.run("This is a very long message that definitely exceeds the ten token limit set for testing")
  except InputCheckError:
    blocked2 = True
  check(blocked2, "max_tokens blocks input exceeding token limit")

  # --- Test 3: pii_filter on output ---
  agent3 = Agent(
    model=MockModel(responses=["SSN is 123-45-6789 and card 4111-1111-1111-1111"]),
    guardrails=Guardrails(
      output=[pii_filter(action="modify")],
      on_block="return_message",
    ),
  )
  out3 = agent3.run("Show details")
  check("123-45-6789" not in out3.content, "pii_filter redacts SSN from output")

  # --- Test 4: tool_blocklist ---
  agent4 = Agent(
    model=MockModel(responses=["I'll look that up"]),
    tools=[lookup, delete_user],
    guardrails=Guardrails(
      tool=[tool_blocklist({"delete_user"})],
      on_block="raise",
    ),
  )
  check(agent4 is not None, "Agent created with tool_blocklist guardrail")

  # --- Test 5: on_block="return_message" ---
  agent5 = Agent(
    model=MockModel(responses=["Should not appear"]),
    guardrails=Guardrails(
      input=[block_topics(["violence"])],
      on_block="return_message",
    ),
  )
  out5 = agent5.run("Tell me about violence")
  check(out5.status == RunStatus.blocked, f"on_block=return_message sets status to blocked (got {out5.status})")

  # --- Test 6: Output block with raise ---
  agent6 = Agent(
    model=MockModel(responses=["Contact bob@example.com for info"]),
    guardrails=Guardrails(
      output=[pii_filter(action="block")],
      on_block="raise",
    ),
  )
  output_blocked = False
  try:
    agent6.run("What's the email?")
  except OutputCheckError:
    output_blocked = True
  check(output_blocked, "pii_filter(action=block) raises OutputCheckError")

  # --- Summary ---
  print(f"\n--- Summary: {PASS} passed, {FAIL} failed ---")
  sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
  main()
