"""
Tool hooks: pre_hook and post_hook.

This example shows how to:
- Add pre_hook for logging/validation before tool execution
- Add post_hook for logging/processing after tool execution
- Use hooks for timing, logging, and transformation

Requirements:
    export OPENAI_API_KEY=sk-...

Note: Hooks receive keyword arguments: fc (function call), agent, session_state, dependencies
"""

import time

from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool


# Simple logging hooks using **kwargs pattern
def log_before(**kwargs) -> None:
  """Log before tool execution."""
  fc = kwargs.get("fc")
  if fc:
    print(f"  [PRE-HOOK] Calling: {fc.function.name}")
    print(f"  [PRE-HOOK] Args: {fc.arguments}")


def log_after(**kwargs) -> None:
  """Log after tool execution."""
  fc = kwargs.get("fc")
  if fc:
    print(f"  [POST-HOOK] Result: {fc.result}")


@tool(pre_hook=log_before, post_hook=log_after)
def add(a: int, b: int) -> int:
  """Add two numbers together."""
  return a + b


# Timing hooks
timing_start: float = 0


def start_timer(**kwargs) -> None:
  """Start timing the tool execution."""
  global timing_start
  timing_start = time.time()
  fc = kwargs.get("fc")
  name = fc.function.name if fc else "unknown"
  print(f"  [TIMER] Starting {name}...")


def end_timer(**kwargs) -> None:
  """End timing and log duration."""
  global timing_start
  duration = time.time() - timing_start
  fc = kwargs.get("fc")
  name = fc.function.name if fc else "unknown"
  print(f"  [TIMER] {name} completed in {duration:.4f}s")


@tool(pre_hook=start_timer, post_hook=end_timer)
def slow_operation(iterations: int) -> str:
  """Perform a slow operation (for timing demonstration)."""
  total = 0
  for i in range(iterations):
    total += i
  return f"Computed sum of 0 to {iterations}: {total}"


# Validation hook
def validate_positive(**kwargs) -> None:
  """Validate that all numeric arguments are positive."""
  fc = kwargs.get("fc")
  if fc:
    args = fc.arguments
    for key, value in args.items():
      if isinstance(value, (int, float)) and value < 0:
        raise ValueError(f"Argument '{key}' must be positive, got {value}")
    print(f"  [VALIDATION] All arguments valid for {fc.function.name}")


@tool(pre_hook=validate_positive)
def calculate_square_root(number: float) -> float:
  """Calculate the square root of a positive number."""
  return number**0.5


# Result logging hook
def log_result(**kwargs) -> None:
  """Log the result (for logging purposes)."""
  fc = kwargs.get("fc")
  if fc:
    print(f"  [RESULT] {fc.function.name} returned: {fc.result}")


@tool(post_hook=log_result)
def get_user_info(user_id: int) -> str:
  """Get information about a user."""
  users = {
    1: "Alice (admin)",
    2: "Bob (user)",
    3: "Charlie (user)",
  }
  return users.get(user_id, "Unknown user")


# Audit logging hook
audit_log = []


def audit_before(**kwargs) -> None:
  """Record tool call in audit log."""
  fc = kwargs.get("fc")
  if fc:
    entry = {
      "tool": fc.function.name,
      "args": fc.arguments,
      "timestamp": time.time(),
      "type": "start",
    }
    audit_log.append(entry)
    print(f"  [AUDIT] Recorded call to {fc.function.name}")


def audit_after(**kwargs) -> None:
  """Record tool result in audit log."""
  fc = kwargs.get("fc")
  if fc:
    entry = {
      "tool": fc.function.name,
      "result": str(fc.result)[:100],  # Truncate long results
      "timestamp": time.time(),
      "type": "end",
    }
    audit_log.append(entry)


@tool(pre_hook=audit_before, post_hook=audit_after)
def sensitive_operation(action: str, resource: str) -> str:
  """Perform a sensitive operation that should be audited."""
  return f"Performed '{action}' on '{resource}'"


def main():
  """Demonstrate tool hooks."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[
      add,
      slow_operation,
      calculate_square_root,
      get_user_info,
      sensitive_operation,
    ],
    instructions="""You are a helpful assistant with access to various tools.
Use the appropriate tool to answer user questions.""",
  )

  print("Tool Hooks Demonstration")
  print("=" * 50)

  # Test logging hooks
  print("\n1. Logging Hooks (add):")
  output = agent.run("Add 5 and 3")
  print(f"Final answer: {output.content}")

  # Test timing hooks
  print("\n2. Timing Hooks (slow_operation):")
  output = agent.run("Run a slow operation with 10000 iterations")
  print(f"Final answer: {output.content}")

  # Test validation hooks
  print("\n3. Validation Hooks (calculate_square_root):")
  output = agent.run("What is the square root of 16?")
  print(f"Final answer: {output.content}")

  # Test result logging hooks
  print("\n4. Result Logging Hooks (get_user_info):")
  output = agent.run("Get info about user 1")
  print(f"Final answer: {output.content}")

  # Test audit logging hooks
  print("\n5. Audit Logging Hooks (sensitive_operation):")
  output = agent.run("Perform a 'delete' action on 'old_records'")
  print(f"Final answer: {output.content}")

  # Show audit log
  print("\nAudit Log:")
  for entry in audit_log:
    print(f"  {entry['type']}: {entry['tool']}")


if __name__ == "__main__":
  main()
