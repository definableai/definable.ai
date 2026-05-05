"""
Human-in-the-Loop (HITL) agent with permissions and questions.

This example shows how to:
- Gate tool execution with Allow / Deny / Always Allow
- Persist "Always Allow" decisions to .definable/settings.json
- Let the agent ask the user questions mid-run via the ask_user tool

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import asyncio

from definable.agent import Agent
from definable.agent.hitl import (
  Answer,
  PermissionAction,
  PermissionDecision,
  PermissionRequest,
  PermissionResponse,
  Question,
)
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def read_file(path: str) -> str:
  """Read a file from disk."""
  return f"Contents of {path}: Hello World!"


@tool
def delete_file(path: str) -> str:
  """Permanently delete a file from disk."""
  return f"File '{path}' has been permanently deleted."


@tool
def run_shell(command: str) -> str:
  """Execute a shell command."""
  return f"$ {command}\nCommand executed successfully."


# ---------------------------------------------------------------------------
# Permission resolver (CLI prompt)
# ---------------------------------------------------------------------------


async def cli_permission_prompt(request: PermissionRequest) -> PermissionResponse:
  """Prompt the user for tool permission in the terminal."""
  print(f"\n  Tool: {request.tool_name}")
  print(f"  Args: {request.tool_args}")
  print("  [a] Allow once  [l] Always allow  [d] Deny")

  choice = input("  > ").strip().lower()
  if choice == "l":
    return PermissionResponse(decision=PermissionDecision.allow_always)
  elif choice == "d":
    feedback = input("  Reason (optional): ").strip() or None
    return PermissionResponse(decision=PermissionDecision.deny, feedback=feedback)
  return PermissionResponse(decision=PermissionDecision.allow_once)


# ---------------------------------------------------------------------------
# Question resolver (CLI prompt)
# ---------------------------------------------------------------------------


async def cli_question_prompt(questions: list[Question]) -> list[Answer]:
  """Prompt the user to answer questions in the terminal."""
  answers: list[Answer] = []
  for q in questions:
    print(f"\n  {q.header or 'Question'}:")
    print(f"  {q.text}")
    if q.options:
      for i, opt in enumerate(q.options):
        desc = f" - {opt.description}" if opt.description else ""
        print(f"    [{i + 1}] {opt.label}{desc}")
      raw = input("  Choose (number): ").strip()
      try:
        idx = int(raw) - 1
        answers.append(Answer(question_text=q.text, selected=[q.options[idx].label]))
      except (ValueError, IndexError):
        answers.append(Answer(question_text=q.text, custom_text=raw))
    else:
      text = input("  Answer: ").strip()
      answers.append(Answer(question_text=q.text, custom_text=text))
  return answers


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def main():
  print("HITL Demo — Permissions + Questions")
  print("=" * 50)

  agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[read_file, delete_file, run_shell],
    instructions=(
      "You are a system admin assistant. "
      "When the user asks you to do something, use the appropriate tool. "
      "If you need clarification, use the ask_user tool to ask questions."
    ),
    # Permission: read_file is auto-allowed, others require user approval
    permission_resolver=cli_permission_prompt,
    permission_defaults={"read_file": PermissionAction.allow},
    # Question: enable the ask_user tool
    question_resolver=cli_question_prompt,
  )

  print("\nAgent ready. Tools: read_file (auto-allowed), delete_file (ask), run_shell (ask)")
  print("'Always Allow' decisions persist to .definable/settings.json\n")

  # Run 1: read_file should auto-allow (no prompt)
  print("--- Run 1: Reading a file (should auto-allow) ---")
  output = await agent.arun("Read the file at /tmp/hello.txt")
  print(f"Agent: {output.content}\n")

  # Run 2: delete_file should prompt for permission
  print("--- Run 2: Deleting a file (will prompt) ---")
  output = await agent.arun("Delete the file /tmp/old.log")
  print(f"Agent: {output.content}\n")

  # Run 3: If you chose "Always Allow" above, this won't prompt
  print("--- Run 3: Another delete (tests persistence) ---")
  output = await agent.arun("Delete /tmp/another.log")
  print(f"Agent: {output.content}\n")

  print("=" * 50)
  print("Done! Check .definable/settings.json for persisted rules.")


if __name__ == "__main__":
  asyncio.run(main())
