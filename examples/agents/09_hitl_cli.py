"""
HITL CLI REPL — interactive permission prompts and agent questions.

This example starts a REPL where:
- Every tool call prompts you: Allow / Always Allow / Deny
- "Always Allow" persists to .definable/settings.json (remembered across runs)
- The agent can ask you questions via the ask_user tool

Try these prompts:
  >>> read the file /tmp/hello.txt
  >>> delete the file /tmp/old.log
  >>> what time is it

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from definable.agent import Agent
from definable.agent.interface.cli import CLIInterface
from definable.tool.decorator import tool


@tool
def read_file(path: str) -> str:
  """Read a file from disk."""
  return f"Contents of {path}: Hello World!"


@tool
def delete_file(path: str) -> str:
  """Permanently delete a file from disk."""
  return f"File '{path}' has been permanently deleted."


@tool
def get_time() -> str:
  """Get the current server time."""
  from datetime import datetime, timezone

  return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


agent = Agent(
  model="openai/gpt-4o-mini",
  tools=[read_file, delete_file, get_time],
  instructions=(
    "You are a system admin assistant. Use the appropriate tool when the user asks. "
    "If you need clarification, use the ask_user tool to ask questions."
  ),
  interfaces=[CLIInterface(mode="repl", enable_completions=False)],
)

agent.serve()
