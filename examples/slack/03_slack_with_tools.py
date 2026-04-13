"""Slack bot with tools, memory, and media support.

Demonstrates a full-featured Slack agent with:
  - Custom tools for the agent to call
  - Memory for conversation continuity
  - File/image handling
  - Thread-based conversations
  - Access control via allowed channels

Usage:
  pip install 'definable[slack]'
  python 03_slack_with_tools.py
"""

import asyncio
import os
from datetime import datetime

from definable.agent import Agent
from definable.agent.interface.slack import SlackInterface
from definable.memory import Memory
from definable.tool.decorator import tool


@tool
def get_current_time() -> str:
  """Get the current date and time."""
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


@tool
def calculate(expression: str) -> str:
  """Evaluate a mathematical expression safely.

  Args:
    expression: A mathematical expression like '2 + 2' or 'sqrt(16)'.
  """
  import math

  allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
  allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
  try:
    result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
    return str(result)
  except Exception as e:
    return f"Error: {e}"


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions=(
    "You are a helpful Slack assistant with access to tools.\n"
    "Use markdown formatting in responses — it will be converted to Slack format.\n"
    "When users share files, acknowledge them and describe what you see."
  ),
  tools=[get_current_time, calculate],
  memory=Memory(),
)

interface = SlackInterface(
  agent=agent,
  bot_token=os.environ["SLACK_BOT_TOKEN"],
  app_token=os.environ["SLACK_APP_TOKEN"],
  # Show ✅ when done processing
  done_reaction="white_check_mark",
  # Only respond in specific channels (optional)
  # allowed_channel_ids=["C0123456789"],
)

asyncio.run(interface.serve_forever())
