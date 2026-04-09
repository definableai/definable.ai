"""Basic cron trigger example.

Runs a scheduled task every minute using a cron expression.

Prerequisites:
  pip install 'definable[cron]'
  export OPENAI_API_KEY=sk-...

Usage:
  python examples/runtime/02_cron_basic.py
"""

from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.agent.trigger import Cron

agent = Agent(
  model=OpenAIChat(id="gpt-4o-mini"),
  instructions="You are a helpful assistant. Keep responses to one sentence.",
)


@agent.on(Cron("* * * * *"))  # Every minute
async def every_minute(event):
  """Run agent with a prompt every minute."""
  return "What is one interesting science fact? Be brief."


if __name__ == "__main__":
  agent.serve(enable_server=False)
