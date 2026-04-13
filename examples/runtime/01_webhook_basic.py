"""Basic webhook trigger example.

Starts an HTTP server with a webhook endpoint that feeds incoming
requests to the agent.

Prerequisites:
  pip install 'definable[serve]'
  export OPENAI_API_KEY=sk-...

Usage:
  python examples/runtime/01_webhook_basic.py

  # In another terminal:
  curl -X POST http://localhost:8000/webhook \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello from webhook!"}'
"""

from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.agent.trigger import Webhook

agent = Agent(
  model=OpenAIChat(id="gpt-4o-mini"),
  instructions="You are a helpful assistant. Respond concisely.",
)


@agent.on(Webhook("/webhook"))
async def handle_webhook(event):
  """Process incoming webhook and run through agent."""
  message = event.body.get("message", "") if event.body else ""
  return message or "No message provided"


if __name__ == "__main__":
  # Use dev=True for hot reload and Swagger docs at /docs:
  #   agent.serve(port=8000, dev=True)
  agent.serve(port=8000)
