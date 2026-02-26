"""Interactive terminal for ComposioToolkit with observability dashboard.

Usage:
    pip install composio
    source .env.test
    .venv/bin/python test_composio.py

Observability dashboard: http://localhost:7777/obs/
"""

import asyncio

from definable.agent import Agent
from definable.agent.compression import Compression
from definable.agent.interface.cli import CLIInterface
from definable.agent.tracing import JSONLExporter, Tracing
from definable.composio import ComposioToolkit


async def main() -> None:
  toolkit = ComposioToolkit(user_id="123")
  await toolkit.initialize()

  agent = Agent(
    model="openai/gpt-5.2",
    memory=True,
    toolkits=[toolkit],
    instructions="You have access to Composio tools. Search for tools, handle auth, execute actions.",
    observability=True,
    tracing=Tracing(exporters=[JSONLExporter()]),
    compression=Compression(token_limit=10000),
  )

  await agent.aserve(CLIInterface(mode="repl"), enable_server=True, port=8001)


if __name__ == "__main__":
  asyncio.run(main())
