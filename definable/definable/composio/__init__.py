"""Composio integration — 1000+ SaaS tools via MCP.

Wraps Composio's Tool Router as an MCPToolkit, giving agents
dynamic access to Gmail, Slack, GitHub, and hundreds more with
per-user authentication isolation.

Quick start::

    from definable.agent import Agent
    from definable.composio import ComposioToolkit

    async def main():
        async with ComposioToolkit(user_id="user_123") as toolkit:
            agent = Agent(
                model="openai/gpt-4o",
                toolkits=[toolkit],
                instructions="Use Composio tools to help the user.",
            )
            result = await agent.arun("Search for email-related tools")
            print(result.content)

Requires::

    pip install 'definable[composio]'
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.composio.toolkit import ComposioToolkit

__all__ = ["ComposioToolkit"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
  if name == "ComposioToolkit":
    from definable.composio.toolkit import ComposioToolkit

    return ComposioToolkit

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
