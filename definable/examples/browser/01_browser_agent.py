# ruff: noqa: E501
"""Agent with BrowserToolkit — headless Playwright.

The simplest possible browser agent: give it a task, watch it work.
Runs headless by default so you can test without a visible window.

Requirements:
    pip install 'definable[browser]'
    playwright install chromium
    export OPENAI_API_KEY=sk-...

Usage:
    python examples/browser/01_browser_agent.py
"""

import asyncio

from definable.agent import Agent
from definable.browser import BrowserConfig, BrowserToolkit
from definable.model.openai import OpenAIChat

INSTRUCTIONS = """\
You are a browser automation agent. You control a real Playwright browser.

Workflow:
1. Use browser_snapshot() to see the page structure with element refs (e1, e2, ...).
2. Interact using refs: browser_click("e1"), browser_type("e2", "hello").
3. After navigation or page changes, take a fresh snapshot before interacting.

Rules:
- Always snapshot before clicking or typing — refs go stale after navigation.
- Use browser_wait_for_element() after navigation to confirm the page loaded.
- If an element is covered by a popup, use browser_remove_elements() first.
- For form fields, prefer refs from snapshot over CSS selectors.
"""


async def main() -> None:
  config = BrowserConfig(headless=True, timeout=20.0)

  async with BrowserToolkit(config=config) as toolkit:
    print(f"Browser ready — {len(toolkit.tools)} tools\n")

    agent = Agent(
      model=OpenAIChat(id="gpt-4o-mini"),
      toolkits=[toolkit],
      instructions=INSTRUCTIONS,
    )

    # Task: navigate, read, and extract structured data
    result = await agent.arun("Go to https://news.ycombinator.com. Read the page and tell me the top 5 story titles with their point counts.")
    print(result.content)

    # Show what tools the agent used
    if result.tools:
      print(f"\n--- Tools used ({len(result.tools)}) ---")
      for t in result.tools:
        print(f"  {t.tool_name}({', '.join(f'{k}={repr(v)[:40]}' for k, v in (t.tool_args or {}).items())})")


if __name__ == "__main__":
  asyncio.run(main())
