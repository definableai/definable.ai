"""Attach to an already-running Chrome via CDP.

Instead of launching a fresh browser, this connects to YOUR Chrome so the
agent operates on your actual tabs, cookies, and logged-in sessions.

Step 1 — Launch Chrome with remote debugging enabled:

    macOS:
        /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \\
            --remote-debugging-port=9222 --no-first-run

    Linux:
        google-chrome --remote-debugging-port=9222 --no-first-run

    Windows:
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" ^
            --remote-debugging-port=9222 --no-first-run

Step 2 — Run this script (auto-discovers the WebSocket URL):

    python examples/browser/03_connect_existing_browser.py

Requirements:
    pip install 'definable[browser]'
    export OPENAI_API_KEY=sk-...
"""

import asyncio
import json
import urllib.request

from definable.agent import Agent
from definable.browser import BrowserConfig, BrowserToolkit
from definable.model.openai import OpenAIChat


def discover_cdp_url(port: int = 9222) -> str:
  """Auto-discover the WebSocket URL from a locally running Chrome."""
  try:
    with urllib.request.urlopen(f"http://localhost:{port}/json/version", timeout=3) as resp:
      data = json.loads(resp.read())
      return data["webSocketDebuggerUrl"]
  except Exception as exc:
    raise RuntimeError(
      f"Cannot reach Chrome on port {port}.\nLaunch Chrome with: --remote-debugging-port={port} --no-first-run\nError: {exc}"
    ) from exc


async def main() -> None:
  ws_url = discover_cdp_url()
  print(f"Connecting to: {ws_url}\n")

  config = BrowserConfig(cdp_url=ws_url)

  async with BrowserToolkit(config=config) as toolkit:
    print(f"Attached — {len(toolkit.tools)} tools available\n")

    agent = Agent(
      model=OpenAIChat(id="gpt-4o-mini"),
      toolkits=[toolkit],
      instructions=(
        "You control the user's browser via Playwright. "
        "Always run browser_snapshot() first to see the current page. "
        "Use element refs (e1, e2, ...) from the snapshot for interactions."
      ),
    )

    result = await agent.arun("Look at whatever page I have open right now. Tell me the URL and summarize the page content in 2-3 sentences.")
    print(result.content)

    if result.tools:
      print(f"\nTools used: {[t.tool_name for t in result.tools]}")


if __name__ == "__main__":
  asyncio.run(main())
