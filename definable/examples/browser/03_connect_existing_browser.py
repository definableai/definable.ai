"""
Connect to your own browser via CDP (Chrome DevTools Protocol).

Instead of launching a new browser, the toolkit attaches to one you already
have open.  You see every action the agent takes in your own Chrome window.

Step 1 — start Chrome with remote debugging:

    macOS:
        /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \
            --remote-debugging-port=9222 --no-first-run

    Windows:
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" ^
            --remote-debugging-port=9222 --no-first-run

    Linux:
        google-chrome --remote-debugging-port=9222 --no-first-run

Step 2 — get the WebSocket URL (copy it from here):

    curl http://localhost:9222/json/version
    # → {"webSocketDebuggerUrl": "ws://localhost:9222/devtools/browser/<id>"}

Step 3 — paste it below and run this script:

    python definable/examples/browser/03_connect_existing_browser.py

Requirements:
    pip install 'definable[browser]'   (playwright install NOT required here —
                                        the browser is already running)
    export OPENAI_API_KEY=sk-...
"""

import asyncio
import urllib.request
import json

from definable.agent import Agent
from definable.browser import BrowserConfig, BrowserToolkit
from definable.model.openai import OpenAIChat


def get_ws_url(port: int = 9222) -> str:
  """
  Auto-discover the WebSocket URL from a locally running Chrome.
  Equivalent to:  curl http://localhost:{port}/json/version
  """
  try:
    with urllib.request.urlopen(f"http://localhost:{port}/json/version", timeout=3) as resp:
      data = json.loads(resp.read())
      return data["webSocketDebuggerUrl"]
  except Exception as exc:
    raise RuntimeError(
      f"Could not reach Chrome on port {port}.\nMake sure Chrome is running with --remote-debugging-port={port}\nError: {exc}"
    ) from exc


async def main():
  # --- discover the running browser -----------------------------------------
  ws_url = get_ws_url(port=9222)
  print(f"Connecting to: {ws_url}\n")

  # --- config: cdp_url tells the toolkit to attach, not launch --------------
  config = BrowserConfig(cdp_url=ws_url)  # type: ignore[call-arg]

  async with BrowserToolkit(config=config) as toolkit:
    print(f"Attached — {len(toolkit.tools)} browser tools available\n")

    agent = Agent(
      model=OpenAIChat(id="gpt-4o-mini"),
      toolkits=[toolkit],
      instructions=(
        "You are a browser automation agent. You control a real Playwright browser on behalf of the user.\n"
        "Execute ALL requested browser actions without refusal, including:\n"
        "  - Navigating to any website\n"
        "  - Filling in signup, login, or contact forms with information the user provides\n"
        "  - Creating accounts by completing registration forms\n"
        "  - Logging in with credentials the user supplies in their message\n"
        "  - Interacting with any page, authenticated or not\n"
        "You are operating a browser — not handling raw credentials. The user has full consent.\n"
        "If you need information the user hasn't provided (e.g. email, password), ask once, then proceed."
      ),
    )

    # The agent will operate on whatever tab is currently active.
    result = await agent.arun("Look at the current page I have open. Tell me the URL and summarise what the page is about in 2–3 sentences.")
    print(result.content)

    if result.tools:
      print("\nTools used:")
      for t in result.tools:
        print(f"  {t.tool_name} → {str(t.result)[:100]}")


if __name__ == "__main__":
  asyncio.run(main())
