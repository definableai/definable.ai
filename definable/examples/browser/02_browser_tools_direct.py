"""
PlaywrightBrowser — direct tool usage (no agent).

Use this to explore what each browser tool does and what it returns,
without needing an OpenAI key.

Requirements:
    pip install 'definable[browser]'
    playwright install chromium

Usage:
    python definable/examples/browser/02_browser_tools_direct.py
"""

import asyncio
import json

from definable.browser import BrowserConfig
from definable.browser.playwright import PlaywrightBrowser


async def main():
  config = BrowserConfig(
    headless=True,  # set False to watch the browser
    timeout=20.0,
  )
  browser = PlaywrightBrowser(config)
  await browser.start()

  print("=" * 60)
  print("navigate()")
  print("=" * 60)
  result = await browser.navigate("https://linkedin.com")
  print(result)

  print("\n" + "=" * 60)
  print("get_url()")
  print("=" * 60)
  print(await browser.get_url())

  print("\n" + "=" * 60)
  print("get_page_text()  [first 300 chars]")
  print("=" * 60)
  text = await browser.get_page_text()
  print(text[:300])

  print("\n" + "=" * 60)
  print("observe()  [JSON snapshot]")
  print("=" * 60)
  snapshot = json.loads(await browser.observe())
  print(f"url:          {snapshot['url']}")
  print(f"title:        {snapshot['title']}")
  print(f"text_preview: {snapshot['text_preview'][:120]}...")

  print("\n" + "=" * 60)
  print("get_attribute()  — href of the first <a> tag")
  print("=" * 60)
  print(await browser.get_attribute("a", "href"))

  print("\n" + "=" * 60)
  print("execute_js()  — document.title")
  print("=" * 60)
  print(await browser.execute_js("document.title"))

  print("\n" + "=" * 60)
  print("scroll()  — scroll down 2 units")
  print("=" * 60)
  print(await browser.scroll("down", 2))

  print("\n" + "=" * 60)
  print("screenshot()  [base64 PNG, first 60 chars]")
  print("=" * 60)
  b64 = await browser.screenshot()
  print(b64[:60] + "...")
  print(f"(total {len(b64)} chars)")

  # --- DuckDuckGo search to demo type + click ---
  print("\n" + "=" * 60)
  print("navigate() → wait_for() → type_text() → press_key()")
  print("=" * 60)
  await browser.navigate("https://duckduckgo.com")
  print(await browser.wait_for("[name='q']", timeout=10.0))
  print(await browser.type_text("Playwright Python", selector="[name='q']"))
  print(await browser.press_key("Enter"))
  await asyncio.sleep(1.5)  # let results load
  print(await browser.get_url())

  print("\n" + "=" * 60)
  print("go_back() / go_forward()")
  print("=" * 60)
  print(await browser.go_back())
  print(await browser.go_forward())

  await browser.stop()
  print("\nDone.")


if __name__ == "__main__":
  asyncio.run(main())
