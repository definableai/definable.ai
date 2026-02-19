"""SeleniumBase browser — direct tool usage (no agent).

Use this to explore what each browser tool does and what it returns,
without needing an OpenAI key.

Requirements:
    pip install 'definable[browser]'

Usage:
    python definable/examples/browser/02_browser_tools_direct.py
"""

import asyncio

from definable.browser import BrowserConfig, SeleniumBaseBrowser


async def main():
  config = BrowserConfig(
    headless=True,  # set False to watch the browser
    timeout=20.0,
  )
  browser = SeleniumBaseBrowser(config)
  await browser.start()

  print("=" * 60)
  print("navigate()")
  print("=" * 60)
  result = await browser.navigate("https://example.com")
  print(result)

  print("\n" + "=" * 60)
  print("get_url()")
  print("=" * 60)
  print(await browser.get_url())

  print("\n" + "=" * 60)
  print("get_text()  [first 300 chars]")
  print("=" * 60)
  text = await browser.get_text()
  print(text[:300])

  print("\n" + "=" * 60)
  print("get_page_info()  [situational snapshot]")
  print("=" * 60)
  print(await browser.get_page_info())

  print("\n" + "=" * 60)
  print("get_attribute()  — href of the first <a> tag")
  print("=" * 60)
  print(await browser.get_attribute("a", "href"))

  print("\n" + "=" * 60)
  print("execute_js()  — document.title")
  print("=" * 60)
  print(await browser.execute_js("document.title"))

  print("\n" + "=" * 60)
  print("scroll_down()  — scroll down 2 units")
  print("=" * 60)
  print(await browser.scroll_down(2))

  print("\n" + "=" * 60)
  print("screenshot()  [file path]")
  print("=" * 60)
  path = await browser.screenshot()
  print(path)

  # --- DuckDuckGo search to demo type_text + press_keys ---
  print("\n" + "=" * 60)
  print("navigate() → wait_for_element() → type_text() → press_keys()")
  print("=" * 60)
  await browser.navigate("https://duckduckgo.com")
  print(await browser.wait_for_element("[name='q']", timeout=10.0))
  print(await browser.type_text("[name='q']", "SeleniumBase Python"))
  print(await browser.press_keys("[name='q']", "\n"))
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
