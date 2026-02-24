"""PlaywrightBrowser — direct usage without an agent.

No API keys needed. Exercises the core browser methods directly to show
what each one returns. Good for understanding the API before wiring it
into an agent.

Requirements:
    pip install 'definable[browser]'
    playwright install chromium

Usage:
    python examples/browser/02_browser_tools_direct.py
"""

import asyncio

from definable.browser import BrowserConfig, PlaywrightBrowser


def section(title: str) -> None:
  print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


async def main() -> None:
  config = BrowserConfig(headless=True, timeout=15.0)

  async with PlaywrightBrowser(config) as browser:
    # -- Navigation ----------------------------------------------------------
    section("navigate + get_url + get_title")
    print(await browser.navigate("https://example.com"))
    print(f"URL:   {await browser.get_url()}")
    print(f"Title: {await browser.get_title()}")

    # -- Snapshot (the key feature) ------------------------------------------
    section("snapshot — role-based element refs")
    snap = await browser.snapshot()
    print(snap)
    print("\n^ Refs like [ref=e1] let you click/type by ref instead of CSS selectors.")

    # -- Page state ----------------------------------------------------------
    section("get_text / get_attribute / get_page_info")
    print(f"Body text: {(await browser.get_text())[:120]}...")
    print(f"Link href: {await browser.get_attribute('a', 'href')}")
    print(await browser.get_page_info())

    # -- Click by ref --------------------------------------------------------
    section("click by ref (e2 = 'Learn more' link)")
    print(await browser.click("e2"))
    await asyncio.sleep(1)
    print(f"Now at: {await browser.get_url()}")
    print(await browser.go_back())

    # -- Search on DuckDuckGo ------------------------------------------------
    section("type + press_key — DuckDuckGo search")
    await browser.navigate("https://duckduckgo.com")
    await browser.wait_for_element("[name='q']", timeout=10.0)
    print(await browser.type_text("[name='q']", "Playwright Python"))
    print(await browser.press_key("Enter"))
    await asyncio.sleep(2)
    print(f"Search URL: {await browser.get_url()}")

    # -- Screenshot ----------------------------------------------------------
    section("screenshot")
    print(await browser.screenshot(name="search_results"))

    # -- JavaScript ----------------------------------------------------------
    section("execute_js")
    print(f"document.title = {await browser.execute_js('document.title')}")
    print(f"2 + 2 = {await browser.execute_js('2 + 2')}")

    # -- Cookies + Storage ---------------------------------------------------
    section("cookies + localStorage")
    await browser.navigate("https://example.com")
    print(await browser.set_cookie("demo", "hello"))
    print(await browser.set_storage("key1", "value1", kind="local"))
    print(f"Storage: {await browser.get_storage('key1', kind='local')}")
    print(await browser.clear_cookies())

    # -- Diagnostics ---------------------------------------------------------
    section("diagnostics — console / errors / network")
    await browser.execute_js("console.log('hello from definable')")
    await browser.execute_js("console.warn('test warning')")
    await asyncio.sleep(0.2)
    print(await browser.get_console(limit=5))
    print(f"Errors: {await browser.get_errors(limit=5)}")
    print(f"Network: {await browser.get_network(limit=3)}")

  print("\nDone.")


if __name__ == "__main__":
  asyncio.run(main())
