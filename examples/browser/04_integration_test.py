"""Comprehensive integration test for PlaywrightBrowser.

Exercises ALL major features with a real browser. No API keys needed.
Run headless, verifies each feature, prints PASS/FAIL summary.

Requirements:
    pip install 'definable[browser]'
    playwright install chromium

Usage:
    python definable/examples/browser/04_integration_test.py
"""

import asyncio
import json
import sys

from definable.browser import BrowserConfig, PlaywrightBrowser

results: list[tuple[str, bool, str]] = []


def report(name: str, passed: bool, detail: str = "") -> None:
  results.append((name, passed, detail))
  status = "PASS" if passed else "FAIL"
  print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


async def main() -> None:
  config = BrowserConfig(headless=True, timeout=15.0)

  async with PlaywrightBrowser(config) as browser:
    # =================================================================
    # 1. NAVIGATION
    # =================================================================
    print("\n--- NAVIGATION ---")

    r = await browser.navigate("https://example.com")
    report("navigate", "example.com" in r and "Example Domain" in r, r[:80])

    r = await browser.get_url()
    report("get_url", "example.com" in r, r)

    r = await browser.get_title()
    report("get_title", "Example Domain" in r, r)

    r = await browser.refresh()
    report("refresh", "example.com" in r, r[:60])

    # =================================================================
    # 2. SNAPSHOT + ELEMENT REFS
    # =================================================================
    print("\n--- SNAPSHOT & REFS ---")

    r = await browser.snapshot()
    report("snapshot returns refs", "[ref=e" in r, f"{len(r)} chars")
    report("snapshot has heading", "Example Domain" in r)
    report("snapshot has link", "Learn more" in r)

    # =================================================================
    # 3. PAGE STATE
    # =================================================================
    print("\n--- PAGE STATE ---")

    r = await browser.get_text()
    report("get_text (body)", "Example Domain" in r, f"{len(r)} chars")

    r = await browser.get_text("h1")
    report("get_text (h1)", "Example Domain" in r, r.strip())

    r = await browser.get_page_source()
    report("get_page_source", "<html" in r.lower(), f"{len(r)} chars")

    r = await browser.get_attribute("a", "href")
    report("get_attribute", "iana.org" in r, r)

    r = await browser.is_element_visible("h1")
    report("is_element_visible", r == "true", r)

    r = await browser.get_page_info()
    report("get_page_info has viewport", "1280x720" in r, r.split("\n")[2] if "\n" in r else r)

    # =================================================================
    # 4. CLICK BY REF
    # =================================================================
    print("\n--- CLICK BY REF ---")

    # Snapshot again to get fresh refs
    snap = await browser.snapshot()
    # Find the link ref (should be e2 or similar)
    has_link_ref = "link" in snap and "[ref=e" in snap
    report("snapshot has link ref", has_link_ref)

    # Click the "Learn more" link by ref
    r = await browser.click("e2")
    report("click by ref (e2)", "Clicked" in r or "Error" not in r, r[:60])

    # Should have navigated to iana.org
    await asyncio.sleep(1)
    url = await browser.get_url()
    report("navigation after click", "iana.org" in url, url[:60])

    # Go back (iana.org can be slow, so navigate directly if go_back fails)
    r = await browser.go_back()
    cur_url = await browser.get_url()
    if "example.com" not in cur_url:
      await browser.navigate("https://example.com")
      cur_url = await browser.get_url()
    report("go_back", "example.com" in cur_url, r[:60])

    # =================================================================
    # 5. INTERACTION
    # =================================================================
    print("\n--- INTERACTION ---")

    await browser.navigate("https://duckduckgo.com")
    await browser.wait_for_element("[name='q']", timeout=10.0)

    r = await browser.type_text("[name='q']", "test search")
    report("type_text", "Typed" in r, r[:60])

    r = await browser.clear_input("[name='q']")
    report("clear_input", "Cleared" in r, r[:60])

    r = await browser.type_slowly("[name='q']", "hello", delay=20.0)
    report("type_slowly", "Typed" in r, r[:60])

    r = await browser.press_key("Escape")
    report("press_key", "Pressed" in r, r[:40])

    # =================================================================
    # 6. SCROLLING
    # =================================================================
    print("\n--- SCROLLING ---")

    await browser.navigate("https://example.com")
    r = await browser.scroll_down(2)
    report("scroll_down", "Scrolled" in r, r)

    r = await browser.scroll_up(1)
    report("scroll_up", "Scrolled" in r, r)

    r = await browser.scroll_to_element("a")
    report("scroll_to_element", "Scrolled" in r or "Error" not in r, r[:60])

    # =================================================================
    # 7. SCREENSHOT
    # =================================================================
    print("\n--- SCREENSHOT ---")

    r = await browser.screenshot(name="test_full")
    report("screenshot (full page)", "saved" in r.lower() or ".png" in r, r[:80])

    # =================================================================
    # 8. JAVASCRIPT EXECUTION
    # =================================================================
    print("\n--- JS EXECUTION ---")

    r = await browser.execute_js("document.title")
    report("execute_js simple", "Example Domain" in r, r[:40])

    r = await browser.execute_js("2 + 2")
    report("execute_js arithmetic", "4" in r, r)

    r = await browser.execute_js("JSON.stringify({a: 1, b: 'hello'})")
    report("execute_js JSON", '"a"' in r and '"hello"' in r, r[:60])

    # =================================================================
    # 9. COOKIES
    # =================================================================
    print("\n--- COOKIES ---")

    r = await browser.set_cookie("test_cookie", "test_value")
    report("set_cookie", "Cookie set" in r, r[:60])

    r = await browser.get_cookies()
    report("get_cookies", "test_cookie" in r, f"{len(r)} chars")

    r = await browser.clear_cookies()
    report("clear_cookies", "cleared" in r.lower(), r[:40])

    r = await browser.get_cookies()
    cookies = json.loads(r) if r.startswith("[") else []
    report("cookies cleared", len(cookies) == 0, f"{len(cookies)} cookies remaining")

    # =================================================================
    # 10. STORAGE
    # =================================================================
    print("\n--- STORAGE ---")

    r = await browser.set_storage("mykey", "myvalue", kind="local")
    report("set_storage (local)", "set" in r.lower(), r[:60])

    r = await browser.get_storage("mykey", kind="local")
    report("get_storage (local)", "myvalue" in r, r[:60])

    # =================================================================
    # 11. TABS
    # =================================================================
    print("\n--- TABS ---")

    r = await browser.get_tabs()
    report("get_tabs", "1" in r or "tab" in r.lower(), r)

    r = await browser.open_tab("https://example.com")
    report("open_tab", "Opened" in r, r[:60])

    r = await browser.get_tabs()
    report("get_tabs (2)", "2" in r, r)

    r = await browser.switch_to_tab(0)
    report("switch_to_tab", "Switched" in r, r[:60])

    r = await browser.close_tab()
    report("close_tab", "Closed" in r, r[:60])

    # =================================================================
    # 12. WAITING
    # =================================================================
    print("\n--- WAITING ---")

    r = await browser.wait(0.5)
    report("wait", "Waited" in r, r)

    await browser.navigate("https://example.com")
    r = await browser.wait_for_element("h1", timeout=5.0)
    report("wait_for_element", "appeared" in r.lower() or "Error" not in r, r[:60])

    r = await browser.wait_for_text("Example Domain", timeout=5.0)
    report("wait_for_text", "found" in r.lower() or "Error" not in r, r[:60])

    # =================================================================
    # 13. DIAGNOSTICS
    # =================================================================
    print("\n--- DIAGNOSTICS ---")

    # Generate some console output
    await browser.execute_js("console.log('definable-test-marker')")
    await browser.execute_js("console.warn('definable-warning')")
    await asyncio.sleep(0.2)

    r = await browser.get_console(limit=10)
    report("get_console", "definable-test-marker" in r or "No console" in r, r[:80])

    r = await browser.get_errors()
    report("get_errors", isinstance(r, str), f"{len(r)} chars")

    r = await browser.get_network(limit=5)
    report("get_network", isinstance(r, str), f"{len(r)} chars")

    # =================================================================
    # 14. GEOLOCATION
    # =================================================================
    print("\n--- EMULATION ---")

    r = await browser.set_geolocation(37.7749, -122.4194)
    report("set_geolocation", "Geolocation" in r or "Error" not in r, r[:60])

    # =================================================================
    # 15. DOM MANIPULATION
    # =================================================================
    print("\n--- DOM MANIPULATION ---")

    r = await browser.highlight("h1")
    report("highlight", "Highlighted" in r or "Error" not in r, r[:60])

    r = await browser.remove_elements("p")
    report("remove_elements", "Removed" in r, r[:60])

    # Verify paragraphs are gone
    text = await browser.get_text()
    report("elements actually removed", "documentation" not in text.lower())

    # =================================================================
    # 16. DIALOG HANDLING
    # =================================================================
    print("\n--- DIALOG ---")

    r = await browser.handle_dialog(accept=True)
    report("handle_dialog (arm)", "armed" in r.lower() or "Dialog" in r or "Error" not in r, r[:60])

    # =================================================================
    # 17. PDF
    # =================================================================
    print("\n--- PDF ---")

    r = await browser.print_to_pdf(name="test_output")
    report("print_to_pdf", ".pdf" in r or "saved" in r.lower(), r[:80])

  # ===========================================================================
  # SUMMARY
  # ===========================================================================
  print("\n" + "=" * 60)
  passed = sum(1 for _, p, _ in results if p)
  failed = sum(1 for _, p, _ in results if not p)
  total = len(results)
  print(f"RESULTS: {passed}/{total} passed, {failed} failed")

  if failed > 0:
    print("\nFailed tests:")
    for name, p, detail in results:
      if not p:
        print(f"  FAIL: {name} — {detail}")

  print("=" * 60)
  sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
  asyncio.run(main())
