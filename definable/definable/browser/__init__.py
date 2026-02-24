"""Browser automation for Definable agents via Playwright CDP.

Playwright connects to Chrome directly via Chrome DevTools Protocol —
native async, role-based element refs, AI-friendly errors, and
self-healing connections.

Quick start::

    from definable.agent import Agent
    from definable.browser import BrowserToolkit, BrowserConfig

    async def main():
        async with BrowserToolkit() as toolkit:
            agent = Agent(
                model="openai/gpt-4o",
                toolkits=[toolkit],
            )
            result = await agent.arun("Go to news.ycombinator.com and list the top 3 stories")
            print(result.content)

Requires::

    pip install 'definable[browser]'
    playwright install chromium
"""

from typing import TYPE_CHECKING

from definable.browser.base import BaseBrowser
from definable.browser.config import BrowserConfig
from definable.browser.events import BrowserActionEvent

if TYPE_CHECKING:
  from definable.browser.playwright_browser import PlaywrightBrowser
  from definable.browser.toolkit import BrowserToolkit

__all__ = [
  # Always safe to import (no heavy deps)
  "BaseBrowser",
  "BrowserConfig",
  "BrowserActionEvent",
  # Lazy (requires playwright)
  "PlaywrightBrowser",
  "BrowserToolkit",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
  if name == "PlaywrightBrowser":
    from definable.browser.playwright_browser import PlaywrightBrowser

    return PlaywrightBrowser

  if name == "BrowserToolkit":
    from definable.browser.toolkit import BrowserToolkit

    return BrowserToolkit

  # Backward compat: old name still works, emits deprecation warning
  if name == "SeleniumBaseBrowser":
    import warnings

    warnings.warn(
      "SeleniumBaseBrowser has been removed. Use PlaywrightBrowser instead.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.browser.playwright_browser import PlaywrightBrowser

    return PlaywrightBrowser

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
