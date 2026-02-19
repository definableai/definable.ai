"""Browser automation for Definable agents via SeleniumBase CDP mode.

SeleniumBase CDP drives Chrome directly via Chrome DevTools Protocol —
no WebDriver, no automation fingerprints. It is the most stealth-capable
browser automation approach available in Python.

Quick start::

    from definable.agent import Agent
    from definable.browser import BrowserToolkit, BrowserConfig
    from definable.model.openai import OpenAIChat

    async def main():
        async with BrowserToolkit() as toolkit:
            agent = Agent(
                model=OpenAIChat(id="gpt-4o"),
                toolkits=[toolkit],
            )
            result = await agent.arun("Go to news.ycombinator.com and list the top 3 stories")
            print(result.content)

Requires::

    pip install 'definable[browser]'
"""

from typing import TYPE_CHECKING

from definable.browser.base import BaseBrowser
from definable.browser.config import BrowserConfig

if TYPE_CHECKING:
  from definable.browser.seleniumbase_browser import SeleniumBaseBrowser
  from definable.browser.toolkit import BrowserToolkit

__all__ = [
  # Always safe to import (no heavy deps)
  "BaseBrowser",
  "BrowserConfig",
  # Lazy (requires seleniumbase)
  "SeleniumBaseBrowser",
  "BrowserToolkit",
]


def __getattr__(name: str):
  if name == "SeleniumBaseBrowser":
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    return SeleniumBaseBrowser

  if name == "BrowserToolkit":
    from definable.browser.toolkit import BrowserToolkit

    return BrowserToolkit

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
