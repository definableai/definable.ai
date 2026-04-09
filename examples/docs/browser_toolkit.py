import asyncio

from definable.browser import BrowserToolkit

from support import MockBrowser


async def main() -> None:
  async with BrowserToolkit(browser=MockBrowser()) as toolkit:
    navigate = next(tool for tool in toolkit.tools if tool.name == "browser_navigate")
    snapshot = next(tool for tool in toolkit.tools if tool.name == "browser_snapshot")

    result = await navigate.entrypoint(url="https://example.com")
    tree = await snapshot.entrypoint()

    assert result == "Navigated to https://example.com"
    assert "ref=e1" in tree


asyncio.run(main())
