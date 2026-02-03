"""ChatWindow class - PyQt-style API for chat UI."""

import webbrowser
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from definable.agents.agent import Agent


class ChatWindow:
    """
    PyQt-style chat window that controls a web-based UI.

    Example:
        >>> from definable.agents import Agent
        >>> from definable.models import OpenAIChat
        >>> 
        >>> chat = ChatWindow(title="AI Assistant", theme="dark")
        >>> agent = Agent(
        >>>     model=OpenAIChat(id="gpt-4o-mini"),
        >>>     ui=chat  # Agent auto-registers itself
        >>> )
        >>> chat.show()  # Opens browser and starts server
    """

    def __init__(
        self,
        title: str = "Definable Chat",
        width: int = 800,
        height: int = 600,
        theme: str = "light",
        port: int = 8000,
        cdn_url: Optional[str] = None,
    ):
        self.title = title
        self.width = width
        self.height = height
        self.theme = theme
        self.port = port
        self.cdn_url = cdn_url

        self._server = None
        self.agent: Optional["Agent"] = None  # Set by Agent when ui= is passed

    def show(self, auto_open: bool = True):
        """
        Start the UI server and open in browser (blocking call).

        Args:
            auto_open: Automatically open browser window
        """
        from definable.ui.server import UIServer

        self._server = UIServer(chat_window=self, port=self.port, cdn_url=self.cdn_url)

        if auto_open:
            webbrowser.open(f"http://localhost:{self.port}")

        print(f"🚀 Chat UI running at http://localhost:{self.port}")
        if self.cdn_url:
            print(f"📦 Loading web component from CDN: {self.cdn_url}")
        print("Press Ctrl+C to stop")

        # Start server (blocking)
        self._server.start()

