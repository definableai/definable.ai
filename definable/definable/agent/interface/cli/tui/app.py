"""DefinableApp — top-level Textual application for the CLI interface."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from textual.app import App

from definable.agent.interface.cli.tui.router import EventRouter
from definable.agent.interface.cli.tui.screens.help_modal import HelpModal
from definable.agent.interface.cli.tui.screens.main import MainScreen

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.cli.interface import CLIInterface

# Path to the TCSS styles directory
_STYLES_DIR = Path(__file__).parent / "styles"


class DefinableApp(App):
  """Textual TUI application for Definable agent interaction.

  This is the top-level Textual application that manages:
  - The main conversation screen
  - Pipeline event routing to widgets
  - Global keybindings
  - Theme management
  """

  TITLE = "Definable AI"
  SUB_TITLE = ""
  CSS_PATH = [_STYLES_DIR / "base.tcss"]

  BINDINGS = [
    ("ctrl+q", "quit", "Quit"),
    ("f1", "help", "Help"),
  ]

  def __init__(
    self,
    *,
    agent: "Agent",
    interface: "CLIInterface",
  ) -> None:
    super().__init__()
    self._agent = agent
    self._interface = interface
    self._router: Optional[EventRouter] = None
    self._event_handler: Optional[object] = None

    # Set app subtitle to model info
    model_id = str(getattr(agent, "_model_id", None) or getattr(agent.model, "id", ""))
    self.sub_title = model_id

  async def on_mount(self) -> None:
    """Initialize the app — push main screen, subscribe to events."""
    # Subscribe to pipeline events
    self._router = EventRouter(self)
    pipeline = getattr(self._agent, "pipeline", None)
    if pipeline is not None:
      event_stream = getattr(pipeline, "event_stream", None)
      if event_stream is not None:
        self._event_handler = self._router.handle
        event_stream.subscribe(self._event_handler)

    # Determine model name for status bar
    model_id = str(getattr(self._agent, "_model_id", None) or getattr(self._agent.model, "id", ""))
    provider = str(getattr(self._agent.model, "provider", ""))
    model_display = f"{model_id} ({provider})" if provider else model_id

    # Push main screen with tools_expand config
    tools_expand = getattr(self._interface, "_cli_config", None)
    tools_expand = getattr(tools_expand, "tools_expand", "success") if tools_expand else "success"
    main = MainScreen(interface=self._interface, model_name=model_display, tools_expand=tools_expand)
    await self.push_screen(main)

  async def on_unmount(self) -> None:
    """Cleanup — unsubscribe from events."""
    if self._event_handler is not None and self._agent is not None:
      pipeline = getattr(self._agent, "pipeline", None)
      if pipeline is not None:
        event_stream = getattr(pipeline, "event_stream", None)
        if event_stream is not None:
          event_stream.unsubscribe(self._event_handler)
      self._event_handler = None

  def action_help(self) -> None:
    """Show the help modal overlay."""
    self.push_screen(HelpModal(interface=self._interface))
