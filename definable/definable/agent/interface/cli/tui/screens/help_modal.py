"""Help modal — F1 overlay with keybindings and command reference."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
  from definable.agent.interface.cli.interface import CLIInterface


class HelpModal(ModalScreen[None]):
  """Full-screen help overlay showing keybindings, commands, and agent info.

  Dismiss with Escape or ``q``.
  """

  BINDINGS = [
    ("escape", "close", "Close"),
    ("q", "close", "Close"),
  ]

  DEFAULT_CSS = """
  HelpModal {
    align: center middle;
  }

  HelpModal > Vertical {
    width: 80;
    max-width: 90%;
    height: auto;
    max-height: 85%;
    background: $surface;
    border: round $accent;
    padding: 1 2;
  }

  HelpModal .help-title {
    text-align: center;
    text-style: bold;
    color: $accent;
    padding: 0 0 1 0;
  }

  HelpModal .help-section {
    text-style: bold;
    color: $text;
    padding: 1 0 0 0;
  }

  HelpModal .help-row {
    padding: 0 0 0 2;
    color: $text-muted;
  }

  HelpModal .help-footer {
    text-align: center;
    color: $text-disabled;
    padding: 1 0 0 0;
    text-style: italic;
  }
  """

  def __init__(self, interface: "CLIInterface | None" = None) -> None:
    super().__init__()
    self._interface = interface

  def compose(self) -> ComposeResult:
    with Vertical():
      with VerticalScroll():
        yield Static("Definable AI \u2014 Help", classes="help-title")

        # --- Keyboard shortcuts ---
        yield Static("Keyboard Shortcuts", classes="help-section")
        bindings = [
          ("Enter", "Submit prompt"),
          ("Shift+Enter", "New line in prompt"),
          ("Ctrl+F", "Search conversation"),
          ("Ctrl+L", "Clear conversation"),
          ("Ctrl+T", "New session"),
          ("Ctrl+C", "Cancel run / Quit"),
          ("Ctrl+Q", "Quit application"),
          ("F1", "This help screen"),
          ("Up / Down", "Input history"),
          ("Alt+Up / Down", "Navigate message blocks"),
          ("Tab", "Accept command completion"),
          ("Escape", "Dismiss popup / search"),
        ]
        for key, desc in bindings:
          yield Static(f"  {key:<20} {desc}", classes="help-row")

        # --- Slash commands ---
        yield Static("Commands", classes="help-section")
        commands = [
          ("/help, /h, /?", "Show available commands"),
          ("/info, /i", "Show agent configuration"),
          ("/tools, /t", "List available tools"),
          ("/model, /m", "Show model details"),
          ("/history, /hist", "Show conversation history"),
          ("/export [path]", "Export chat to JSON"),
          ("/reset, /new", "Reset current session"),
          ("/clear, /cls", "Clear conversation display"),
          ("/sessions", "List active sessions"),
          ("/session new [name]", "Create new session"),
          ("/session <name>", "Switch to session"),
          ("/quit, /q", "Exit application"),
        ]
        for cmd, desc in commands:
          yield Static(f"  {cmd:<24} {desc}", classes="help-row")

        # --- Agent info (if available) ---
        if self._interface and self._interface.agent:
          agent = self._interface.agent
          yield Static("Agent", classes="help-section")
          name = agent.agent_name or "unnamed"
          model_id = getattr(agent, "_model_id", None) or getattr(agent.model, "id", "?")
          provider = getattr(agent.model, "provider", "?")
          tool_count = len(agent.tools) if agent.tools else 0
          yield Static(f"  Name:      {name}", classes="help-row")
          yield Static(f"  Model:     {model_id} ({provider})", classes="help-row")
          yield Static(f"  Tools:     {tool_count}", classes="help-row")

          # Features
          features = []
          for feat, attr in [
            ("memory", "_memory"),
            ("knowledge", "_knowledge"),
            ("thinking", "_thinking"),
            ("tracing", "_tracing"),
            ("research", "_deep_research"),
            ("guardrails", "_guardrails"),
            ("sub-agents", "_sub_agents"),
          ]:
            if getattr(agent, attr, None):
              features.append(feat)
          if features:
            yield Static(f"  Features:  {', '.join(features)}", classes="help-row")

        yield Static("Press Escape or q to close", classes="help-footer")

  def action_close(self) -> None:
    """Dismiss the modal."""
    self.dismiss(None)
