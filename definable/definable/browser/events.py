"""Browser action events for the Definable event system.

Emitted when the browser performs user-visible actions (click, type, navigate, etc.).
Flows through EventStream to all subscribers: tracing, debug, observability, SSE.

Usage::

    from definable.browser.events import BrowserActionEvent

    async def handler(event: BrowserActionEvent):
        print(f"{event.action}: {event.selector} -> {event.result}")

    toolkit = BrowserToolkit(config=config, on_action=handler)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time

from definable.run.base import BaseRunOutputEvent


@dataclass
class BrowserActionEvent(BaseRunOutputEvent):
  """Emitted when the browser performs an action.

  Attributes:
    event: Always "BrowserAction" — used for event type discrimination.
    action: The action name (navigate, click, type, scroll_down, snapshot, etc.).
    selector: Element ref (e1) or CSS selector used, if applicable.
    value: Primary value — URL for navigate, text for type, key for press_key, etc.
    result: Human-readable result message from the action.
    url: Page URL when the action was performed.
    timestamp: Unix timestamp (seconds) when the event was created.
    error: Error message if the action failed, empty string on success.
  """

  event: str = "BrowserAction"
  action: str = ""
  selector: str = ""
  value: str = ""
  result: str = ""
  url: str = ""
  timestamp: float = field(default_factory=time)
  error: str = ""
