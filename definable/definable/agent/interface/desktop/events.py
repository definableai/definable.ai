"""Desktop Bridge events for the Definable event system.

Emitted when the MacOS skill or BridgeClient interacts with the Desktop Bridge.
Two tiers of events:

- **BridgeCallEvent** — transport layer, emitted on every HTTP call to the bridge.
  Useful for debugging connectivity, latency, and raw request/response flow.

- **DesktopActionEvent** — domain layer, emitted for user-visible desktop actions
  (click, screenshot, open app, etc.). Similar to BrowserActionEvent. Useful for
  tracing agent behavior at the semantic level.

Usage::

    from definable.agent.interface.desktop.events import DesktopActionEvent, BridgeCallEvent

    async def handler(event):
        if isinstance(event, DesktopActionEvent):
            print(f"{event.category}/{event.action}: {event.result}")

    skill = MacOS(on_event=handler)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time

from definable.agent.run.base import BaseRunOutputEvent


@dataclass
class BridgeCallEvent(BaseRunOutputEvent):
  """Emitted on every HTTP call to the Desktop Bridge.

  This is the transport-level event — one per ``_post()`` call. Captures
  the raw endpoint, payload shape, response time, and any errors.

  Attributes:
    event: Always ``"BridgeCall"`` — used for event type discrimination.
    endpoint: Bridge HTTP path (e.g. ``"/screen/capture"``, ``"/input/click"``).
    method: HTTP method (always ``"POST"`` for current bridge).
    status_code: HTTP response status code (200, 500, etc.). 0 if connection failed.
    duration_ms: Round-trip time in milliseconds.
    error: Error message if the call failed, empty string on success.
    timestamp: Unix timestamp (seconds) when the event was created.
  """

  event: str = "BridgeCall"
  endpoint: str = ""
  method: str = "POST"
  status_code: int = 0
  duration_ms: float = 0.0
  error: str = ""
  timestamp: float = field(default_factory=time)


@dataclass
class DesktopActionEvent(BaseRunOutputEvent):
  """Emitted when the desktop bridge performs a user-visible action.

  This is the domain-level event — one per semantic action (screenshot,
  click, open app, etc.). Mirrors ``BrowserActionEvent`` in structure.

  Attributes:
    event: Always ``"DesktopAction"`` — used for event type discrimination.
    category: Action domain — ``"screen"``, ``"input"``, ``"app"``, ``"window"``,
      ``"accessibility"``, ``"file"``, ``"clipboard"``, ``"system"``, ``"shell"``,
      ``"camera"``, ``"applescript"``.
    action: Specific action name (e.g. ``"screenshot"``, ``"click"``, ``"open_app"``).
    target: Primary target — app name, file path, element descriptor, coordinates, etc.
    value: Secondary value — text typed, URL opened, volume level, etc.
    result: Human-readable result summary.
    timestamp: Unix timestamp (seconds) when the event was created.
    error: Error message if the action failed, empty string on success.
  """

  event: str = "DesktopAction"
  category: str = ""
  action: str = ""
  target: str = ""
  value: str = ""
  result: str = ""
  timestamp: float = field(default_factory=time)
  error: str = ""
