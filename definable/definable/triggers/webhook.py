"""Webhook trigger — registers an HTTP endpoint on the agent server."""

from typing import Any, Optional

from definable.triggers.base import BaseTrigger


class Webhook(BaseTrigger):
  """HTTP webhook trigger.

  Registers a route on the agent's HTTP server.  The decorated handler
  receives a :class:`TriggerEvent` with the parsed JSON body and headers.

  Args:
    path: URL path for the webhook (e.g. ``"/github"``).
    method: HTTP method (default ``"POST"``).
    auth: Per-webhook auth override.  ``None`` inherits from the agent,
      ``False`` explicitly disables auth for this endpoint.

  Example::

    @agent.on(Webhook("/github"))
    async def handle_github(event):
        return f"Got push to {event.body['repository']}"
  """

  def __init__(
    self,
    path: str,
    *,
    method: str = "POST",
    auth: Optional[Any] = None,
  ) -> None:
    self.path = path if path.startswith("/") else f"/{path}"
    self.method = method.upper()
    self.auth = auth

  @property
  def name(self) -> str:
    return f"{self.method} {self.path}"
