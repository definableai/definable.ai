"""definable.observability — event-bus subscribers + live dashboard.

The harness has no built-in observability — it has an :class:`EventBus`.
This module ships subscribers that pipe those events somewhere:

* JSONL on disk via :func:`attach_jsonl` (durable, replayable).
* A live FastAPI dashboard via :class:`ObservabilityServer` (REST + SSE
  + browser playground) — the new path used when an Agent is constructed
  with ``observability=True``.

Usage::

    from definable import Agent
    agent = Agent(name="t", model="...", observability=True)  # both sinks wire up

    # or low-level
    from definable.observability import ObservabilityServer
    server = ObservabilityServer.singleton()
    await server.attach(agent)
"""

from typing import TYPE_CHECKING

from definable.observability.subscriber import Observability, attach_jsonl

if TYPE_CHECKING:
  from definable.observability.server import ObservabilityServer  # noqa: F401

__all__ = ["Observability", "ObservabilityServer", "attach_jsonl"]


def __getattr__(name: str):  # noqa: ANN202 — pep562 lazy import
  # Lazy-load ObservabilityServer so importing `definable.observability` doesn't
  # require fastapi/uvicorn (which live in the optional `[serve]` extra).
  if name == "ObservabilityServer":
    from definable.observability.server import ObservabilityServer as _ObservabilityServer

    return _ObservabilityServer
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
