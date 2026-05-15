"""Server-Sent Events live event tail.

Mounted as a raw Starlette ``Route`` rather than a FastAPI ``@app.get``
because FastAPI's dependency-injection wrapper has historically rejected
long-lived streaming requests on the upgrade path (the same class of
issue that bit ``WebSocketInterface`` — see harness-v2 smoke #163).

Each connection subscribes to the store's broadcast fanout. The first
write inside an SSE handler triggers HTTP headers, so we send a comment
keep-alive on connect plus one every 15 seconds while the client is
listening.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
  from fastapi import FastAPI

  from definable.observability.store import TraceStore

log = logging.getLogger(__name__)

PING_INTERVAL_S = 15.0


def install_sse_routes(app: FastAPI, store: TraceStore) -> None:
  """Register ``GET /api/stream`` as a raw Starlette Route."""
  from starlette.requests import Request
  from starlette.responses import StreamingResponse
  from starlette.routing import Route

  async def _stream(request: Request) -> StreamingResponse:
    agent_filter = request.query_params.get("agent")

    async def generator() -> AsyncIterator[bytes]:
      queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

      def _push(payload: dict[str, Any]) -> None:
        if agent_filter and payload.get("agent") != agent_filter:
          return
        # Drop on overflow — slow consumer.
        with contextlib.suppress(asyncio.QueueFull):
          queue.put_nowait(payload)

      unsub = store.subscribe_broadcast(_push)
      try:
        yield b": connected\n\n"
        while True:
          try:
            payload = await asyncio.wait_for(queue.get(), timeout=PING_INTERVAL_S)
          except TimeoutError:
            yield b": ping\n\n"
            continue
          if await request.is_disconnected():
            break
          line = f"data: {json.dumps(payload, default=str)}\n\n"
          yield line.encode("utf-8")
      finally:
        unsub()

    return StreamingResponse(generator(), media_type="text/event-stream")

  app.router.routes.append(Route("/api/stream", endpoint=_stream))
