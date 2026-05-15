"""``POST /api/playground/run`` — invoke an agent and stream events back.

The endpoint looks up the live :class:`Agent` instance in
:class:`AgentRegistry` and calls :meth:`Agent.arun(input, stream=True)`.
Each event from the agent's bus is forwarded as a Server-Sent Event so
the playground UI can render the response (and per-turn token / cost
metadata) live.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from pydantic import BaseModel

if TYPE_CHECKING:
  from fastapi import FastAPI

log = logging.getLogger(__name__)


class PlaygroundRequest(BaseModel):
  """JSON body for ``POST /api/playground/run``."""

  agent: str
  input: str


def install_playground_routes(app: FastAPI) -> None:
  """Register ``POST /api/playground/run``."""
  from fastapi import HTTPException
  from starlette.responses import StreamingResponse

  from definable.observability.registry import AgentRegistry

  @app.post("/api/playground/run")
  async def run_playground(req: PlaygroundRequest) -> StreamingResponse:
    agent = AgentRegistry.get().lookup(req.agent)
    if agent is None:
      raise HTTPException(status_code=404, detail=f"agent {req.agent!r} not registered")

    async def generator() -> AsyncIterator[bytes]:
      yield b": started\n\n"
      try:
        stream: Any = await agent.arun(req.input, stream=True)
        async for event in stream:
          payload = {
            "type": event.__class__.__name__,
            "run_id": getattr(event, "run_id", None),
            "timestamp": getattr(event, "timestamp", None),
            "data": _event_as_dict(event),
          }
          yield f"data: {json.dumps(payload, default=str)}\n\n".encode("utf-8")
      except Exception as e:
        err = {"type": "error", "error": str(e) or e.__class__.__name__}
        yield f"data: {json.dumps(err)}\n\n".encode("utf-8")
        log.exception("playground run failed")

    return StreamingResponse(generator(), media_type="text/event-stream")


def _event_as_dict(event: Any) -> dict[str, Any]:
  try:
    return dataclasses.asdict(event)
  except TypeError:
    return {"repr": repr(event)}
