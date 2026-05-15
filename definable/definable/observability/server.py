"""Process-wide observability server.

Lazy singleton: first :class:`Agent` constructed with
``observability=True`` calls :meth:`ObservabilityServer.singleton`, which
boots the FastAPI app, finds a free port, prints the URL, and starts
uvicorn in a background task. Subsequent agents :meth:`attach` to the
same server.

Tear-down via :meth:`aclose` is best-effort and registered with
``atexit`` so scripts that just call ``arun`` and exit do not leak the
listener.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import socket
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from definable.observability.playground import install_playground_routes
from definable.observability.registry import AgentRegistry
from definable.observability.routes import install_rest_routes
from definable.observability.sse import install_sse_routes
from definable.observability.store import TraceStore

if TYPE_CHECKING:
  from definable.agent.agent import Agent

log = logging.getLogger(__name__)

PORT_PROBE_LIMIT = 10
_STATIC_DIR = Path(__file__).parent / "static"


class ObservabilityServer:
  """One FastAPI app per process, shared across every opted-in Agent."""

  _instance: ObservabilityServer | None = None
  _bootstrap_lock = threading.Lock()

  def __init__(self, *, host: str = "127.0.0.1", preferred_port: int = 7777) -> None:
    self.host = host
    self.preferred_port = preferred_port
    self.port: int | None = None
    self.store = TraceStore()
    self.registry = AgentRegistry.get()
    self.app: Any = None
    self._server: Any = None
    self._serve_task: asyncio.Task[Any] | None = None
    self._started = False
    self._lock = asyncio.Lock()

  @classmethod
  def singleton(cls, *, host: str = "127.0.0.1", preferred_port: int = 7777) -> ObservabilityServer:
    if cls._instance is None:
      with cls._bootstrap_lock:
        if cls._instance is None:
          cls._instance = cls(host=host, preferred_port=preferred_port)
          atexit.register(cls._instance._shutdown_atexit)
    return cls._instance

  @property
  def url(self) -> str:
    return f"http://{self.host}:{self.port}" if self.port else f"http://{self.host}:?"

  # ---- lifecycle ---------------------------------------------------------

  async def aopen(self) -> None:
    """Idempotent boot: open store, build app, start uvicorn task."""
    async with self._lock:
      if self._started:
        return
      try:
        import uvicorn
        from fastapi import FastAPI
      except ImportError as e:
        raise ImportError("ObservabilityServer requires fastapi + uvicorn — `pip install definable[serve]`") from e

      await self.store.aopen()

      app = FastAPI(title="Definable Console")
      install_rest_routes(app, self.store)
      install_sse_routes(app, self.store)
      install_playground_routes(app)
      self._mount_static(app)
      self.app = app

      self.port = _pick_free_port(self.host, self.preferred_port)
      config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning", access_log=False)
      self._server = uvicorn.Server(config)
      self._serve_task = asyncio.create_task(self._server.serve())
      for _ in range(100):
        if getattr(self._server, "started", False):
          break
        await asyncio.sleep(0.05)

      self._started = True
      log.info("[definable] observability live → %s", self.url)
      print(f"[definable] observability live → {self.url}", flush=True)  # noqa: T201 — explicit user-visible URL

  async def aclose(self) -> None:
    async with self._lock:
      if not self._started:
        return
      if self._server is not None:
        self._server.should_exit = True
      if self._serve_task is not None:
        self._serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
          await self._serve_task
        self._serve_task = None
      await self.store.aclose()
      self._started = False

  # ---- agent attach ------------------------------------------------------

  async def attach(self, agent: Agent) -> None:
    """Register ``agent`` and subscribe its EventBus to the store.

    Boots the server lazily on first attach so users don't think about
    server lifecycle separately from agent construction.
    """
    if not self._started:
      await self.aopen()
    model_id = getattr(getattr(agent, "model", None), "id", None) or repr(agent.model)
    self.registry.register(agent)
    callback = self.store.attach(agent_name=agent.name, agent_model=str(model_id), instructions=agent.instructions)
    agent.events.subscribe(callback)

  # ---- internals ---------------------------------------------------------

  def _mount_static(self, app: Any) -> None:
    if not _STATIC_DIR.exists():
      log.debug("Static dir missing at %s — dashboard UI not mounted yet", _STATIC_DIR)
      return
    from starlette.responses import FileResponse
    from starlette.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    async def root() -> FileResponse:
      return FileResponse(_STATIC_DIR / "index.html")

  def _shutdown_atexit(self) -> None:
    if not self._started:
      return
    with contextlib.suppress(Exception):
      loop = asyncio.new_event_loop()
      try:
        loop.run_until_complete(self.aclose())
      finally:
        loop.close()


def _pick_free_port(host: str, preferred: int) -> int:
  """Probe ``preferred..preferred+PORT_PROBE_LIMIT`` and return the first free port."""
  for offset in range(PORT_PROBE_LIMIT + 1):
    candidate = preferred + offset
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      try:
        s.bind((host, candidate))
        return candidate
      except OSError:
        continue
  # Fallback: ask the OS for any free port.
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, 0))
    return int(s.getsockname()[1])
