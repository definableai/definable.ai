"""Agent runtime — orchestrates server, interfaces, and cron in one event loop."""

from __future__ import annotations

import asyncio
import signal
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from definable.utils.log import log_error, log_info

if TYPE_CHECKING:
  from definable.agents.agent import Agent
  from definable.interfaces.base import BaseInterface


class AgentRuntime:
  """Orchestrates server, interface supervisor, and cron scheduler.

  Runs everything in a single event loop via ``asyncio.gather``.
  Handles SIGINT/SIGTERM for graceful shutdown.

  Args:
    agent: The Agent instance to run.
    interfaces: Optional list of interfaces to supervise.
    host: Host for the HTTP server.
    port: Port for the HTTP server.
    enable_server: Force-enable/disable the server.  When *None*,
      the server starts if any Webhook triggers exist.
    name: Optional name for log messages.
  """

  def __init__(
    self,
    agent: "Agent",
    *,
    interfaces: Optional[List["BaseInterface"]] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_server: Optional[bool] = None,
    name: Optional[str] = None,
    dev: bool = False,
  ) -> None:
    self.agent = agent
    self.interfaces = interfaces or []
    self.host = host
    self.port = port
    self.name = name or agent.agent_name
    self.dev = dev
    self._shutdown_event = asyncio.Event()

    # Auto-detect server need
    from definable.triggers.webhook import Webhook

    has_webhooks = any(isinstance(t, Webhook) for t in agent._triggers)
    if enable_server is None:
      self.enable_server = has_webhooks
    else:
      self.enable_server = enable_server

  async def start(self) -> None:
    """Start the runtime and block until shutdown."""
    self._print_banner()
    self._install_signal_handlers()

    tasks: List[asyncio.Task] = []

    # 1. HTTP server (if enabled)
    if self.enable_server:
      tasks.append(asyncio.create_task(self._run_server()))

    # 2. Interface supervisor (if any interfaces)
    if self.interfaces:
      tasks.append(asyncio.create_task(self._run_interfaces()))

    # 3. Cron scheduler (if any cron triggers)
    from definable.triggers.cron import Cron

    cron_triggers = [t for t in self.agent._triggers if isinstance(t, Cron)]
    if cron_triggers:
      tasks.append(asyncio.create_task(self._run_cron_scheduler(cron_triggers)))

    if not tasks:
      log_error(f"[{self.name}] No interfaces, triggers, or server configured. Nothing to run.")
      return

    # Wait for shutdown signal
    shutdown_task = asyncio.create_task(self._shutdown_event.wait())
    tasks.append(shutdown_task)

    try:
      done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

      # If shutdown was requested, cancel remaining tasks
      for task in pending:
        task.cancel()
      await asyncio.gather(*pending, return_exceptions=True)

      # Re-raise exceptions from completed tasks (except shutdown)
      for task in done:
        if task is not shutdown_task and not task.cancelled():
          exc = task.exception()
          if exc is not None:
            raise exc

    except asyncio.CancelledError:
      pass
    finally:
      log_info(f"[{self.name}] Runtime stopped")

  async def _run_server(self) -> None:
    """Start the uvicorn HTTP server."""
    try:
      import uvicorn
    except ImportError as e:
      raise ImportError("uvicorn is required for the agent server. Install it with: pip install 'definable[serve]'") from e

    from definable.runtime.server import AgentServer

    server = AgentServer(self.agent, self.host, self.port, dev=self.dev)
    app = server.create_app()

    config = uvicorn.Config(
      app,
      host=self.host,
      port=self.port,
      log_level="info" if self.dev else "warning",
    )
    uv_server = uvicorn.Server(config)

    # Override shutdown to respect our signal
    original_shutdown = uv_server.shutdown

    async def graceful_shutdown() -> None:
      self._shutdown_event.set()
      await original_shutdown()

    uv_server.handle_exit = lambda *_: self._shutdown_event.set()  # type: ignore[assignment]

    await uv_server.serve()

  async def _run_interfaces(self) -> None:
    """Run the interface supervisor."""
    from definable.utils.supervisor import supervise_interfaces

    await supervise_interfaces(*self.interfaces, name=self.name)

  async def _run_cron_scheduler(self, cron_triggers: list) -> None:
    """Run the cron scheduler loop.

    Tracks next fire time per trigger.  Sleeps until the next soonest
    trigger (capped at 60s).  Each execution is fire-and-forget.
    """
    from definable.triggers.base import TriggerEvent
    from definable.triggers.executor import TriggerExecutor

    executor = TriggerExecutor(self.agent)
    now = time.time()

    # Initialize next-fire times
    next_fire: Dict[int, float] = {}
    for trigger in cron_triggers:
      next_fire[id(trigger)] = trigger.next_run(now)
      log_info(f"[{self.name}] Cron scheduled: {trigger.name}")

    while not self._shutdown_event.is_set():
      now = time.time()

      for trigger in cron_triggers:
        tid = id(trigger)
        if now >= next_fire[tid]:
          # Fire!
          event = TriggerEvent(source=trigger.name)
          asyncio.create_task(executor.execute(trigger, event))
          next_fire[tid] = trigger.next_run(now)

      # Sleep until next soonest trigger (max 60s)
      soonest = min(next_fire.values())
      sleep_time = max(0.1, min(soonest - time.time(), 60.0))
      await asyncio.sleep(sleep_time)

  def _install_signal_handlers(self) -> None:
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""
    loop = asyncio.get_running_loop()

    def _handle_signal() -> None:
      log_info(f"[{self.name}] Shutdown signal received")
      self._shutdown_event.set()

    import contextlib

    for sig in (signal.SIGINT, signal.SIGTERM):
      with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(sig, _handle_signal)

  def _print_banner(self) -> None:
    """Print a startup banner with runtime configuration."""
    from definable.triggers.cron import Cron
    from definable.triggers.webhook import Webhook

    lines = [
      "",
      f"  Agent: {self.agent.agent_name}",
      f"  Model: {self.agent.model.id}",
    ]

    if self.dev:
      lines.append("  Mode: development (hot reload)")

    if self.interfaces:
      iface_names = [i.config.platform or type(i).__name__ for i in self.interfaces]
      lines.append(f"  Interfaces: {', '.join(iface_names)}")

    webhooks = [t for t in self.agent._triggers if isinstance(t, Webhook)]
    if webhooks:
      lines.append(f"  Webhooks: {', '.join(t.name for t in webhooks)}")

    crons = [t for t in self.agent._triggers if isinstance(t, Cron)]
    if crons:
      lines.append(f"  Cron jobs: {', '.join(t.name for t in crons)}")

    if self.enable_server:
      lines.append(f"  Server: http://{self.host}:{self.port}")
      if self.dev:
        lines.append(f"  Docs: http://{self.host}:{self.port}/docs")

    if self.agent._auth is not None:
      lines.append(f"  Auth: {type(self.agent._auth).__name__}")

    lines.append("")

    banner = "\n".join(lines)
    log_info(f"[{self.name}] Starting runtime\n{banner}")
