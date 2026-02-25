"""Agent runtime — orchestrates server, interfaces, and cron in one event loop."""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING, Any, List, Optional

from definable.utils.log import log_error, log_info

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.base import BaseInterface
  from definable.agent.interface.gateway import InterfaceGateway


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
    gateway: Optional["InterfaceGateway"] = None,
  ) -> None:
    self.agent = agent
    self.interfaces = interfaces or []
    self.host = host
    self.port = port
    self.name = name or agent.agent_name
    self.dev = dev
    self.gateway = gateway
    self._shutdown_event = asyncio.Event()
    self._uv_server: Any = None

    # Auto-detect server need
    from definable.agent.trigger.webhook import Webhook

    has_webhooks = any(isinstance(t, Webhook) for t in agent._triggers)
    has_server_interface = self._has_server_interface()
    if enable_server is None:
      self.enable_server = has_webhooks or has_server_interface
    else:
      self.enable_server = enable_server

  def _has_server_interface(self) -> bool:
    """Check if any registered interface needs the HTTP server.

    Detects interfaces with ``needs_server()`` or ``create_router()``
    (duck typing for router-based interfaces like WebSocket, WhatsApp, Call).
    """
    all_ifaces = self.gateway.interfaces if self.gateway else self.interfaces
    for iface in all_ifaces:
      if hasattr(iface, "needs_server") and iface.needs_server():
        return True
      if hasattr(iface, "create_router"):
        return True
    return False

  async def start(self) -> None:
    """Start the runtime and block until shutdown."""
    self._print_banner()
    self._install_signal_handlers()

    tasks: List[asyncio.Task] = []
    server_task: Optional[asyncio.Task] = None

    # 1. HTTP server (if enabled)
    if self.enable_server:
      server_task = asyncio.create_task(self._run_server())
      tasks.append(server_task)

    # 2. Interface supervisor (gateway or direct interfaces)
    has_interfaces = bool(self.interfaces) or (self.gateway is not None and bool(self.gateway.interfaces))
    if has_interfaces:
      tasks.append(asyncio.create_task(self._run_interfaces()))

    # 3. Scheduler (cron, interval, oneshot triggers)
    schedulable = self._get_schedulable_triggers()
    if schedulable:
      tasks.append(asyncio.create_task(self._run_scheduler(schedulable)))

    if not tasks:
      log_error(f"[{self.name}] No interfaces, triggers, or server configured. Nothing to run.")
      return

    # Wait for shutdown signal
    shutdown_task = asyncio.create_task(self._shutdown_event.wait())
    tasks.append(shutdown_task)

    try:
      done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

      # Signal uvicorn to exit gracefully so its lifespan handler
      # can complete without a CancelledError traceback.
      if self._uv_server is not None:
        self._uv_server.should_exit = True

      # Cancel non-server pending tasks immediately
      for task in pending:
        if task is not server_task:
          task.cancel()

      # Wait for the server to finish its graceful shutdown (with a timeout),
      # then cancel any stragglers.
      if server_task is not None and not server_task.done():
        _, still_pending = await asyncio.wait({server_task}, timeout=3.0)
        for task in still_pending:
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

    from definable.agent.runtime.server import AgentServer

    all_ifaces = list(self.gateway.interfaces) if self.gateway else list(self.interfaces)
    server = AgentServer(self.agent, self.host, self.port, dev=self.dev, interfaces=all_ifaces)
    app = server.create_app()

    config = uvicorn.Config(
      app,
      host=self.host,
      port=self.port,
      log_level="info" if self.dev else "warning",
    )
    uv_server = uvicorn.Server(config)
    self._uv_server = uv_server

    # Wire uvicorn's exit signal into our shutdown event
    uv_server.handle_exit = lambda *_: self._shutdown_event.set()  # type: ignore[method-assign]

    await uv_server.serve()

  async def _run_interfaces(self) -> None:
    """Run the interface supervisor (or gateway if configured)."""
    if self.gateway is not None:
      await self.gateway.aserve(name=self.name)
    else:
      from definable.utils.supervisor import supervise_interfaces

      await supervise_interfaces(*self.interfaces, name=self.name)

  def _get_schedulable_triggers(self) -> list:
    """Return all triggers that the Scheduler can manage."""
    from definable.agent.trigger.interval import Interval
    from definable.agent.trigger.oneshot import OneShot

    schedulable_types: tuple = (Interval, OneShot)
    try:
      from definable.agent.trigger.cron import Cron

      schedulable_types = (Cron, Interval, OneShot)
    except ImportError:
      pass
    return [t for t in self.agent._triggers if isinstance(t, schedulable_types)]

  async def _run_scheduler(self, triggers: list) -> None:
    """Run the Scheduler loop for all time-based triggers.

    Uses the new Scheduler system instead of the legacy cron-only loop.
    """
    from definable.agent.scheduler.scheduler import Scheduler
    from definable.agent.trigger.executor import TriggerExecutor

    executor = TriggerExecutor(self.agent)
    scheduler = Scheduler(tick_interval=1.0)

    for trigger in triggers:
      scheduler.add(trigger)
      log_info(f"[{self.name}] Scheduled: {trigger.name}")

    # Run until shutdown
    async def _stop_on_shutdown() -> None:
      await self._shutdown_event.wait()
      scheduler.stop()

    stop_task = asyncio.create_task(_stop_on_shutdown())
    try:
      await scheduler.start(executor)
    finally:
      stop_task.cancel()

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
    from definable.agent.trigger.webhook import Webhook

    lines = [
      "",
      f"  Agent: {self.agent.agent_name}",
      f"  Model: {self.agent.model.id}",
    ]

    if self.dev:
      lines.append("  Mode: development (hot reload)")

    # Show interfaces from gateway or direct list
    all_ifaces = self.gateway.interfaces if self.gateway else self.interfaces
    if all_ifaces:
      iface_names = [i.config.platform or type(i).__name__ for i in all_ifaces]
      label = "Interfaces (gateway)" if self.gateway else "Interfaces"
      lines.append(f"  {label}: {', '.join(iface_names)}")

    webhooks = [t for t in self.agent._triggers if isinstance(t, Webhook)]
    if webhooks:
      lines.append(f"  Webhooks: {', '.join(t.name for t in webhooks)}")

    scheduled = self._get_schedulable_triggers()
    if scheduled:
      lines.append(f"  Scheduled: {', '.join(t.name for t in scheduled)}")

    if self.enable_server:
      lines.append(f"  Server: http://{self.host}:{self.port}")
      if self.dev:
        lines.append(f"  Docs: http://{self.host}:{self.port}/docs")

    if self.agent._auth is not None:
      lines.append(f"  Auth: {type(self.agent._auth).__name__}")

    if self._has_server_interface():
      all_srv_ifaces = self.gateway.interfaces if self.gateway else self.interfaces
      for iface in all_srv_ifaces:
        if hasattr(iface, "create_router"):
          platform = getattr(iface.config, "platform", type(iface).__name__)
          if platform == "call":
            call_cfg = getattr(iface, "_call_config", None)
            if call_cfg:
              lines.append(f"  Call: {call_cfg.phone_number} (pipeline={call_cfg.pipeline_mode})")
            else:
              lines.append("  Call: routes mounted")
          elif platform == "websocket":
            ws_path = getattr(iface.config, "path", "/ws")
            lines.append(f"  WebSocket: ws://{self.host}:{self.port}{ws_path}")
          elif platform == "whatsapp":
            lines.append(f"  WhatsApp: {getattr(iface.config, 'webhook_path', '/whatsapp/webhook')}")
          else:
            lines.append(f"  {platform}: routes mounted")

    obs_config = getattr(self.agent, "_observability_config", None)
    if obs_config is not None and obs_config.enabled and self.enable_server:
      lines.append(f"  Observability: http://{self.host}:{self.port}/obs/")

    lines.append("")

    banner = "\n".join(lines)
    log_info(f"[{self.name}] Starting runtime\n{banner}")
