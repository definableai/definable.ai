"""Agent HTTP server — hidden FastAPI app with webhook routes and /run endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from definable.agent.auth.base import AuthRequest, resolve_auth
from definable.utils.log import log_info

if TYPE_CHECKING:
  from fastapi import Request

  from definable.agent.agent import Agent


class AgentServer:
  """Creates and manages a FastAPI application for the agent.

  Provides:
    - ``POST /run`` — invoke the agent with ``{input, session_id, user_id}``
    - ``GET /health`` — health check
    - Webhook routes from registered triggers
    - Call interface routes (when CallInterface is registered)

  Args:
    agent: The Agent instance to serve.
    host: Host to bind to.
    port: Port to listen on.
    interfaces: Optional list of interfaces (from runtime) to scan for CallInterface.
  """

  def __init__(
    self,
    agent: "Agent",
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    dev: bool = False,
    interfaces: Optional[list] = None,
  ) -> None:
    self.agent = agent
    self.host = host
    self.port = port
    self.dev = dev
    self._interfaces = interfaces or []
    self._app: Optional[Any] = None

  def create_app(self) -> Any:
    """Build the FastAPI application with routes and middleware.

    Returns:
      A FastAPI application instance.

    Raises:
      ImportError: If fastapi is not installed.
    """
    try:
      from fastapi import FastAPI, Request
      from fastapi.responses import JSONResponse
    except ImportError as e:
      raise ImportError("fastapi is required for the agent server. Install it with: pip install 'definable[serve]'") from e

    # Make Request available in module globals for annotation resolution.
    globals()["Request"] = Request

    app = FastAPI(
      title=f"{self.agent.agent_name} API",
      docs_url="/docs" if self.dev else None,
      redoc_url="/redoc" if self.dev else None,
    )

    # --- Auth middleware ---
    if self.agent._auth is not None:
      auth_provider = self.agent._auth

      @app.middleware("http")
      async def auth_middleware(request: Request, call_next):
        # Skip auth for health endpoint and docs in dev mode
        if request.url.path == "/health":
          return await call_next(request)
        if self.dev and request.url.path in ("/docs", "/redoc", "/openapi.json"):
          return await call_next(request)

        # Check if this is a webhook with auth explicitly disabled
        for trigger in self.agent._triggers:
          from definable.agent.trigger.webhook import Webhook

          if isinstance(trigger, Webhook) and trigger.auth is False:
            if request.url.path == trigger.path and request.method == trigger.method:
              return await call_next(request)

        auth_request = AuthRequest(
          platform="http",
          headers=dict(request.headers),
        )
        auth_context = await resolve_auth(auth_provider, auth_request)
        if auth_context is None:
          return JSONResponse(status_code=401, content={"error": "Unauthorized"})

        request.state.auth = auth_context
        return await call_next(request)

    # --- Health endpoint ---
    @app.get("/health")
    async def health():
      return {"status": "ok", "agent": self.agent.agent_name}

    # --- /run endpoint ---
    @app.post("/run")
    async def run_agent(request: Request):
      body = await request.json()
      instruction = body.get("input", "")
      session_id = body.get("session_id")
      user_id = body.get("user_id")

      output = await self.agent.arun(
        instruction=instruction,
        session_id=session_id,
        user_id=user_id,
      )
      return {
        "content": output.content,
        "session_id": output.session_id,
        "run_id": output.run_id,
      }

    # --- Webhook routes ---
    self._register_webhooks(app)

    # --- Interface routes (WebSocket, WhatsApp, Call, etc.) ---
    self._mount_interface_routes(app)

    # --- Observability dashboard (conditionally mounted) ---
    self._mount_observability(app)

    self._app = app
    return app

  def _mount_observability(self, app: Any) -> None:
    """Conditionally mount observability dashboard and API endpoints."""
    obs_config = getattr(self.agent, "_observability_config", None)
    if obs_config is None or not obs_config.enabled:
      return

    import logging

    from fastapi.responses import HTMLResponse

    from definable.agent.observability.api import create_observability_router
    from definable.agent.observability.collector import ObservabilityExporter
    from definable.agent.observability.dashboard import get_dashboard_html
    from definable.agent.observability.metrics import MetricsAggregator
    from definable.agent.observability.trace_browser import TraceBrowser

    # Suppress noisy "socket.send() raised exception" from asyncio transport.
    # These fire when the browser refreshes and the old SSE connection's transport
    # keeps trying to write to the dead socket (asyncio.selector_events._SelectorSocketTransport).
    class _SocketSendFilter(logging.Filter):
      def filter(self, record: logging.LogRecord) -> bool:
        return "socket.send() raised exception" not in record.getMessage()

    logging.getLogger("asyncio").addFilter(_SocketSendFilter())

    # Get or create the collector exporter
    exporter = getattr(self.agent, "_observability_exporter", None)
    if exporter is None:
      exporter = ObservabilityExporter(buffer_size=obs_config.buffer_size)

    trace_browser = TraceBrowser(trace_dir=obs_config.trace_dir)
    metrics_aggregator = MetricsAggregator()

    # Mount API router
    obs_router = create_observability_router(
      collector=exporter,
      trace_browser=trace_browser,
      metrics_aggregator=metrics_aggregator,
      agent=self.agent,
    )
    app.include_router(obs_router, prefix="/obs/api")

    # Serve dashboard HTML at /obs/
    dashboard_html = get_dashboard_html(obs_config)

    @app.get("/obs/", response_class=HTMLResponse)
    async def obs_dashboard():
      return dashboard_html

    @app.get("/obs", response_class=HTMLResponse)
    async def obs_dashboard_redirect():
      return dashboard_html

    log_info("Observability dashboard mounted at /obs/")

  def _mount_interface_routes(self, app: Any) -> None:
    """Mount routes for any interface that exposes a create_router() method.

    Scans both the agent's interfaces and the runtime-provided
    interfaces. Supports CallInterface, WebSocketInterface, WhatsAppInterface,
    and any future interface implementing ``create_router()``.
    """
    # Combine agent-registered and runtime-provided interfaces
    all_interfaces = list(getattr(self.agent, "_interfaces", []) or [])
    for iface in self._interfaces:
      if iface not in all_interfaces:
        all_interfaces.append(iface)

    for iface in all_interfaces:
      if hasattr(iface, "create_router"):
        router = iface.create_router()
        app.include_router(router)
        platform = getattr(iface.config, "platform", type(iface).__name__)
        log_info(f"Mounted {platform} interface routes")

  def _register_webhooks(self, app: Any) -> None:
    """Register webhook trigger routes on the FastAPI app."""
    from definable.agent.trigger.base import TriggerEvent
    from definable.agent.trigger.executor import TriggerExecutor
    from definable.agent.trigger.webhook import Webhook

    executor = TriggerExecutor(self.agent)

    for trigger in self.agent._triggers:
      if not isinstance(trigger, Webhook):
        continue

      # Capture trigger in closure
      def _make_handler(t: Webhook):
        async def handler(request: "Request"):
          from fastapi.responses import JSONResponse

          try:
            body = await request.json()
          except Exception:
            body = None

          headers = dict(request.headers)
          event = TriggerEvent(
            body=body,
            headers=headers,
            source=t.name,
            raw=request,
          )
          result = await executor.execute(t, event)

          if result is None:
            return JSONResponse(content={"status": "ok"})
          if hasattr(result, "content"):
            return JSONResponse(content={"content": result.content})
          return JSONResponse(content={"result": str(result)})

        return handler

      app.add_api_route(
        trigger.path,
        _make_handler(trigger),
        methods=[trigger.method],
      )
      log_info(f"Registered webhook: {trigger.name}")
