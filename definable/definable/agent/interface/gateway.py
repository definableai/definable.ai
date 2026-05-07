"""InterfaceGateway — central multi-channel coordination layer.

Sits between AgentRuntime and BaseInterface, providing:
- Shared gateway-level hooks (applied to ALL interfaces)
- Per-interface status tracking
- Lifecycle events (started/stopped/restarted/error)
- Optional shared sessions
- Optional cross-platform identity linking (/link flow)
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from definable.agent.interface.hooks import InterfaceHook
from definable.agent.interface.session import SessionManager
from definable.run.agent import BaseAgentRunEvent, RunEvent
from definable.utils.log import log_error, log_info, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.base import BaseInterface
  from definable.agent.interface.identity import IdentityResolver
  from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
  from definable.agent.interface.session import InterfaceSession


# ---------------------------------------------------------------------------
# InterfaceStatus enum
# ---------------------------------------------------------------------------


class InterfaceStatus(str, Enum):
  """Lifecycle status of an interface managed by the gateway."""

  pending = "pending"
  starting = "starting"
  running = "running"
  restarting = "restarting"
  stopped = "stopped"
  error = "error"


# ---------------------------------------------------------------------------
# Gateway lifecycle events
# ---------------------------------------------------------------------------


@dataclass
class InterfaceStartedEvent(BaseAgentRunEvent):
  """Emitted when an interface starts successfully."""

  event: str = RunEvent.interface_started.value
  platform: str = ""
  interface_class: str = ""


@dataclass
class InterfaceStoppedEvent(BaseAgentRunEvent):
  """Emitted when an interface stops cleanly."""

  event: str = RunEvent.interface_stopped.value
  platform: str = ""
  interface_class: str = ""


@dataclass
class InterfaceRestartedEvent(BaseAgentRunEvent):
  """Emitted after a crashed interface is restarted."""

  event: str = RunEvent.interface_restarted.value
  platform: str = ""
  interface_class: str = ""
  restart_count: int = 0
  backoff_seconds: float = 0.0


@dataclass
class InterfaceErrorEvent(BaseAgentRunEvent):
  """Emitted when an interface crashes."""

  event: str = RunEvent.interface_error.value
  platform: str = ""
  interface_class: str = ""
  error_message: str = ""


# ---------------------------------------------------------------------------
# _GatewayHookBridge — injects gateway hooks into each interface
# ---------------------------------------------------------------------------


class _GatewayHookBridge:
  """Bridges gateway-level hooks into a single interface's hook list.

  Inserted at position 0 of ``interface._hooks`` so gateway hooks run
  first. Holds a *reference* to the gateway (not a snapshot), so
  ``gateway.add_hook()`` after registration still takes effect.

  Also wraps ``handle_platform_message`` on the interface so that
  hooks which stash a ``_link_reply`` in message metadata (e.g. the
  identity-linking hook) can send the reply directly back to the user.
  The wrapper captures ``raw_message`` — the native platform object —
  which is needed by some implementations (Discord uses it for the
  channel reference).
  """

  def __init__(self, gateway: "InterfaceGateway", interface: "BaseInterface") -> None:
    self._gateway = gateway
    self._interface = interface
    self._install_message_wrapper(interface)

  def _install_message_wrapper(self, interface: "BaseInterface") -> None:
    """Replace handle_platform_message with a wrapper that captures raw_message.

    This is necessary so that ``_send_link_reply_if_present`` can pass the
    native platform object (e.g. ``discord.Message``) to ``_send_response``.
    Skips wrapping if the interface doesn't have the method (test stubs).
    """
    original = getattr(interface, "handle_platform_message", None)
    if original is None:
      return

    async def _wrapped_handle(raw_message: Any) -> None:
      self._current_raw_message = raw_message
      try:
        await original(raw_message)
      finally:
        self._current_raw_message = None

    interface.handle_platform_message = _wrapped_handle  # type: ignore[assignment]

  async def on_message_received(self, message: "InterfaceMessage") -> Optional[bool]:
    for hook in self._gateway._hooks:
      if hasattr(hook, "on_message_received"):
        result = await hook.on_message_received(message)
        if result is False:
          await self._send_link_reply_if_present(message)
          return False
    return None

  async def _send_link_reply_if_present(self, message: "InterfaceMessage") -> None:
    """Send a ``_link_reply`` back to the user if one was stashed in metadata."""
    reply_text = message.metadata.pop("_link_reply", None)
    if reply_text is None:
      return
    try:
      from definable.agent.interface.message import InterfaceResponse

      response = InterfaceResponse(content=reply_text)
      raw = getattr(self, "_current_raw_message", None)
      await self._interface._send_response(message, response, raw if raw is not None else message)
    except Exception as exc:
      log_warning(f"Failed to send link reply: {exc}")

  async def on_before_respond(self, message: "InterfaceMessage", session: "InterfaceSession") -> Optional["InterfaceMessage"]:
    for hook in self._gateway._hooks:
      if hasattr(hook, "on_before_respond"):
        modified = await hook.on_before_respond(message, session)
        if modified is not None:
          message = modified
    return message or None

  async def on_after_respond(
    self,
    message: "InterfaceMessage",
    response: "InterfaceResponse",
    session: "InterfaceSession",
  ) -> Optional["InterfaceResponse"]:
    for hook in self._gateway._hooks:
      if hasattr(hook, "on_after_respond"):
        modified = await hook.on_after_respond(message, response, session)
        if modified is not None:
          response = modified
    return response or None

  async def on_error(self, error: Exception, message: Optional["InterfaceMessage"]) -> None:
    for hook in self._gateway._hooks:
      if hasattr(hook, "on_error"):
        try:
          await hook.on_error(error, message)
        except Exception as exc:
          log_warning(f"Gateway hook on_error failed: {exc}")


# ---------------------------------------------------------------------------
# _IdentityLinkHook — self-service /link flow
# ---------------------------------------------------------------------------


@dataclass
class _LinkRequest:
  """Pending cross-platform link request."""

  platform: str
  platform_user_id: str
  code: str
  created_at: float = field(default_factory=time.time)


class _IdentityLinkHook:
  """Gateway hook implementing the self-service ``/link`` flow.

  - ``/link`` (no arg) on platform A → generates a 6-char code, replies with it
  - ``/link CODE`` on platform B → validates code, links both identities

  Returns ``False`` from ``on_message_received`` to prevent link
  commands from reaching the agent.
  """

  def __init__(
    self,
    gateway: "InterfaceGateway",
    command: str = "/link",
    code_ttl: int = 300,
  ) -> None:
    self._gateway = gateway
    self._command = command
    self._code_ttl = code_ttl
    self._pending: Dict[str, _LinkRequest] = {}

  def _cleanup_expired(self) -> None:
    now = time.time()
    expired = [code for code, req in self._pending.items() if now - req.created_at > self._code_ttl]
    for code in expired:
      del self._pending[code]

  def _generate_code(self) -> str:
    return secrets.token_hex(3).upper()  # 6 hex chars

  async def on_message_received(self, message: "InterfaceMessage") -> Optional[bool]:
    text = (message.text or "").strip()
    if not text.startswith(self._command):
      return None  # not a link command — pass through

    self._cleanup_expired()

    parts = text.split(maxsplit=1)
    resolver = self._gateway._identity_resolver
    if resolver is None:
      return False  # no resolver configured — swallow silently

    if len(parts) == 1:
      # Generate a new code
      code = self._generate_code()
      self._pending[code] = _LinkRequest(
        platform=message.platform,
        platform_user_id=message.platform_user_id,
        code=code,
      )
      message.metadata["_link_reply"] = f"Enter this code on another platform: {code}"
      return False

    # Validate code
    code = parts[1].strip().upper()
    req = self._pending.get(code)
    if req is None:
      message.metadata["_link_reply"] = "Invalid or expired code."
      return False

    # Determine canonical user ID
    del self._pending[code]
    source_canonical = await resolver.resolve(req.platform, req.platform_user_id)
    target_canonical = await resolver.resolve(message.platform, message.platform_user_id)

    canonical = source_canonical or target_canonical
    if canonical is None:
      from uuid import uuid4

      canonical = str(uuid4())

    await resolver.link(req.platform, req.platform_user_id, canonical)
    await resolver.link(message.platform, message.platform_user_id, canonical)

    message.metadata["_link_reply"] = "Linked! Your accounts are now connected."
    return False


# ---------------------------------------------------------------------------
# InterfaceGateway
# ---------------------------------------------------------------------------

_STABILITY_THRESHOLD = 60.0
_MAX_BACKOFF = 60.0


class InterfaceGateway:
  """Central coordination layer for multiple interfaces.

  Usage::

      gateway = InterfaceGateway(agent)
      gateway.add(TelegramInterface(bot_token="..."))
      gateway.add(CLIInterface())
      gateway.add_hook(LoggingHook())
      gateway.serve()

  Args:
    agent: The Agent instance that owns these interfaces.
    shared_sessions: When True, all interfaces share one SessionManager.
    identity_resolver: Optional resolver for cross-platform user identity.
    session_ttl_seconds: TTL for the shared SessionManager.
    hooks: Optional initial list of gateway-level hooks.
    enable_identity_linking: Enable the self-service ``/link`` flow.
    link_command: Command prefix for identity linking (default ``/link``).
    link_code_ttl: How long link codes remain valid in seconds (default 300).
  """

  def __init__(
    self,
    agent: Optional["Agent"] = None,
    *,
    shared_sessions: bool = False,
    identity_resolver: Optional["IdentityResolver"] = None,
    session_ttl_seconds: int = 3600,
    hooks: Optional[List[InterfaceHook]] = None,
    enable_identity_linking: bool = False,
    link_command: str = "/link",
    link_code_ttl: int = 300,
  ) -> None:
    self.agent = agent
    self._shared_sessions = shared_sessions
    self._identity_resolver = identity_resolver
    self._session_ttl_seconds = session_ttl_seconds
    self._hooks: List[InterfaceHook] = list(hooks or [])
    self._interfaces: List["BaseInterface"] = []
    self._statuses: Dict[int, InterfaceStatus] = {}
    self._restart_counts: Dict[int, int] = {}
    self._shared_session_manager: Optional[SessionManager] = None

    if shared_sessions:
      self._shared_session_manager = SessionManager(session_ttl_seconds=session_ttl_seconds)

    # Identity linking hook (opt-in)
    self._link_hook: Optional[_IdentityLinkHook] = None
    if enable_identity_linking and identity_resolver is not None:
      self._link_hook = _IdentityLinkHook(
        gateway=self,
        command=link_command,
        code_ttl=link_code_ttl,
      )
      self._hooks.insert(0, self._link_hook)  # type: ignore[arg-type]

  # --- Agent binding ---

  def _bind_agent(self, agent: "Agent") -> None:
    """Bind this gateway to an agent (called by Agent constructor).

    Args:
      agent: The Agent instance to bind.

    Raises:
      ValueError: If already bound to a *different* agent.
    """
    if self.agent is not None and self.agent is not agent:
      raise ValueError("Gateway is already bound to a different agent")
    self.agent = agent

  # --- Interface management ---

  def add(self, interface: "BaseInterface") -> "InterfaceGateway":
    """Register an interface with the gateway.

    Binds the interface to the agent, injects the gateway hook bridge,
    optionally shares sessions, and propagates the identity resolver.

    Args:
      interface: BaseInterface instance to add.

    Returns:
      Self for method chaining.
    """
    # Bind to agent (skip if gateway has no agent yet — deferred binding)
    if interface.agent is None and self.agent is not None:
      interface.bind(self.agent)

    # Inject gateway hook bridge at position 0
    bridge = _GatewayHookBridge(self, interface)
    interface._hooks.insert(0, bridge)  # type: ignore[arg-type]

    # Shared sessions
    if self._shared_session_manager is not None:
      interface.session_manager = self._shared_session_manager

    # Propagate identity resolver
    if self._identity_resolver is not None and interface._identity_resolver is None:
      interface._identity_resolver = self._identity_resolver

    self._interfaces.append(interface)
    self._statuses[id(interface)] = InterfaceStatus.pending
    self._restart_counts[id(interface)] = 0
    return self

  def remove(self, interface: "BaseInterface") -> bool:
    """Remove an interface from the gateway.

    Args:
      interface: The interface to remove.

    Returns:
      True if the interface was found and removed.
    """
    if interface not in self._interfaces:
      return False
    self._interfaces.remove(interface)
    self._statuses.pop(id(interface), None)
    self._restart_counts.pop(id(interface), None)
    # Remove gateway hook bridge
    interface._hooks[:] = [h for h in interface._hooks if not isinstance(h, _GatewayHookBridge)]
    return True

  # --- Hook management ---

  def add_hook(self, hook: InterfaceHook) -> "InterfaceGateway":
    """Add a gateway-level hook (applies to all interfaces).

    Args:
      hook: Hook instance to add.

    Returns:
      Self for method chaining.
    """
    self._hooks.append(hook)
    return self

  def remove_hook(self, hook: InterfaceHook) -> bool:
    """Remove a gateway-level hook.

    Args:
      hook: The hook to remove.

    Returns:
      True if the hook was found and removed.
    """
    if hook in self._hooks:
      self._hooks.remove(hook)
      return True
    return False

  # --- Status ---

  def status(self, interface: "BaseInterface") -> InterfaceStatus:
    """Get the current status of a specific interface.

    Args:
      interface: The interface to query.

    Returns:
      Current InterfaceStatus.

    Raises:
      ValueError: If the interface is not registered.
    """
    iface_id = id(interface)
    if iface_id not in self._statuses:
      raise ValueError("Interface is not registered with this gateway")
    return self._statuses[iface_id]

  @property
  def statuses(self) -> Dict[str, InterfaceStatus]:
    """Return a dict mapping interface platform names to their statuses."""
    result: Dict[str, InterfaceStatus] = {}
    for iface in self._interfaces:
      name = iface.config.platform or type(iface).__name__
      result[name] = self._statuses[id(iface)]
    return result

  @property
  def interfaces(self) -> List["BaseInterface"]:
    """Return the list of registered interfaces."""
    return list(self._interfaces)

  @property
  def is_healthy(self) -> bool:
    """True if all interfaces are running or pending."""
    return all(s in (InterfaceStatus.running, InterfaceStatus.pending) for s in self._statuses.values())

  # --- Lifecycle ---

  def _iface_name(self, iface: "BaseInterface") -> str:
    return iface.config.platform or type(iface).__name__

  async def _emit_event(self, event: BaseAgentRunEvent) -> None:
    """Emit an event through the agent's EventBus."""
    if self.agent is not None:
      await self.agent._event_bus.emit(event)

  async def aserve(self, *, name: Optional[str] = None) -> None:
    """Start all interfaces and supervise them (async).

    Blocks until all interfaces stop cleanly or cancellation occurs.

    Args:
      name: Optional prefix for log messages.
    """
    if self.agent is None:
      raise ValueError("Gateway has no agent. Pass gateway= to Agent() constructor or call _bind_agent().")

    if not self._interfaces:
      raise ValueError("InterfaceGateway has no interfaces registered. Call add() first.")

    prefix = f"[{name or 'gateway'}]"

    # Backoff state: interface id -> (delay, last_start_time)
    backoff: Dict[int, Tuple[float, float]] = {}
    task_to_iface: Dict[asyncio.Task, "BaseInterface"] = {}

    def _start_task(iface: "BaseInterface") -> asyncio.Task:
      task = asyncio.create_task(iface.serve_forever())
      task_to_iface[task] = iface
      iface_id = id(iface)
      backoff.setdefault(iface_id, (0.0, 0.0))
      backoff[iface_id] = (backoff[iface_id][0], time.monotonic())
      self._statuses[iface_id] = InterfaceStatus.running
      return task

    # Start all interfaces
    for iface in self._interfaces:
      iface_name = self._iface_name(iface)
      self._statuses[id(iface)] = InterfaceStatus.starting
      _start_task(iface)
      log_info(f"{prefix} Started {iface_name}")
      await self._emit_event(
        InterfaceStartedEvent(
          platform=iface_name,
          interface_class=type(iface).__name__,
        )
      )

    try:
      while task_to_iface:
        done, _ = await asyncio.wait(
          task_to_iface.keys(),
          return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
          iface = task_to_iface.pop(task)
          iface_id = id(iface)
          iface_name = self._iface_name(iface)
          exc = task.exception() if not task.cancelled() else None

          if exc is not None:
            # Crashed — emit error event, restart with backoff
            self._statuses[iface_id] = InterfaceStatus.error
            await self._emit_event(
              InterfaceErrorEvent(
                platform=iface_name,
                interface_class=type(iface).__name__,
                error_message=str(exc),
              )
            )

            current_backoff, last_start = backoff.get(iface_id, (0.0, 0.0))
            elapsed = time.monotonic() - last_start

            if elapsed >= _STABILITY_THRESHOLD:
              next_backoff = 1.0
            else:
              next_backoff = min((current_backoff * 2) or 1.0, _MAX_BACKOFF)

            backoff[iface_id] = (next_backoff, 0.0)
            self._restart_counts[iface_id] = self._restart_counts.get(iface_id, 0) + 1

            log_error(f"{prefix} {iface_name} crashed: {exc}")
            log_warning(f"{prefix} Restarting {iface_name} in {next_backoff:.1f}s")

            self._statuses[iface_id] = InterfaceStatus.restarting
            await asyncio.sleep(next_backoff)
            _start_task(iface)
            log_info(f"{prefix} Restarted {iface_name}")

            await self._emit_event(
              InterfaceRestartedEvent(
                platform=iface_name,
                interface_class=type(iface).__name__,
                restart_count=self._restart_counts[iface_id],
                backoff_seconds=next_backoff,
              )
            )
          else:
            # Clean stop
            self._statuses[iface_id] = InterfaceStatus.stopped
            log_info(f"{prefix} {iface_name} stopped cleanly")
            await self._emit_event(
              InterfaceStoppedEvent(
                platform=iface_name,
                interface_class=type(iface).__name__,
              )
            )

    except asyncio.CancelledError:
      log_info(f"{prefix} Shutting down all interfaces")
      for task in task_to_iface:
        task.cancel()
      await asyncio.gather(*task_to_iface, return_exceptions=True)
      for iface in self._interfaces:
        self._statuses[id(iface)] = InterfaceStatus.stopped
      log_info(f"{prefix} All interfaces stopped")

  def serve(self, *, name: Optional[str] = None) -> None:
    """Start all interfaces (sync wrapper around aserve).

    Args:
      name: Optional prefix for log messages.
    """
    asyncio.run(self.aserve(name=name))
