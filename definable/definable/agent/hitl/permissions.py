"""Permission service — gates tool execution with user permission checks.

Resolution order:
  1. Persistent settings (``.definable/settings.json``) — "always allow" rules
  2. Config-level defaults (set by agent creator per-tool)
  3. Resolver callback (the UI layer prompts the user)
  4. No resolver = headless mode → allow everything
"""

from typing import Dict, Optional

from definable.agent.hitl.settings import Settings
from definable.agent.hitl.types import (
  PermissionAction,
  PermissionDecision,
  PermissionRequest,
  PermissionResolver,
  PermissionResponse,
)
from definable.utils.log import log_debug


class PermissionService:
  """Gates tool execution with persistent rules and user prompts.

  Args:
    resolver: Async callback that prompts the user for a decision.
        ``None`` = headless mode (all "ask" tools auto-allowed).
    defaults: Per-tool default actions set by the agent creator.
    settings: Pre-loaded ``Settings`` instance.  Loaded from disk if omitted.
  """

  def __init__(
    self,
    *,
    resolver: Optional[PermissionResolver] = None,
    defaults: Optional[Dict[str, PermissionAction]] = None,
    settings: Optional[Settings] = None,
  ) -> None:
    self._resolver = resolver
    self._defaults: Dict[str, PermissionAction] = defaults or {}
    self._settings = settings or Settings.load()

  @property
  def settings(self) -> Settings:
    return self._settings

  async def check(self, request: PermissionRequest) -> PermissionResponse:
    """Check whether a tool call is permitted.

    Returns a ``PermissionResponse``.  The loop uses the decision to
    either execute the tool or send a denial message back to the model.
    """
    tool_name = request.tool_name

    # 1. Persistent settings (last write wins)
    persisted = self._settings.get_tool_permission(tool_name)
    if persisted == PermissionAction.allow:
      log_debug(f"Permission '{tool_name}': allowed (settings)")
      return PermissionResponse(decision=PermissionDecision.allow_once)
    if persisted == PermissionAction.deny:
      log_debug(f"Permission '{tool_name}': denied (settings)")
      return PermissionResponse(
        decision=PermissionDecision.deny,
        feedback=f"Tool '{tool_name}' is permanently denied in settings.",
      )

    # 2. Config-level defaults
    default_action = self._defaults.get(tool_name, PermissionAction.ask)
    if default_action == PermissionAction.allow:
      log_debug(f"Permission '{tool_name}': allowed (config default)")
      return PermissionResponse(decision=PermissionDecision.allow_once)
    if default_action == PermissionAction.deny:
      log_debug(f"Permission '{tool_name}': denied (config default)")
      return PermissionResponse(
        decision=PermissionDecision.deny,
        feedback=f"Tool '{tool_name}' is denied by configuration.",
      )

    # 3. No resolver = headless / non-interactive → allow
    if self._resolver is None:
      log_debug(f"Permission '{tool_name}': allowed (headless)")
      return PermissionResponse(decision=PermissionDecision.allow_once)

    # 4. Ask the user
    response = await self._resolver(request)

    # Persist "always allow"
    if response.decision == PermissionDecision.allow_always:
      self._settings.set_tool_permission(tool_name, PermissionAction.allow)
      log_debug(f"Permission '{tool_name}': always allow (persisted)")

    return response
