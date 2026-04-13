"""Tests for HITL permission service."""

from definable.agent.hitl.permissions import PermissionService
from definable.agent.hitl.settings import Settings
from definable.agent.hitl.types import (
  PermissionAction,
  PermissionDecision,
  PermissionRequest,
  PermissionResponse,
)


def _req(name: str = "bash", args: dict | None = None) -> PermissionRequest:
  return PermissionRequest(tool_name=name, tool_args=args or {})


class TestPermissionService:
  async def test_settings_allow_skips_resolver(self):
    """Persistent 'allow' in settings should bypass the resolver entirely."""
    settings = Settings(tool_permissions={"bash": "allow"})

    async def should_not_be_called(request):
      raise AssertionError("resolver should not be called")

    svc = PermissionService(resolver=should_not_be_called, settings=settings)
    resp = await svc.check(_req("bash"))
    assert resp.decision == PermissionDecision.allow_once

  async def test_settings_deny_blocks(self):
    """Persistent 'deny' in settings should block without calling resolver."""
    settings = Settings(tool_permissions={"bash": "deny"})
    svc = PermissionService(settings=settings)
    resp = await svc.check(_req("bash"))
    assert resp.decision == PermissionDecision.deny
    assert "permanently denied" in (resp.feedback or "")

  async def test_config_default_allow(self):
    """Config-level 'allow' default should skip the resolver."""
    svc = PermissionService(
      defaults={"bash": PermissionAction.allow},
      settings=Settings(),
    )
    resp = await svc.check(_req("bash"))
    assert resp.decision == PermissionDecision.allow_once

  async def test_config_default_deny(self):
    """Config-level 'deny' default should block."""
    svc = PermissionService(
      defaults={"bash": PermissionAction.deny},
      settings=Settings(),
    )
    resp = await svc.check(_req("bash"))
    assert resp.decision == PermissionDecision.deny

  async def test_headless_mode_allows_all(self):
    """No resolver = headless mode → auto-allow."""
    svc = PermissionService(resolver=None, settings=Settings())
    resp = await svc.check(_req("bash"))
    assert resp.decision == PermissionDecision.allow_once

  async def test_resolver_called_on_ask(self):
    """When action is 'ask', the resolver callback is called."""
    called = False

    async def resolver(request):
      nonlocal called
      called = True
      return PermissionResponse(decision=PermissionDecision.allow_once)

    svc = PermissionService(resolver=resolver, settings=Settings())
    resp = await svc.check(_req("bash"))
    assert called
    assert resp.decision == PermissionDecision.allow_once

  async def test_always_allow_persists(self, tmp_path):
    """'always allow' should persist to settings and skip future prompts."""
    settings = Settings()

    call_count = 0

    async def resolver(request):
      nonlocal call_count
      call_count += 1
      return PermissionResponse(decision=PermissionDecision.allow_always)

    svc = PermissionService(resolver=resolver, settings=settings)

    # First call: resolver is called, decision is always_allow
    resp = await svc.check(_req("bash"))
    assert resp.decision == PermissionDecision.allow_always
    assert call_count == 1

    # Settings now have "bash": "allow"
    assert settings.get_tool_permission("bash") == PermissionAction.allow

    # Second call: settings say "allow", resolver should NOT be called
    resp2 = await svc.check(_req("bash"))
    assert resp2.decision == PermissionDecision.allow_once
    assert call_count == 1  # Not incremented

  async def test_deny_with_feedback(self):
    """Denial with user feedback should pass feedback through."""

    async def resolver(request):
      return PermissionResponse(decision=PermissionDecision.deny, feedback="Too dangerous!")

    svc = PermissionService(resolver=resolver, settings=Settings())
    resp = await svc.check(_req("bash"))
    assert resp.decision == PermissionDecision.deny
    assert resp.feedback == "Too dangerous!"

  async def test_settings_take_precedence_over_defaults(self):
    """Settings should override config defaults."""
    settings = Settings(tool_permissions={"bash": "allow"})
    svc = PermissionService(
      defaults={"bash": PermissionAction.deny},
      settings=settings,
    )
    resp = await svc.check(_req("bash"))
    # Settings say "allow" even though defaults say "deny"
    assert resp.decision == PermissionDecision.allow_once
