"""Tests for the unified authentication layer.

No API keys required — all tests use mocks and local auth providers.
"""

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from definable.auth.allowlist import AllowlistAuth
from definable.auth.api_key import APIKeyAuth
from definable.auth.base import AuthContext, AuthRequest, resolve_auth
from definable.auth.composite import CompositeAuth
from definable.interfaces.message import InterfaceMessage


# ---------------------------------------------------------------------------
# AuthRequest
# ---------------------------------------------------------------------------


class TestAuthRequest:
  def test_construction_defaults(self):
    req = AuthRequest(platform="telegram")
    assert req.platform == "telegram"
    assert req.user_id is None
    assert req.username is None
    assert req.chat_id is None
    assert req.headers == {}
    assert req.metadata == {}

  def test_construction_full(self):
    req = AuthRequest(
      platform="discord",
      user_id="u1",
      username="alice",
      chat_id="ch1",
      headers={"x-custom": "val"},
      metadata={"extra": True},
    )
    assert req.platform == "discord"
    assert req.user_id == "u1"
    assert req.username == "alice"
    assert req.chat_id == "ch1"
    assert req.headers == {"x-custom": "val"}
    assert req.metadata == {"extra": True}

  def test_http_request(self):
    req = AuthRequest(
      platform="http",
      headers={"x-api-key": "sk-123", "content-type": "application/json"},
    )
    assert req.platform == "http"
    assert req.headers["x-api-key"] == "sk-123"


# ---------------------------------------------------------------------------
# AllowlistAuth
# ---------------------------------------------------------------------------


class TestAllowlistAuth:
  def test_user_in_allowlist(self):
    auth = AllowlistAuth(user_ids={"u1", "u2"})
    req = AuthRequest(platform="telegram", user_id="u1")
    ctx = auth.authenticate(req)
    assert ctx is not None
    assert ctx.user_id == "u1"
    assert ctx.metadata["platform"] == "telegram"
    assert ctx.metadata["auth_method"] == "allowlist"

  def test_user_not_in_allowlist(self):
    auth = AllowlistAuth(user_ids={"u1", "u2"})
    req = AuthRequest(platform="telegram", user_id="u99")
    assert auth.authenticate(req) is None

  def test_no_user_id(self):
    auth = AllowlistAuth(user_ids={"u1"})
    req = AuthRequest(platform="telegram")
    assert auth.authenticate(req) is None

  def test_platform_scoping_match(self):
    auth = AllowlistAuth(user_ids={"u1"}, platforms={"telegram"})
    req = AuthRequest(platform="telegram", user_id="u1")
    assert auth.authenticate(req) is not None

  def test_platform_scoping_mismatch(self):
    auth = AllowlistAuth(user_ids={"u1"}, platforms={"telegram"})
    req = AuthRequest(platform="discord", user_id="u1")
    assert auth.authenticate(req) is None

  def test_chat_id_filter_match(self):
    auth = AllowlistAuth(user_ids={"u1"}, chat_ids={"chat1"})
    req = AuthRequest(platform="telegram", user_id="u1", chat_id="chat1")
    assert auth.authenticate(req) is not None

  def test_chat_id_filter_mismatch(self):
    auth = AllowlistAuth(user_ids={"u1"}, chat_ids={"chat1"})
    req = AuthRequest(platform="telegram", user_id="u1", chat_id="chat99")
    assert auth.authenticate(req) is None

  def test_non_auth_request_passthrough(self):
    """Non-AuthRequest objects should return None (not applicable)."""
    auth = AllowlistAuth(user_ids={"u1"})
    fake_request = MagicMock()
    fake_request.headers = {"x-api-key": "test"}
    assert auth.authenticate(fake_request) is None


# ---------------------------------------------------------------------------
# CompositeAuth
# ---------------------------------------------------------------------------


class TestCompositeAuth:
  @pytest.mark.asyncio
  async def test_first_success_wins(self):
    p1 = MagicMock()
    p1.authenticate = MagicMock(return_value=None)
    p2 = MagicMock()
    p2.authenticate = MagicMock(return_value=AuthContext(user_id="winner"))
    p3 = MagicMock()
    p3.authenticate = MagicMock(return_value=AuthContext(user_id="too-late"))

    composite = CompositeAuth(p1, p2, p3)
    req = AuthRequest(platform="test")
    ctx = await composite.authenticate(req)

    assert ctx is not None
    assert ctx.user_id == "winner"
    # p3 should not have been called
    p3.authenticate.assert_not_called()

  @pytest.mark.asyncio
  async def test_all_fail(self):
    p1 = MagicMock()
    p1.authenticate = MagicMock(return_value=None)
    p2 = MagicMock()
    p2.authenticate = MagicMock(return_value=None)

    composite = CompositeAuth(p1, p2)
    req = AuthRequest(platform="test")
    assert await composite.authenticate(req) is None

  def test_empty_raises(self):
    with pytest.raises(ValueError, match="at least one provider"):
      CompositeAuth()

  @pytest.mark.asyncio
  async def test_mixed_sync_async(self):
    sync_provider = AllowlistAuth(user_ids={"u1"}, platforms={"telegram"})

    class AsyncProvider:
      async def authenticate(self, request: Any) -> Optional[AuthContext]:
        return AuthContext(user_id="async-user")

    # Sync provider returns None (wrong platform), async wins
    composite = CompositeAuth(sync_provider, AsyncProvider())
    req = AuthRequest(platform="http", user_id="u1")
    ctx = await composite.authenticate(req)
    assert ctx is not None
    assert ctx.user_id == "async-user"


# ---------------------------------------------------------------------------
# resolve_auth
# ---------------------------------------------------------------------------


class TestResolveAuth:
  @pytest.mark.asyncio
  async def test_sync_provider(self):
    provider = AllowlistAuth(user_ids={"u1"})
    req = AuthRequest(platform="telegram", user_id="u1")
    ctx = await resolve_auth(provider, req)
    assert ctx is not None
    assert ctx.user_id == "u1"

  @pytest.mark.asyncio
  async def test_async_provider(self):
    class AsyncProvider:
      async def authenticate(self, request: Any) -> Optional[AuthContext]:
        return AuthContext(user_id="async-u1")

    req = AuthRequest(platform="test")
    ctx = await resolve_auth(AsyncProvider(), req)
    assert ctx is not None
    assert ctx.user_id == "async-u1"

  @pytest.mark.asyncio
  async def test_provider_returns_none(self):
    provider = AllowlistAuth(user_ids={"u1"})
    req = AuthRequest(platform="telegram", user_id="u99")
    ctx = await resolve_auth(provider, req)
    assert ctx is None


# ---------------------------------------------------------------------------
# BaseInterface auth integration
# ---------------------------------------------------------------------------


class _MockInterface:
  """Minimal stand-in for BaseInterface to test auth pipeline."""

  def __init__(self, *, auth=None):
    from definable.interfaces.base import BaseInterface

    # We need to test _check_auth, so we borrow the method
    self._auth = auth
    self.config = MagicMock()
    self.config.platform = "test"
    # Bind the method
    self._check_auth = BaseInterface._check_auth.__get__(self)


class TestBaseInterfaceAuth:
  @pytest.mark.asyncio
  async def test_no_auth_passes(self):
    iface = _MockInterface(auth=None)
    msg = InterfaceMessage(
      platform="telegram",
      platform_user_id="u1",
      platform_chat_id="c1",
      platform_message_id="m1",
    )
    assert await iface._check_auth(msg) is True
    assert "auth_context" not in msg.metadata

  @pytest.mark.asyncio
  async def test_auth_false_passes(self):
    iface = _MockInterface(auth=False)
    msg = InterfaceMessage(
      platform="telegram",
      platform_user_id="u1",
      platform_chat_id="c1",
      platform_message_id="m1",
    )
    assert await iface._check_auth(msg) is True

  @pytest.mark.asyncio
  async def test_auth_allowed(self):
    auth = AllowlistAuth(user_ids={"u1"})
    iface = _MockInterface(auth=auth)
    msg = InterfaceMessage(
      platform="telegram",
      platform_user_id="u1",
      platform_chat_id="c1",
      platform_message_id="m1",
    )
    assert await iface._check_auth(msg) is True
    assert "auth_context" in msg.metadata
    assert msg.metadata["auth_context"].user_id == "u1"

  @pytest.mark.asyncio
  async def test_auth_denied(self):
    auth = AllowlistAuth(user_ids={"u1"})
    iface = _MockInterface(auth=auth)
    msg = InterfaceMessage(
      platform="telegram",
      platform_user_id="u99",
      platform_chat_id="c1",
      platform_message_id="m1",
    )
    assert await iface._check_auth(msg) is False
    assert "auth_context" not in msg.metadata

  @pytest.mark.asyncio
  async def test_auth_provider_exception(self):
    """Auth provider exceptions are non-fatal: message is rejected."""

    class BrokenAuth:
      def authenticate(self, request):
        raise RuntimeError("boom")

    iface = _MockInterface(auth=BrokenAuth())
    msg = InterfaceMessage(
      platform="telegram",
      platform_user_id="u1",
      platform_chat_id="c1",
      platform_message_id="m1",
    )
    assert await iface._check_auth(msg) is False

  @pytest.mark.asyncio
  async def test_auth_context_fields_populated(self):
    auth = AllowlistAuth(user_ids={"u1"})
    iface = _MockInterface(auth=auth)
    msg = InterfaceMessage(
      platform="telegram",
      platform_user_id="u1",
      platform_chat_id="c1",
      platform_message_id="m1",
      username="alice",
    )
    assert await iface._check_auth(msg) is True
    ctx = msg.metadata["auth_context"]
    assert ctx.metadata["platform"] == "telegram"
    assert ctx.metadata["auth_method"] == "allowlist"


# ---------------------------------------------------------------------------
# Backward compat: APIKeyAuth with AuthRequest
# ---------------------------------------------------------------------------


class TestAPIKeyAuthWithAuthRequest:
  def test_api_key_via_auth_request(self):
    """APIKeyAuth should work with AuthRequest via the headers dict."""
    auth = APIKeyAuth(keys={"sk-123"})
    req = AuthRequest(
      platform="http",
      headers={"x-api-key": "sk-123"},
    )
    ctx = auth.authenticate(req)
    assert ctx is not None
    assert ctx.user_id.startswith("apikey_")

  def test_api_key_via_auth_request_bearer(self):
    auth = APIKeyAuth(keys={"sk-456"})
    req = AuthRequest(
      platform="http",
      headers={"authorization": "Bearer sk-456"},
    )
    ctx = auth.authenticate(req)
    assert ctx is not None

  def test_api_key_via_auth_request_wrong_key(self):
    auth = APIKeyAuth(keys={"sk-123"})
    req = AuthRequest(
      platform="http",
      headers={"x-api-key": "wrong"},
    )
    assert auth.authenticate(req) is None

  def test_api_key_no_headers(self):
    auth = APIKeyAuth(keys={"sk-123"})
    req = AuthRequest(platform="http")
    assert auth.authenticate(req) is None


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
  def test_eager_imports(self):
    from definable.auth import AuthContext as _AC
    from definable.auth import AuthProvider as _AP
    from definable.auth import AuthRequest as _AR
    from definable.auth import resolve_auth as _ra

    assert _AR is not None
    assert _ra is not None
    assert _AC is not None
    assert _AP is not None

  def test_lazy_imports(self):
    from definable.auth import AllowlistAuth, CompositeAuth

    assert AllowlistAuth is not None
    assert CompositeAuth is not None
