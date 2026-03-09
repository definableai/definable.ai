"""Unit tests for the agent/auth module.

Tests cover all four auth providers (APIKeyAuth, AllowlistAuth, JWTAuth,
CompositeAuth), the base types (AuthContext, AuthRequest), the resolve_auth
helper, and the case-insensitive header utility.
No real HTTP servers or external services needed.
"""

import hashlib

import pytest

from definable.agent.auth.base import AuthContext, AuthProvider, AuthRequest, resolve_auth
from definable.agent.auth.api_key import APIKeyAuth, _get_header
from definable.agent.auth.allowlist import AllowlistAuth
from definable.agent.auth.composite import CompositeAuth
from definable.agent.auth.jwt import JWTAuth


# ---------------------------------------------------------------------------
# Base types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAuthContext:
  """Tests for AuthContext dataclass."""

  def test_create_minimal(self):
    ctx = AuthContext(user_id="u1")
    assert ctx.user_id == "u1"
    assert ctx.metadata == {}

  def test_create_with_metadata(self):
    ctx = AuthContext(user_id="u1", metadata={"role": "admin"})
    assert ctx.metadata["role"] == "admin"


@pytest.mark.unit
class TestAuthRequest:
  """Tests for AuthRequest dataclass."""

  def test_create_minimal(self):
    req = AuthRequest(platform="telegram")
    assert req.platform == "telegram"
    assert req.user_id is None
    assert req.username is None
    assert req.chat_id is None
    assert req.headers == {}
    assert req.metadata == {}

  def test_create_full(self):
    req = AuthRequest(
      platform="http",
      user_id="u1",
      username="alice",
      chat_id="c1",
      headers={"X-API-Key": "sk-abc"},
      metadata={"ip": "127.0.0.1"},
    )
    assert req.platform == "http"
    assert req.user_id == "u1"
    assert req.headers["X-API-Key"] == "sk-abc"
    assert req.metadata["ip"] == "127.0.0.1"


@pytest.mark.unit
class TestAuthProviderProtocol:
  """Tests for AuthProvider protocol compliance."""

  def test_class_satisfies_protocol(self):
    class MyAuth:
      def authenticate(self, request):
        return AuthContext(user_id="test")

    assert isinstance(MyAuth(), AuthProvider)

  def test_class_without_authenticate_fails(self):
    class NotAuth:
      pass

    assert not isinstance(NotAuth(), AuthProvider)


@pytest.mark.unit
class TestResolveAuth:
  """Tests for the resolve_auth helper that handles sync/async providers."""

  @pytest.mark.asyncio
  async def test_sync_provider(self):
    class SyncAuth:
      def authenticate(self, request):
        return AuthContext(user_id="sync")

    result = await resolve_auth(SyncAuth(), AuthRequest(platform="test"))
    assert result is not None
    assert result.user_id == "sync"

  @pytest.mark.asyncio
  async def test_async_provider(self):
    class AsyncAuth:
      async def authenticate(self, request):
        return AuthContext(user_id="async")

    result = await resolve_auth(AsyncAuth(), AuthRequest(platform="test"))
    assert result is not None
    assert result.user_id == "async"

  @pytest.mark.asyncio
  async def test_returns_none_on_failure(self):
    class FailAuth:
      def authenticate(self, request):
        return None

    result = await resolve_auth(FailAuth(), AuthRequest(platform="test"))
    assert result is None


# ---------------------------------------------------------------------------
# _get_header helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetHeader:
  """Tests for case-insensitive header lookup."""

  def test_exact_match(self):
    assert _get_header({"X-API-Key": "abc"}, "X-API-Key") == "abc"

  def test_case_insensitive_lookup(self):
    assert _get_header({"x-api-key": "abc"}, "X-API-Key") == "abc"

  def test_missing_header(self):
    assert _get_header({}, "X-API-Key") == ""

  def test_empty_value_falls_through(self):
    """Empty string is falsy, so exact match fails and case-insensitive kicks in."""
    assert _get_header({"X-API-Key": ""}, "X-API-Key") == ""


# ---------------------------------------------------------------------------
# APIKeyAuth
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAPIKeyAuth:
  """Tests for API key authentication."""

  def test_init_with_set(self):
    auth = APIKeyAuth(keys={"sk-abc", "sk-def"})
    assert len(auth.keys) == 2

  def test_init_with_string(self):
    auth = APIKeyAuth(keys="sk-single")
    assert auth.keys == {"sk-single"}

  def test_init_rejects_empty_set(self):
    with pytest.raises(ValueError, match="at least one"):
      APIKeyAuth(keys=set())

  def test_init_rejects_empty_string(self):
    with pytest.raises(ValueError, match="at least one"):
      APIKeyAuth(keys="")

  def test_init_strips_empty_strings(self):
    """Empty strings should be discarded from the key set."""
    with pytest.raises(ValueError, match="at least one"):
      APIKeyAuth(keys={"", ""})

  def test_custom_header(self):
    auth = APIKeyAuth(keys={"key1"}, header="Authorization")
    assert auth.header == "Authorization"

  def test_valid_key_returns_context(self):
    auth = APIKeyAuth(keys={"sk-abc123"})
    req = AuthRequest(platform="http", headers={"X-API-Key": "sk-abc123"})
    result = auth.authenticate(req)
    assert result is not None
    assert result.user_id.startswith("apikey_")

  def test_valid_key_user_id_is_hashed(self):
    auth = APIKeyAuth(keys={"sk-abc123"})
    req = AuthRequest(platform="http", headers={"X-API-Key": "sk-abc123"})
    result = auth.authenticate(req)
    expected_hash = hashlib.sha256("sk-abc123".encode()).hexdigest()[:12]
    assert result is not None
    assert result.user_id == f"apikey_{expected_hash}"

  def test_invalid_key_returns_none(self):
    auth = APIKeyAuth(keys={"sk-abc123"})
    req = AuthRequest(platform="http", headers={"X-API-Key": "wrong-key"})
    result = auth.authenticate(req)
    assert result is None

  def test_missing_header_returns_none(self):
    auth = APIKeyAuth(keys={"sk-abc123"})
    req = AuthRequest(platform="http", headers={})
    result = auth.authenticate(req)
    assert result is None

  def test_bearer_prefix_stripped(self):
    """Keys in Authorization header with 'Bearer ' prefix are handled."""
    auth = APIKeyAuth(keys={"sk-abc123"})
    req = AuthRequest(platform="http", headers={"Authorization": "Bearer sk-abc123"})
    result = auth.authenticate(req)
    assert result is not None

  def test_bearer_case_insensitive(self):
    auth = APIKeyAuth(keys={"sk-abc123"})
    req = AuthRequest(platform="http", headers={"Authorization": "bearer sk-abc123"})
    result = auth.authenticate(req)
    assert result is not None

  def test_fallback_to_authorization_header(self):
    """When the primary header is missing, falls back to Authorization."""
    auth = APIKeyAuth(keys={"sk-abc123"})
    req = AuthRequest(platform="http", headers={"Authorization": "sk-abc123"})
    result = auth.authenticate(req)
    assert result is not None

  def test_header_case_insensitive(self):
    auth = APIKeyAuth(keys={"sk-abc123"})
    req = AuthRequest(platform="http", headers={"x-api-key": "sk-abc123"})
    result = auth.authenticate(req)
    assert result is not None

  def test_works_with_non_authrequest_object(self):
    """APIKeyAuth works with any object that has a headers attribute."""

    class FakeRequest:
      headers = {"X-API-Key": "sk-abc123"}

    auth = APIKeyAuth(keys={"sk-abc123"})
    result = auth.authenticate(FakeRequest())
    assert result is not None

  def test_object_without_headers_returns_none(self):
    auth = APIKeyAuth(keys={"sk-abc123"})
    result = auth.authenticate(object())
    assert result is None

  def test_whitespace_in_key_stripped(self):
    auth = APIKeyAuth(keys={"sk-abc123"})
    req = AuthRequest(platform="http", headers={"X-API-Key": "  sk-abc123  "})
    result = auth.authenticate(req)
    assert result is not None

  def test_multiple_keys_any_valid(self):
    auth = APIKeyAuth(keys={"key1", "key2", "key3"})
    for key in ["key1", "key2", "key3"]:
      req = AuthRequest(platform="http", headers={"X-API-Key": key})
      result = auth.authenticate(req)
      assert result is not None, f"key {key} should be valid"


# ---------------------------------------------------------------------------
# AllowlistAuth
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAllowlistAuth:
  """Tests for allowlist-based authentication."""

  def test_valid_user_returns_context(self):
    auth = AllowlistAuth(user_ids={"12345"})
    req = AuthRequest(platform="telegram", user_id="12345")
    result = auth.authenticate(req)
    assert result is not None
    assert result.user_id == "12345"

  def test_valid_user_metadata(self):
    auth = AllowlistAuth(user_ids={"12345"})
    req = AuthRequest(platform="telegram", user_id="12345")
    result = auth.authenticate(req)
    assert result is not None
    assert result.metadata["platform"] == "telegram"
    assert result.metadata["auth_method"] == "allowlist"

  def test_invalid_user_returns_none(self):
    auth = AllowlistAuth(user_ids={"12345"})
    req = AuthRequest(platform="telegram", user_id="99999")
    result = auth.authenticate(req)
    assert result is None

  def test_none_user_id_returns_none(self):
    auth = AllowlistAuth(user_ids={"12345"})
    req = AuthRequest(platform="telegram", user_id=None)
    result = auth.authenticate(req)
    assert result is None

  def test_non_authrequest_returns_none(self):
    """AllowlistAuth is only for messaging; non-AuthRequest returns None."""
    auth = AllowlistAuth(user_ids={"12345"})
    result = auth.authenticate(object())
    assert result is None

  def test_platform_scoping_matches(self):
    auth = AllowlistAuth(user_ids={"12345"}, platforms={"telegram"})
    req = AuthRequest(platform="telegram", user_id="12345")
    result = auth.authenticate(req)
    assert result is not None

  def test_platform_scoping_rejects(self):
    auth = AllowlistAuth(user_ids={"12345"}, platforms={"telegram"})
    req = AuthRequest(platform="discord", user_id="12345")
    result = auth.authenticate(req)
    assert result is None

  def test_chat_id_filter_matches(self):
    auth = AllowlistAuth(user_ids={"12345"}, chat_ids={"chat-abc"})
    req = AuthRequest(platform="telegram", user_id="12345", chat_id="chat-abc")
    result = auth.authenticate(req)
    assert result is not None

  def test_chat_id_filter_rejects(self):
    auth = AllowlistAuth(user_ids={"12345"}, chat_ids={"chat-abc"})
    req = AuthRequest(platform="telegram", user_id="12345", chat_id="other-chat")
    result = auth.authenticate(req)
    assert result is None

  def test_chat_id_none_with_filter(self):
    """Request with no chat_id fails chat_id filter."""
    auth = AllowlistAuth(user_ids={"12345"}, chat_ids={"chat-abc"})
    req = AuthRequest(platform="telegram", user_id="12345")
    result = auth.authenticate(req)
    assert result is None

  def test_combined_platform_and_chat_filter(self):
    auth = AllowlistAuth(
      user_ids={"12345"},
      platforms={"telegram"},
      chat_ids={"chat-abc"},
    )
    # All match
    req = AuthRequest(platform="telegram", user_id="12345", chat_id="chat-abc")
    assert auth.authenticate(req) is not None
    # Wrong platform
    req2 = AuthRequest(platform="discord", user_id="12345", chat_id="chat-abc")
    assert auth.authenticate(req2) is None
    # Wrong chat
    req3 = AuthRequest(platform="telegram", user_id="12345", chat_id="wrong")
    assert auth.authenticate(req3) is None


# ---------------------------------------------------------------------------
# JWTAuth
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJWTAuth:
  """Tests for JWT authentication."""

  @pytest.fixture
  def jwt_module(self):
    """Skip if pyjwt is not installed."""
    pytest.importorskip("jwt", reason="pyjwt not installed")
    import jwt

    return jwt

  @pytest.fixture
  def auth(self, jwt_module):
    return JWTAuth(secret="test-secret")

  @pytest.fixture
  def valid_token(self, jwt_module):
    return jwt_module.encode({"sub": "user-1", "role": "admin"}, "test-secret", algorithm="HS256")

  def test_valid_token_returns_context(self, auth, valid_token):
    req = AuthRequest(platform="http", headers={"Authorization": f"Bearer {valid_token}"})
    result = auth.authenticate(req)
    assert result is not None
    assert result.user_id == "user-1"

  def test_metadata_excludes_identity_claims(self, auth, valid_token):
    req = AuthRequest(platform="http", headers={"Authorization": f"Bearer {valid_token}"})
    result = auth.authenticate(req)
    assert "sub" not in result.metadata
    assert result.metadata.get("role") == "admin"

  def test_user_id_from_user_id_claim(self, jwt_module):
    auth = JWTAuth(secret="s")
    token = jwt_module.encode({"user_id": "u2"}, "s", algorithm="HS256")
    req = AuthRequest(platform="http", headers={"Authorization": f"Bearer {token}"})
    result = auth.authenticate(req)
    assert result is not None
    assert result.user_id == "u2"

  def test_user_id_from_id_claim(self, jwt_module):
    auth = JWTAuth(secret="s")
    token = jwt_module.encode({"id": "u3"}, "s", algorithm="HS256")
    req = AuthRequest(platform="http", headers={"Authorization": f"Bearer {token}"})
    result = auth.authenticate(req)
    assert result is not None
    assert result.user_id == "u3"

  def test_sub_takes_priority(self, jwt_module):
    """sub claim takes priority over user_id and id."""
    auth = JWTAuth(secret="s")
    token = jwt_module.encode({"sub": "primary", "user_id": "secondary", "id": "tertiary"}, "s", algorithm="HS256")
    req = AuthRequest(platform="http", headers={"Authorization": f"Bearer {token}"})
    result = auth.authenticate(req)
    assert result is not None
    assert result.user_id == "primary"

  def test_invalid_token_returns_none(self, auth):
    req = AuthRequest(platform="http", headers={"Authorization": "Bearer invalid.token.here"})
    result = auth.authenticate(req)
    assert result is None

  def test_wrong_secret_returns_none(self, jwt_module):
    auth = JWTAuth(secret="correct-secret")
    token = jwt_module.encode({"sub": "u1"}, "wrong-secret", algorithm="HS256")
    req = AuthRequest(platform="http", headers={"Authorization": f"Bearer {token}"})
    result = auth.authenticate(req)
    assert result is None

  def test_missing_auth_header_returns_none(self, auth):
    req = AuthRequest(platform="http", headers={})
    result = auth.authenticate(req)
    assert result is None

  def test_non_bearer_header_returns_none(self, auth):
    req = AuthRequest(platform="http", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    result = auth.authenticate(req)
    assert result is None

  def test_empty_bearer_token_returns_none(self, auth):
    req = AuthRequest(platform="http", headers={"Authorization": "Bearer "})
    result = auth.authenticate(req)
    assert result is None

  def test_no_user_id_in_claims_returns_none(self, jwt_module):
    auth = JWTAuth(secret="s")
    token = jwt_module.encode({"role": "admin"}, "s", algorithm="HS256")
    req = AuthRequest(platform="http", headers={"Authorization": f"Bearer {token}"})
    result = auth.authenticate(req)
    assert result is None

  def test_audience_validation(self, jwt_module):
    auth = JWTAuth(secret="s", audience="my-app")
    token = jwt_module.encode({"sub": "u1", "aud": "my-app"}, "s", algorithm="HS256")
    req = AuthRequest(platform="http", headers={"Authorization": f"Bearer {token}"})
    result = auth.authenticate(req)
    assert result is not None

  def test_wrong_audience_returns_none(self, jwt_module):
    auth = JWTAuth(secret="s", audience="my-app")
    token = jwt_module.encode({"sub": "u1", "aud": "other-app"}, "s", algorithm="HS256")
    req = AuthRequest(platform="http", headers={"Authorization": f"Bearer {token}"})
    result = auth.authenticate(req)
    assert result is None

  def test_issuer_validation(self, jwt_module):
    auth = JWTAuth(secret="s", issuer="auth.example.com")
    token = jwt_module.encode({"sub": "u1", "iss": "auth.example.com"}, "s", algorithm="HS256")
    req = AuthRequest(platform="http", headers={"Authorization": f"Bearer {token}"})
    result = auth.authenticate(req)
    assert result is not None

  def test_numeric_user_id_converted_to_string(self, jwt_module):
    """Numeric ID in a non-standard claim is converted to string."""
    auth = JWTAuth(secret="s")
    # Use "id" claim (not "sub") — PyJWT validates sub must be string
    token = jwt_module.encode({"id": 12345}, "s", algorithm="HS256")
    req = AuthRequest(platform="http", headers={"Authorization": f"Bearer {token}"})
    result = auth.authenticate(req)
    assert result is not None
    assert result.user_id == "12345"
    assert isinstance(result.user_id, str)

  def test_header_case_insensitive(self, auth, valid_token):
    req = AuthRequest(platform="http", headers={"authorization": f"Bearer {valid_token}"})
    result = auth.authenticate(req)
    assert result is not None


# ---------------------------------------------------------------------------
# CompositeAuth
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompositeAuth:
  """Tests for composite auth provider chaining."""

  def test_init_requires_providers(self):
    with pytest.raises(ValueError, match="at least one"):
      CompositeAuth()

  @pytest.mark.asyncio
  async def test_first_success_wins(self):
    class Auth1:
      def authenticate(self, request):
        return AuthContext(user_id="from-auth1")

    class Auth2:
      def authenticate(self, request):
        return AuthContext(user_id="from-auth2")

    composite = CompositeAuth(Auth1(), Auth2())
    result = await composite.authenticate(AuthRequest(platform="test"))
    assert result is not None
    assert result.user_id == "from-auth1"

  @pytest.mark.asyncio
  async def test_skips_failed_providers(self):
    class FailAuth:
      def authenticate(self, request):
        return None

    class SuccessAuth:
      def authenticate(self, request):
        return AuthContext(user_id="success")

    composite = CompositeAuth(FailAuth(), SuccessAuth())
    result = await composite.authenticate(AuthRequest(platform="test"))
    assert result is not None
    assert result.user_id == "success"

  @pytest.mark.asyncio
  async def test_all_fail_returns_none(self):
    class FailAuth:
      def authenticate(self, request):
        return None

    composite = CompositeAuth(FailAuth(), FailAuth())
    result = await composite.authenticate(AuthRequest(platform="test"))
    assert result is None

  @pytest.mark.asyncio
  async def test_mixed_sync_async(self):
    class SyncAuth:
      def authenticate(self, request):
        return None

    class AsyncAuth:
      async def authenticate(self, request):
        return AuthContext(user_id="async-success")

    composite = CompositeAuth(SyncAuth(), AsyncAuth())
    result = await composite.authenticate(AuthRequest(platform="test"))
    assert result is not None
    assert result.user_id == "async-success"

  @pytest.mark.asyncio
  async def test_with_real_providers(self):
    """Composite with APIKeyAuth + AllowlistAuth."""
    composite = CompositeAuth(
      APIKeyAuth(keys={"sk-abc"}),
      AllowlistAuth(user_ids={"12345"}, platforms={"telegram"}),
    )
    # HTTP request with valid API key
    http_req = AuthRequest(platform="http", headers={"X-API-Key": "sk-abc"})
    result = await composite.authenticate(http_req)
    assert result is not None
    assert result.user_id.startswith("apikey_")

    # Telegram request with valid user
    tg_req = AuthRequest(platform="telegram", user_id="12345")
    result = await composite.authenticate(tg_req)
    assert result is not None
    assert result.user_id == "12345"

    # No valid credentials
    bad_req = AuthRequest(platform="http", headers={})
    result = await composite.authenticate(bad_req)
    assert result is None


# ---------------------------------------------------------------------------
# Lazy imports in __init__
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAuthModuleImports:
  """Tests for the auth module's lazy import mechanism."""

  def test_import_apikey(self):
    from definable.agent.auth import APIKeyAuth

    assert APIKeyAuth is not None

  def test_import_allowlist(self):
    from definable.agent.auth import AllowlistAuth

    assert AllowlistAuth is not None

  def test_import_composite(self):
    from definable.agent.auth import CompositeAuth

    assert CompositeAuth is not None

  def test_import_jwt(self):
    from definable.agent.auth import JWTAuth

    assert JWTAuth is not None

  def test_import_base_types(self):
    from definable.agent.auth import AuthContext, AuthProvider, AuthRequest, resolve_auth

    assert AuthContext is not None
    assert AuthProvider is not None
    assert AuthRequest is not None
    assert resolve_auth is not None

  def test_invalid_attr_raises(self):
    with pytest.raises(AttributeError):
      from definable.agent import auth

      auth.__getattr__("NonExistent")
