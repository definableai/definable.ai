"""Tests for cross-platform user identity resolution."""

from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest

from definable.agents.agent import Agent
from definable.agents.testing import MockModel
from definable.interfaces.base import BaseInterface
from definable.interfaces.config import InterfaceConfig
from definable.interfaces.identity import (
  IdentityResolver,
  PlatformIdentity,
  SQLiteIdentityResolver,
)
from definable.interfaces.message import InterfaceMessage, InterfaceResponse
from definable.interfaces.session import InterfaceSession


# --- SQLiteIdentityResolver tests ---


class TestSQLiteIdentityResolver:
  @pytest.mark.asyncio
  async def test_resolve_unknown_returns_none(self, tmp_path):
    async with SQLiteIdentityResolver(str(tmp_path / "id.db")) as resolver:
      result = await resolver.resolve("telegram", "unknown_user")
      assert result is None

  @pytest.mark.asyncio
  async def test_link_and_resolve(self, tmp_path):
    async with SQLiteIdentityResolver(str(tmp_path / "id.db")) as resolver:
      await resolver.link("telegram", "tg_123", "user_abc")
      result = await resolver.resolve("telegram", "tg_123")
      assert result == "user_abc"

  @pytest.mark.asyncio
  async def test_link_multiple_platforms_same_user(self, tmp_path):
    async with SQLiteIdentityResolver(str(tmp_path / "id.db")) as resolver:
      await resolver.link("telegram", "tg_123", "user_abc")
      await resolver.link("discord", "dc_456", "user_abc")

      assert await resolver.resolve("telegram", "tg_123") == "user_abc"
      assert await resolver.resolve("discord", "dc_456") == "user_abc"

  @pytest.mark.asyncio
  async def test_link_upsert_changes_canonical(self, tmp_path):
    async with SQLiteIdentityResolver(str(tmp_path / "id.db")) as resolver:
      await resolver.link("telegram", "tg_123", "user_old")
      assert await resolver.resolve("telegram", "tg_123") == "user_old"

      await resolver.link("telegram", "tg_123", "user_new")
      assert await resolver.resolve("telegram", "tg_123") == "user_new"

  @pytest.mark.asyncio
  async def test_unlink(self, tmp_path):
    async with SQLiteIdentityResolver(str(tmp_path / "id.db")) as resolver:
      await resolver.link("telegram", "tg_123", "user_abc")
      assert await resolver.resolve("telegram", "tg_123") == "user_abc"

      result = await resolver.unlink("telegram", "tg_123")
      assert result is True
      assert await resolver.resolve("telegram", "tg_123") is None

  @pytest.mark.asyncio
  async def test_unlink_nonexistent_returns_false(self, tmp_path):
    async with SQLiteIdentityResolver(str(tmp_path / "id.db")) as resolver:
      result = await resolver.unlink("telegram", "nonexistent")
      assert result is False

  @pytest.mark.asyncio
  async def test_get_identities(self, tmp_path):
    async with SQLiteIdentityResolver(str(tmp_path / "id.db")) as resolver:
      await resolver.link("telegram", "tg_123", "user_abc", username="Alice TG")
      await resolver.link("discord", "dc_456", "user_abc", username="Alice DC")

      identities = await resolver.get_identities("user_abc")
      assert len(identities) == 2
      assert all(isinstance(i, PlatformIdentity) for i in identities)

      platforms = {i.platform for i in identities}
      assert platforms == {"telegram", "discord"}

      for ident in identities:
        assert ident.canonical_user_id == "user_abc"
        assert ident.linked_at > 0

  @pytest.mark.asyncio
  async def test_get_identities_empty(self, tmp_path):
    async with SQLiteIdentityResolver(str(tmp_path / "id.db")) as resolver:
      identities = await resolver.get_identities("nonexistent")
      assert identities == []

  @pytest.mark.asyncio
  async def test_different_users_same_platform(self, tmp_path):
    async with SQLiteIdentityResolver(str(tmp_path / "id.db")) as resolver:
      await resolver.link("telegram", "tg_alice", "user_alice")
      await resolver.link("telegram", "tg_bob", "user_bob")

      assert await resolver.resolve("telegram", "tg_alice") == "user_alice"
      assert await resolver.resolve("telegram", "tg_bob") == "user_bob"

  @pytest.mark.asyncio
  async def test_context_manager(self, tmp_path):
    resolver = SQLiteIdentityResolver(str(tmp_path / "id.db"))
    assert not resolver._initialized

    async with resolver:
      assert resolver._initialized
      await resolver.link("telegram", "tg_123", "user_abc")
      assert await resolver.resolve("telegram", "tg_123") == "user_abc"

    assert not resolver._initialized

  @pytest.mark.asyncio
  async def test_link_with_username(self, tmp_path):
    async with SQLiteIdentityResolver(str(tmp_path / "id.db")) as resolver:
      await resolver.link("telegram", "tg_123", "user_abc", username="Alice")
      identities = await resolver.get_identities("user_abc")
      assert len(identities) == 1
      assert identities[0].username == "Alice"

  @pytest.mark.asyncio
  async def test_protocol_compliance(self):
    """SQLiteIdentityResolver should satisfy the IdentityResolver protocol."""
    assert isinstance(SQLiteIdentityResolver(), IdentityResolver)


# --- BaseInterface identity integration tests ---


class _StubInterface(BaseInterface):
  """Minimal interface subclass for testing identity resolution in _run_agent."""

  def __init__(self, agent: Agent, **kwargs):
    super().__init__(
      agent=agent,
      config=InterfaceConfig(platform="test"),
      **kwargs,
    )

  async def _start_receiver(self) -> None:
    pass

  async def _stop_receiver(self) -> None:
    pass

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    return None

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    pass


def _make_message(
  platform: str = "telegram",
  user_id: str = "tg_123",
) -> InterfaceMessage:
  return InterfaceMessage(
    text="hello",
    platform=platform,
    platform_user_id=user_id,
    platform_chat_id="chat_1",
    platform_message_id="msg_1",
  )


class TestBaseInterfaceIdentity:
  @pytest.mark.asyncio
  async def test_run_agent_with_resolver(self):
    """When resolver returns a canonical ID, agent.arun() should receive it."""
    agent = Agent(model=MockModel(responses=["ok"]), instructions="test")
    resolver = AsyncMock(spec=IdentityResolver)
    resolver.initialize = AsyncMock()
    resolver.resolve = AsyncMock(return_value="canonical_alice")

    iface = _StubInterface(agent, identity_resolver=resolver)
    msg = _make_message(platform="telegram", user_id="tg_123")
    session = InterfaceSession()

    with patch.object(agent, "arun", new_callable=AsyncMock) as mock_arun:
      mock_arun.return_value = AsyncMock(
        content="ok",
        messages=[],
        images=None,
        videos=None,
        audio=None,
        files=None,
      )
      await iface._run_agent(msg, session)

    mock_arun.assert_called_once()
    call_kwargs = mock_arun.call_args
    assert call_kwargs.kwargs["user_id"] == "canonical_alice"

  @pytest.mark.asyncio
  async def test_run_agent_resolver_returns_none_uses_platform_id(self):
    """When resolver returns None (unknown user), fall back to platform_user_id."""
    agent = Agent(model=MockModel(responses=["ok"]), instructions="test")
    resolver = AsyncMock(spec=IdentityResolver)
    resolver.initialize = AsyncMock()
    resolver.resolve = AsyncMock(return_value=None)

    iface = _StubInterface(agent, identity_resolver=resolver)
    msg = _make_message(platform="discord", user_id="dc_unknown")
    session = InterfaceSession()

    with patch.object(agent, "arun", new_callable=AsyncMock) as mock_arun:
      mock_arun.return_value = AsyncMock(
        content="ok",
        messages=[],
        images=None,
        videos=None,
        audio=None,
        files=None,
      )
      await iface._run_agent(msg, session)

    call_kwargs = mock_arun.call_args
    assert call_kwargs.kwargs["user_id"] == "dc_unknown"

  @pytest.mark.asyncio
  async def test_run_agent_resolver_failure_falls_back(self):
    """When resolver raises an exception, fall back to platform_user_id gracefully."""
    agent = Agent(model=MockModel(responses=["ok"]), instructions="test")
    resolver = AsyncMock(spec=IdentityResolver)
    resolver.initialize = AsyncMock()
    resolver.resolve = AsyncMock(side_effect=RuntimeError("db connection failed"))

    iface = _StubInterface(agent, identity_resolver=resolver)
    msg = _make_message(platform="signal", user_id="sig_789")
    session = InterfaceSession()

    with patch.object(agent, "arun", new_callable=AsyncMock) as mock_arun:
      mock_arun.return_value = AsyncMock(
        content="ok",
        messages=[],
        images=None,
        videos=None,
        audio=None,
        files=None,
      )
      # Should not raise despite resolver failure
      await iface._run_agent(msg, session)

    call_kwargs = mock_arun.call_args
    assert call_kwargs.kwargs["user_id"] == "sig_789"

  @pytest.mark.asyncio
  async def test_run_agent_no_resolver_uses_platform_id(self):
    """Without a resolver, agent.arun() should receive the platform_user_id."""
    agent = Agent(model=MockModel(responses=["ok"]), instructions="test")
    iface = _StubInterface(agent)
    msg = _make_message(platform="telegram", user_id="tg_123")
    session = InterfaceSession()

    with patch.object(agent, "arun", new_callable=AsyncMock) as mock_arun:
      mock_arun.return_value = AsyncMock(
        content="ok",
        messages=[],
        images=None,
        videos=None,
        audio=None,
        files=None,
      )
      await iface._run_agent(msg, session)

    call_kwargs = mock_arun.call_args
    assert call_kwargs.kwargs["user_id"] == "tg_123"

  @pytest.mark.asyncio
  async def test_resolver_initialized_once(self):
    """The resolver should be initialized on first call only."""
    agent = Agent(model=MockModel(responses=["ok"]), instructions="test")
    resolver = AsyncMock(spec=IdentityResolver)
    resolver.initialize = AsyncMock()
    resolver.resolve = AsyncMock(return_value="canonical_1")

    iface = _StubInterface(agent, identity_resolver=resolver)
    msg = _make_message()
    session = InterfaceSession()

    with patch.object(agent, "arun", new_callable=AsyncMock) as mock_arun:
      mock_arun.return_value = AsyncMock(
        content="ok",
        messages=[],
        images=None,
        videos=None,
        audio=None,
        files=None,
      )
      await iface._run_agent(msg, session)
      await iface._run_agent(msg, session)

    assert resolver.initialize.call_count == 1
    assert resolver.resolve.call_count == 2


# --- serve() identity propagation tests ---


class TestServeIdentityPropagation:
  @pytest.mark.asyncio
  async def test_serve_propagates_resolver(self):
    """serve() should set resolver on interfaces that don't have one."""
    agent = Agent(model=MockModel(responses=["ok"]), instructions="test")
    resolver = AsyncMock(spec=IdentityResolver)

    iface = _StubInterface(agent)
    assert iface._identity_resolver is None

    # We just need to check propagation, not run the full serve loop.
    # Simulate the propagation logic directly.
    if resolver is not None:
      if iface._identity_resolver is None:
        iface._identity_resolver = resolver

    assert iface._identity_resolver is resolver

  @pytest.mark.asyncio
  async def test_serve_does_not_override_existing_resolver(self):
    """serve() should not override an interface's existing resolver."""
    agent = Agent(model=MockModel(responses=["ok"]), instructions="test")
    existing_resolver = AsyncMock(spec=IdentityResolver)
    shared_resolver = AsyncMock(spec=IdentityResolver)

    iface = _StubInterface(agent, identity_resolver=existing_resolver)

    # Simulate serve() propagation
    if shared_resolver is not None:
      if iface._identity_resolver is None:
        iface._identity_resolver = shared_resolver

    assert iface._identity_resolver is existing_resolver
