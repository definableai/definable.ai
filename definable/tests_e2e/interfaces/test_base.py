"""Tests for base interface types: InterfaceConfig, InterfaceMessage, InterfaceResponse, SessionManager, hooks, errors."""

import time
from dataclasses import FrozenInstanceError

import pytest

from definable.interfaces.config import InterfaceConfig
from definable.interfaces.errors import (
  InterfaceAuthenticationError,
  InterfaceConnectionError,
  InterfaceError,
  InterfaceMessageError,
  InterfaceRateLimitError,
)
from definable.interfaces.hooks import AllowlistHook, LoggingHook
from definable.interfaces.message import InterfaceMessage, InterfaceResponse
from definable.interfaces.session import InterfaceSession, SessionManager
from definable.media import Image


# --- InterfaceConfig tests ---


class TestInterfaceConfig:
  def test_default_config(self):
    config = InterfaceConfig()
    assert config.platform == ""
    assert config.max_session_history == 50
    assert config.session_ttl_seconds == 3600
    assert config.max_concurrent_requests == 10
    assert config.typing_indicator is True
    assert config.max_message_length == 4096
    assert config.rate_limit_messages_per_minute == 30

  def test_custom_config(self):
    config = InterfaceConfig(platform="test", max_session_history=100)
    assert config.platform == "test"
    assert config.max_session_history == 100

  def test_frozen_immutability(self):
    config = InterfaceConfig()
    with pytest.raises(FrozenInstanceError):
      config.platform = "changed"  # type: ignore[misc]

  def test_with_updates(self):
    config = InterfaceConfig(platform="test", max_session_history=50)
    updated = config.with_updates(max_session_history=100, typing_indicator=False)
    assert updated.max_session_history == 100
    assert updated.typing_indicator is False
    # Original unchanged
    assert config.max_session_history == 50
    assert config.typing_indicator is True


# --- InterfaceMessage tests ---


class TestInterfaceMessage:
  def test_basic_message(self):
    msg = InterfaceMessage(
      text="Hello",
      platform="test",
      platform_user_id="u1",
      platform_chat_id="c1",
      platform_message_id="m1",
    )
    assert msg.text == "Hello"
    assert msg.platform == "test"
    assert msg.platform_user_id == "u1"
    assert msg.images is None
    assert msg.audio is None
    assert msg.files is None

  def test_message_with_media(self):
    img = Image(url="https://example.com/image.png")
    msg = InterfaceMessage(
      text="Look at this",
      platform="test",
      platform_user_id="u1",
      platform_chat_id="c1",
      platform_message_id="m1",
      images=[img],
    )
    assert msg.images is not None
    assert len(msg.images) == 1
    assert msg.images[0].url == "https://example.com/image.png"

  def test_message_with_reply(self):
    msg = InterfaceMessage(
      platform="test",
      platform_user_id="u1",
      platform_chat_id="c1",
      platform_message_id="m1",
      reply_to_message_id="m0",
    )
    assert msg.reply_to_message_id == "m0"

  def test_message_created_at(self):
    before = time.time()
    msg = InterfaceMessage(
      platform="test",
      platform_user_id="u1",
      platform_chat_id="c1",
      platform_message_id="m1",
    )
    after = time.time()
    assert before <= msg.created_at <= after


# --- InterfaceResponse tests ---


class TestInterfaceResponse:
  def test_basic_response(self):
    resp = InterfaceResponse(content="Hello back")
    assert resp.content == "Hello back"
    assert resp.images is None
    assert resp.files is None

  def test_response_with_media(self):
    img = Image(url="https://example.com/reply.png")
    resp = InterfaceResponse(content="Here you go", images=[img])
    assert resp.images is not None
    assert len(resp.images) == 1


# --- SessionManager tests ---


class TestSessionManager:
  def test_get_or_create_new_session(self):
    mgr = SessionManager(session_ttl_seconds=3600)
    session = mgr.get_or_create("test", "u1", "c1")
    assert isinstance(session, InterfaceSession)
    assert session.platform == "test"
    assert session.platform_user_id == "u1"
    assert session.platform_chat_id == "c1"

  def test_get_or_create_returns_same_session(self):
    mgr = SessionManager(session_ttl_seconds=3600)
    s1 = mgr.get_or_create("test", "u1", "c1")
    s2 = mgr.get_or_create("test", "u1", "c1")
    assert s1.session_id == s2.session_id

  def test_get_or_create_different_keys(self):
    mgr = SessionManager(session_ttl_seconds=3600)
    s1 = mgr.get_or_create("test", "u1", "c1")
    s2 = mgr.get_or_create("test", "u2", "c1")
    assert s1.session_id != s2.session_id

  def test_get_existing(self):
    mgr = SessionManager(session_ttl_seconds=3600)
    created = mgr.get_or_create("test", "u1", "c1")
    found = mgr.get("test", "u1", "c1")
    assert found is not None
    assert found.session_id == created.session_id

  def test_get_nonexistent(self):
    mgr = SessionManager(session_ttl_seconds=3600)
    assert mgr.get("test", "u1", "c1") is None

  def test_remove_session(self):
    mgr = SessionManager(session_ttl_seconds=3600)
    mgr.get_or_create("test", "u1", "c1")
    assert mgr.remove("test", "u1", "c1") is True
    assert mgr.get("test", "u1", "c1") is None

  def test_remove_nonexistent(self):
    mgr = SessionManager(session_ttl_seconds=3600)
    assert mgr.remove("test", "u1", "c1") is False

  def test_ttl_expiration(self):
    mgr = SessionManager(session_ttl_seconds=0)  # Immediate expiry
    session = mgr.get_or_create("test", "u1", "c1")
    sid = session.session_id
    time.sleep(0.01)
    # Getting should create a new session since old one expired
    new_session = mgr.get_or_create("test", "u1", "c1")
    assert new_session.session_id != sid

  def test_cleanup_expired(self):
    mgr = SessionManager(session_ttl_seconds=0)
    mgr.get_or_create("test", "u1", "c1")
    mgr.get_or_create("test", "u2", "c2")
    time.sleep(0.01)
    removed = mgr.cleanup_expired()
    assert removed == 2

  def test_truncate_history(self):
    session = InterfaceSession()
    from definable.models.message import Message

    session.messages = [Message(role="user", content=f"msg{i}") for i in range(10)]
    session.truncate_history(5)
    assert session.messages is not None
    assert len(session.messages) == 5

  def test_active_session_count(self):
    mgr = SessionManager(session_ttl_seconds=3600)
    mgr.get_or_create("test", "u1", "c1")
    mgr.get_or_create("test", "u2", "c2")
    assert mgr.active_session_count == 2


# --- Hook tests ---


class TestLoggingHook:
  @pytest.mark.asyncio
  async def test_on_message_received(self, sample_interface_message):
    hook = LoggingHook()
    msg = sample_interface_message(platform="test", text="Hi")
    result = await hook.on_message_received(msg)
    assert result is None  # LoggingHook doesn't veto

  @pytest.mark.asyncio
  async def test_on_error(self, sample_interface_message):
    hook = LoggingHook()
    msg = sample_interface_message()
    # Should not raise
    await hook.on_error(ValueError("test error"), msg)

  @pytest.mark.asyncio
  async def test_on_error_no_message(self):
    hook = LoggingHook()
    await hook.on_error(ValueError("test error"), None)


class TestAllowlistHook:
  @pytest.mark.asyncio
  async def test_allowed_user(self, sample_interface_message):
    hook = AllowlistHook(allowed_user_ids={"user123"})
    msg = sample_interface_message(user_id="user123")
    result = await hook.on_message_received(msg)
    assert result is None  # Allowed

  @pytest.mark.asyncio
  async def test_blocked_user(self, sample_interface_message):
    hook = AllowlistHook(allowed_user_ids={"user123"})
    msg = sample_interface_message(user_id="user999")
    result = await hook.on_message_received(msg)
    assert result is False  # Blocked


# --- Error tests ---


class TestErrors:
  def test_interface_error(self):
    err = InterfaceError("test", platform="telegram")
    assert str(err) == "test"
    assert err.platform == "telegram"
    assert err.status_code == 500

  def test_connection_error(self):
    err = InterfaceConnectionError("conn failed", platform="discord")
    assert err.status_code == 503
    assert err.platform == "discord"

  def test_authentication_error(self):
    err = InterfaceAuthenticationError("bad token", platform="signal")
    assert err.status_code == 401

  def test_rate_limit_error(self):
    err = InterfaceRateLimitError("slow down", platform="telegram", retry_after=5.0)
    assert err.status_code == 429
    assert err.retry_after == 5.0

  def test_message_error(self):
    err = InterfaceMessageError("bad msg", platform="telegram")
    assert err.status_code == 400

  def test_error_hierarchy(self):
    assert issubclass(InterfaceConnectionError, InterfaceError)
    assert issubclass(InterfaceAuthenticationError, InterfaceError)
    assert issubclass(InterfaceRateLimitError, InterfaceError)
    assert issubclass(InterfaceMessageError, InterfaceError)
