"""Tests for WebSocketInterface — config, lifecycle, conversion."""

import pytest

from definable.agent.interface.websocket.config import WebSocketConfig
from definable.agent.interface.websocket.interface import WebSocketInterface


class TestWebSocketConfig:
  def test_defaults(self):
    config = WebSocketConfig()
    assert config.platform == "websocket"
    assert config.path == "/ws"
    assert config.heartbeat_interval == 30.0
    assert config.max_connections == 100
    assert config.auth_on_connect is True
    assert config.message_format == "json"
    assert config.max_message_length == 65536

  def test_custom(self):
    config = WebSocketConfig(path="/agent/ws", heartbeat_interval=10.0, max_connections=50)
    assert config.path == "/agent/ws"
    assert config.heartbeat_interval == 10.0
    assert config.max_connections == 50


class TestWebSocketInterface:
  def test_init_defaults(self):
    iface = WebSocketInterface()
    assert iface.config.platform == "websocket"
    assert iface._ws_config.path == "/ws"
    assert iface.active_connections == 0
    assert iface.needs_server() is True

  def test_init_custom_path(self):
    iface = WebSocketInterface(path="/chat/ws")
    assert iface._ws_config.path == "/chat/ws"

  def test_init_heartbeat(self):
    iface = WebSocketInterface(heartbeat_interval=5.0)
    assert iface._ws_config.heartbeat_interval == 5.0

  def test_init_max_connections(self):
    iface = WebSocketInterface(max_connections=50)
    assert iface._ws_config.max_connections == 50

  def test_deprecated_config_param(self):
    config = WebSocketConfig(path="/old")
    with pytest.warns(DeprecationWarning):
      iface = WebSocketInterface(config=config)
    assert iface._ws_config.path == "/old"

  def test_create_router(self):
    """create_router() returns a FastAPI APIRouter."""
    iface = WebSocketInterface()
    router = iface.create_router()
    # APIRouter has routes attribute
    assert hasattr(router, "routes")

  @pytest.mark.asyncio
  async def test_convert_inbound_valid(self):
    iface = WebSocketInterface()
    raw = {
      "conn_id": "abc",
      "websocket": None,
      "data": {
        "type": "message",
        "text": "Hello",
        "session_id": "s1",
        "user_id": "u1",
      },
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text == "Hello"
    assert msg.platform == "websocket"
    assert msg.platform_user_id == "u1"
    assert msg.platform_chat_id == "s1"

  @pytest.mark.asyncio
  async def test_convert_inbound_empty_text(self):
    iface = WebSocketInterface()
    raw = {
      "conn_id": "abc",
      "websocket": None,
      "data": {"text": ""},
    }
    msg = await iface._convert_inbound(raw)
    assert msg is None

  @pytest.mark.asyncio
  async def test_convert_inbound_defaults_to_conn_id(self):
    """If no user_id/session_id provided, defaults to conn_id."""
    iface = WebSocketInterface()
    raw = {
      "conn_id": "xyz",
      "websocket": None,
      "data": {"text": "Hi"},
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.platform_user_id == "xyz"
    assert msg.platform_chat_id == "xyz"


class TestWebSocketExports:
  def test_import_from_websocket_package(self):
    from definable.agent.interface.websocket import WebSocketConfig, WebSocketInterface

    assert WebSocketConfig is not None
    assert WebSocketInterface is not None

  def test_import_from_interface_package(self):
    from definable.agent.interface import WebSocketConfig, WebSocketInterface

    assert WebSocketConfig is not None
    assert WebSocketInterface is not None
