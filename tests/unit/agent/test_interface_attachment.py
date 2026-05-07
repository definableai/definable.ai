"""Unit tests for unified interface attachment — Agent(interfaces=, gateway=)."""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.config import InterfaceConfig
from definable.agent.interface.gateway import InterfaceGateway
from definable.agent.interface.message import InterfaceMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubInterface(BaseInterface):
  """Minimal stub for testing interface attachment."""

  def __init__(self, platform: str = "test") -> None:
    super().__init__(config=InterfaceConfig(platform=platform))

  def bind(self, agent: Any) -> "_StubInterface":  # type: ignore[override]
    self.agent = agent
    return self  # type: ignore[return-value]

  async def _start_receiver(self) -> None:
    pass

  async def _stop_receiver(self) -> None:
    pass

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    return None

  async def _send_response(self, original_msg: Any, response: Any, raw_message: Any) -> None:
    pass

  async def handle_platform_message(self, raw_message: Any) -> None:
    pass

  async def serve_forever(self) -> None:
    await asyncio.sleep(0.05)


def _make_model() -> MagicMock:
  """Create a minimal mock model for Agent construction."""
  model = MagicMock()
  model.id = "gpt-4o-mini"
  model.name = "gpt-4o-mini"
  model.provider = "OpenAI"
  model.metrics = {}
  model.provider_request_headers = None
  return model


def _make_agent(**kwargs: Any) -> Any:
  from definable.agent.agent import Agent

  return Agent(model=_make_model(), **kwargs)


# ===========================================================================
# 1. Constructor — interfaces=
# ===========================================================================


class TestInterfacesParam:
  def test_single_interface_binds(self) -> None:
    iface = _StubInterface("telegram")
    agent = _make_agent(interfaces=iface)
    assert iface.agent is agent
    assert iface in agent.interfaces
    assert len(agent.interfaces) == 1

  def test_list_of_interfaces_binds_all(self) -> None:
    tg = _StubInterface("telegram")
    dc = _StubInterface("discord")
    agent = _make_agent(interfaces=[tg, dc])
    assert tg.agent is agent
    assert dc.agent is agent
    assert len(agent.interfaces) == 2

  def test_none_interfaces_is_default(self) -> None:
    agent = _make_agent()
    assert agent.interfaces == []

  def test_empty_list_is_fine(self) -> None:
    agent = _make_agent(interfaces=[])
    assert agent.interfaces == []


# ===========================================================================
# 2. Constructor — gateway=
# ===========================================================================


class TestGatewayParam:
  def test_gateway_binds_to_agent(self) -> None:
    gw = InterfaceGateway()
    agent = _make_agent(gateway=gw)
    assert gw.agent is agent
    assert agent.gateway is gw

  def test_gateway_receives_constructor_interfaces(self) -> None:
    tg = _StubInterface("telegram")
    dc = _StubInterface("discord")
    gw = InterfaceGateway()
    agent = _make_agent(interfaces=[tg, dc], gateway=gw)
    assert tg in gw.interfaces
    assert dc in gw.interfaces
    assert agent.gateway is gw

  def test_gateway_with_shared_sessions(self) -> None:
    gw = InterfaceGateway(shared_sessions=True)
    tg = _StubInterface("telegram")
    agent = _make_agent(interfaces=tg, gateway=gw)
    assert gw._shared_session_manager is not None
    assert agent.gateway is gw


# ===========================================================================
# 3. interfaces property
# ===========================================================================


class TestInterfacesProperty:
  def test_returns_copy(self) -> None:
    tg = _StubInterface("telegram")
    agent = _make_agent(interfaces=tg)
    interfaces = agent.interfaces
    interfaces.append(_StubInterface("discord"))
    assert len(agent.interfaces) == 1  # original unchanged

  def test_reflects_constructor_interfaces(self) -> None:
    tg = _StubInterface("telegram")
    dc = _StubInterface("discord")
    agent = _make_agent(interfaces=[tg, dc])
    assert agent.interfaces == [tg, dc]


# ===========================================================================
# 4. Deprecation warnings
# ===========================================================================


# ===========================================================================
# 5. InterfaceGateway — deferred binding
# ===========================================================================


class TestGatewayDeferredBinding:
  def test_gateway_without_agent(self) -> None:
    gw = InterfaceGateway()
    assert gw.agent is None

  def test_bind_agent(self) -> None:
    gw = InterfaceGateway()
    agent = _make_agent()
    gw._bind_agent(agent)
    assert gw.agent is agent

  def test_bind_same_agent_is_ok(self) -> None:
    agent = _make_agent()
    gw = InterfaceGateway()
    gw._bind_agent(agent)
    gw._bind_agent(agent)  # should not raise
    assert gw.agent is agent

  def test_bind_different_agent_raises(self) -> None:
    agent1 = _make_agent()
    agent2 = _make_agent()
    gw = InterfaceGateway()
    gw._bind_agent(agent1)
    with pytest.raises(ValueError, match="already bound"):
      gw._bind_agent(agent2)

  @pytest.mark.asyncio
  async def test_aserve_without_agent_raises(self) -> None:
    gw = InterfaceGateway()
    iface = _StubInterface("test")
    gw.add(iface)
    with pytest.raises(ValueError, match="no agent"):
      await gw.aserve()


# ===========================================================================
# 6. Auto-gateway for 2+ interfaces
# ===========================================================================


class TestAutoGateway:
  @pytest.mark.asyncio
  async def test_auto_gateway_for_two_interfaces(self) -> None:
    tg = _StubInterface("telegram")
    dc = _StubInterface("discord")
    agent = _make_agent(interfaces=[tg, dc])

    # Before serve, no gateway
    assert agent.gateway is None

    # aserve should auto-create a gateway
    with patch("definable.runtime.runner.AgentRuntime.start", new_callable=AsyncMock) as mock_start:
      await agent.aserve()

    # Verify runtime was created (gateway is internal to aserve)
    mock_start.assert_called_once()

  @pytest.mark.asyncio
  async def test_no_auto_gateway_for_single_interface(self) -> None:
    tg = _StubInterface("telegram")
    agent = _make_agent(interfaces=tg)

    with patch("definable.runtime.runner.AgentRuntime.start", new_callable=AsyncMock) as mock_start:
      await agent.aserve()

    mock_start.assert_called_once()
    # Agent's own gateway should still be None
    assert agent.gateway is None

  @pytest.mark.asyncio
  async def test_explicit_gateway_prevents_auto_creation(self) -> None:
    tg = _StubInterface("telegram")
    dc = _StubInterface("discord")
    gw = InterfaceGateway(shared_sessions=True)
    agent = _make_agent(interfaces=[tg, dc], gateway=gw)

    with patch("definable.runtime.runner.AgentRuntime.start", new_callable=AsyncMock):
      await agent.aserve()

    assert agent.gateway is gw


# ===========================================================================
# 7. Backward compatibility
# ===========================================================================


class TestBackwardCompat:
  def test_gateway_with_agent_in_constructor_still_works(self) -> None:
    """Old pattern: InterfaceGateway(agent) still works."""
    agent = _make_agent()
    gw = InterfaceGateway(agent)
    assert gw.agent is agent
