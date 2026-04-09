"""Integration tests: HITL permission service with a real Agent + MockModel.

Verifies the full flow: model calls tool → permission check → allow/deny → result.
"""

import pytest
from unittest.mock import MagicMock

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.events import RunStatus
from definable.agent.hitl.settings import Settings
from definable.agent.hitl.types import (
  PermissionAction,
  PermissionDecision,
  PermissionRequest,
  PermissionResponse,
)
from definable.agent.testing import MockModel
from definable.agent.tracing import Tracing
from definable.model.metrics import Metrics
from definable.tool.decorator import tool


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


@tool
def safe_read() -> str:
  """Read a file."""
  return "file contents"


@tool
def dangerous_delete(target: str) -> str:
  """Delete something permanently."""
  return f"deleted {target}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


NO_TRACE = AgentConfig(tracing=Tracing(enabled=False))


def _make_tool_call_side_effect(tool_name: str, tool_args: str = "{}"):
  call_count = 0

  def side_effect(messages, tools, **kwargs):
    nonlocal call_count
    call_count += 1
    response = MagicMock()
    response.response_usage = Metrics()
    response.reasoning_content = None
    response.citations = None
    response.images = None
    response.videos = None
    response.audios = None
    if call_count == 1:
      response.content = ""
      response.tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": tool_name, "arguments": tool_args}},
      ]
    else:
      response.content = "Done."
      response.tool_calls = []
    return response

  return side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
class TestHITLPermissions:
  """Permission service integration with Agent."""

  async def test_allow_once_executes_tool(self):
    """When resolver returns allow_once, tool executes normally."""

    async def resolver(request: PermissionRequest) -> PermissionResponse:
      return PermissionResponse(decision=PermissionDecision.allow_once)

    model = MockModel(side_effect=_make_tool_call_side_effect("dangerous_delete", '{"target": "tmp"}'))
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[dangerous_delete],
      config=NO_TRACE,
      permission_resolver=resolver,
    )

    output = await agent.arun("Delete tmp")
    assert output.status == RunStatus.completed
    assert output.content is not None

  async def test_deny_sends_message_to_model(self):
    """When resolver denies, denial message goes to model as tool result."""

    async def resolver(request: PermissionRequest) -> PermissionResponse:
      return PermissionResponse(decision=PermissionDecision.deny, feedback="Not allowed")

    model = MockModel(side_effect=_make_tool_call_side_effect("dangerous_delete", '{"target": "prod"}'))
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[dangerous_delete],
      config=NO_TRACE,
      permission_resolver=resolver,
    )

    output = await agent.arun("Delete prod")
    # Should complete (model gets denial and responds)
    assert output.status == RunStatus.completed

  async def test_config_default_allow_skips_resolver(self):
    """Tool with 'allow' default should never trigger the resolver."""

    async def should_not_be_called(request):
      raise AssertionError("resolver should not be called for safe_read")

    model = MockModel(side_effect=_make_tool_call_side_effect("safe_read"))
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[safe_read],
      config=NO_TRACE,
      permission_resolver=should_not_be_called,
      permission_defaults={"safe_read": PermissionAction.allow},
    )

    output = await agent.arun("Read the file")
    assert output.status == RunStatus.completed

  async def test_headless_mode_no_resolver(self):
    """Agent without resolver should auto-allow all tools."""
    model = MockModel(side_effect=_make_tool_call_side_effect("dangerous_delete", '{"target": "x"}'))
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[dangerous_delete],
      config=NO_TRACE,
      permission_defaults={"dangerous_delete": PermissionAction.ask},
    )

    output = await agent.arun("Delete x")
    assert output.status == RunStatus.completed

  async def test_always_allow_persists_and_skips_future(self):
    """'Always allow' should persist and skip resolver on subsequent calls."""
    call_count = 0

    async def resolver(request: PermissionRequest) -> PermissionResponse:
      nonlocal call_count
      call_count += 1
      return PermissionResponse(decision=PermissionDecision.allow_always)

    settings = Settings()
    from definable.agent.hitl.permissions import PermissionService

    perm_service = PermissionService(resolver=resolver, settings=settings)

    model = MockModel(side_effect=_make_tool_call_side_effect("dangerous_delete", '{"target": "a"}'))
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[dangerous_delete],
      config=NO_TRACE,
    )
    # Inject permission service directly for this test
    agent._permission_service = perm_service

    # First run: resolver should be called
    await agent.arun("Delete a")
    assert call_count == 1

    # Second run: settings say "allow", resolver should NOT be called
    model2 = MockModel(side_effect=_make_tool_call_side_effect("dangerous_delete", '{"target": "b"}'))
    agent2 = Agent(model=model2, tools=[dangerous_delete], config=NO_TRACE)
    agent2._permission_service = perm_service

    await agent2.arun("Delete b")
    assert call_count == 1  # Still 1 — resolver was not called
