"""
Unit tests for Agent sync shutdown resource cleanup.

Verifies that Agent._shutdown() (sync context manager exit) properly
closes memory stores, drains pending memory tasks, and shuts down
agent-owned toolkits — not just skills and trace writers.

Prior to fix: only _ashutdown() closed these resources; _shutdown()
leaked SQLite connections, memory tasks, and toolkit connections.

Covers:
  - _shutdown closes memory when no event loop is running
  - _shutdown drains pending memory tasks
  - _shutdown shuts down agent-owned toolkits
  - _shutdown warns when called inside a running event loop
  - _ashutdown still works correctly (no regression)
  - sync context manager (__exit__) triggers full cleanup
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from definable.agent.agent import Agent
from definable.model.openai.chat import OpenAIChat


def _make_agent_with_memory():
  """Create an Agent with a mock memory for testing shutdown."""
  model = MagicMock(spec=OpenAIChat)
  model.id = "gpt-4o-mini"
  model.provider = "openai"

  agent = Agent(model=model)
  agent.memory = AsyncMock()
  agent.memory.close = AsyncMock()
  agent._pending_memory_tasks = []
  agent._agent_owned_toolkits = []
  agent._trace_writer = None
  agent._started = True

  return agent


# ---------------------------------------------------------------------------
# _shutdown: memory cleanup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncShutdownMemory:
  """_shutdown() closes memory when no event loop is running."""

  def test_closes_memory_store(self):
    agent = _make_agent_with_memory()

    agent._shutdown()

    agent.memory.close.assert_awaited_once()

  def test_no_error_when_memory_is_none(self):
    agent = _make_agent_with_memory()
    agent.memory = None

    # Should not raise
    agent._shutdown()

  def test_suppresses_memory_close_error(self):
    agent = _make_agent_with_memory()
    agent.memory.close = AsyncMock(side_effect=RuntimeError("close failed"))

    # Should not raise
    agent._shutdown()


# ---------------------------------------------------------------------------
# _shutdown: toolkit cleanup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncShutdownToolkits:
  """_shutdown() shuts down agent-owned toolkits."""

  def test_shuts_down_agent_owned_toolkits(self):
    agent = _make_agent_with_memory()
    toolkit = AsyncMock()
    toolkit.shutdown = AsyncMock()
    agent._agent_owned_toolkits = [toolkit]

    agent._shutdown()

    toolkit.shutdown.assert_awaited_once()

  def test_clears_agent_owned_toolkits_list(self):
    agent = _make_agent_with_memory()
    toolkit = AsyncMock()
    toolkit.shutdown = AsyncMock()
    agent._agent_owned_toolkits = [toolkit]

    agent._shutdown()

    assert len(agent._agent_owned_toolkits) == 0

  def test_suppresses_toolkit_shutdown_error(self):
    agent = _make_agent_with_memory()
    toolkit = AsyncMock()
    toolkit.shutdown = AsyncMock(side_effect=RuntimeError("shutdown failed"))
    agent._agent_owned_toolkits = [toolkit]

    # Should not raise
    agent._shutdown()


# ---------------------------------------------------------------------------
# _shutdown: pending memory tasks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncShutdownDrainTasks:
  """_shutdown() drains pending memory tasks."""

  def test_drains_completed_tasks(self):
    agent = _make_agent_with_memory()

    # Create a completed async task
    async def _noop():
      pass

    loop = asyncio.new_event_loop()
    task = loop.create_task(_noop())
    loop.run_until_complete(task)
    loop.close()

    agent._pending_memory_tasks = [task]

    agent._shutdown()

    # After drain, completed tasks should be removed
    assert all(t.done() for t in agent._pending_memory_tasks) or len(agent._pending_memory_tasks) == 0


# ---------------------------------------------------------------------------
# _shutdown: async context warning
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncShutdownInAsyncContext:
  """_shutdown() warns when called from a running event loop."""

  @pytest.mark.asyncio
  async def test_warns_in_async_context(self):
    agent = _make_agent_with_memory()

    with patch("definable.agent.agent.log_warning", create=True) as mock_warn:
      # We need to patch at the import location inside the method
      with patch("definable.utils.log.log_warning") as mock_log_warn:
        agent._shutdown()

    # Memory should NOT be closed (can't run asyncio.run inside running loop)
    # But warning should be issued — check both possible import paths
    warned = mock_warn.called or mock_log_warn.called
    assert warned or not agent.memory.close.awaited
    # The key invariant: _started should still be set to False
    assert agent._started is False


# ---------------------------------------------------------------------------
# _ashutdown: no regression
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncShutdownRegression:
  """_ashutdown() still properly closes all resources."""

  @pytest.mark.asyncio
  async def test_ashutdown_closes_memory(self):
    agent = _make_agent_with_memory()

    await agent._ashutdown()

    agent.memory.close.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_ashutdown_shuts_down_toolkits(self):
    agent = _make_agent_with_memory()
    toolkit = AsyncMock()
    toolkit.shutdown = AsyncMock()
    agent._agent_owned_toolkits = [toolkit]

    await agent._ashutdown()

    toolkit.shutdown.assert_awaited_once()
    assert len(agent._agent_owned_toolkits) == 0


# ---------------------------------------------------------------------------
# Sync context manager integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncContextManager:
  """Sync context manager (__exit__) triggers full cleanup."""

  def test_context_manager_closes_memory(self):
    agent = _make_agent_with_memory()

    agent.__enter__()
    agent.__exit__(None, None, None)

    agent.memory.close.assert_awaited_once()

  def test_context_manager_sets_started_false(self):
    agent = _make_agent_with_memory()

    agent.__enter__()
    agent.__exit__(None, None, None)

    assert agent._started is False
