"""TurnSnapshot + RunResult — dataclass sanity tests."""

from __future__ import annotations

import dataclasses

import pytest

from definable.agent.core.debug import TurnSnapshot
from definable.agent.core.result import RunResult


@pytest.mark.unit
def test_turn_snapshot_constructs_with_required_fields() -> None:
  snap = TurnSnapshot(turn=3, context_tokens=1024, messages=7)
  assert snap.turn == 3
  assert snap.context_tokens == 1024
  assert snap.messages == 7
  assert snap.available_tools == []
  assert snap.memory_files == []
  assert snap.last_decision is None
  assert snap.reasoning_text is None


@pytest.mark.unit
def test_turn_snapshot_is_frozen() -> None:
  snap = TurnSnapshot(turn=1, context_tokens=0, messages=0)
  with pytest.raises(dataclasses.FrozenInstanceError):
    snap.turn = 99  # type: ignore[misc]


@pytest.mark.unit
def test_run_result_defaults() -> None:
  result = RunResult(content="hello")
  assert result.content == "hello"
  assert result.parsed is None
  assert result.messages == []
  assert result.turns == 0
  assert result.exit_reason == "natural"


@pytest.mark.unit
def test_run_result_max_turns_exit() -> None:
  result = RunResult(content=None, turns=50, exit_reason="max_turns")
  assert result.content is None
  assert result.turns == 50
  assert result.exit_reason == "max_turns"


@pytest.mark.unit
def test_run_result_is_frozen() -> None:
  result = RunResult(content="x")
  with pytest.raises(dataclasses.FrozenInstanceError):
    result.content = "y"  # type: ignore[misc]
