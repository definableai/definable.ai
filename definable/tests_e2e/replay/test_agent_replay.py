"""Integration tests for agent.replay() and agent.compare()."""

import pytest

from definable.agents import Agent
from definable.agents.testing import MockModel
from definable.models.metrics import Metrics
from definable.models.response import ToolExecution
from definable.replay import Replay, ReplayComparison
from definable.run.agent import RunInput, RunOutput
from definable.run.base import RunStatus


def _make_mock_agent(content: str = "Hello!", **kwargs) -> Agent:
  return Agent(
    model=MockModel(responses=[content], **kwargs),
    name="test-agent",
  )


class TestAgentReplayFromRunOutput:
  def test_returns_replay(self):
    output = RunOutput(
      run_id="run-1",
      session_id="sess-1",
      agent_id="agent-1",
      agent_name="TestAgent",
      model="mock",
      model_provider="mock",
      content="Hello!",
      status=RunStatus.completed,
      metrics=Metrics(input_tokens=10, output_tokens=5, total_tokens=15),
      input=RunInput(input_content="Hi"),
    )
    agent = _make_mock_agent()
    result = agent.replay(run_output=output)

    assert isinstance(result, Replay)
    assert result.run_id == "run-1"
    assert result.content == "Hello!"
    assert result.tokens.total_tokens == 15
    assert result.source == "run_output"

  def test_with_tool_executions(self):
    output = RunOutput(
      run_id="run-1",
      content="done",
      status=RunStatus.completed,
      tools=[
        ToolExecution(
          tool_call_id="tc-1",
          tool_name="search",
          tool_args={"q": "test"},
          result="found",
        ),
      ],
    )
    agent = _make_mock_agent()
    result = agent.replay(run_output=output)

    assert isinstance(result, Replay)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "search"


class TestAgentReplayFromEvents:
  def test_from_events(self):
    from definable.run.agent import RunCompletedEvent, RunStartedEvent

    events = [
      RunStartedEvent(
        created_at=1000,
        run_id="run-1",
        session_id="sess-1",
        agent_id="agent-1",
        agent_name="TestAgent",
        model="mock",
        model_provider="mock",
        run_input=RunInput(input_content="Hi"),
      ),
      RunCompletedEvent(
        created_at=1001,
        run_id="run-1",
        session_id="sess-1",
        agent_id="agent-1",
        agent_name="TestAgent",
        content="Hello!",
        metrics=Metrics(input_tokens=10, output_tokens=5, total_tokens=15),
      ),
    ]

    agent = _make_mock_agent()
    result = agent.replay(events=events)

    assert isinstance(result, Replay)
    assert result.run_id == "run-1"
    assert result.content == "Hello!"

  def test_from_trace_file(self, tmp_path):
    from definable.run.agent import RunCompletedEvent, RunStartedEvent

    events = [
      RunStartedEvent(
        created_at=1000,
        run_id="run-1",
        session_id="sess-1",
        agent_id="agent-1",
        agent_name="TestAgent",
        model="mock",
        model_provider="mock",
        run_input=RunInput(input_content="Hi"),
      ),
      RunCompletedEvent(
        created_at=1001,
        run_id="run-1",
        session_id="sess-1",
        agent_id="agent-1",
        agent_name="TestAgent",
        content="Hello!",
        metrics=Metrics(input_tokens=10, output_tokens=5, total_tokens=15),
      ),
    ]
    jsonl_path = tmp_path / "test.jsonl"
    with open(jsonl_path, "w") as f:
      for evt in events:
        f.write(evt.to_json(indent=None) + "\n")

    agent = _make_mock_agent()
    result = agent.replay(trace_file=str(jsonl_path))

    assert isinstance(result, Replay)
    assert result.run_id == "run-1"


class TestAgentReplayErrors:
  def test_no_source_raises(self):
    agent = _make_mock_agent()
    with pytest.raises(ValueError, match="Provide one of"):
      agent.replay()


class TestAgentCompare:
  def test_compare_two_outputs(self):
    a = RunOutput(
      run_id="run-1",
      content="Hello world",
      status=RunStatus.completed,
      metrics=Metrics(input_tokens=10, output_tokens=5, total_tokens=15, cost=0.01),
    )
    b = RunOutput(
      run_id="run-2",
      content="Hello universe",
      status=RunStatus.completed,
      metrics=Metrics(input_tokens=12, output_tokens=8, total_tokens=20, cost=0.02),
    )

    agent = _make_mock_agent()
    result = agent.compare(a, b)

    assert isinstance(result, ReplayComparison)
    assert result.content_diff is not None
    assert result.token_diff == 5
    assert result.cost_diff is not None
    assert abs(result.cost_diff - 0.01) < 1e-9

  def test_compare_replay_and_output(self):
    replay = Replay(
      run_id="run-1",
      content="Hello",
      tokens=Replay.__dataclass_fields__["tokens"].default_factory(),
    )
    output = RunOutput(
      run_id="run-2",
      content="World",
      status=RunStatus.completed,
    )

    agent = _make_mock_agent()
    result = agent.compare(replay, output)

    assert isinstance(result, ReplayComparison)
    assert result.content_diff is not None
