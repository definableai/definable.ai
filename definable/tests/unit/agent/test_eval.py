"""Tests for definable.agent.eval — evaluation module."""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.agent.eval.result import (
  AccuracyResult,
  EvalResult,
  JudgeResult,
  PerformanceResult,
  ReliabilityResult,
)
from definable.agent.eval.base import BaseEval, EvalCase, EvalSuite
from definable.agent.eval.accuracy import AccuracyEval, ACCURACY_JUDGE_PROMPT
from definable.agent.eval.performance import PerformanceEval
from definable.agent.eval.reliability import ReliabilityEval
from definable.agent.eval.judge import AgentAsJudgeEval, NUMERIC_JUDGE_PROMPT, BINARY_JUDGE_PROMPT


# ── Helpers ──────────────────────────────────────────────────


def _make_mock_agent(content: str = "Hello world", tool_executions: Optional[list] = None):
  """Create a mock agent with a configurable arun() response."""
  agent = MagicMock()
  output = MagicMock()
  output.content = content
  output.tool_executions = tool_executions or []
  output.messages = []
  agent.arun = AsyncMock(return_value=output)
  return agent


def _make_mock_team(content: str = "Team response"):
  """Create a mock team with a configurable arun() response."""
  team = MagicMock()
  output = MagicMock()
  output.content = content
  output.tool_executions = []
  output.messages = []
  team.arun = AsyncMock(return_value=output)
  return team


def _make_mock_judge_model(response_content: str = '{"score": 8, "reason": "Good"}'):
  """Create a mock model for judge evals."""
  model = MagicMock()
  response = MagicMock()
  response.content = response_content
  model.aresponse = AsyncMock(return_value=response)
  return model


# ═════════════════════════════════════════════════════════════
# EvalResult types
# ═════════════════════════════════════════════════════════════


class TestEvalResult:
  def test_base_result(self):
    r = EvalResult(eval_name="test", success=True, score=8.5, reason="Good")
    assert r.eval_name == "test"
    assert r.success is True
    assert r.score == 8.5
    assert r.reason == "Good"

  def test_base_result_to_dict(self):
    r = EvalResult(eval_name="test", success=True, score=8.0)
    d = r.to_dict()
    assert d["eval_name"] == "test"
    assert d["success"] is True
    assert d["score"] == 8.0
    assert "reason" not in d  # None values excluded

  def test_base_result_defaults(self):
    r = EvalResult()
    assert r.eval_name == ""
    assert r.success is False
    assert r.score is None
    assert r.reason is None
    assert r.metadata == {}


class TestAccuracyResult:
  def test_fields(self):
    r = AccuracyResult(
      eval_name="accuracy",
      score=9.0,
      success=True,
      threshold=7.0,
      expected="4",
      actual="The answer is 4",
    )
    assert r.threshold == 7.0
    assert r.expected == "4"
    assert r.actual == "The answer is 4"

  def test_to_dict(self):
    r = AccuracyResult(eval_name="accuracy", success=True, score=8.0, threshold=7.0, expected="foo", actual="bar")
    d = r.to_dict()
    assert d["threshold"] == 7.0
    assert d["expected"] == "foo"
    assert d["actual"] == "bar"

  def test_defaults(self):
    r = AccuracyResult()
    assert r.threshold == 7.0
    assert r.expected is None
    assert r.actual is None


class TestPerformanceResult:
  def test_fields(self):
    r = PerformanceResult(
      eval_name="perf",
      success=True,
      duration_ms=1500.0,
      peak_memory_mb=25.5,
      runs=3,
      durations=[1400, 1500, 1600],
    )
    assert r.duration_ms == 1500.0
    assert r.peak_memory_mb == 25.5
    assert r.runs == 3

  def test_to_dict(self):
    r = PerformanceResult(
      eval_name="perf",
      success=True,
      duration_ms=1000.0,
      peak_memory_mb=10.0,
      duration_threshold_ms=5000.0,
      runs=2,
      durations=[900, 1000],
    )
    d = r.to_dict()
    assert d["duration_ms"] == 1000.0
    assert d["peak_memory_mb"] == 10.0
    assert d["duration_threshold_ms"] == 5000.0
    assert d["durations"] == [900, 1000]


class TestReliabilityResult:
  def test_fields(self):
    r = ReliabilityResult(
      eval_name="reliability",
      success=True,
      expected_tools=["search_web"],
      actual_tools=["search_web", "summarize"],
      missing_tools=[],
      extra_tools=["summarize"],
      strict=False,
    )
    assert r.expected_tools == ["search_web"]
    assert r.extra_tools == ["summarize"]

  def test_to_dict(self):
    r = ReliabilityResult(eval_name="rel", success=True, strict=True)
    d = r.to_dict()
    assert d["strict"] is True


class TestJudgeResult:
  def test_fields(self):
    r = JudgeResult(
      eval_name="judge",
      success=True,
      score=9.0,
      criteria="Must be concise",
      mode="numeric",
      threshold=7.0,
    )
    assert r.criteria == "Must be concise"
    assert r.mode == "numeric"

  def test_to_dict_numeric(self):
    r = JudgeResult(eval_name="judge", success=True, mode="numeric", threshold=8.0)
    d = r.to_dict()
    assert d["threshold"] == 8.0

  def test_to_dict_binary(self):
    r = JudgeResult(eval_name="judge", success=True, mode="binary")
    d = r.to_dict()
    assert "threshold" not in d


# ═════════════════════════════════════════════════════════════
# EvalCase
# ═════════════════════════════════════════════════════════════


class TestEvalCase:
  def test_basic(self):
    c = EvalCase(input="What is 2+2?", expected="4", name="math")
    assert c.input == "What is 2+2?"
    assert c.expected == "4"
    assert c.name == "math"

  def test_defaults(self):
    c = EvalCase(input="hello")
    assert c.expected is None
    assert c.metadata == {}
    assert c.name == ""

  def test_with_metadata(self):
    c = EvalCase(input="test", metadata={"expected_tools": ["search"]})
    assert c.metadata["expected_tools"] == ["search"]


# ═════════════════════════════════════════════════════════════
# EvalSuite
# ═════════════════════════════════════════════════════════════


class TestEvalSuite:
  def test_empty(self):
    s = EvalSuite(eval_name="test")
    assert s.total == 0
    assert s.passed == 0
    assert s.failed == 0
    assert s.pass_rate == 0.0

  def test_with_results(self):
    s = EvalSuite(
      eval_name="test",
      results=[
        EvalResult(success=True),
        EvalResult(success=True),
        EvalResult(success=False),
      ],
    )
    assert s.total == 3
    assert s.passed == 2
    assert s.failed == 1
    assert s.pass_rate == pytest.approx(2 / 3)

  def test_all_pass(self):
    s = EvalSuite(
      eval_name="test",
      results=[EvalResult(success=True), EvalResult(success=True)],
    )
    assert s.pass_rate == 1.0

  def test_to_dict(self):
    s = EvalSuite(
      eval_name="test",
      results=[EvalResult(eval_name="r1", success=True)],
    )
    d = s.to_dict()
    assert d["eval_name"] == "test"
    assert d["total"] == 1
    assert d["passed"] == 1
    assert d["failed"] == 0
    assert d["pass_rate"] == 1.0
    assert len(d["results"]) == 1


# ═════════════════════════════════════════════════════════════
# AccuracyEval
# ═════════════════════════════════════════════════════════════


class TestAccuracyEval:
  @pytest.mark.asyncio
  async def test_pass(self):
    agent = _make_mock_agent(content="The answer is 4")
    ev = AccuracyEval(threshold=7.0)
    ev._judge = _make_mock_judge_model('{"score": 9, "reason": "Correct answer"}')

    result = await ev.arun(agent, EvalCase(input="What is 2+2?", expected="4"))
    assert result.success is True
    assert result.score == 9.0
    assert result.eval_name == "accuracy"
    assert result.expected == "4"
    assert result.actual == "The answer is 4"

  @pytest.mark.asyncio
  async def test_fail(self):
    agent = _make_mock_agent(content="I don't know")
    ev = AccuracyEval(threshold=7.0)
    ev._judge = _make_mock_judge_model('{"score": 2, "reason": "Wrong"}')

    result = await ev.arun(agent, EvalCase(input="What is 2+2?", expected="4"))
    assert result.success is False
    assert result.score == 2.0

  @pytest.mark.asyncio
  async def test_no_expected(self):
    agent = _make_mock_agent()
    ev = AccuracyEval()
    result = await ev.arun(agent, EvalCase(input="hello"))
    assert result.success is False
    assert "No expected output" in (result.reason or "")

  @pytest.mark.asyncio
  async def test_agent_failure(self):
    agent = MagicMock()
    agent.arun = AsyncMock(side_effect=RuntimeError("boom"))
    ev = AccuracyEval()
    result = await ev.arun(agent, EvalCase(input="hello", expected="hi"))
    assert result.success is False
    assert "Agent execution failed" in (result.reason or "")

  @pytest.mark.asyncio
  async def test_custom_threshold(self):
    agent = _make_mock_agent(content="some answer")
    ev = AccuracyEval(threshold=9.0)
    ev._judge = _make_mock_judge_model('{"score": 8, "reason": "Good but not perfect"}')

    result = await ev.arun(agent, EvalCase(input="q", expected="a"))
    assert result.success is False  # 8 < 9.0 threshold
    assert result.threshold == 9.0

  @pytest.mark.asyncio
  async def test_team_eval(self):
    team = _make_mock_team(content="Team result")
    ev = AccuracyEval(threshold=7.0)
    ev._judge = _make_mock_judge_model('{"score": 8, "reason": "Good"}')

    result = await ev.evaluate_team(team, EvalCase(input="q", expected="a"))
    assert result.success is True
    assert result.actual == "Team result"

  @pytest.mark.asyncio
  async def test_batch(self):
    agent = _make_mock_agent(content="answer")
    ev = AccuracyEval(threshold=5.0)
    ev._judge = _make_mock_judge_model('{"score": 7, "reason": "ok"}')

    cases = [
      EvalCase(input="q1", expected="a1", name="case1"),
      EvalCase(input="q2", expected="a2", name="case2"),
    ]
    suite = await ev.arun_batch(agent, cases)
    assert suite.total == 2
    assert suite.passed == 2
    assert suite.pass_rate == 1.0


class TestAccuracyEvalParsing:
  def test_parse_json(self):
    ev = AccuracyEval()
    score, reason = ev._parse_judge_response('{"score": 8, "reason": "Good answer"}')
    assert score == 8.0
    assert reason == "Good answer"

  def test_parse_json_with_code_fence(self):
    ev = AccuracyEval()
    score, reason = ev._parse_judge_response('```json\n{"score": 9, "reason": "Great"}\n```')
    assert score == 9.0

  def test_parse_fallback_regex(self):
    ev = AccuracyEval()
    score, reason = ev._parse_judge_response("The score is score: 7 because it's good")
    assert score == 7.0

  def test_parse_clamps_to_range(self):
    ev = AccuracyEval()
    score, _ = ev._parse_judge_response('{"score": 15, "reason": "off scale"}')
    assert score == 10.0

  def test_parse_clamps_minimum(self):
    ev = AccuracyEval()
    score, _ = ev._parse_judge_response('{"score": 0, "reason": "terrible"}')
    assert score == 1.0

  def test_parse_unparseable(self):
    ev = AccuracyEval()
    score, reason = ev._parse_judge_response("I cannot evaluate this")
    assert score == 0.0
    assert "Unparseable" in reason


# ═════════════════════════════════════════════════════════════
# PerformanceEval
# ═════════════════════════════════════════════════════════════


class TestPerformanceEval:
  @pytest.mark.asyncio
  async def test_basic_run(self):
    agent = _make_mock_agent(content="fast")
    ev = PerformanceEval(runs=2, warmup_runs=0)

    result = await ev.arun(agent, EvalCase(input="hello"))
    assert result.success is True
    assert result.runs == 2
    assert len(result.durations) == 2
    assert result.duration_ms > 0
    assert result.peak_memory_mb >= 0

  @pytest.mark.asyncio
  async def test_duration_threshold_pass(self):
    agent = _make_mock_agent(content="fast")
    ev = PerformanceEval(duration_threshold_ms=10000, runs=1)

    result = await ev.arun(agent, EvalCase(input="hello"))
    assert result.success is True

  @pytest.mark.asyncio
  async def test_duration_threshold_fail(self):
    """Agent that sleeps should exceed a very tight threshold."""
    agent = MagicMock()

    async def slow_run(prompt: str):
      await asyncio.sleep(0.05)  # 50ms
      output = MagicMock()
      output.content = "slow"
      return output

    agent.arun = slow_run
    ev = PerformanceEval(duration_threshold_ms=10, runs=1)  # 10ms threshold

    result = await ev.arun(agent, EvalCase(input="hello"))
    assert result.success is False
    assert "exceeds threshold" in (result.reason or "")

  @pytest.mark.asyncio
  async def test_memory_threshold(self):
    agent = _make_mock_agent(content="ok")
    ev = PerformanceEval(memory_threshold_mb=1000, runs=1)

    result = await ev.arun(agent, EvalCase(input="hello"))
    assert result.success is True

  @pytest.mark.asyncio
  async def test_warmup_runs(self):
    agent = _make_mock_agent(content="ok")
    ev = PerformanceEval(runs=1, warmup_runs=2)

    await ev.arun(agent, EvalCase(input="hello"))
    assert agent.arun.await_count == 3  # 2 warmup + 1 profiling

  @pytest.mark.asyncio
  async def test_team_eval(self):
    team = _make_mock_team(content="team fast")
    ev = PerformanceEval(runs=1)

    result = await ev.evaluate_team(team, EvalCase(input="hello"))
    assert result.success is True
    assert result.runs == 1

  @pytest.mark.asyncio
  async def test_agent_failure_still_records(self):
    """Even if the agent fails, we should still record the duration."""
    agent = MagicMock()
    agent.arun = AsyncMock(side_effect=RuntimeError("boom"))
    ev = PerformanceEval(runs=2)

    result = await ev.arun(agent, EvalCase(input="hello"))
    assert result.runs == 2
    assert len(result.durations) == 2


# ═════════════════════════════════════════════════════════════
# ReliabilityEval
# ═════════════════════════════════════════════════════════════


class TestReliabilityEval:
  @pytest.mark.asyncio
  async def test_all_expected_called(self):
    te1 = MagicMock()
    te1.function_name = "search_web"
    te1.tool_name = "search_web"
    te2 = MagicMock()
    te2.function_name = "summarize"
    te2.tool_name = "summarize"

    agent = _make_mock_agent(content="result", tool_executions=[te1, te2])
    ev = ReliabilityEval(expected_tools=["search_web", "summarize"])

    result = await ev.arun(agent, EvalCase(input="Research AI"))
    assert result.success is True
    assert result.missing_tools == []

  @pytest.mark.asyncio
  async def test_missing_tool(self):
    te1 = MagicMock()
    te1.function_name = "search_web"
    te1.tool_name = "search_web"

    agent = _make_mock_agent(content="result", tool_executions=[te1])
    ev = ReliabilityEval(expected_tools=["search_web", "summarize"])

    result = await ev.arun(agent, EvalCase(input="Research AI"))
    assert result.success is False
    assert "summarize" in result.missing_tools

  @pytest.mark.asyncio
  async def test_extra_tools_permissive(self):
    te1 = MagicMock()
    te1.function_name = "search_web"
    te1.tool_name = "search_web"
    te2 = MagicMock()
    te2.function_name = "extra_tool"
    te2.tool_name = "extra_tool"

    agent = _make_mock_agent(content="result", tool_executions=[te1, te2])
    ev = ReliabilityEval(expected_tools=["search_web"], strict=False)

    result = await ev.arun(agent, EvalCase(input="test"))
    assert result.success is True  # Extra is OK in permissive mode
    assert "extra_tool" in result.extra_tools

  @pytest.mark.asyncio
  async def test_extra_tools_strict(self):
    te1 = MagicMock()
    te1.function_name = "search_web"
    te1.tool_name = "search_web"
    te2 = MagicMock()
    te2.function_name = "extra_tool"
    te2.tool_name = "extra_tool"

    agent = _make_mock_agent(content="result", tool_executions=[te1, te2])
    ev = ReliabilityEval(expected_tools=["search_web"], strict=True)

    result = await ev.arun(agent, EvalCase(input="test"))
    assert result.success is False  # Extra tools cause failure in strict mode

  @pytest.mark.asyncio
  async def test_no_expected_tools(self):
    agent = _make_mock_agent()
    ev = ReliabilityEval(expected_tools=[])
    result = await ev.arun(agent, EvalCase(input="test"))
    assert result.success is False
    assert "No expected tools" in (result.reason or "")

  @pytest.mark.asyncio
  async def test_case_metadata_override(self):
    """expected_tools in case.metadata overrides eval-level setting."""
    te1 = MagicMock()
    te1.function_name = "custom_tool"
    te1.tool_name = "custom_tool"

    agent = _make_mock_agent(content="ok", tool_executions=[te1])
    ev = ReliabilityEval(expected_tools=["other_tool"])

    case = EvalCase(input="test", metadata={"expected_tools": ["custom_tool"]})
    result = await ev.arun(agent, case)
    assert result.success is True

  @pytest.mark.asyncio
  async def test_agent_failure(self):
    agent = MagicMock()
    agent.arun = AsyncMock(side_effect=RuntimeError("boom"))
    ev = ReliabilityEval(expected_tools=["search"])
    result = await ev.arun(agent, EvalCase(input="test"))
    assert result.success is False
    assert "Execution failed" in (result.reason or "")

  @pytest.mark.asyncio
  async def test_team_eval(self):
    te1 = MagicMock()
    te1.function_name = "search_web"
    te1.tool_name = "search_web"

    team = _make_mock_team(content="team result")
    team_output = team.arun.return_value
    team_output.tool_executions = [te1]

    ev = ReliabilityEval(expected_tools=["search_web"])
    result = await ev.evaluate_team(team, EvalCase(input="test"))
    assert result.success is True

  @pytest.mark.asyncio
  async def test_fallback_message_tool_calls(self):
    """When tool_executions is empty, fall back to scanning messages."""
    msg = MagicMock()
    msg.tool_calls = [
      {"function": {"name": "search_web"}, "id": "1"},
    ]

    agent = _make_mock_agent(content="ok", tool_executions=[])
    agent.arun.return_value.messages = [msg]

    ev = ReliabilityEval(expected_tools=["search_web"])
    result = await ev.arun(agent, EvalCase(input="test"))
    assert result.success is True
    assert "search_web" in result.actual_tools


# ═════════════════════════════════════════════════════════════
# AgentAsJudgeEval
# ═════════════════════════════════════════════════════════════


class TestAgentAsJudgeEval:
  @pytest.mark.asyncio
  async def test_numeric_pass(self):
    agent = _make_mock_agent(content="concise answer")
    ev = AgentAsJudgeEval(criteria="Must be concise", threshold=7.0, mode="numeric")
    ev._judge = _make_mock_judge_model('{"score": 9, "reason": "Very concise"}')

    result = await ev.arun(agent, EvalCase(input="Summarize"))
    assert result.success is True
    assert result.score == 9.0
    assert result.criteria == "Must be concise"
    assert result.mode == "numeric"

  @pytest.mark.asyncio
  async def test_numeric_fail(self):
    agent = _make_mock_agent(content="verbose")
    ev = AgentAsJudgeEval(criteria="Must be concise", threshold=8.0, mode="numeric")
    ev._judge = _make_mock_judge_model('{"score": 5, "reason": "Too verbose"}')

    result = await ev.arun(agent, EvalCase(input="Summarize"))
    assert result.success is False
    assert result.score == 5.0

  @pytest.mark.asyncio
  async def test_binary_pass(self):
    agent = _make_mock_agent(content="Clean output")
    ev = AgentAsJudgeEval(criteria="No profanity", mode="binary")
    ev._judge = _make_mock_judge_model('{"pass": true, "reason": "Clean"}')

    result = await ev.arun(agent, EvalCase(input="Write something"))
    assert result.success is True
    assert result.mode == "binary"
    assert result.score == 10.0

  @pytest.mark.asyncio
  async def test_binary_fail(self):
    agent = _make_mock_agent(content="Bad output")
    ev = AgentAsJudgeEval(criteria="No profanity", mode="binary")
    ev._judge = _make_mock_judge_model('{"pass": false, "reason": "Contains bad language"}')

    result = await ev.arun(agent, EvalCase(input="Write something"))
    assert result.success is False
    assert result.score == 0.0

  @pytest.mark.asyncio
  async def test_no_criteria(self):
    agent = _make_mock_agent()
    ev = AgentAsJudgeEval(criteria="")
    result = await ev.arun(agent, EvalCase(input="test"))
    assert result.success is False
    assert "No evaluation criteria" in (result.reason or "")

  @pytest.mark.asyncio
  async def test_criteria_from_case_metadata(self):
    agent = _make_mock_agent(content="answer")
    ev = AgentAsJudgeEval(criteria="default", mode="numeric", threshold=5.0)
    ev._judge = _make_mock_judge_model('{"score": 8, "reason": "Good"}')

    case = EvalCase(input="q", metadata={"criteria": "Custom criteria"})
    result = await ev.arun(agent, case)
    assert result.success is True
    assert result.criteria == "Custom criteria"

  @pytest.mark.asyncio
  async def test_team_eval(self):
    team = _make_mock_team(content="Team output")
    ev = AgentAsJudgeEval(criteria="Must be professional", mode="numeric", threshold=7.0)
    ev._judge = _make_mock_judge_model('{"score": 8, "reason": "Professional"}')

    result = await ev.evaluate_team(team, EvalCase(input="test"))
    assert result.success is True

  @pytest.mark.asyncio
  async def test_agent_failure(self):
    agent = MagicMock()
    agent.arun = AsyncMock(side_effect=RuntimeError("boom"))
    ev = AgentAsJudgeEval(criteria="test")
    result = await ev.arun(agent, EvalCase(input="test"))
    assert result.success is False


class TestJudgeParsing:
  def test_parse_numeric_json(self):
    ev = AgentAsJudgeEval()
    score, reason = ev._parse_numeric('{"score": 8, "reason": "Good"}')
    assert score == 8.0
    assert reason == "Good"

  def test_parse_numeric_code_fence(self):
    ev = AgentAsJudgeEval()
    score, reason = ev._parse_numeric('```json\n{"score": 7, "reason": "OK"}\n```')
    assert score == 7.0

  def test_parse_numeric_clamp(self):
    ev = AgentAsJudgeEval()
    score, _ = ev._parse_numeric('{"score": 12, "reason": "off scale"}')
    assert score == 10.0

  def test_parse_binary_true(self):
    ev = AgentAsJudgeEval()
    passed, reason = ev._parse_binary('{"pass": true, "reason": "Clean"}')
    assert passed is True
    assert reason == "Clean"

  def test_parse_binary_false(self):
    ev = AgentAsJudgeEval()
    passed, reason = ev._parse_binary('{"pass": false, "reason": "Bad"}')
    assert passed is False

  def test_parse_binary_fallback(self):
    ev = AgentAsJudgeEval()
    passed, _ = ev._parse_binary('"pass": true somewhere in text')
    assert passed is True

  def test_parse_binary_unparseable(self):
    ev = AgentAsJudgeEval()
    passed, reason = ev._parse_binary("I cannot evaluate")
    assert passed is False
    assert "Unparseable" in reason


# ═════════════════════════════════════════════════════════════
# Compression Events
# ═════════════════════════════════════════════════════════════


class TestCompressionEvents:
  def test_compression_started_event(self):
    from definable.agent.events import CompressionStartedEvent

    evt = CompressionStartedEvent(
      run_id="r1",
      agent_id="a1",
      agent_name="test",
      tool_results_count=3,
    )
    assert evt.event == "CompressionStarted"
    assert evt.tool_results_count == 3

  def test_compression_completed_event(self):
    from definable.agent.events import CompressionCompletedEvent

    evt = CompressionCompletedEvent(
      run_id="r1",
      agent_id="a1",
      agent_name="test",
      tool_results_compressed=3,
      original_size=5000,
      compressed_size=1000,
      duration_ms=150.0,
    )
    assert evt.event == "CompressionCompleted"
    assert evt.tool_results_compressed == 3
    assert evt.original_size == 5000
    assert evt.compressed_size == 1000
    assert evt.duration_ms == 150.0

  def test_compression_events_in_run_event_enum(self):
    from definable.agent.run.agent import RunEvent

    assert RunEvent.compression_started.value == "CompressionStarted"
    assert RunEvent.compression_completed.value == "CompressionCompleted"

  def test_compression_events_in_registry(self):
    from definable.agent.run.agent import (
      CompressionCompletedEvent,
      CompressionStartedEvent,
      RUN_EVENT_TYPE_REGISTRY,
    )

    assert RUN_EVENT_TYPE_REGISTRY["CompressionStarted"] == CompressionStartedEvent
    assert RUN_EVENT_TYPE_REGISTRY["CompressionCompleted"] == CompressionCompletedEvent


# ═════════════════════════════════════════════════════════════
# Import tests
# ═════════════════════════════════════════════════════════════


class TestImports:
  def test_import_from_eval_package(self):
    from definable.agent.eval import (  # noqa: F401
      AccuracyEval,
      AccuracyResult,
      AgentAsJudgeEval,
      BaseEval,
      EvalCase,
      EvalResult,
      EvalSuite,
      JudgeResult,
      PerformanceEval,
      PerformanceResult,
      ReliabilityEval,
      ReliabilityResult,
    )

    assert AccuracyEval is not None
    assert AccuracyResult is not None
    assert AgentAsJudgeEval is not None
    assert BaseEval is not None
    assert EvalCase is not None
    assert EvalResult is not None
    assert EvalSuite is not None
    assert JudgeResult is not None
    assert PerformanceEval is not None
    assert PerformanceResult is not None
    assert ReliabilityEval is not None
    assert ReliabilityResult is not None

  def test_import_from_agent_package(self):
    from definable.agent import (  # noqa: F401
      AccuracyEval,
      AccuracyResult,
      AgentAsJudgeEval,
      BaseEval,
      EvalCase,
      EvalResult,
      EvalSuite,
      JudgeResult,
      PerformanceEval,
      PerformanceResult,
      ReliabilityEval,
      ReliabilityResult,
    )

    assert AccuracyEval is not None
    assert PerformanceEval is not None
    assert ReliabilityEval is not None
    assert AgentAsJudgeEval is not None

  def test_import_compression_events_from_events(self):
    from definable.agent.events import (  # noqa: F401
      CompressionCompletedEvent,
      CompressionStartedEvent,
    )

    assert CompressionStartedEvent is not None
    assert CompressionCompletedEvent is not None

  def test_eval_prompt_constants(self):
    """Verify prompt templates are accessible."""
    assert "{input}" in ACCURACY_JUDGE_PROMPT
    assert "{expected}" in ACCURACY_JUDGE_PROMPT
    assert "{actual}" in ACCURACY_JUDGE_PROMPT
    assert "{input}" in NUMERIC_JUDGE_PROMPT
    assert "{criteria}" in NUMERIC_JUDGE_PROMPT
    assert "{input}" in BINARY_JUDGE_PROMPT


# ═════════════════════════════════════════════════════════════
# BaseEval abstract enforcement
# ═════════════════════════════════════════════════════════════


class TestBaseEval:
  def test_cannot_instantiate_directly(self):
    with pytest.raises(TypeError):
      BaseEval()  # type: ignore[abstract]

  @pytest.mark.asyncio
  async def test_team_eval_not_implemented_by_default(self):
    """Custom eval without evaluate_team should raise NotImplementedError."""

    class MinimalEval(BaseEval):
      name = "minimal"

      async def evaluate(self, agent, case):
        return EvalResult(eval_name=self.name, success=True)

    ev = MinimalEval()
    with pytest.raises(NotImplementedError, match="does not support team"):
      await ev.evaluate_team(MagicMock(), EvalCase(input="test"))

  @pytest.mark.asyncio
  async def test_custom_eval(self):
    """A minimal custom eval should work."""

    class AlwaysPassEval(BaseEval):
      name = "always_pass"

      async def evaluate(self, agent, case):
        output = await agent.arun(case.input)
        return EvalResult(eval_name=self.name, success=True, reason=output.content)

    agent = _make_mock_agent(content="hello")
    ev = AlwaysPassEval()
    result = await ev.arun(agent, EvalCase(input="test"))
    assert result.success is True
    assert result.reason == "hello"
