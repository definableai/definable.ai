import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from definable.agent import Agent, MockModel
from definable.agent.eval import AccuracyEval, AgentAsJudgeEval, EvalCase, PerformanceEval, ReliabilityEval
from definable.model.metrics import Metrics
from definable.tool.decorator import tool


class FakeJudge:
  def __init__(self, content: str) -> None:
    self._content = content

  async def aresponse(self, messages):
    return SimpleNamespace(content=self._content)


@tool
def add(a: int, b: int) -> int:
  return a + b


def tool_model(messages=None, **kwargs):
  response = MagicMock()

  if not any(getattr(message, "role", None) == "tool" for message in (messages or [])):
    response.content = ""
    response.tool_calls = [
      {
        "id": "call_1",
        "type": "function",
        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
      }
    ]
  else:
    response.content = "5"
    response.tool_calls = []

  response.tool_executions = []
  response.response_usage = Metrics()
  response.reasoning_content = None
  response.citations = None
  response.images = None
  response.videos = None
  response.audios = None
  response.parsed = None
  return response


async def main() -> None:
  accuracy_agent = Agent(model=MockModel(responses=["Paris"]))
  accuracy = AccuracyEval(threshold=7.0)
  accuracy._judge = FakeJudge('{"score": 9, "reason": "Matches expected answer."}')
  accuracy_result = await accuracy.arun(
    accuracy_agent,
    EvalCase(input="Capital of France?", expected="Paris"),
  )

  reliability_agent = Agent(model=MockModel(side_effect=tool_model), tools=[add])
  reliability = ReliabilityEval(expected_tools=["add"])
  reliability_result = await reliability.arun(
    reliability_agent,
    EvalCase(input="What is 2 + 3?"),
  )

  performance_agent = Agent(model=MockModel(responses=["ok", "ok"]))
  performance = PerformanceEval(
    duration_threshold_ms=1000,
    memory_threshold_mb=100,
    runs=2,
  )
  performance_result = await performance.arun(
    performance_agent,
    EvalCase(input="Hello"),
  )

  judge_agent = Agent(model=MockModel(responses=["A concise answer."]))
  judge = AgentAsJudgeEval(criteria="Output must be concise.", mode="binary")
  judge._judge = FakeJudge('{"pass": true, "reason": "The output is concise."}')
  judge_result = await judge.arun(judge_agent, EvalCase(input="Summarize."))

  assert accuracy_result.success is True
  assert reliability_result.actual_tools == ["add"]
  assert performance_result.runs == 2
  assert judge_result.success is True


asyncio.run(main())
