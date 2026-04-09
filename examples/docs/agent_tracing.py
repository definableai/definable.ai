from pathlib import Path
from tempfile import TemporaryDirectory

from definable.agent import Agent, JSONLExporter, MockModel
from definable.agent.tracing import Tracing
from definable.agent.tracing.jsonl import read_trace_file


with TemporaryDirectory() as tmp:
  agent = Agent(
    model=MockModel(responses=["hello"]),
    session_id="trace-session",
    tracing=Tracing(exporters=[JSONLExporter(tmp, mirror_stdout=False)]),
  )

  output = agent.run("Hi")
  events = read_trace_file(Path(tmp) / "trace-session.jsonl")

  assert output.content == "hello"
  assert [event["event"] for event in events] == [
    "RunStarted",
    "ModelCallStarted",
    "ModelCallCompleted",
    "RunCompleted",
  ]
