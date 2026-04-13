from __future__ import annotations

import asyncio

try:
  from examples.docs.support import MockEmbedder, MockVectorDB
except ImportError:
  from support import MockEmbedder, MockVectorDB

from definable.agent import Agent, AgentConfig
from definable.agent.run import RunStatus
from definable.agent.run.agent import RunOutput
from definable.agent.team import Team, TeamMode
from definable.agent.testing import MockModel
from definable.agent.workflow import Step, Workflow
from definable.knowledge import Knowledge
from definable.memory import InMemoryStore, Memory
from definable.model import OpenAIChat, resolve_model_string
from definable.model.message import Message
from definable.model.metrics import Metrics
from definable.tool.decorator import tool


@tool(name="lookup_docs", cache_results=True, cache_ttl=30)
def lookup_docs(topic: str) -> str:
  return f"docs:{topic}"


async def _memory_summary() -> int:
  async with Memory(store=InMemoryStore()) as memory:
    await memory.add(Message(role="user", content="Ada"), session_id="docs")
    entries = await memory.get_entries("docs")
  return len(entries)


async def _team_summary() -> dict[str, object]:
  researcher = Agent(model=MockModel(responses=["research notes"]), name="researcher")
  writer = Agent(model=MockModel(responses=["writer draft"]), name="writer")
  team = Team(
    name="docs-team",
    model=MockModel(responses=["team synthesis"]),
    members=[researcher, writer],
    mode=TeamMode.collaborate,
  )
  output = await team.arun("Create a summary.")
  return {
    "team_output": output.content,
    "team_mode": team.mode.value,
    "member_names": team.member_names,
  }


async def _workflow_summary() -> dict[str, object]:
  workflow = Workflow(
    name="docs-workflow",
    steps=[
      Step(name="research", executor=lambda step_input: "notes"),
      Step(name="draft", executor=lambda step_input: f"draft from {step_input.get_last_step_content()}"),
    ],
  )
  output = await workflow.arun("Write docs.")
  return {
    "workflow_content": output.content,
    "workflow_success": output.success,
  }


def main() -> dict[str, object]:
  config = AgentConfig(max_iterations=5, max_retries=1).with_updates(agent_name="docs-agent")
  agent = Agent(
    model=MockModel(responses=["12"]),
    instructions="Reply with the answer only.",
    tools=[lookup_docs],
    config=config,
  )
  agent_output = agent.run("What is 5 + 7?")

  run_output = RunOutput(
    content="done",
    status=RunStatus.completed,
    model="mock-model",
    metrics=Metrics(input_tokens=1, output_tokens=1, total_tokens=2),
  )

  knowledge = Knowledge(
    vector_db=MockVectorDB(embedder=MockEmbedder(dimensions=4)),
    embedder=MockEmbedder(dimensions=4),
  )
  knowledge.add("Agent docs", chunk=False)

  openai = OpenAIChat(id="gpt-4o-mini")
  resolved = resolve_model_string("openai/gpt-4o-mini")

  summary = {
    "agent_output": agent_output.content,
    "tool_name": lookup_docs.name,
    "tool_requires_confirmation": lookup_docs.requires_confirmation,
    "config_max_iterations": config.max_iterations,
    "config_agent_name": config.agent_name,
    "run_output_status": run_output.status.value,
    "run_output_tokens": run_output.metrics.total_tokens if run_output.metrics else None,
    "knowledge_result": knowledge.search("Agent", limit=1)[0].content,
    "memory_entries": asyncio.run(_memory_summary()),
    "openai_model_id": openai.id,
    "resolved_model_type": type(resolved).__name__,
    **asyncio.run(_team_summary()),
    **asyncio.run(_workflow_summary()),
  }

  assert summary["agent_output"] == "12"
  assert summary["tool_name"] == "lookup_docs"
  assert summary["config_max_iterations"] == 5
  assert summary["config_agent_name"] == "docs-agent"
  assert summary["run_output_status"] == "COMPLETED"
  assert summary["run_output_tokens"] == 2
  assert summary["knowledge_result"] == "Agent docs"
  assert summary["memory_entries"] == 1
  assert summary["openai_model_id"] == "gpt-4o-mini"
  assert summary["resolved_model_type"] == "OpenAIChat"
  assert summary["team_output"] == "team synthesis"
  assert summary["team_mode"] == "collaborate"
  assert summary["workflow_content"] == "draft from notes"
  assert summary["workflow_success"] is True

  return summary


if __name__ == "__main__":
  print(main())
