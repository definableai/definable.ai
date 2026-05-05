import asyncio

from definable.agent import Agent
from definable.agent.team import Team, TeamMode
from definable.agent.testing import MockModel

from support import mock_model_response


async def run_collaborate_mode() -> None:
  researcher = Agent(
    model=MockModel(responses=["Use qubits and superposition."]),
    name="researcher",
    instructions="Research specialist.",
  )
  writer = Agent(
    model=MockModel(responses=["Explain the idea in plain language."]),
    name="writer",
    instructions="Technical writer.",
  )
  team = Team(
    name="content-team",
    model=MockModel(responses=["Combine the research with a plain-language explanation."]),
    members=[researcher, writer],
    mode=TeamMode.collaborate,
  )

  output = await team.arun("Explain quantum computing.")

  assert output.content == "Combine the research with a plain-language explanation."


async def run_route_mode() -> None:
  state = {"calls": 0}

  def leader_side_effect(messages, tools, **kwargs):
    state["calls"] += 1
    if state["calls"] == 1:
      return mock_model_response(
        tool_calls=[
          {
            "id": "call-1",
            "type": "function",
            "function": {
              "name": "delegate_to_member",
              "arguments": '{"member_name": "specialist", "task_input": "explain gravity"}',
            },
          }
        ]
      )
    return mock_model_response(content="unused")

  specialist = Agent(
    model=MockModel(responses=["Gravity is the force that attracts masses."]),
    name="specialist",
    instructions="Physics specialist.",
  )
  team = Team(
    name="router-team",
    model=MockModel(side_effect=leader_side_effect),
    members=[specialist],
    mode=TeamMode.route,
  )

  output = await team.arun("Explain gravity.")

  assert output.content == "Gravity is the force that attracts masses."


async def main() -> None:
  await run_collaborate_mode()
  await run_route_mode()


asyncio.run(main())
