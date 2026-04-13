from definable.agent import Agent
from definable.agent.run import RunStatus
from definable.agent.testing import MockModel


agent = Agent(
  model=MockModel(responses=["12"]),
  instructions="Reply with the answer only.",
)

output = agent.run("What is 5 + 7?")

assert output.content == "12"
assert output.status == RunStatus.completed
