from definable.agent import Agent
from definable.agent.testing import MockModel
from definable.tool import tool
from definable.toolkit import Toolkit


class MathToolkit(Toolkit):
  @property
  def tools(self):
    @tool
    def add(a: int, b: int) -> int:
      """Add two integers."""
      return a + b

    return [add]


agent = Agent(
  model=MockModel(responses=["7"]),
  toolkits=[MathToolkit()],
)

output = agent.run("What is 3 + 4?")

assert agent.tool_names == ["add"]
assert output.content == "7"
