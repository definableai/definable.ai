from definable.agent import Agent
from definable.agent.testing import MockModel
from definable.skill import Skill
from definable.tool import tool


@tool
def lookup_policy(topic: str) -> str:
  """Return the policy section for a topic."""
  return f"Policy for {topic}"


support = Skill(
  name="support_docs",
  instructions="Check policy documentation before answering.",
  tools=[lookup_policy],
)

agent = Agent(
  model=MockModel(responses=["Refunds close after 30 days."]),
  skills=[support],
)

output = agent.run("Summarize the refund policy.")

assert "lookup_policy" in agent.tool_names
assert output.content == "Refunds close after 30 days."
