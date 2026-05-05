from definable.agent import Agent
from definable.agent.replay import compare_runs
from definable.agent.testing import MockModel


agent = Agent(model=MockModel(responses=["first", "second"]), instructions="Reply briefly.")

first = agent.run("hello")
second = agent.run("hello again")
replay = agent.replay(run_output=first)
diff = compare_runs(first, second)

assert replay.content == "first"
assert diff.token_diff == 0
