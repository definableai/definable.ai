"""
Example: Using agent.replay() to inspect a past run.

This example shows how to:
1. Run an agent and capture the output
2. Create a Replay from the RunOutput for inspection
3. Compare two runs side-by-side
"""

from definable.agent import Agent
from definable.model.openai import OpenAIChat

# 1. Create an agent
agent = Agent(
  model=OpenAIChat(id="gpt-4o-mini"),
  instructions="You are a helpful assistant. Be concise.",
)

# 2. Run it
output = agent.run("What is the capital of France?")
print(f"Output: {output.content}")
print(f"Tokens: {output.metrics.total_tokens if output.metrics else 'N/A'}")

# 3. Build a Replay for inspection
replay = agent.replay(run_output=output)

print("\n--- Replay ---")
print(f"Run ID: {replay.run_id}")
print(f"Model: {replay.model} ({replay.model_provider})")
print(f"Status: {replay.status}")
print(f"Input: {replay.input.input_content if replay.input else 'N/A'}")
print(f"Content: {replay.content}")
print(f"Tokens: {replay.tokens.total_tokens} (in={replay.tokens.input_tokens}, out={replay.tokens.output_tokens})")  # type: ignore[union-attr]
print(f"Cost: ${replay.cost:.6f}" if replay.cost else "Cost: N/A")  # type: ignore[union-attr]
print(f"Tool calls: {len(replay.tool_calls)}")  # type: ignore[union-attr]

# 4. Run again with different instructions and compare
output2 = agent.run("What is the capital of France?")
comparison = agent.compare(output, output2)

print("\n--- Comparison ---")
print(f"Token diff: {comparison.token_diff:+d}")
print(f"Cost diff: ${comparison.cost_diff:+.6f}" if comparison.cost_diff is not None else "Cost diff: N/A")
if comparison.content_diff:
  print(f"Content changed:\n{comparison.content_diff}")
else:
  print("Content: identical")
