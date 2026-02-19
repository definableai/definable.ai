"""Basic readers example — Agent with readers=True, attach a text file.

Demonstrates the simplest way to use file readers with an agent.
When `readers=True`, the agent auto-creates a BaseReader with all
available built-in parsers and extracts content from attached files
before sending them to the model.

Usage:
    export OPENAI_API_KEY=your-key
    python definable/examples/readers/01_basic_readers.py
"""

from definable.agent import Agent
from definable.media import File
from definable.model import OpenAIChat

model = OpenAIChat(id="gpt-4o-mini")

# Create agent with file readers enabled
agent = Agent(
  model=model,
  instructions="You are a helpful assistant. Analyze any files the user provides.",
  readers=True,  # Auto-creates a BaseReader with all available parsers
)

# Attach a text file to the message
file = File(
  content=b"Q3 Revenue: $2.5M\nQ3 Expenses: $1.8M\nQ3 Net Income: $700K\nGrowth: 15% YoY",
  filename="financials.txt",
  mime_type="text/plain",
)

output = agent.run(
  "Summarize the key financial metrics from this file.",
  files=[file],
)

print(output.content)
