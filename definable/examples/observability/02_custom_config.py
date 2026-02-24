"""
Example: Custom observability configuration.

This example shows how to:
1. Use ObservabilityConfig for fine-grained control
2. Set a custom trace directory
3. Adjust buffer size and theme
4. Compose with existing tracing

Run:
    .venv/bin/python definable/examples/observability/02_custom_config.py

Then open http://localhost:8000/obs/ in your browser.
"""

from definable.agent import Agent
from definable.agent.observability import ObservabilityConfig
from definable.agent.tracing.base import Tracing
from definable.agent.tracing.jsonl import JSONLExporter
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool


@tool
def search(query: str) -> str:
  """Search the web for information."""
  return f"Results for '{query}': [example result 1, example result 2]"


# Custom observability configuration
obs_config = ObservabilityConfig(
  enabled=True,
  trace_dir="./my-traces",  # custom trace directory
  buffer_size=5000,  # smaller buffer for memory-constrained environments
  theme="light",  # light theme (default is dark)
)

# Composes with existing tracing — the observability exporter is
# automatically added alongside any other exporters you configure.
tracing = Tracing(exporters=[JSONLExporter(trace_dir="./my-traces")])

agent = Agent(
  model=OpenAIChat(id="gpt-4o-mini"),
  tools=[search],
  instructions="You are a research assistant.",
  tracing=tracing,
  observability=obs_config,
)

# Serve the agent
agent.serve(enable_server=True)
