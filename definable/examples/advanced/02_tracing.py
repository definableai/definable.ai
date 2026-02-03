"""
Tracing and debugging with JSONLExporter.

This example shows how to:
- Enable tracing for agents
- Export traces to JSONL files
- Debug agent execution
- Analyze trace data

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from definable.agents import Agent, AgentConfig, JSONLExporter, TracingConfig
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool

if TYPE_CHECKING:
  from definable.run.base import BaseRunOutputEvent


@tool
def calculate(expression: str) -> str:
  """Evaluate a mathematical expression."""
  try:
    result = eval(expression)
    return f"Result: {result}"
  except Exception as e:
    return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
  """Get weather for a city (mock)."""
  weather_data = {
    "london": "Cloudy, 15°C",
    "paris": "Sunny, 22°C",
    "tokyo": "Rainy, 18°C",
  }
  return weather_data.get(city.lower(), f"Unknown city: {city}")


def basic_tracing():
  """Basic tracing setup."""
  print("Basic Tracing")
  print("=" * 50)

  # Create a temporary directory for traces
  trace_dir = tempfile.mkdtemp(prefix="definable_traces_")
  print(f"Trace directory: {trace_dir}")

  # Create JSONLExporter
  exporter = JSONLExporter(trace_dir=trace_dir)

  model = OpenAIChat(id="gpt-4o-mini")

  # Create agent with tracing enabled
  agent = Agent(
    model=model,
    tools=[calculate, get_weather],
    instructions="You are a helpful assistant with calculation and weather tools.",
    config=AgentConfig(
      tracing=TracingConfig(
        exporters=[exporter],  # Add trace exporter
      ),
    ),
  )

  # Run some queries
  output = agent.run("What is 15 * 7?")
  print(f"\nResponse: {output.content}")

  output = agent.run("What's the weather in Paris?")
  print(f"Response: {output.content}")

  # Flush to ensure all traces are written
  exporter.flush()

  # Check for trace files
  trace_files = list(Path(trace_dir).glob("*.jsonl"))
  print(f"\nTrace files created: {len(trace_files)}")

  return trace_dir


def read_trace_files(trace_dir: str):
  """Read and display trace data."""
  print("\n" + "=" * 50)
  print("Reading Trace Data")
  print("=" * 50)

  trace_files = list(Path(trace_dir).glob("*.jsonl"))

  if not trace_files:
    print("No trace files found.")
    return

  for trace_file in trace_files[:1]:  # Show first file
    print(f"\nFile: {trace_file.name}")
    print("-" * 40)

    with open(trace_file) as f:
      for i, line in enumerate(f):
        if i >= 5:  # Limit output
          print("  ... (more entries)")
          break

        try:
          entry = json.loads(line)
          event_type = entry.get("event", "unknown")
          run_id = entry.get("run_id", "")[:8]

          print(f"  [{event_type}] run_id={run_id}...")

          # Show relevant fields based on event type
          if "content" in entry and entry["content"]:
            content = str(entry["content"])[:60]
            print(f"    Content: {content}...")
          if "tool_name" in entry:
            print(f"    Tool: {entry['tool_name']}")

        except json.JSONDecodeError:
          print("  [Invalid JSON]")


def trace_with_tool_execution():
  """Trace tool executions in detail."""
  print("\n" + "=" * 50)
  print("Tracing Tool Execution")
  print("=" * 50)

  trace_dir = tempfile.mkdtemp(prefix="tool_traces_")

  exporter = JSONLExporter(trace_dir=trace_dir)

  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[calculate, get_weather],
    instructions="Use tools to answer questions.",
    config=AgentConfig(
      tracing=TracingConfig(exporters=[exporter]),
    ),
  )

  # Query that requires multiple tool calls
  output = agent.run("Calculate 100 / 4 and tell me the weather in London and Tokyo")

  print(f"Response: {output.content}")

  # Show tool executions from output
  if output.tools:
    print("\nTool Executions:")
    for execution in output.tools:
      print(f"  - {execution.tool_name}")
      print(f"    Args: {execution.tool_args}")
      print(f"    Result: {execution.result}")

  exporter.flush()
  print(f"\nTrace saved to: {trace_dir}")


def analyze_traces():
  """Analyze trace data for insights."""
  print("\n" + "=" * 50)
  print("Trace Analysis")
  print("=" * 50)

  trace_dir = tempfile.mkdtemp(prefix="analysis_traces_")
  exporter = JSONLExporter(trace_dir=trace_dir)

  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[calculate],
    instructions="You are a calculator.",
    config=AgentConfig(
      tracing=TracingConfig(exporters=[exporter]),
    ),
  )

  # Run multiple queries
  queries = [
    "What is 2 + 2?",
    "Calculate 10 * 5",
    "What is 100 / 4?",
  ]

  for query in queries:
    agent.run(query)

  exporter.flush()

  # Analyze traces
  trace_files = list(Path(trace_dir).glob("*.jsonl"))

  total_entries = 0
  tool_calls = 0
  events_by_type: Dict[str, int] = {}

  for trace_file in trace_files:
    with open(trace_file) as f:
      for line in f:
        try:
          entry = json.loads(line)
          total_entries += 1

          event_type = entry.get("event", "unknown")
          events_by_type[event_type] = events_by_type.get(event_type, 0) + 1

          if event_type == "ToolCallStarted":
            tool_calls += 1

        except json.JSONDecodeError:
          pass

  print("Analysis Results:")
  print(f"  Total trace entries: {total_entries}")
  print(f"  Tool calls started: {tool_calls}")
  print("\n  Events by type:")
  for event_type, count in sorted(events_by_type.items()):
    print(f"    {event_type}: {count}")


def debugging_workflow():
  """Using traces for debugging."""
  print("\n" + "=" * 50)
  print("Debugging Workflow")
  print("=" * 50)

  print("""
Debugging with Traces:

1. Enable tracing during development:
   ```python
   config=AgentConfig(
       tracing=TracingConfig(
           exporters=[JSONLExporter("./traces")]
       )
   )
   ```

2. Run your agent and reproduce the issue

3. Examine trace files:
   - Look for error events
   - Check tool call sequences
   - Verify message flow
   - Analyze timing

4. Common issues to look for:
   - Tool calls with unexpected arguments
   - Missing or malformed responses
   - Long delays between events
   - Repeated tool calls (loops)

5. Use trace data to:
   - Reproduce issues consistently
   - Understand agent decision-making
   - Optimize performance
   - Create test cases
""")


class PrintExporter:
  """Simple exporter that prints events.

  Implements the TraceExporter protocol.
  """

  def export(self, event: "BaseRunOutputEvent") -> None:
    """Print the event type."""
    event_type = getattr(event, "event", "unknown")
    print(f"  [TRACE] {event_type}")

  def flush(self) -> None:
    """No buffering, nothing to flush."""
    pass

  def shutdown(self) -> None:
    """No resources to clean up."""
    pass


def custom_trace_export():
  """Custom trace handling."""
  print("\n" + "=" * 50)
  print("Custom Trace Export")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[calculate],
    instructions="Use calculate for math.",
    config=AgentConfig(
      tracing=TracingConfig(
        exporters=[PrintExporter()],
      ),
    ),
  )

  print("\nRunning with PrintExporter:")
  output = agent.run("What is 5 + 5?")
  print(f"\nFinal response: {output.content}")


def main():
  trace_dir = basic_tracing()
  read_trace_files(trace_dir)
  trace_with_tool_execution()
  analyze_traces()
  debugging_workflow()
  custom_trace_export()


if __name__ == "__main__":
  main()
