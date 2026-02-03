"""
Cost tracking and metrics.

This example shows how to:
- Track token usage and costs
- Access response metrics
- Monitor agent performance
- Estimate API costs

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from dataclasses import dataclass
from typing import Dict

from definable.agents import Agent, MetricsMiddleware
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool

# Approximate pricing per 1M tokens (as of 2024)
# Check OpenAI pricing page for current rates
PRICING = {
  "gpt-4o": {"input": 2.50, "output": 10.00},
  "gpt-4o-mini": {"input": 0.15, "output": 0.60},
  "gpt-4-turbo": {"input": 10.00, "output": 30.00},
  "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


@dataclass
class CostTracker:
  """Track costs across multiple API calls."""

  model_id: str
  total_input_tokens: int = 0
  total_output_tokens: int = 0
  total_calls: int = 0

  def add_usage(self, input_tokens: int, output_tokens: int):
    """Add token usage from a call."""
    self.total_input_tokens += input_tokens
    self.total_output_tokens += output_tokens
    self.total_calls += 1

  def get_estimated_cost(self) -> float:
    """Estimate total cost in USD."""
    pricing = PRICING.get(self.model_id, PRICING["gpt-4o-mini"])

    input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost

  def get_summary(self) -> Dict:
    """Get a summary of usage and costs."""
    return {
      "model": self.model_id,
      "total_calls": self.total_calls,
      "total_input_tokens": self.total_input_tokens,
      "total_output_tokens": self.total_output_tokens,
      "total_tokens": self.total_input_tokens + self.total_output_tokens,
      "estimated_cost_usd": f"${self.get_estimated_cost():.6f}",
    }


@tool
def get_data(query: str) -> str:
  """Get data for a query."""
  return f"Data for: {query}"


def basic_metrics():
  """Access basic metrics from responses."""
  print("Basic Metrics")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(model=model, instructions="Be concise.")

  output = agent.run("What is the capital of France?")

  print(f"Response: {output.content}")
  print()

  # Access metrics
  if output.metrics:
    print("Metrics:")
    print(f"  Input tokens: {output.metrics.input_tokens}")
    print(f"  Output tokens: {output.metrics.output_tokens}")
    print(f"  Total tokens: {output.metrics.total_tokens}")

    # Calculate cost
    pricing = PRICING["gpt-4o-mini"]
    input_cost = (output.metrics.input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output.metrics.output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    print(f"\nEstimated cost: ${total_cost:.6f}")


def track_multiple_calls():
  """Track costs across multiple API calls."""
  print("\n" + "=" * 50)
  print("Tracking Multiple Calls")
  print("=" * 50)

  model_id = "gpt-4o-mini"
  tracker = CostTracker(model_id=model_id)

  model = OpenAIChat(id=model_id)
  agent = Agent(model=model, instructions="Be brief.")

  queries = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?",
    "Compare them briefly.",
  ]

  for query in queries:
    output = agent.run(query)
    print(f"Q: {query}")
    print(f"A: {(output.content or '')[:50]}...")

    if output.metrics:
      tracker.add_usage(
        output.metrics.input_tokens or 0,
        output.metrics.output_tokens or 0,
      )

    print()

  # Summary
  summary = tracker.get_summary()
  print("Cost Summary:")
  for key, value in summary.items():
    print(f"  {key}: {value}")


def compare_model_costs():
  """Compare costs between different models."""
  print("\n" + "=" * 50)
  print("Model Cost Comparison")
  print("=" * 50)

  print("\nPricing per 1M tokens:")
  print("-" * 40)
  print(f"{'Model':<20} {'Input':>10} {'Output':>10}")
  print("-" * 40)

  for model, prices in PRICING.items():
    print(f"{model:<20} ${prices['input']:>8.2f} ${prices['output']:>8.2f}")

  # Example calculation
  print("\n\nExample: 10,000 input + 1,000 output tokens")
  print("-" * 40)

  for model, prices in PRICING.items():
    input_cost = (10000 / 1_000_000) * prices["input"]
    output_cost = (1000 / 1_000_000) * prices["output"]
    total = input_cost + output_cost
    print(f"{model:<20} ${total:.6f}")


def metrics_middleware_example():
  """Using MetricsMiddleware for automatic tracking."""
  print("\n" + "=" * 50)
  print("MetricsMiddleware")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(model=model, instructions="Be helpful.")

  # Add metrics middleware
  metrics_mw = MetricsMiddleware()
  agent.use(metrics_mw)

  # Make some calls
  agent.run("Hello")
  agent.run("How are you?")
  agent.run("Goodbye")

  print("MetricsMiddleware tracks metrics automatically.")
  print("Access via middleware instance for aggregated stats.")


def budget_monitoring():
  """Monitor budget and alert on threshold."""
  print("\n" + "=" * 50)
  print("Budget Monitoring")
  print("=" * 50)

  class BudgetMonitor:
    """Monitor API costs against a budget."""

    def __init__(self, budget_usd: float, model_id: str):
      self.budget = budget_usd
      self.tracker = CostTracker(model_id=model_id)

    def add_usage(self, input_tokens: int, output_tokens: int):
      self.tracker.add_usage(input_tokens, output_tokens)
      current_cost = self.tracker.get_estimated_cost()

      if current_cost >= self.budget:
        print(f"[ALERT] Budget exceeded! ${current_cost:.4f} >= ${self.budget:.4f}")
      elif current_cost >= self.budget * 0.8:
        print(f"[WARNING] 80% of budget used: ${current_cost:.4f}")

    def get_remaining(self) -> float:
      return max(0, self.budget - self.tracker.get_estimated_cost())

  # Example usage
  monitor = BudgetMonitor(budget_usd=0.001, model_id="gpt-4o-mini")

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(model=model, instructions="Be brief.")

  for i in range(5):
    output = agent.run(f"Count to {i + 1}")
    print(f"Response {i + 1}: {output.content}")

    if output.metrics:
      monitor.add_usage(
        output.metrics.input_tokens or 0,
        output.metrics.output_tokens or 0,
      )

    remaining = monitor.get_remaining()
    print(f"Budget remaining: ${remaining:.6f}\n")


def token_estimation():
  """Estimate tokens before making calls."""
  print("\n" + "=" * 50)
  print("Token Estimation")
  print("=" * 50)

  def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars for English)."""
    return len(text) // 4

  # Example prompts
  prompts = [
    "Hello",
    "What is the meaning of life?",
    "Explain quantum computing in detail, including its history, principles, and applications.",
  ]

  print("Token Estimates (approximate):")
  for prompt in prompts:
    est = estimate_tokens(prompt)
    print(f"  '{prompt[:30]}...' → ~{est} tokens")

  print("""
Note: Actual token counts depend on the tokenizer.
For accurate counts, use tiktoken library:
  pip install tiktoken
  import tiktoken
  enc = tiktoken.encoding_for_model("gpt-4o")
  tokens = len(enc.encode(text))
""")


def main():
  basic_metrics()
  track_multiple_calls()
  compare_model_costs()
  metrics_middleware_example()
  budget_monitoring()
  token_estimation()


if __name__ == "__main__":
  main()
