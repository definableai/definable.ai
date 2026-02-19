"""
Custom middleware for agents.

This example shows how to:
- Create custom middleware classes
- Use built-in middleware (Logging, Retry, Metrics)
- Chain multiple middleware together
- Intercept and modify agent behavior

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import logging
import time
from typing import Awaitable, Callable, List, Optional

from definable.agent import Agent
from definable.agent.middleware import LoggingMiddleware, MetricsMiddleware, RetryMiddleware
from definable.model.openai import OpenAIChat
from definable.agent.events import RunContext, RunOutput
from definable.tool.decorator import tool

# Type alias for middleware handler
NextHandler = Callable[[RunContext], Awaitable[RunOutput]]


# Simple tool for testing
@tool
def get_data(query: str) -> str:
  """Fetch data based on query."""
  return f"Data for: {query}"


class TimingMiddleware:
  """Custom middleware to track execution time."""

  def __init__(self, name: str = "timing"):
    self.name = name
    self.total_time = 0.0
    self.call_count = 0

  async def __call__(
    self,
    context: RunContext,
    next_handler: NextHandler,
  ) -> RunOutput:
    """Time the execution of the handler chain."""
    print(f"[{self.name}] Starting execution...")
    start_time = time.perf_counter()

    try:
      result = await next_handler(context)
      return result
    finally:
      elapsed = time.perf_counter() - start_time
      self.total_time += elapsed
      self.call_count += 1
      print(f"[{self.name}] Completed in {elapsed:.3f}s")


class AuditMiddleware:
  """Custom middleware for audit logging."""

  def __init__(self):
    self.audit_log = []

  async def __call__(
    self,
    context: RunContext,
    next_handler: NextHandler,
  ) -> RunOutput:
    """Log request and response."""
    # Log request
    entry = {
      "type": "request",
      "timestamp": time.time(),
      "run_id": context.run_id,
    }
    self.audit_log.append(entry)
    print(f"[AUDIT] Request logged (run_id={context.run_id})")

    # Execute handler
    result = await next_handler(context)

    # Log response
    entry = {
      "type": "response",
      "timestamp": time.time(),
      "run_id": context.run_id,
      "status": result.status,
    }
    self.audit_log.append(entry)
    print(f"[AUDIT] Response logged (status={result.status})")

    return result

  def get_audit_log(self):
    return self.audit_log


class ContentFilterMiddleware:
  """Middleware to filter or modify content."""

  def __init__(self, blocked_words: Optional[List[str]] = None):
    self.blocked_words = blocked_words or []

  async def __call__(
    self,
    context: RunContext,
    next_handler: NextHandler,
  ) -> RunOutput:
    """Check input and filter output."""
    print("[FILTER] Checking input...")

    # Execute handler
    result = await next_handler(context)

    print("[FILTER] Filtering output...")
    # In a real implementation, you would filter the result content
    return result


def basic_middleware_usage():
  """Basic middleware usage."""
  print("Basic Middleware Usage")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[get_data],
    instructions="You are a helpful assistant.",
  )

  # Add middleware
  timing = TimingMiddleware()
  agent.use(timing)

  output = agent.run("Get data for 'user statistics'")
  print(f"\nResponse: {output.content}")
  print(f"Total time tracked: {timing.total_time:.3f}s")


def builtin_middleware():
  """Using built-in middleware."""
  print("\n" + "=" * 50)
  print("Built-in Middleware")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(model=model, instructions="You are helpful.")

  # Logging middleware
  logger = logging.getLogger("agent")
  logger.setLevel(logging.INFO)
  handler = logging.StreamHandler()
  handler.setFormatter(logging.Formatter("%(message)s"))
  if not logger.handlers:
    logger.addHandler(handler)

  agent.use(LoggingMiddleware(logger=logger))

  # Retry middleware for transient errors
  agent.use(RetryMiddleware(max_retries=3))

  # Metrics middleware
  metrics = MetricsMiddleware()
  agent.use(metrics)

  print("Added: LoggingMiddleware, RetryMiddleware, MetricsMiddleware")

  output = agent.run("Say hello")
  print(f"Response: {output.content}")
  print(f"Metrics - Run count: {metrics.run_count}, Avg latency: {metrics.average_latency_ms:.2f}ms")


def middleware_chain():
  """Chain multiple middleware together."""
  print("\n" + "=" * 50)
  print("Middleware Chain")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(model=model, instructions="You are helpful.")

  # Create middleware instances
  timing = TimingMiddleware("timer")
  audit = AuditMiddleware()
  filter_mw = ContentFilterMiddleware(blocked_words=["secret"])

  # Add in order - they execute in the order added
  agent.use(timing)  # First: start timing
  agent.use(audit)  # Second: log request
  agent.use(filter_mw)  # Third: filter content

  print("Middleware chain: Timing -> Audit -> ContentFilter")
  print()

  output = agent.run("Tell me a fun fact")

  print(f"\nResponse: {output.content}")
  print(f"\nAudit log entries: {len(audit.get_audit_log())}")


class RequestCounterMiddleware:
  """Count total requests processed."""

  def __init__(self):
    self.request_count = 0
    self.success_count = 0
    self.error_count = 0

  async def __call__(
    self,
    context: RunContext,
    next_handler: NextHandler,
  ) -> RunOutput:
    """Count requests and track success/errors."""
    self.request_count += 1
    print(f"[COUNTER] Request #{self.request_count}")

    try:
      result = await next_handler(context)
      self.success_count += 1
      return result
    except Exception:
      self.error_count += 1
      raise

  def get_stats(self):
    return {
      "total": self.request_count,
      "success": self.success_count,
      "errors": self.error_count,
    }


def custom_middleware_example():
  """Create a custom middleware for specific use case."""
  print("\n" + "=" * 50)
  print("Custom Middleware Example")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(model=model, instructions="Be brief.")

  counter = RequestCounterMiddleware()
  agent.use(counter)

  # Make multiple requests
  for i in range(3):
    output = agent.run(f"Say the number {i + 1}")
    print(f"  Response: {output.content}")

  print(f"\nStats: {counter.get_stats()}")


class ErrorTrackingMiddleware:
  """Track and handle errors."""

  def __init__(self):
    self.errors = []

  async def __call__(
    self,
    context: RunContext,
    next_handler: NextHandler,
  ) -> RunOutput:
    """Track any errors that occur."""
    try:
      return await next_handler(context)
    except Exception as e:
      self.errors.append({
        "error": str(e),
        "type": type(e).__name__,
        "timestamp": time.time(),
        "run_id": context.run_id,
      })
      print(f"[ERROR TRACKING] Caught: {type(e).__name__}: {e}")
      raise


def middleware_error_handling():
  """Middleware error handling."""
  print("\n" + "=" * 50)
  print("Middleware Error Handling")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(model=model)

  error_tracker = ErrorTrackingMiddleware()
  agent.use(error_tracker)

  # Normal request
  output = agent.run("Hello")
  print(f"Response: {output.content}")
  print(f"Errors tracked: {len(error_tracker.errors)}")


def main():
  basic_middleware_usage()
  builtin_middleware()
  middleware_chain()
  custom_middleware_example()
  middleware_error_handling()


if __name__ == "__main__":
  main()
