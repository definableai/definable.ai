"""Middleware protocol and common implementations for agent execution."""

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
  from definable.run.agent import RunOutput
  from definable.run.base import RunContext


# Type alias for the handler function
NextHandler = Callable[["RunContext"], Awaitable["RunOutput"]]


@runtime_checkable
class Middleware(Protocol):
  """
  Protocol for execution middleware - enables cross-cutting concerns.

  Middleware wraps the agent's core execution, allowing you to add
  behavior before and after each run. Common use cases include:
  - Logging and observability
  - Retry logic for transient failures
  - Caching responses
  - Rate limiting
  - Authentication token injection
  - Metrics collection

  Example:
      class MyMiddleware:
          async def __call__(
              self,
              context: RunContext,
              next_handler: NextHandler
          ) -> RunOutput:
              # Pre-processing
              print(f"Starting run {context.run_id}")

              # Call the next middleware or core handler
              result = await next_handler(context)

              # Post-processing
              print(f"Completed run {context.run_id}")

              return result

  Usage with Agent:
      agent = Agent(model=my_model)
      agent.use(LoggingMiddleware(logger))
      agent.use(RetryMiddleware(max_retries=3))
  """

  def __call__(
    self,
    context: "RunContext",
    next_handler: NextHandler,
  ) -> Awaitable["RunOutput"]:
    """
    Process the run, optionally delegating to next_handler.

    Args:
        context: The run context with run_id, session_id, etc.
        next_handler: The next middleware or core handler to call.

    Returns:
        Awaitable that resolves to the RunOutput.
    """
    ...


class LoggingMiddleware:
  """
  Middleware that logs run start, completion, and errors.

  Example:
      import logging
      logger = logging.getLogger("my_agent")
      agent.use(LoggingMiddleware(logger))
  """

  def __init__(self, logger: logging.Logger, level: int = logging.INFO):
    """
    Initialize the logging middleware.

    Args:
        logger: The logger instance to use.
        level: Log level for normal messages (errors always use ERROR).
    """
    self.logger = logger
    self.level = level

  async def __call__(
    self,
    context: "RunContext",
    next_handler: NextHandler,
  ) -> "RunOutput":
    """Log run lifecycle events."""
    self.logger.log(self.level, f"Run started: run_id={context.run_id}, session_id={context.session_id}")

    try:
      result = await next_handler(context)
      self.logger.log(self.level, f"Run completed: run_id={context.run_id}, status={result.status}")
      return result
    except Exception as e:
      self.logger.error(f"Run failed: run_id={context.run_id}, error={type(e).__name__}: {e}")
      raise


class RetryMiddleware:
  """
  Middleware that retries on transient errors with exponential backoff.

  Only retries on specific transient errors (ConnectionError, TimeoutError).
  Other exceptions are re-raised immediately.

  Example:
      agent.use(RetryMiddleware(max_retries=3, backoff_base=1.0))
  """

  # Errors considered transient and safe to retry
  TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError)

  def __init__(
    self,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 60.0,
  ):
    """
    Initialize the retry middleware.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds (doubles each retry).
        backoff_max: Maximum delay between retries.
    """
    self.max_retries = max_retries
    self.backoff_base = backoff_base
    self.backoff_max = backoff_max

  async def __call__(
    self,
    context: "RunContext",
    next_handler: NextHandler,
  ) -> "RunOutput":
    """Execute with retry logic."""
    last_error = None

    for attempt in range(self.max_retries + 1):
      try:
        return await next_handler(context)
      except self.TRANSIENT_ERRORS as e:
        last_error = e
        if attempt < self.max_retries:
          delay = min(self.backoff_base * (2**attempt), self.backoff_max)
          await asyncio.sleep(delay)
      # Non-transient errors are re-raised immediately
      except Exception:
        raise

    # All retries exhausted
    raise last_error  # type: ignore


class MetricsMiddleware:
  """
  Middleware that collects timing metrics for runs.

  Example:
      metrics = MetricsMiddleware()
      agent.use(metrics)
      # ... run agent ...
      print(f"Average latency: {metrics.average_latency_ms}ms")
  """

  def __init__(self):
    """Initialize the metrics middleware."""
    self._run_count = 0
    self._total_latency_ms = 0.0
    self._error_count = 0

  async def __call__(
    self,
    context: "RunContext",
    next_handler: NextHandler,
  ) -> "RunOutput":
    """Collect timing metrics."""
    import time

    start_time = time.perf_counter()

    try:
      result = await next_handler(context)
      return result
    except Exception:
      self._error_count += 1
      raise
    finally:
      elapsed_ms = (time.perf_counter() - start_time) * 1000
      self._run_count += 1
      self._total_latency_ms += elapsed_ms

  @property
  def run_count(self) -> int:
    """Return total number of runs."""
    return self._run_count

  @property
  def error_count(self) -> int:
    """Return total number of failed runs."""
    return self._error_count

  @property
  def average_latency_ms(self) -> float:
    """Return average latency in milliseconds."""
    if self._run_count == 0:
      return 0.0
    return self._total_latency_ms / self._run_count

  def reset(self) -> None:
    """Reset all metrics."""
    self._run_count = 0
    self._total_latency_ms = 0.0
    self._error_count = 0
