"""Middleware protocol and common implementations for agent execution."""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
  from definable.agents.config import KnowledgeConfig
  from definable.knowledge import Document
  from definable.models.message import Message
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


class KnowledgeMiddleware:
  """
  Middleware for RAG (Retrieval-Augmented Generation) integration.

  Retrieves relevant documents from a knowledge base before agent execution
  and injects them into the context for enhanced responses.

  Pipeline:
    1. Extract query from user message(s)
    2. Search knowledge base for relevant documents
    3. Filter by minimum relevance score
    4. Format documents as context string
    5. Store in RunContext for system message injection

  Example:
    from definable.agents.middleware import KnowledgeMiddleware
    from definable.agents.config import KnowledgeConfig

    middleware = KnowledgeMiddleware(KnowledgeConfig(
      knowledge=kb,
      top_k=5,
      rerank=True,
      context_format="xml",
    ))
    agent.use(middleware)
  """

  def __init__(self, config: "KnowledgeConfig"):
    """
    Initialize knowledge middleware.

    Args:
      config: Knowledge configuration with retrieval settings.
    """
    self.config = config

  async def __call__(
    self,
    context: "RunContext",
    next_handler: NextHandler,
  ) -> "RunOutput":
    """
    Retrieve knowledge and enrich context before execution.

    Args:
      context: Run context with metadata containing messages.
      next_handler: Next middleware or core execution handler.

    Returns:
      RunOutput from the handler chain.
    """
    if not self.config.enabled:
      return await next_handler(context)

    # Extract query from messages stored in context metadata
    messages = context.metadata.get("_messages") if context.metadata else None
    if not messages:
      return await next_handler(context)

    query = self._extract_query(messages)
    if not query:
      return await next_handler(context)

    # Retrieve relevant documents
    try:
      documents = await self.config.knowledge.asearch(
        query=query,
        top_k=self.config.top_k,
        rerank=self.config.rerank,
      )
    except Exception:
      # Don't fail the run if knowledge retrieval fails
      return await next_handler(context)

    # Filter by minimum score if configured
    if self.config.min_score is not None:
      documents = [doc for doc in documents if doc.reranking_score is not None and doc.reranking_score >= self.config.min_score]

    if not documents:
      return await next_handler(context)

    # Format and store context
    context_text = self._format_context(documents)
    context.knowledge_context = context_text
    context.knowledge_documents = documents

    # Store position preference for agent to use
    if context.metadata is None:
      context.metadata = {}
    context.metadata["_knowledge_position"] = self.config.context_position

    return await next_handler(context)

  def _extract_query(self, messages: List["Message"]) -> Optional[str]:
    """
    Extract search query from conversation messages.

    Args:
      messages: List of conversation messages.

    Returns:
      Query string or None if no user message found.
    """
    if self.config.query_from == "last_user":
      # Find last user message
      for msg in reversed(messages):
        if msg.role == "user" and msg.content:
          content = msg.content if isinstance(msg.content, str) else str(msg.content)
          return content[: self.config.max_query_length]
      return None

    elif self.config.query_from == "full_conversation":
      # Concatenate all user messages
      user_contents: List[str] = []
      for msg in messages:
        if msg.role == "user" and msg.content:
          content = msg.content if isinstance(msg.content, str) else str(msg.content)
          user_contents.append(content)
      if user_contents:
        full_query = " ".join(user_contents)
        return full_query[: self.config.max_query_length]
      return None

    return None

  def _format_context(self, documents: List["Document"]) -> str:
    """
    Format retrieved documents as context string.

    Args:
      documents: List of retrieved documents.

    Returns:
      Formatted context string.
    """
    if self.config.context_format == "xml":
      return self._format_xml(documents)
    elif self.config.context_format == "markdown":
      return self._format_markdown(documents)
    elif self.config.context_format == "json":
      return self._format_json(documents)
    return self._format_xml(documents)

  def _format_xml(self, documents: List["Document"]) -> str:
    """Format as XML-style context block."""
    lines = ["<knowledge_context>"]
    for i, doc in enumerate(documents):
      attrs = [f'index="{i + 1}"']
      if doc.reranking_score is not None:
        attrs.append(f'relevance="{doc.reranking_score:.3f}"')
      if doc.source:
        attrs.append(f'source="{doc.source}"')
      if doc.name:
        attrs.append(f'name="{doc.name}"')
      attr_str = " ".join(attrs)
      lines.append(f"  <document {attr_str}>")
      lines.append(f"    {doc.content}")
      lines.append("  </document>")
    lines.append("</knowledge_context>")
    return "\n".join(lines)

  def _format_markdown(self, documents: List["Document"]) -> str:
    """Format as Markdown."""
    lines = ["## Retrieved Context\n"]
    for i, doc in enumerate(documents):
      score_str = f" (relevance: {doc.reranking_score:.3f})" if doc.reranking_score else ""
      name_str = f" - {doc.name}" if doc.name else ""
      lines.append(f"### Document {i + 1}{name_str}{score_str}")
      lines.append("")
      lines.append(doc.content)
      if doc.source:
        lines.append(f"\n*Source: {doc.source}*")
      lines.append("")
    return "\n".join(lines)

  def _format_json(self, documents: List["Document"]) -> str:
    """Format as JSON."""
    data = [
      {
        "index": i + 1,
        "content": doc.content,
        "name": doc.name,
        "source": doc.source,
        "relevance": doc.reranking_score,
      }
      for i, doc in enumerate(documents)
    ]
    return json.dumps({"knowledge_context": data}, indent=2)
