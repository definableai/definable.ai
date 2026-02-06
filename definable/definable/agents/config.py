"""Agent configuration with immutable settings."""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
  from definable.agents.tracing.base import TraceExporter
  from definable.knowledge import Knowledge
  from definable.models.base import Model
  from definable.run.base import BaseRunOutputEvent


@dataclass
class CompressionConfig:
  """
  Configuration for tool result compression in agents.

  Controls automatic compression of tool results to save context space
  while preserving critical information during agent execution.

  Attributes:
    enabled: Whether compression is enabled.
    model: Model instance or string to use for compression.
           If None, uses the agent's model.
    tool_results_limit: Compress after N uncompressed tool results.
    token_limit: Compress when context exceeds this token threshold.
    instructions: Custom compression prompt/instructions.

  Example:
    from definable.agents.config import AgentConfig, CompressionConfig

    config = AgentConfig(
      compression=CompressionConfig(
        enabled=True,
        tool_results_limit=3,
        # model not specified - uses agent's model
      ),
    )
  """

  enabled: bool = True
  model: Optional[Union[str, "Model"]] = None  # Model instance or string (default: agent's model)
  tool_results_limit: Optional[int] = 3  # Compress after N tool results
  token_limit: Optional[int] = None  # Or compress at token threshold
  instructions: Optional[str] = None  # Custom compression prompt


@dataclass
class KnowledgeConfig:
  """
  Configuration for knowledge base integration with agents.

  Controls how knowledge is retrieved and injected into the agent's context,
  enabling RAG (Retrieval-Augmented Generation) capabilities.

  Attributes:
    knowledge: Knowledge base instance for retrieval.
    top_k: Number of documents to retrieve per query.
    rerank: Whether to rerank results for better relevance.
    min_score: Minimum relevance score threshold (0.0 to 1.0).
    context_format: Format for injected context ("xml", "markdown", "json").
    context_position: Where to inject context ("system" or "before_user").
    query_from: How to extract query ("last_user", "full_conversation").
    max_query_length: Maximum characters to use from user message for query.
    enabled: Runtime toggle to enable/disable retrieval.

  Example:
    from definable.agents.config import AgentConfig, KnowledgeConfig
    from definable.knowledge import Knowledge

    kb = Knowledge(vector_db=..., embedder=...)
    config = AgentConfig(
      knowledge=KnowledgeConfig(
        knowledge=kb,
        top_k=5,
        rerank=True,
        context_format="xml",
      ),
    )
  """

  knowledge: "Knowledge"
  top_k: int = 5
  rerank: bool = True
  min_score: Optional[float] = None
  context_format: Literal["xml", "markdown", "json"] = "xml"
  context_position: Literal["system", "before_user"] = "system"
  query_from: Literal["last_user", "full_conversation"] = "last_user"
  max_query_length: int = 500
  enabled: bool = True


@dataclass
class TracingConfig:
  """Configuration for the tracing system."""

  enabled: bool = True
  exporters: Optional[List["TraceExporter"]] = None

  # Filtering
  event_filter: Optional[Callable[["BaseRunOutputEvent"], bool]] = None

  # Batching (for high-throughput scenarios)
  batch_size: int = 1  # 1 = immediate export
  flush_interval_ms: int = 5000


@dataclass(frozen=True)
class AgentConfig:
  """
  Optional configuration for advanced Agent settings.

  Uses frozen dataclass for immutability, ensuring thread-safety
  and preventing config changes during execution.

  Attributes:
      agent_id: Unique identifier for the agent instance.
      agent_name: Human-readable name for the agent.
      tracing: Configuration for trace exporters.
      session_state: Default session state for runs.
      dependencies: Dependencies to inject into tools.
      max_iterations: Maximum tool call loops before stopping.
      max_tokens: Optional token limit per run.
      stream_timeout_seconds: Timeout for sync streaming runs.
      retry_transient_errors: Whether to retry on transient errors.
      max_retries: Maximum number of retry attempts.
      retry_backoff_base: Base for exponential backoff (seconds).
      validate_tool_args: Whether to validate tool arguments.
      strict_output_schema: Whether to strictly enforce output schema.
  """

  # Identity
  agent_id: Optional[str] = None
  agent_name: Optional[str] = None

  # Tracing configuration (not frozen, but typically set once)
  tracing: Optional[TracingConfig] = field(default=None, hash=False)

  # Knowledge base configuration for RAG
  knowledge: Optional[KnowledgeConfig] = field(default=None, hash=False)

  # Compression configuration for tool results
  compression: Optional[CompressionConfig] = field(default=None, hash=False)

  # Context defaults
  session_state: Optional[Dict[str, Any]] = field(default=None, hash=False)
  dependencies: Optional[Dict[str, Any]] = field(default=None, hash=False)

  # Execution settings
  max_iterations: int = 10  # Max tool call loops before stopping
  max_tokens: Optional[int] = None  # Optional token limit per run
  stream_timeout_seconds: float = 300.0  # Timeout for run_stream in seconds

  # Error handling
  retry_transient_errors: bool = True
  max_retries: int = 3
  retry_backoff_base: float = 1.0  # Exponential backoff base (seconds)

  # Validation
  validate_tool_args: bool = True  # Validate tool arguments with Pydantic
  strict_output_schema: bool = False  # Enforce output_schema strictly

  def with_updates(self, **kwargs) -> "AgentConfig":
    """
    Create new config with updated values (immutable pattern).

    Example:
        new_config = config.with_updates(max_retries=5, agent_name="NewAgent")

    Args:
        **kwargs: Fields to update in the new config.

    Returns:
        New AgentConfig instance with updated values.
    """
    # Fields that cannot be serialized with asdict
    non_serializable = ("tracing", "session_state", "dependencies", "knowledge", "compression")
    current = {k: v for k, v in asdict(self).items() if k not in non_serializable}
    # Handle non-serializable fields separately
    current["tracing"] = self.tracing
    current["session_state"] = self.session_state
    current["dependencies"] = self.dependencies
    current["knowledge"] = self.knowledge
    current["compression"] = self.compression
    current.update(kwargs)
    return AgentConfig(**current)

  def __repr__(self) -> str:
    return (
      f"AgentConfig(agent_id={self.agent_id!r}, agent_name={self.agent_name!r}, max_iterations={self.max_iterations}, max_retries={self.max_retries})"
    )
