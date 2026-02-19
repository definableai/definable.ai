"""Agent configuration with immutable settings."""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

if TYPE_CHECKING:
  from definable.agent.tracing.base import Tracing
  from definable.model.base import Model


# Default descriptions for the layer guide (system prompt capabilities menu)
DEFAULT_MEMORY_DESCRIPTION = (
  "User-specific context recalled from prior sessions: the user's stated preferences, personal details, "
  "goals, past decisions, things they've explicitly told the agent before, and conversation history that "
  "spans multiple interactions. "
  "Activates when the user refers to something previously discussed, asks for personalized recommendations, "
  "or when the response would benefit from knowing who this user is and what they care about — "
  "for example: 'like we talked about before', 'based on what I told you', 'remember my preference for...'. "
  "When memory context is injected, reference it to avoid asking for information the user has already provided; "
  "use it to personalize the response without making the user repeat themselves. "
  "Does not contain general domain knowledge, company policies, documents, or real-time web data."
)
DEFAULT_KNOWLEDGE_DESCRIPTION = (
  "Relevant passages retrieved from a pre-loaded, curated knowledge base: documents, policies, product "
  "information, company data, uploaded files, technical references, or any domain-specific material "
  "explicitly added by the operator before deployment. "
  "Activates when the query asks about topics likely covered by the knowledge base — for example: "
  "product features, internal procedures, domain terminology, or content from uploaded documents — "
  "rather than requiring live web lookup or personal user context. "
  "When knowledge passages are injected, cite and use them to answer accurately; prefer these retrieved "
  "excerpts over relying on training knowledge alone, which may be imprecise or outdated for domain-specific topics. "
  "Does not contain personal user history, real-time information, or live web data."
)
DEFAULT_THINKING_DESCRIPTION = (
  "An explicit structured reasoning chain completed before the final response, for queries where "
  "direct answering risks errors or oversimplification. Used for: mathematical reasoning, logical "
  "analysis, multi-step planning, code review, nuanced trade-off evaluation, or any question where "
  "working through the problem step-by-step improves correctness and depth. "
  "Activates when the query is complex, ambiguous, or high-stakes — for example: 'should I...', "
  "'what's the best approach to...', 'help me design...', 'analyze the trade-offs between...'. "
  "When thinking has been performed, a reasoning process has already concluded; produce the final "
  "answer based on that reasoning rather than re-deriving or repeating the full chain verbatim. "
  "Not activated for simple factual lookups, short conversational replies, or questions with obvious answers."
)
DEFAULT_RESEARCH_DESCRIPTION = (
  "Live web search results gathered from current online sources: recent news, official announcements, "
  "newly published data, real-time prices or availability, and any information that may have changed "
  "since the model's training cutoff or is absent from the knowledge base. "
  "Activates when the query requires current or time-sensitive information — for example: recent events, "
  "today's metrics, software version updates, ongoing situations, or facts explicitly tied to a recent date. "
  "When research results are injected, they appear as excerpts with citations from live sources; "
  "use these to answer accurately about recent developments and attribute information to its source. "
  "Does not replace curated domain documents in the knowledge base, and does not recall personal user history."
)


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
    from definable.agent.config import AgentConfig, CompressionConfig

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
class ReadersConfig:
  """Configuration for file readers integration with agents.

  Controls how files attached to messages are pre-processed into text
  before reaching the LLM.

  Attributes:
    enabled: Whether file reading is enabled.
    registry: Optional FileReaderRegistry instance. If None, a default
      registry with all available built-in readers is created.
    max_total_content_length: Maximum total characters across all files.
    context_format: Format for the injected context ("xml" or "markdown").
  """

  enabled: bool = True
  registry: Optional[Any] = None
  max_total_content_length: Optional[int] = None
  context_format: Literal["xml", "markdown"] = "xml"


@dataclass(frozen=True)
class AgentConfig:
  """
  Optional configuration for advanced Agent settings.

  Controls execution behavior, error handling, and observability.
  Capability layers (memory, knowledge, deep_research) are now
  direct ``Agent`` constructor parameters rather than fields here.

  Uses frozen dataclass for immutability, ensuring thread-safety
  and preventing config changes during execution.

  Attributes:
      agent_id: Unique identifier for the agent instance.
      agent_name: Human-readable name for the agent.
      tracing: Tracing instance (backward compat — prefer Agent(tracing=...)).
      compression: Tool-result compression settings.
      readers: File readers settings.
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

  # Tracing configuration (backward compat — prefer Agent(tracing=...))
  tracing: Optional["Tracing"] = field(default=None, hash=False)

  # Compression configuration for tool results
  compression: Optional[CompressionConfig] = field(default=None, hash=False)

  # File readers configuration
  readers: Optional[ReadersConfig] = field(default=None, hash=False)

  # Context defaults
  session_state: Optional[Dict[str, Any]] = field(default=None, hash=False)
  dependencies: Optional[Dict[str, Any]] = field(default=None, hash=False)

  # Execution settings
  max_iterations: int = 10  # Max tool call loops before stopping
  max_tool_rounds: int = 30  # Max agent loop iterations (tool-call rounds) before forced stop
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
    non_serializable = ("tracing", "session_state", "dependencies", "compression", "readers")
    current = {k: v for k, v in asdict(self).items() if k not in non_serializable}
    # Handle non-serializable fields separately
    current["tracing"] = self.tracing
    current["session_state"] = self.session_state
    current["dependencies"] = self.dependencies
    current["compression"] = self.compression
    current["readers"] = self.readers
    current.update(kwargs)
    return AgentConfig(**current)

  def __repr__(self) -> str:
    return (
      f"AgentConfig(agent_id={self.agent_id!r}, agent_name={self.agent_name!r}, max_iterations={self.max_iterations}, max_retries={self.max_retries})"
    )
