"""Agent configuration with immutable settings."""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
  from definable.agents.tracing.base import TraceExporter
  from definable.run.base import BaseRunOutputEvent


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

  # Context defaults
  session_state: Optional[Dict[str, Any]] = field(default=None, hash=False)
  dependencies: Optional[Dict[str, Any]] = field(default=None, hash=False)

  # Execution settings
  max_iterations: int = 10  # Max tool call loops before stopping
  max_tokens: Optional[int] = None  # Optional token limit per run

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
    current = {k: v for k, v in asdict(self).items() if k not in ("tracing", "session_state", "dependencies")}
    # Handle non-serializable fields separately
    current["tracing"] = self.tracing
    current["session_state"] = self.session_state
    current["dependencies"] = self.dependencies
    current.update(kwargs)
    return AgentConfig(**current)

  def __repr__(self) -> str:
    return (
      f"AgentConfig(agent_id={self.agent_id!r}, agent_name={self.agent_name!r}, max_iterations={self.max_iterations}, max_retries={self.max_retries})"
    )
