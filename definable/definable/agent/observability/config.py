"""Observability configuration."""

from dataclasses import dataclass


@dataclass
class ObservabilityConfig:
  """Configuration for the observability dashboard.

  Snaps directly into an Agent like other config blocks:

      from definable.agent.observability import ObservabilityConfig
      agent = Agent(model="openai/gpt-4o", observability=ObservabilityConfig(enabled=True))

  Or use the boolean shorthand:

      agent = Agent(model="openai/gpt-4o", observability=True)

  Attributes:
    enabled: Whether observability endpoints are active.
    trace_dir: Directory containing JSONL trace files for historical browsing.
    buffer_size: Maximum events held in the live ring buffer.
    theme: Default dashboard theme (``"dark"`` or ``"light"``).
  """

  enabled: bool = False
  trace_dir: str = "./traces"
  buffer_size: int = 10_000
  theme: str = "dark"
