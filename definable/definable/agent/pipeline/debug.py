"""Debug configuration and hook for pipeline inspection."""

from dataclasses import dataclass, field
from typing import Callable, Optional, Set


@dataclass
class DebugConfig:
  """Configuration for pipeline debug mode.

  Extends the existing ``Agent(debug=True)`` with fine-grained
  pipeline inspection capabilities.

  Example::

      agent = Agent(
          model="openai/gpt-4o",
          debug=DebugConfig(
              breakpoints={"before:invoke_loop", "after:guard_output"},
              step_mode=True,
              log_state_changes=True,
          ),
      )

  Attributes:
    breakpoints: Set of hook specs where execution pauses for
        inspection. Format: ``"{timing}:{phase_name}"``.
    step_mode: If True, pause after every phase (overrides breakpoints).
    inspector: Callback invoked at each phase with ``(state, phase_name)``.
        Useful for custom logging, state snapshots, or test assertions.
    log_state_changes: If True, diff state between phases and log changes.
    enable_trace: If True, also attach DebugExporter (color-coded stderr).
        Default True to maintain backward compat with ``debug=True``.
  """

  breakpoints: Set[str] = field(default_factory=set)
  step_mode: bool = False
  inspector: Optional[Callable] = None
  log_state_changes: bool = False
  enable_trace: bool = True
