"""Tracing events emitted by the guardrails layer."""

from dataclasses import dataclass
from typing import Optional

from definable.run.agent import BaseAgentRunEvent


@dataclass
class GuardrailCheckedEvent(BaseAgentRunEvent):
  """Emitted after each guardrail check completes (allow, modify, or warn)."""

  event: str = "GuardrailChecked"
  guardrail_name: str = ""
  guardrail_type: str = ""  # "input" | "output" | "tool"
  action: str = ""  # "allow" | "block" | "modify" | "warn"
  message: Optional[str] = None
  duration_ms: Optional[float] = None


@dataclass
class GuardrailBlockedEvent(BaseAgentRunEvent):
  """Emitted when a guardrail blocks execution."""

  event: str = "GuardrailBlocked"
  guardrail_name: str = ""
  guardrail_type: str = ""  # "input" | "output" | "tool"
  reason: str = ""
