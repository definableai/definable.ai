"""Workflow execution context — input/output types for step chaining."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

if TYPE_CHECKING:
  from definable.agent.run.agent import RunOutput


class StepStatus(str, Enum):
  """Status of a workflow step execution."""

  pending = "pending"
  running = "running"
  completed = "completed"
  failed = "failed"
  skipped = "skipped"


@dataclass
class StepInput:
  """Context passed to each step during workflow execution.

  Carries the original user input, previous step outputs, and shared state
  so each step has full context for its execution.
  """

  input: Optional[str] = None
  previous_step_content: Optional[str] = None
  previous_step_outputs: Dict[str, "StepOutput"] = field(default_factory=dict)
  additional_data: Dict[str, Any] = field(default_factory=dict)
  session_state: Dict[str, Any] = field(default_factory=dict)

  def get_step_output(self, step_name: str) -> Optional["StepOutput"]:
    """Get output from a specific previous step by name."""
    return self.previous_step_outputs.get(step_name)

  def get_step_content(self, step_name: str) -> Optional[str]:
    """Get content string from a specific previous step."""
    output = self.previous_step_outputs.get(step_name)
    return output.content if output else None

  def get_last_step_content(self) -> Optional[str]:
    """Get the content from the most recent previous step."""
    return self.previous_step_content

  def get_all_previous_content(self) -> Dict[str, Optional[str]]:
    """Get content from all previous steps as a dict."""
    return {name: out.content for name, out in self.previous_step_outputs.items()}


@dataclass
class StepOutput:
  """Result from executing a single workflow step.

  Supports composite pattern — steps can contain nested step outputs
  (e.g., Parallel, Loop, Steps all produce nested outputs).
  """

  step_name: str = ""
  step_id: str = field(default_factory=lambda: str(uuid4())[:8])
  step_type: str = "step"
  content: Optional[str] = None
  status: StepStatus = StepStatus.completed
  success: bool = True
  error: Optional[str] = None
  stop: bool = False
  metrics: Optional[Dict[str, Any]] = None
  duration_ms: float = 0.0
  run_output: Optional["RunOutput"] = None
  steps: List["StepOutput"] = field(default_factory=list)

  def to_dict(self) -> Dict[str, Any]:
    """Serialize to dict."""
    result: Dict[str, Any] = {
      "step_name": self.step_name,
      "step_id": self.step_id,
      "step_type": self.step_type,
      "content": self.content,
      "status": self.status.value,
      "success": self.success,
      "error": self.error,
      "stop": self.stop,
      "duration_ms": self.duration_ms,
    }
    if self.metrics:
      result["metrics"] = self.metrics
    if self.steps:
      result["steps"] = [s.to_dict() for s in self.steps]
    return result


@dataclass
class WorkflowOutput:
  """Result from executing a complete workflow."""

  workflow_id: str = ""
  workflow_name: str = ""
  run_id: str = ""
  content: Optional[str] = None
  success: bool = True
  error: Optional[str] = None
  step_outputs: List[StepOutput] = field(default_factory=list)
  duration_ms: float = 0.0
  session_state: Dict[str, Any] = field(default_factory=dict)

  def get_step_output(self, step_name: str) -> Optional[StepOutput]:
    """Get output from a specific step by name (searches nested)."""
    for output in self.step_outputs:
      if output.step_name == step_name:
        return output
      for nested in output.steps:
        if nested.step_name == step_name:
          return nested
    return None

  def get_step_content(self, step_name: str) -> Optional[str]:
    """Get content from a specific step by name."""
    output = self.get_step_output(step_name)
    return output.content if output else None
