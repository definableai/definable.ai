"""Workflow lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import List, Optional

from definable.agent.run.base import BaseRunOutputEvent


@dataclass
class BaseWorkflowEvent(BaseRunOutputEvent):
  """Base class for all workflow events."""

  created_at: int = field(default_factory=lambda: int(time()))
  event: str = ""
  run_id: Optional[str] = None
  workflow_id: str = ""
  workflow_name: str = ""


@dataclass
class WorkflowRunStartedEvent(BaseWorkflowEvent):
  """Emitted when a workflow run begins."""

  event: str = "workflow_run_started"
  step_count: int = 0
  step_names: List[str] = field(default_factory=list)


@dataclass
class WorkflowRunCompletedEvent(BaseWorkflowEvent):
  """Emitted when a workflow run finishes successfully."""

  event: str = "workflow_run_completed"
  content: Optional[str] = None
  success: bool = True
  duration_ms: float = 0.0


@dataclass
class WorkflowRunErrorEvent(BaseWorkflowEvent):
  """Emitted when a workflow run fails with an unrecoverable error."""

  event: str = "workflow_run_error"
  error: str = ""


@dataclass
class StepStartedEvent(BaseWorkflowEvent):
  """Emitted when a workflow step begins execution."""

  event: str = "step_started"
  step_id: str = ""
  step_name: str = ""
  step_type: str = ""
  step_index: int = 0


@dataclass
class StepCompletedEvent(BaseWorkflowEvent):
  """Emitted when a workflow step finishes successfully."""

  event: str = "step_completed"
  step_id: str = ""
  step_name: str = ""
  step_type: str = ""
  step_index: int = 0
  content: Optional[str] = None
  success: bool = True
  duration_ms: float = 0.0


@dataclass
class StepErrorEvent(BaseWorkflowEvent):
  """Emitted when a workflow step fails."""

  event: str = "step_error"
  step_id: str = ""
  step_name: str = ""
  step_type: str = ""
  step_index: int = 0
  error: str = ""


@dataclass
class StepSkippedEvent(BaseWorkflowEvent):
  """Emitted when a workflow step is skipped (e.g., condition branch not taken)."""

  event: str = "step_skipped"
  step_id: str = ""
  step_name: str = ""
  step_type: str = ""
  reason: str = ""


@dataclass
class LoopIterationEvent(BaseWorkflowEvent):
  """Emitted at the start of each loop iteration."""

  event: str = "loop_iteration"
  step_name: str = ""
  iteration: int = 0
  max_iterations: int = 0
  should_continue: bool = True
