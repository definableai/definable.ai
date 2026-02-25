"""Workflow — top-level orchestrator for multi-step agent execution.

A Workflow composes multiple steps (agents, teams, or callables) into a
pipeline with support for sequential, parallel, looping, conditional,
and routed execution.

Example::

    from definable.agent import Agent
    from definable.agent.workflow import Workflow, Step

    researcher = Agent(model="openai/gpt-4o", instructions="You are a research specialist.")
    writer = Agent(model="openai/gpt-4o", instructions="You are a technical writer.")

    workflow = Workflow(
        name="research-pipeline",
        steps=[
            Step(name="researcher", agent=researcher),
            Step(name="writer", agent=writer),
        ],
    )
    result = await workflow.arun("Write about quantum computing")
    print(result.content)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from definable.agent.event_bus import EventBus
from definable.agent.workflow.context import StepInput, WorkflowOutput
from definable.agent.workflow.events import (
  WorkflowRunCompletedEvent,
  WorkflowRunErrorEvent,
  WorkflowRunStartedEvent,
)
from definable.agent.workflow.step import BaseStep, Steps, _normalize_step
from definable.utils.log import log_error, log_info


@dataclass
class Workflow:
  """Multi-step agent workflow orchestrator.

  Executes a sequence of steps (agents, teams, or callables) in order,
  with support for parallel execution, loops, conditions, and routing.

  Args:
      name: Human-readable workflow name.
      description: Optional description.
      instructions: Optional instructions appended to step prompts.
      steps: A single step, list of steps, or composite step (Steps, Parallel, etc.).
      session_state: Initial shared state passed to all steps.
      debug: Enable debug logging.

  Example::

      workflow = Workflow(
          name="research-pipeline",
          steps=[
              Step(name="researcher", agent=researcher_agent),
              Step(name="writer", agent=writer_agent),
          ],
      )
      result = await workflow.arun("Research quantum computing")
  """

  # ── Identity ──────────────────────────────────────────────
  name: str = ""
  description: Optional[str] = None
  instructions: Optional[str] = None

  # ── Steps ─────────────────────────────────────────────────
  steps: Union[Any, List[Any]] = field(default_factory=list)

  # ── State ─────────────────────────────────────────────────
  session_state: Dict[str, Any] = field(default_factory=dict)

  # ── Config ────────────────────────────────────────────────
  debug: bool = False

  # ── Internal ──────────────────────────────────────────────
  _id: str = field(default_factory=lambda: str(uuid4()))
  _event_bus: EventBus = field(default_factory=EventBus)

  @property
  def id(self) -> str:
    return self._id

  @property
  def events(self) -> EventBus:
    """Access the workflow's event bus for subscribing to events."""
    return self._event_bus

  async def arun(
    self,
    input: str,  # noqa: A002
    *,
    session_state: Optional[Dict[str, Any]] = None,
    additional_data: Optional[Dict[str, Any]] = None,
  ) -> WorkflowOutput:
    """Execute the workflow.

    Args:
        input: The initial input/prompt for the workflow.
        session_state: Override session state for this run.
        additional_data: Additional data to pass to steps.

    Returns:
        WorkflowOutput with results from all steps.
    """
    run_id = str(uuid4())
    start = time()

    # Merge session state
    merged_state: Dict[str, Any] = {**self.session_state}
    if session_state:
      merged_state.update(session_state)

    # Build initial StepInput
    step_input = StepInput(
      input=input,
      additional_data=additional_data or {},
      session_state=merged_state,
    )

    try:
      # Normalize steps
      executable = self._normalize_steps()
      step_names = self._collect_step_names(executable)

      # Emit start event
      await self._event_bus.emit(
        WorkflowRunStartedEvent(
          run_id=run_id,
          workflow_id=self._id,
          workflow_name=self.name,
          step_count=len(step_names),
          step_names=step_names,
        )
      )

      log_info(f"Workflow '{self.name}' started (run_id={run_id[:8]})")

      result = await executable.execute(
        step_input,
        event_bus=self._event_bus,
        run_id=run_id,
        workflow_id=self._id,
        workflow_name=self.name,
      )

      duration = (time() - start) * 1000

      output = WorkflowOutput(
        workflow_id=self._id,
        workflow_name=self.name,
        run_id=run_id,
        content=result.content,
        success=result.success,
        error=result.error,
        step_outputs=result.steps or [result],
        duration_ms=duration,
        session_state=merged_state,
      )

      await self._event_bus.emit(
        WorkflowRunCompletedEvent(
          run_id=run_id,
          workflow_id=self._id,
          workflow_name=self.name,
          content=result.content,
          success=result.success,
          duration_ms=duration,
        )
      )

      log_info(f"Workflow '{self.name}' completed in {duration:.0f}ms (success={result.success})")
      return output

    except Exception as exc:
      duration = (time() - start) * 1000
      error_msg = str(exc)
      log_error(f"Workflow '{self.name}' failed: {error_msg}")

      await self._event_bus.emit(
        WorkflowRunErrorEvent(
          run_id=run_id,
          workflow_id=self._id,
          workflow_name=self.name,
          error=error_msg,
        )
      )

      return WorkflowOutput(
        workflow_id=self._id,
        workflow_name=self.name,
        run_id=run_id,
        success=False,
        error=error_msg,
        duration_ms=duration,
        session_state=merged_state,
      )

  def _normalize_steps(self) -> BaseStep:
    """Normalize the steps field into an executable BaseStep."""
    if isinstance(self.steps, list):
      return Steps(name=f"{self.name}_sequence", steps=self.steps)
    if isinstance(self.steps, BaseStep):
      return self.steps
    if callable(self.steps):
      return _normalize_step(self.steps)
    raise TypeError(f"Invalid steps type: {type(self.steps)}")

  def _collect_step_names(self, step: BaseStep) -> List[str]:
    """Collect all top-level step names from a step tree (for events)."""
    names: List[str] = []
    if hasattr(step, "steps") and isinstance(step.steps, list):
      for s in step.steps:
        if isinstance(s, BaseStep):
          names.append(s.name)
        elif hasattr(s, "name"):
          name_val = str(getattr(s, "name", ""))
          if name_val:
            names.append(name_val)
    elif step.name:
      names.append(step.name)
    return names
