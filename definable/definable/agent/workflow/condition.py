"""Condition — if/else branching in workflows."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from definable.agent.workflow.context import StepInput, StepOutput, StepStatus
from definable.agent.workflow.events import StepSkippedEvent
from definable.agent.workflow.step import BaseStep, Steps, _normalize_step

if TYPE_CHECKING:
  from definable.agent.event_bus import EventBus


@dataclass
class Condition(BaseStep):
  """Conditional branching — evaluates a condition and executes the matching branch.

  The ``condition`` callable receives :class:`StepInput` and returns a bool.
  If ``True``, ``true_steps`` are executed; otherwise ``false_steps``.

  Example::

      cond = Condition(
          name="quality-gate",
          condition=lambda ctx: "PASS" in (ctx.get_last_step_content() or ""),
          true_steps=Step(name="publish", agent=publisher),
          false_steps=Step(name="rewrite", agent=writer),
      )
  """

  condition: Optional[Callable[[StepInput], bool]] = None
  true_steps: Optional[Union[Any, List[Any]]] = None
  false_steps: Optional[Union[Any, List[Any]]] = None

  def __post_init__(self) -> None:
    if not self.name:
      self.name = "condition"

  @property
  def step_type(self) -> str:
    return "condition"

  async def execute(
    self,
    step_input: StepInput,
    *,
    event_bus: Optional["EventBus"] = None,
    run_id: str = "",
    workflow_id: str = "",
    workflow_name: str = "",
    step_index: int = 0,
  ) -> StepOutput:
    start = time()

    if self.condition is None:
      raise ValueError(f"Condition step '{self.name}' has no condition callable.")

    # Evaluate condition
    result = await self._evaluate_condition(step_input)

    if result:
      branch_steps = self.true_steps
      branch_name = "true"
    else:
      branch_steps = self.false_steps
      branch_name = "false"

    if branch_steps is None:
      # No steps for this branch — skip
      duration = (time() - start) * 1000
      if event_bus:
        await event_bus.emit(
          StepSkippedEvent(
            run_id=run_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            step_id=self._id,
            step_name=self.name,
            step_type=self.step_type,
            reason=f"No steps for {branch_name} branch.",
          )
        )
      return StepOutput(
        step_name=self.name,
        step_id=self._id,
        step_type=self.step_type,
        status=StepStatus.skipped,
        success=True,
        duration_ms=duration,
      )

    # Normalize branch into executable
    executable: BaseStep = (
      Steps(name=f"{self.name}_{branch_name}", steps=branch_steps) if isinstance(branch_steps, list) else _normalize_step(branch_steps)
    )

    output = await executable.execute(
      step_input,
      event_bus=event_bus,
      run_id=run_id,
      workflow_id=workflow_id,
      workflow_name=workflow_name,
      step_index=step_index,
    )

    duration = (time() - start) * 1000

    return StepOutput(
      step_name=self.name,
      step_id=self._id,
      step_type=self.step_type,
      content=output.content,
      status=output.status,
      success=output.success,
      error=output.error,
      duration_ms=duration,
      steps=[output],
      stop=output.stop,
    )

  async def _evaluate_condition(self, step_input: StepInput) -> bool:
    """Evaluate the condition callable (sync or async)."""
    if self.condition is None:
      return False
    if inspect.iscoroutinefunction(self.condition):
      return await self.condition(step_input)
    return self.condition(step_input)
