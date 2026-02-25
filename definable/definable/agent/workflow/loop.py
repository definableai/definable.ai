"""Loop — iterative step execution with end condition."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from definable.agent.workflow.context import StepInput, StepOutput, StepStatus
from definable.agent.workflow.events import LoopIterationEvent
from definable.agent.workflow.step import BaseStep, Steps
from definable.utils.log import log_debug

if TYPE_CHECKING:
  from definable.agent.event_bus import EventBus


@dataclass
class Loop(BaseStep):
  """Iterative execution — runs steps repeatedly until end condition or max iterations.

  The ``end_condition`` callable receives the list of all iteration outputs
  and returns ``True`` to stop iterating.

  Example::

      loop = Loop(
          name="improve",
          steps=[
              Step(name="generate", agent=generator),
              Step(name="evaluate", agent=evaluator),
          ],
          end_condition=lambda outputs: any("APPROVED" in (o.content or "") for o in outputs),
          max_iterations=5,
      )
  """

  steps: List[Any] = field(default_factory=list)
  end_condition: Optional[Callable[[List[StepOutput]], bool]] = None
  max_iterations: int = 3

  def __post_init__(self) -> None:
    if not self.name:
      self.name = "loop"

  @property
  def step_type(self) -> str:
    return "loop"

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
    all_iteration_outputs: List[StepOutput] = []
    current_input = step_input

    for iteration in range(self.max_iterations):
      if event_bus:
        await event_bus.emit(
          LoopIterationEvent(
            run_id=run_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            step_name=self.name,
            iteration=iteration,
            max_iterations=self.max_iterations,
            should_continue=True,
          )
        )

      # Execute inner steps as a sequence
      inner = Steps(name=f"{self.name}_iter_{iteration}", steps=self.steps)
      output = await inner.execute(
        current_input,
        event_bus=event_bus,
        run_id=run_id,
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        step_index=step_index,
      )
      all_iteration_outputs.append(output)

      # Check stop flag
      if output.stop:
        log_debug(f"Loop '{self.name}' stopped by step at iteration {iteration}")
        break

      # Check failure — stop looping on failure
      if not output.success:
        log_debug(f"Loop '{self.name}' stopped due to failure at iteration {iteration}")
        break

      # Check end condition
      if self.end_condition:
        should_stop = await self._evaluate_condition(all_iteration_outputs)
        if should_stop:
          log_debug(f"Loop '{self.name}' end condition met at iteration {iteration}")
          break

      # Chain context for next iteration
      current_input = StepInput(
        input=step_input.input,
        previous_step_content=output.content,
        previous_step_outputs={
          **current_input.previous_step_outputs,
          f"{self.name}_iter_{iteration}": output,
        },
        additional_data=step_input.additional_data,
        session_state=step_input.session_state,
      )

    duration = (time() - start) * 1000
    last_output = all_iteration_outputs[-1] if all_iteration_outputs else None
    all_success = all(o.success for o in all_iteration_outputs)

    return StepOutput(
      step_name=self.name,
      step_id=self._id,
      step_type=self.step_type,
      content=last_output.content if last_output else None,
      status=StepStatus.completed if all_success else StepStatus.failed,
      success=all_success,
      duration_ms=duration,
      steps=all_iteration_outputs,
    )

  async def _evaluate_condition(self, outputs: List[StepOutput]) -> bool:
    """Evaluate the end condition, handling both sync and async callables."""
    if self.end_condition is None:
      return False
    if inspect.iscoroutinefunction(self.end_condition):
      return await self.end_condition(outputs)
    return self.end_condition(outputs)
