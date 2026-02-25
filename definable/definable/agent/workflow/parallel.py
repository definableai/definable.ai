"""Parallel — concurrent step execution in workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, Any, List, Optional

from definable.agent.workflow.context import StepInput, StepOutput, StepStatus
from definable.agent.workflow.step import BaseStep, _normalize_step

if TYPE_CHECKING:
  from definable.agent.event_bus import EventBus


@dataclass
class Parallel(BaseStep):
  """Execute multiple steps concurrently using asyncio.gather.

  All steps receive the same input (no chaining). Results are collected
  and combined into a single output with nested step outputs.

  Example::

      par = Parallel(name="analysis", steps=[
          Step(name="technical", agent=tech_agent),
          Step(name="business", agent=biz_agent),
      ])
  """

  steps: List[Any] = field(default_factory=list)
  max_concurrency: Optional[int] = None

  def __post_init__(self) -> None:
    if not self.name:
      self.name = "parallel"

  @property
  def step_type(self) -> str:
    return "parallel"

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
    semaphore = asyncio.Semaphore(self.max_concurrency) if self.max_concurrency else None

    async def run_one(raw_step: Any, idx: int) -> StepOutput:
      step = _normalize_step(raw_step)
      if semaphore:
        async with semaphore:
          return await step.execute(
            step_input,
            event_bus=event_bus,
            run_id=run_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            step_index=step_index + idx,
          )
      return await step.execute(
        step_input,
        event_bus=event_bus,
        run_id=run_id,
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        step_index=step_index + idx,
      )

    tasks = [run_one(step, i) for i, step in enumerate(self.steps)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_outputs: List[StepOutput] = []
    for i, result in enumerate(results):
      if isinstance(result, BaseException):
        step = _normalize_step(self.steps[i])
        all_outputs.append(
          StepOutput(
            step_name=step.name,
            step_id=step._id,
            step_type=step.step_type,
            status=StepStatus.failed,
            success=False,
            error=str(result),
          )
        )
      else:
        all_outputs.append(result)

    duration = (time() - start) * 1000
    all_success = all(o.success for o in all_outputs)

    # Combine content from all parallel steps
    contents = [f"[{o.step_name}]: {o.content}" for o in all_outputs if o.content]
    combined_content = "\n\n".join(contents) if contents else None

    return StepOutput(
      step_name=self.name,
      step_id=self._id,
      step_type=self.step_type,
      content=combined_content,
      status=StepStatus.completed if all_success else StepStatus.failed,
      success=all_success,
      duration_ms=duration,
      steps=all_outputs,
    )
