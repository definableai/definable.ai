"""Router — dynamic N-way routing in workflows."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from definable.agent.workflow.context import StepInput, StepOutput, StepStatus
from definable.agent.workflow.step import BaseStep, _normalize_step
from definable.utils.log import log_warning

if TYPE_CHECKING:
  from definable.agent.event_bus import EventBus


@dataclass
class Router(BaseStep):
  """Dynamic routing — selects which step(s) to execute based on a selector function.

  The ``selector`` callable receives :class:`StepInput` and returns one or
  more route names (strings) that map into the ``routes`` dict.

  Example::

      router = Router(
          name="support",
          selector=lambda ctx: "technical" if "bug" in (ctx.input or "") else "general",
          routes={
              "technical": Step(name="tech_support", agent=tech_agent),
              "general": Step(name="general_support", agent=general_agent),
          },
      )
  """

  selector: Optional[Callable[[StepInput], Union[str, List[str]]]] = None
  routes: Dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.name:
      self.name = "router"

  @property
  def step_type(self) -> str:
    return "router"

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

    if self.selector is None:
      raise ValueError(f"Router step '{self.name}' has no selector callable.")

    # Evaluate selector
    selected = await self._evaluate_selector(step_input)

    # Normalize to list
    if isinstance(selected, str):
      selected = [selected]

    # Execute selected routes
    all_outputs: List[StepOutput] = []
    for route_name in selected:
      if route_name not in self.routes:
        log_warning(f"Router '{self.name}': route '{route_name}' not found. Available: {list(self.routes.keys())}")
        all_outputs.append(
          StepOutput(
            step_name=route_name,
            step_type="router_route",
            status=StepStatus.failed,
            success=False,
            error=f"Route '{route_name}' not found.",
          )
        )
        continue

      step = _normalize_step(self.routes[route_name])
      output = await step.execute(
        step_input,
        event_bus=event_bus,
        run_id=run_id,
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        step_index=step_index,
      )
      all_outputs.append(output)

    duration = (time() - start) * 1000

    if not all_outputs:
      return StepOutput(
        step_name=self.name,
        step_id=self._id,
        step_type=self.step_type,
        status=StepStatus.skipped,
        success=True,
        duration_ms=duration,
      )

    # Single route: use its output directly
    if len(all_outputs) == 1:
      result = all_outputs[0]
      return StepOutput(
        step_name=self.name,
        step_id=self._id,
        step_type=self.step_type,
        content=result.content,
        status=result.status,
        success=result.success,
        error=result.error,
        duration_ms=duration,
        steps=all_outputs,
        stop=result.stop,
      )

    # Multiple routes: combine
    all_success = all(o.success for o in all_outputs)
    contents = [f"[{o.step_name}]: {o.content}" for o in all_outputs if o.content]
    combined = "\n\n".join(contents) if contents else None

    return StepOutput(
      step_name=self.name,
      step_id=self._id,
      step_type=self.step_type,
      content=combined,
      status=StepStatus.completed if all_success else StepStatus.failed,
      success=all_success,
      duration_ms=duration,
      steps=all_outputs,
    )

  async def _evaluate_selector(self, step_input: StepInput) -> Union[str, List[str]]:
    """Evaluate the selector callable (sync or async)."""
    if self.selector is None:
      return []
    if inspect.iscoroutinefunction(self.selector):
      return await self.selector(step_input)
    return self.selector(step_input)
