"""Step — basic execution unit and sequential composition for workflows."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional
from uuid import uuid4

from definable.agent.workflow.context import StepInput, StepOutput, StepStatus
from definable.agent.workflow.events import StepCompletedEvent, StepErrorEvent, StepStartedEvent
from definable.utils.log import log_debug, log_error, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.event_bus import EventBus
  from definable.agent.team.team import Team

# Runtime type alias — concrete types checked via isinstance(step, BaseStep).
WorkflowStep = Any


def _default_input_builder(step_input: StepInput) -> str:
  """Default prompt builder: combines original input with previous step output."""
  parts = []
  if step_input.input:
    parts.append(step_input.input)
  if step_input.previous_step_content:
    parts.append(f"\nPrevious step output:\n{step_input.previous_step_content}")
  return "\n".join(parts) if parts else ""


def _normalize_step(step: Any) -> "BaseStep":
  """Normalize any WorkflowStep into a BaseStep instance.

  Accepts BaseStep subclasses directly, or wraps a callable in a Step.
  """
  if isinstance(step, BaseStep):
    return step
  if callable(step):
    name = getattr(step, "__name__", "callable")
    return Step(name=name, executor=step)
  raise TypeError(f"Invalid step type: {type(step)}. Expected Step, Steps, Parallel, Loop, Condition, Router, or callable.")


@dataclass
class BaseStep:
  """Base class for all workflow step types."""

  name: str = ""
  description: str = ""
  _id: str = field(default_factory=lambda: str(uuid4())[:8])

  @property
  def step_type(self) -> str:
    return "base"

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
    raise NotImplementedError


@dataclass
class Step(BaseStep):
  """A single execution unit — wraps an Agent, Team, or callable.

  Exactly one of ``agent``, ``team``, or ``executor`` must be set.

  Example::

      step = Step(name="researcher", agent=my_agent)
      step = Step(name="processor", executor=my_async_fn)
  """

  agent: Optional["Agent"] = None
  team: Optional["Team"] = None
  executor: Optional[Callable[..., Any]] = None
  input_builder: Optional[Callable[[StepInput], str]] = None
  timeout: Optional[float] = None
  retries: int = 0

  @property
  def step_type(self) -> str:
    return "step"

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
    step_id = self._id

    if event_bus:
      await event_bus.emit(
        StepStartedEvent(
          run_id=run_id,
          workflow_id=workflow_id,
          workflow_name=workflow_name,
          step_id=step_id,
          step_name=self.name,
          step_type=self.step_type,
          step_index=step_index,
        )
      )

    try:
      result = await self._execute_with_retry(step_input)
      duration = (time() - start) * 1000

      content = result.content if hasattr(result, "content") else str(result)
      run_output = result if hasattr(result, "content") else None

      output = StepOutput(
        step_name=self.name,
        step_id=step_id,
        step_type=self.step_type,
        content=content,
        status=StepStatus.completed,
        success=True,
        duration_ms=duration,
        run_output=run_output,
      )

      if event_bus:
        await event_bus.emit(
          StepCompletedEvent(
            run_id=run_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            step_id=step_id,
            step_name=self.name,
            step_type=self.step_type,
            step_index=step_index,
            content=content,
            success=True,
            duration_ms=duration,
          )
        )

      return output

    except Exception as exc:
      duration = (time() - start) * 1000
      error_msg = str(exc)
      log_error(f"Step '{self.name}' failed: {error_msg}")

      if event_bus:
        await event_bus.emit(
          StepErrorEvent(
            run_id=run_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            step_id=step_id,
            step_name=self.name,
            step_type=self.step_type,
            step_index=step_index,
            error=error_msg,
          )
        )

      return StepOutput(
        step_name=self.name,
        step_id=step_id,
        step_type=self.step_type,
        status=StepStatus.failed,
        success=False,
        error=error_msg,
        duration_ms=duration,
      )

  async def _execute_with_retry(self, step_input: StepInput) -> Any:
    """Execute with optional retries."""
    last_exc: Optional[Exception] = None
    attempts = self.retries + 1

    for attempt in range(attempts):
      try:
        return await self._execute_once(step_input)
      except Exception as exc:
        last_exc = exc
        if attempt < attempts - 1:
          log_warning(f"Step '{self.name}' attempt {attempt + 1} failed: {exc}. Retrying...")

    raise last_exc  # type: ignore[misc]

  async def _execute_once(self, step_input: StepInput) -> Any:
    """Execute the step once."""
    prompt = (self.input_builder or _default_input_builder)(step_input)

    if self.agent:
      coro = self.agent.arun(prompt)
    elif self.team:
      coro = self.team.arun(prompt)
    elif self.executor:
      if inspect.iscoroutinefunction(self.executor):
        coro = self.executor(step_input)
      else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.executor, step_input)
    else:
      raise ValueError(f"Step '{self.name}' has no agent, team, or executor configured.")

    if self.timeout:
      return await asyncio.wait_for(coro, timeout=self.timeout)
    return await coro


@dataclass
class Steps(BaseStep):
  """Sequential composition — executes steps in order, chaining context.

  Each step receives the output of the previous step as context.
  Stops early on failure or if a step sets the ``stop`` flag.

  Example::

      seq = Steps(steps=[
          Step(name="draft", agent=drafter),
          Step(name="review", agent=reviewer),
      ])
  """

  steps: List[Any] = field(default_factory=list)

  @property
  def step_type(self) -> str:
    return "steps"

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
    all_outputs: List[StepOutput] = []
    current_input = step_input

    for i, raw_step in enumerate(self.steps):
      step = _normalize_step(raw_step)

      output = await step.execute(
        current_input,
        event_bus=event_bus,
        run_id=run_id,
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        step_index=step_index + i,
      )
      all_outputs.append(output)

      # Early termination
      if output.stop:
        log_debug(f"Early termination requested by step '{step.name}'")
        break

      # If step failed, stop the sequence
      if not output.success:
        log_warning(f"Step '{step.name}' failed, stopping sequence.")
        break

      # Chain context: update input for next step
      current_input = StepInput(
        input=step_input.input,
        previous_step_content=output.content,
        previous_step_outputs={**current_input.previous_step_outputs, output.step_name: output},
        additional_data=step_input.additional_data,
        session_state=step_input.session_state,
      )

    duration = (time() - start) * 1000
    last_output = all_outputs[-1] if all_outputs else None
    all_success = all(o.success for o in all_outputs)

    return StepOutput(
      step_name=self.name or "sequence",
      step_id=self._id,
      step_type=self.step_type,
      content=last_output.content if last_output else None,
      status=StepStatus.completed if all_success else StepStatus.failed,
      success=all_success,
      error=last_output.error if last_output and not last_output.success else None,
      duration_ms=duration,
      steps=all_outputs,
      stop=last_output.stop if last_output else False,
    )
