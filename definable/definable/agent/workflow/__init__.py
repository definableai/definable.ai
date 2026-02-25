"""Workflow — multi-step agent orchestration.

Usage::

    from definable.agent.workflow import Workflow, Step, Steps, Parallel, Loop, Condition, Router

    workflow = Workflow(
        name="my-workflow",
        steps=[
            Step(name="researcher", agent=researcher_agent),
            Step(name="writer", agent=writer_agent),
        ],
    )
    result = await workflow.arun("Research and write about X")
"""

from definable.agent.workflow.condition import Condition
from definable.agent.workflow.context import StepInput, StepOutput, StepStatus, WorkflowOutput
from definable.agent.workflow.events import (
  BaseWorkflowEvent,
  LoopIterationEvent,
  StepCompletedEvent,
  StepErrorEvent,
  StepSkippedEvent,
  StepStartedEvent,
  WorkflowRunCompletedEvent,
  WorkflowRunErrorEvent,
  WorkflowRunStartedEvent,
)
from definable.agent.workflow.loop import Loop
from definable.agent.workflow.parallel import Parallel
from definable.agent.workflow.router import Router
from definable.agent.workflow.step import BaseStep, Step, Steps
from definable.agent.workflow.workflow import Workflow

__all__ = [
  # Core
  "Workflow",
  "Step",
  "Steps",
  "BaseStep",
  # Control flow
  "Parallel",
  "Loop",
  "Condition",
  "Router",
  # Context
  "StepInput",
  "StepOutput",
  "StepStatus",
  "WorkflowOutput",
  # Events
  "BaseWorkflowEvent",
  "WorkflowRunStartedEvent",
  "WorkflowRunCompletedEvent",
  "WorkflowRunErrorEvent",
  "StepStartedEvent",
  "StepCompletedEvent",
  "StepErrorEvent",
  "StepSkippedEvent",
  "LoopIterationEvent",
]
