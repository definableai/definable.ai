from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class NextAction(str, Enum):
  CONTINUE = "continue"
  VALIDATE = "validate"
  FINAL_ANSWER = "final_answer"
  RESET = "reset"


class ReasoningStep(BaseModel):
  title: Optional[str] = Field(None, description="A concise title summarizing the step's purpose")
  action: Optional[str] = Field(None, description="The action derived from this step. Talk in first person like I will ...")
  result: Optional[str] = Field(None, description="The result of executing the action. Talk in first person like I did this and got ... ")
  reasoning: Optional[str] = Field(None, description="The thought process and considerations behind this step")
  next_action: Optional[NextAction] = Field(
    None,
    description="Indicates whether to continue reasoning, validate the provided result, or confirm that the result is the final answer",
  )
  confidence: Optional[float] = Field(None, description="Confidence score for this step (0.0 to 1.0)")


class ReasoningSteps(BaseModel):
  reasoning_steps: List[ReasoningStep] = Field(..., description="A list of reasoning steps")


class ThinkingOutput(BaseModel):
  """Compact output from the context-aware thinking phase."""

  analysis: str = Field(..., description="1-2 sentence analysis of what the user needs.")
  approach: str = Field(..., description="1-2 sentence plan for how to respond.")
  tool_plan: Optional[List[str]] = Field(
    None,
    description="Ordered list of tool names to use (from the provided catalog). Null if no tools needed.",
  )


def thinking_output_to_reasoning_steps(output: ThinkingOutput) -> List[ReasoningStep]:
  """Map ThinkingOutput to legacy ReasoningStep list for backward compatibility."""
  steps = [
    ReasoningStep(
      title="Analysis",
      reasoning=output.analysis,
      action=output.approach,
      result=None,
      next_action=NextAction.CONTINUE if output.tool_plan else NextAction.FINAL_ANSWER,
      confidence=None,
    )
  ]
  if output.tool_plan:
    steps.append(
      ReasoningStep(
        title="Tool Plan",
        reasoning=f"Tools to use: {', '.join(output.tool_plan)}",
        action=f"Execute tools in order: {', '.join(output.tool_plan)}",
        result=None,
        next_action=NextAction.FINAL_ANSWER,
        confidence=None,
      )
    )
  return steps
