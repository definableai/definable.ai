from enum import Enum
from typing import List, Literal, Optional, Union

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


class ToolStep(BaseModel):
  """A single step in the tool execution plan."""

  tool: str = Field(..., description="Tool name from the available catalog.")
  reason: str = Field(..., description="Why this tool is needed and what information it provides.")
  depends_on: Optional[List[str]] = Field(
    None,
    description="Tool names that must complete before this one. Null if independent.",
  )


class ThinkingOutput(BaseModel):
  """Output from the agent's thinking phase.

  Fields are ordered so that reasoning comes before conclusions — this ordering
  improves accuracy by ~8% when models generate structured output (the model
  reasons through ``chain_of_thought`` before committing to ``approach``).
  """

  # PRIMARY: free-form reasoning FIRST (field ordering matters for accuracy)
  chain_of_thought: str = Field(
    ...,
    description="Step-by-step reasoning process. Work through the problem before concluding.",
  )

  # CONCLUSIONS: derived from the reasoning above
  approach: str = Field(..., description="Concise plan: what to do and how.")
  tool_plan: Optional[List[Union[str, ToolStep]]] = Field(
    None,
    description="Ordered tool execution plan. Each entry is a tool name or a ToolStep with reason and dependencies. Null if no tools needed.",
  )

  # OPTIONAL: effort-dependent fields
  verification: Optional[str] = Field(
    None,
    description="Self-check: verify reasoning against the original request. Flag gaps or errors.",
  )
  considerations: Optional[str] = Field(
    None,
    description="Risks, trade-offs, edge cases, or alternative approaches.",
  )
  confidence: Optional[Literal["low", "medium", "high"]] = Field(
    None,
    description="Confidence in this plan.",
  )

  # BACKWARD COMPAT: 'analysis' alias for 'chain_of_thought'
  @property
  def analysis(self) -> str:
    """Backward-compatible alias for chain_of_thought."""
    return self.chain_of_thought

  def flat_tool_names(self) -> List[str]:
    """Extract flat list of tool names from tool_plan (handles both str and ToolStep)."""
    if not self.tool_plan:
      return []
    names: List[str] = []
    for entry in self.tool_plan:
      if isinstance(entry, str):
        names.append(entry)
      elif isinstance(entry, ToolStep):
        names.append(entry.tool)
    return names


def thinking_output_to_reasoning_steps(output: ThinkingOutput) -> List[ReasoningStep]:
  """Map ThinkingOutput to legacy ReasoningStep list for backward compatibility."""
  tool_names = output.flat_tool_names()

  steps = [
    ReasoningStep(
      title="Chain of Thought",
      reasoning=output.chain_of_thought,
      action=output.approach,
      result=None,
      next_action=NextAction.CONTINUE if tool_names else NextAction.FINAL_ANSWER,
      confidence=None,
    )
  ]
  if tool_names:
    steps.append(
      ReasoningStep(
        title="Tool Plan",
        reasoning=f"Tools to use: {', '.join(tool_names)}",
        action=f"Execute tools in order: {', '.join(tool_names)}",
        result=None,
        next_action=NextAction.FINAL_ANSWER if not output.considerations else NextAction.CONTINUE,
        confidence=None,
      )
    )
  if output.verification:
    steps.append(
      ReasoningStep(
        title="Verification",
        reasoning=output.verification,
        action=None,
        result=None,
        next_action=NextAction.FINAL_ANSWER if not output.considerations else NextAction.CONTINUE,
        confidence=None,
      )
    )
  if output.considerations:
    steps.append(
      ReasoningStep(
        title="Considerations",
        reasoning=output.considerations,
        action=None,
        result=None,
        next_action=NextAction.FINAL_ANSWER,
        confidence=None,
      )
    )
  return steps
