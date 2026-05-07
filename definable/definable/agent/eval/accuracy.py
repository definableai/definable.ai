"""AccuracyEval — LLM judge that scores agent output against expected output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from definable.agent.eval.base import BaseEval, EvalCase
from definable.agent.eval.result import AccuracyResult
from definable.model.message import Message
from definable.model.utils import get_model
from definable.utils.log import log_error, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.model.base import Model

ACCURACY_JUDGE_PROMPT = """\
You are an evaluation judge. Score the ACTUAL output against the EXPECTED output.

TASK INPUT: {input}

EXPECTED OUTPUT:
{expected}

ACTUAL OUTPUT:
{actual}

Score from 1-10 where:
- 1-3: Completely wrong or irrelevant
- 4-5: Partially correct but missing key information
- 6-7: Mostly correct with minor issues
- 8-9: Very accurate with only trivial differences
- 10: Perfect match in meaning (wording can differ)

Respond with ONLY a JSON object:
{{"score": <number>, "reason": "<brief explanation>"}}
"""


@dataclass
class AccuracyEval(BaseEval):
  """Evaluate agent output accuracy using an LLM judge.

  The judge model scores the agent's actual output against the expected
  output on a 1-10 scale. The eval passes if the score meets the threshold.

  Args:
      judge_model: Model to use as the judge. String shorthand or Model instance.
                   Defaults to "openai/gpt-4o-mini".
      threshold: Minimum score (1-10) to pass. Default 7.0.
      judge_prompt: Custom judge prompt template. Must contain {input}, {expected}, {actual}.

  Example::

      eval = AccuracyEval(judge_model="openai/gpt-4o-mini", threshold=8.0)
      result = await eval.arun(agent, EvalCase(input="What is 2+2?", expected="4"))
      print(result.score, result.success)
  """

  name: str = "accuracy"
  judge_model: Optional[str] = "openai/gpt-4o-mini"
  threshold: float = 7.0
  judge_prompt: str = ACCURACY_JUDGE_PROMPT
  _id: str = field(default_factory=lambda: str(uuid4()))
  _judge: Optional["Model"] = field(default=None, repr=False)

  async def evaluate(self, agent: "Agent", case: EvalCase) -> AccuracyResult:
    """Run the agent, then judge the output against expected."""
    if case.expected is None or (isinstance(case.expected, str) and not case.expected.strip()):
      return AccuracyResult(
        eval_name=self.name,
        success=False,
        reason="No expected output provided (None or empty).",
        threshold=self.threshold,
        actual=None,
      )

    # Run the agent
    try:
      output = await agent.arun(case.input)
      actual = output.content or ""
    except Exception as e:
      log_error(f"AccuracyEval: agent.arun() failed: {e}")
      return AccuracyResult(
        eval_name=self.name,
        success=False,
        reason=f"Agent execution failed: {e}",
        threshold=self.threshold,
      )

    # Judge
    score, reason = await self._judge_output(case.input, case.expected, actual)

    return AccuracyResult(
      eval_name=self.name,
      score=score,
      success=score >= self.threshold,
      reason=reason,
      threshold=self.threshold,
      expected=case.expected,
      actual=actual,
    )

  async def _judge_output(self, input_text: str, expected: str, actual: str) -> tuple[float, str]:
    """Ask the judge model to score actual vs expected."""
    if self._judge is None:
      self._judge = get_model(self.judge_model)

    if self._judge is None:
      log_warning("AccuracyEval: No judge model available. Defaulting to score=0.")
      return 0.0, "No judge model configured."

    prompt = self.judge_prompt.format(input=input_text, expected=expected, actual=actual)

    try:
      response = await self._judge.aresponse(messages=[Message(role="user", content=prompt)])
      content = response.content or ""
      return self._parse_judge_response(content)
    except Exception as e:
      log_error(f"AccuracyEval: judge call failed: {e}")
      return 0.0, f"Judge model call failed: {e}"

  def _parse_judge_response(self, content: str) -> tuple[float, str]:
    """Parse the judge's JSON response to extract score and reason."""
    import json

    # Try JSON parse first
    try:
      # Strip markdown code fences if present
      cleaned = content.strip()
      if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
      data = json.loads(cleaned)
      score = float(data.get("score", 0))
      reason = str(data.get("reason", ""))
      return max(1.0, min(10.0, score)), reason
    except (json.JSONDecodeError, ValueError, KeyError):
      pass

    # Fallback: extract score with regex
    score_match = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', content)
    if score_match:
      score = float(score_match.group(1))
      return max(1.0, min(10.0, score)), content[:200]

    log_warning(f"AccuracyEval: Could not parse judge response: {content[:200]}")
    return 0.0, f"Unparseable judge response: {content[:200]}"
