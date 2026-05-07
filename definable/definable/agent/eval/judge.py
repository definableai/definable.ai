"""AgentAsJudgeEval — custom criteria evaluation with an LLM judge."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from definable.agent.eval.base import BaseEval, EvalCase
from definable.agent.eval.result import JudgeResult
from definable.model.message import Message
from definable.model.utils import get_model
from definable.utils.log import log_error, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.model.base import Model

NUMERIC_JUDGE_PROMPT = """\
You are an evaluation judge. Assess the following output against the given criteria.

TASK INPUT: {input}

AGENT OUTPUT:
{output}

EVALUATION CRITERIA:
{criteria}

Score from 1-10 where:
- 1-3: Fails to meet criteria
- 4-5: Partially meets criteria
- 6-7: Mostly meets criteria
- 8-9: Strongly meets criteria
- 10: Perfectly meets criteria

Respond with ONLY a JSON object:
{{"score": <number>, "reason": "<brief explanation>"}}
"""

BINARY_JUDGE_PROMPT = """\
You are an evaluation judge. Assess the following output against the given criteria.

TASK INPUT: {input}

AGENT OUTPUT:
{output}

EVALUATION CRITERIA:
{criteria}

Does the output meet the criteria? Respond with ONLY a JSON object:
{{"pass": true or false, "reason": "<brief explanation>"}}
"""


@dataclass
class AgentAsJudgeEval(BaseEval):
  """Evaluate agent output against custom criteria using an LLM judge.

  Supports two modes:
  - **numeric**: Score 1-10 with a threshold (default)
  - **binary**: Pass/fail judgment

  Args:
      criteria: The evaluation criteria (natural language).
      judge_model: Model to use as the judge. Default "openai/gpt-4o-mini".
      mode: 'numeric' or 'binary'. Default 'numeric'.
      threshold: Minimum score for 'numeric' mode. Default 7.0.

  Example::

      eval = AgentAsJudgeEval(
          criteria="Output must be professional, concise, and include specific data.",
          mode="numeric",
          threshold=8.0,
      )
      result = await eval.arun(agent, EvalCase(input="Write a summary"))
      print(result.score, result.reason)

      # Binary mode:
      eval = AgentAsJudgeEval(
          criteria="Output must NOT contain any profanity.",
          mode="binary",
      )
  """

  name: str = "judge"
  criteria: str = ""
  judge_model: Optional[str] = "openai/gpt-4o-mini"
  mode: str = "numeric"  # "numeric" or "binary"
  threshold: float = 7.0
  _id: str = field(default_factory=lambda: str(uuid4()))
  _judge: Optional["Model"] = field(default=None, repr=False)

  async def evaluate(self, agent: "Agent", case: EvalCase) -> JudgeResult:
    """Run the agent, then judge the output against criteria."""
    return await self._judge_run(agent, case)

  async def _judge_run(
    self,
    agent: "Agent",
    case: EvalCase,
  ) -> JudgeResult:
    """Core judge logic."""
    criteria = case.metadata.get("criteria", self.criteria)
    if not criteria:
      return JudgeResult(
        eval_name=self.name,
        success=False,
        reason="No evaluation criteria specified.",
        criteria="",
        mode=self.mode,
        threshold=self.threshold,
      )

    # Run agent
    try:
      output = await agent.arun(case.input)
      actual = output.content or ""
    except Exception as e:
      log_error(f"AgentAsJudgeEval: execution failed: {e}")
      return JudgeResult(
        eval_name=self.name,
        success=False,
        reason=f"Execution failed: {e}",
        criteria=criteria,
        mode=self.mode,
      )

    # Judge
    if self.mode == "binary":
      return await self._judge_binary(case.input, actual, criteria)
    else:
      return await self._judge_numeric(case.input, actual, criteria)

  async def _judge_numeric(self, input_text: str, output: str, criteria: str) -> JudgeResult:
    """Numeric mode: score 1-10."""
    if self._judge is None:
      self._judge = get_model(self.judge_model)

    if self._judge is None:
      log_warning("AgentAsJudgeEval: No judge model available.")
      return JudgeResult(
        eval_name=self.name,
        success=False,
        reason="No judge model configured.",
        criteria=criteria,
        mode="numeric",
        threshold=self.threshold,
      )

    prompt = NUMERIC_JUDGE_PROMPT.format(input=input_text, output=output, criteria=criteria)

    try:
      response = await self._judge.aresponse(messages=[Message(role="user", content=prompt)])
      content = response.content or ""
      score, reason = self._parse_numeric(content)
      return JudgeResult(
        eval_name=self.name,
        score=score,
        success=score >= self.threshold,
        reason=reason,
        criteria=criteria,
        mode="numeric",
        threshold=self.threshold,
      )
    except Exception as e:
      log_error(f"AgentAsJudgeEval: judge call failed: {e}")
      return JudgeResult(
        eval_name=self.name,
        success=False,
        reason=f"Judge call failed: {e}",
        criteria=criteria,
        mode="numeric",
        threshold=self.threshold,
      )

  async def _judge_binary(self, input_text: str, output: str, criteria: str) -> JudgeResult:
    """Binary mode: pass/fail."""
    if self._judge is None:
      self._judge = get_model(self.judge_model)

    if self._judge is None:
      log_warning("AgentAsJudgeEval: No judge model available.")
      return JudgeResult(
        eval_name=self.name,
        success=False,
        reason="No judge model configured.",
        criteria=criteria,
        mode="binary",
      )

    prompt = BINARY_JUDGE_PROMPT.format(input=input_text, output=output, criteria=criteria)

    try:
      response = await self._judge.aresponse(messages=[Message(role="user", content=prompt)])
      content = response.content or ""
      passed, reason = self._parse_binary(content)
      return JudgeResult(
        eval_name=self.name,
        score=10.0 if passed else 0.0,
        success=passed,
        reason=reason,
        criteria=criteria,
        mode="binary",
      )
    except Exception as e:
      log_error(f"AgentAsJudgeEval: judge call failed: {e}")
      return JudgeResult(
        eval_name=self.name,
        success=False,
        reason=f"Judge call failed: {e}",
        criteria=criteria,
        mode="binary",
      )

  def _parse_numeric(self, content: str) -> tuple[float, str]:
    """Parse numeric judge response."""
    import json

    try:
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

    score_match = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', content)
    if score_match:
      score = float(score_match.group(1))
      return max(1.0, min(10.0, score)), content[:200]

    log_warning(f"AgentAsJudgeEval: Could not parse response: {content[:200]}")
    return 0.0, f"Unparseable response: {content[:200]}"

  def _parse_binary(self, content: str) -> tuple[bool, str]:
    """Parse binary judge response."""
    import json

    try:
      cleaned = content.strip()
      if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
      data = json.loads(cleaned)
      passed = bool(data.get("pass", False))
      reason = str(data.get("reason", ""))
      return passed, reason
    except (json.JSONDecodeError, ValueError, KeyError):
      pass

    # Fallback
    lower = content.lower()
    if '"pass": true' in lower or '"pass":true' in lower:
      return True, content[:200]
    if '"pass": false' in lower or '"pass":false' in lower:
      return False, content[:200]

    log_warning(f"AgentAsJudgeEval: Could not parse response: {content[:200]}")
    return False, f"Unparseable response: {content[:200]}"
