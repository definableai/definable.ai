"""BaseEval — abstract base class for all evaluation types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from definable.agent.eval.result import EvalResult
from definable.utils.log import log_info

if TYPE_CHECKING:
  from definable.agent.agent import Agent


@dataclass
class EvalCase:
  """A single evaluation test case.

  Attributes:
      input: The prompt to send to the agent/team.
      expected: Expected output (for accuracy/judge evals).
      metadata: Arbitrary metadata attached to this case.
      name: Optional human-readable name for the case.
  """

  input: str  # noqa: A003
  expected: Optional[str] = None
  metadata: Dict[str, Any] = field(default_factory=dict)
  name: str = ""


@dataclass
class EvalSuite:
  """Collection of eval results from running multiple cases.

  Attributes:
      eval_name: Name of the eval that produced these results.
      results: Individual results per case.
      passed: Number of cases that passed.
      failed: Number of cases that failed.
      total: Total number of cases.
      pass_rate: Fraction of cases that passed (0.0–1.0).
  """

  eval_name: str = ""
  results: List[EvalResult] = field(default_factory=list)

  @property
  def total(self) -> int:
    return len(self.results)

  @property
  def passed(self) -> int:
    return sum(1 for r in self.results if r.success)

  @property
  def failed(self) -> int:
    return self.total - self.passed

  @property
  def pass_rate(self) -> float:
    return self.passed / self.total if self.total > 0 else 0.0

  def to_dict(self) -> Dict[str, Any]:
    return {
      "eval_name": self.eval_name,
      "total": self.total,
      "passed": self.passed,
      "failed": self.failed,
      "pass_rate": self.pass_rate,
      "results": [r.to_dict() for r in self.results],
    }


class BaseEval(ABC):
  """Abstract base class for evaluations.

  Subclasses must implement ``evaluate()`` which runs a single
  :class:`EvalCase` against an agent or team and returns an
  :class:`EvalResult`.

  Usage::

      eval = AccuracyEval(judge_model="openai/gpt-4o-mini")
      result = await eval.arun(agent, EvalCase(input="...", expected="..."))

      # Or run a batch:
      suite = await eval.arun_batch(agent, [case1, case2, case3])
      print(suite.pass_rate)
  """

  name: str = "base"
  _id: str = ""

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)

  def __post_init__(self) -> None:
    if not self._id:
      self._id = str(uuid4())

  @abstractmethod
  async def evaluate(
    self,
    agent: "Agent",
    case: EvalCase,
  ) -> EvalResult:
    """Run a single evaluation case against an agent.

    Args:
        agent: The agent to evaluate.
        case: The test case.

    Returns:
        An EvalResult (or subclass).
    """
    ...

  async def arun(
    self,
    agent: "Agent",
    case: EvalCase,
  ) -> EvalResult:
    """Execute a single eval case. Convenience wrapper around evaluate()."""
    return await self.evaluate(agent, case)

  async def arun_batch(
    self,
    agent: "Agent",
    cases: List[EvalCase],
  ) -> EvalSuite:
    """Run multiple eval cases sequentially and return an EvalSuite.

    Args:
        agent: The agent to evaluate.
        cases: List of test cases.

    Returns:
        EvalSuite with all results.
    """
    if not cases:
      raise ValueError("arun_batch requires at least one EvalCase.")
    suite = EvalSuite(eval_name=self.name)
    for i, case in enumerate(cases):
      log_info(f"[{self.name}] Running case {i + 1}/{len(cases)}: {case.name or case.input[:50]}")
      result = await self.evaluate(agent, case)
      suite.results.append(result)
    return suite
