"""Evaluation result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalResult:
  """Base evaluation result.

  All eval types return a subclass of this.
  """

  eval_name: str = ""
  success: bool = False
  score: Optional[float] = None
  reason: Optional[str] = None
  metadata: Dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    d: Dict[str, Any] = {
      "eval_name": self.eval_name,
      "success": self.success,
    }
    if self.score is not None:
      d["score"] = self.score
    if self.reason is not None:
      d["reason"] = self.reason
    if self.metadata:
      d["metadata"] = self.metadata
    return d


@dataclass
class AccuracyResult(EvalResult):
  """Result from AccuracyEval — LLM judge scoring.

  Attributes:
      score: 1-10 score from the judge.
      threshold: The minimum score to pass.
      expected: The expected output that was compared against.
      actual: The actual output that was evaluated.
  """

  threshold: float = 7.0
  expected: Optional[str] = None
  actual: Optional[str] = None

  def to_dict(self) -> Dict[str, Any]:
    d = super().to_dict()
    d["threshold"] = self.threshold
    if self.expected is not None:
      d["expected"] = self.expected
    if self.actual is not None:
      d["actual"] = self.actual
    return d


@dataclass
class PerformanceResult(EvalResult):
  """Result from PerformanceEval — runtime + memory profiling.

  Attributes:
      duration_ms: Execution time in milliseconds.
      peak_memory_mb: Peak memory delta during execution (MB).
      duration_threshold_ms: Max allowed duration.
      memory_threshold_mb: Max allowed memory delta.
      runs: Number of profiling runs executed.
      durations: List of individual run durations (ms).
  """

  duration_ms: float = 0.0
  peak_memory_mb: float = 0.0
  duration_threshold_ms: Optional[float] = None
  memory_threshold_mb: Optional[float] = None
  runs: int = 1
  durations: List[float] = field(default_factory=list)

  def to_dict(self) -> Dict[str, Any]:
    d = super().to_dict()
    d["duration_ms"] = self.duration_ms
    d["peak_memory_mb"] = self.peak_memory_mb
    d["runs"] = self.runs
    if self.duration_threshold_ms is not None:
      d["duration_threshold_ms"] = self.duration_threshold_ms
    if self.memory_threshold_mb is not None:
      d["memory_threshold_mb"] = self.memory_threshold_mb
    if self.durations:
      d["durations"] = self.durations
    return d


@dataclass
class ReliabilityResult(EvalResult):
  """Result from ReliabilityEval — tool call verification.

  Attributes:
      expected_tools: Tool names that were expected to be called.
      actual_tools: Tool names that were actually called.
      missing_tools: Expected tools that were NOT called.
      extra_tools: Unexpected tools that WERE called.
      strict: Whether extra tools cause failure.
  """

  expected_tools: List[str] = field(default_factory=list)
  actual_tools: List[str] = field(default_factory=list)
  missing_tools: List[str] = field(default_factory=list)
  extra_tools: List[str] = field(default_factory=list)
  strict: bool = False

  def to_dict(self) -> Dict[str, Any]:
    d = super().to_dict()
    d["expected_tools"] = self.expected_tools
    d["actual_tools"] = self.actual_tools
    d["missing_tools"] = self.missing_tools
    d["extra_tools"] = self.extra_tools
    d["strict"] = self.strict
    return d


@dataclass
class JudgeResult(EvalResult):
  """Result from AgentAsJudgeEval — custom criteria evaluation.

  Attributes:
      criteria: The evaluation criteria that was judged.
      mode: 'numeric' (1-10 + threshold) or 'binary' (pass/fail).
      threshold: Minimum score for 'numeric' mode.
  """

  criteria: str = ""
  mode: str = "numeric"
  threshold: float = 7.0

  def to_dict(self) -> Dict[str, Any]:
    d = super().to_dict()
    d["criteria"] = self.criteria
    d["mode"] = self.mode
    if self.mode == "numeric":
      d["threshold"] = self.threshold
    return d
