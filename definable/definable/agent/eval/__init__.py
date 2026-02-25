"""Definable Evaluation — assess agent quality, performance, and reliability.

Example::

    from definable.agent.eval import AccuracyEval, PerformanceEval, ReliabilityEval, EvalCase

    # Accuracy: LLM judge scores output against expected
    eval = AccuracyEval(judge_model="openai/gpt-4o-mini", threshold=8.0)
    result = await eval.arun(agent, EvalCase(input="What is 2+2?", expected="4"))

    # Performance: runtime + memory profiling
    eval = PerformanceEval(duration_threshold_ms=5000, runs=3)
    result = await eval.arun(agent, EvalCase(input="Complex query"))

    # Reliability: tool call verification
    eval = ReliabilityEval(expected_tools=["search_web"])
    result = await eval.arun(agent, EvalCase(input="Search for AI news"))

    # Custom judge: flexible criteria
    eval = AgentAsJudgeEval(criteria="Output must be concise", mode="binary")
    result = await eval.arun(agent, EvalCase(input="Summarize this"))
"""

from definable.agent.eval.accuracy import AccuracyEval
from definable.agent.eval.base import BaseEval, EvalCase, EvalSuite
from definable.agent.eval.judge import AgentAsJudgeEval
from definable.agent.eval.performance import PerformanceEval
from definable.agent.eval.reliability import ReliabilityEval
from definable.agent.eval.result import (
  AccuracyResult,
  EvalResult,
  JudgeResult,
  PerformanceResult,
  ReliabilityResult,
)

__all__ = [
  # Base
  "BaseEval",
  "EvalCase",
  "EvalSuite",
  # Eval types
  "AccuracyEval",
  "PerformanceEval",
  "ReliabilityEval",
  "AgentAsJudgeEval",
  # Results
  "EvalResult",
  "AccuracyResult",
  "PerformanceResult",
  "ReliabilityResult",
  "JudgeResult",
]
