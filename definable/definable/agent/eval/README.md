# Eval Module

Evaluate agent quality, performance, and reliability. Provides four built-in eval types (accuracy, performance, reliability, custom judge), a composable `EvalCase`/`EvalSuite` framework, and an abstract base class for building custom evaluations.

---

## Architecture

```
agent/eval/
├── __init__.py       # Unified exports (12 symbols)
├── base.py           # BaseEval (ABC), EvalCase, EvalSuite
├── result.py         # EvalResult, AccuracyResult, PerformanceResult, ReliabilityResult, JudgeResult
├── accuracy.py       # AccuracyEval — LLM judge scoring
├── performance.py    # PerformanceEval — runtime + memory profiling
├── reliability.py    # ReliabilityEval — tool call verification
└── judge.py          # AgentAsJudgeEval — custom criteria evaluation
```

### How It Connects to the Agent

```
EvalCase(input="...", expected="...")
  │
  ▼
BaseEval.arun(agent, case) ──► agent.arun(case.input) ──► EvalResult
  │                                                           │
  │  (batch)                                                  │
  ▼                                                           ▼
BaseEval.arun_batch(agent, [cases]) ──────────────────► EvalSuite
                                                         ├── .total
                                                         ├── .passed
                                                         ├── .failed
                                                         └── .pass_rate
```

All four eval types (Accuracy, Performance, Reliability, AgentAsJudge) follow the same interface.

---

## Quick Start

```python
from definable.agent import Agent
from definable.agent.eval import AccuracyEval, EvalCase

agent = Agent(model="openai/gpt-4o-mini", instructions="You are a math tutor.")

# Single case
eval = AccuracyEval(judge_model="openai/gpt-4o-mini", threshold=7.0)
result = await eval.arun(agent, EvalCase(input="What is 2+2?", expected="4"))
print(result.score, result.success, result.reason)

# Batch
suite = await eval.arun_batch(
  agent,
  [
    EvalCase(input="What is 2+2?", expected="4", name="addition"),
    EvalCase(input="What is 10/3?", expected="3.33", name="division"),
    EvalCase(input="What is sqrt(144)?", expected="12", name="sqrt"),
  ],
)
print(f"Pass rate: {suite.pass_rate:.0%}")  # e.g. "Pass rate: 100%"
```

---

## API Reference

### EvalCase

A single evaluation test case. Passed to any eval's `arun()` or `evaluate()` method.

```python
from definable.agent.eval import EvalCase

case = EvalCase(
  input="What is the capital of France?",  # prompt sent to the agent
  expected="Paris",  # ground truth (for accuracy/judge evals)
  metadata={},  # arbitrary metadata
  name="capitals-france",  # human-readable label
)
```

`expected` is optional -- `PerformanceEval` and `ReliabilityEval` do not require it. `metadata` can carry per-case overrides (e.g. `metadata={"expected_tools": ["search"]}` for `ReliabilityEval`).

---

### EvalSuite

Collection of results from running multiple cases. Returned by `arun_batch()`.

```python
from definable.agent.eval import EvalSuite

suite = EvalSuite(eval_name="accuracy", results=[...])

suite.total  # number of cases
suite.passed  # cases where result.success == True
suite.failed  # total - passed
suite.pass_rate  # passed / total (0.0-1.0)

suite.to_dict()  # serializable dict with all results
```

---

### BaseEval

Abstract base class. All eval types inherit from this and implement `evaluate()`.

```python
from definable.agent.eval import BaseEval, EvalCase
from definable.agent.eval.result import EvalResult


class MyCustomEval(BaseEval):
  name = "custom"

  async def evaluate(self, agent, case: EvalCase) -> EvalResult:
    output = await agent.arun(case.input)
    passed = "keyword" in (output.content or "")
    return EvalResult(
      eval_name=self.name,
      success=passed,
      score=10.0 if passed else 0.0,
      reason="Keyword found" if passed else "Keyword missing",
    )
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `evaluate` | `(agent, case) -> EvalResult` | Abstract. Single case against an agent. |
| `arun` | `(agent, case) -> EvalResult` | Convenience wrapper around `evaluate`. |
| `arun_batch` | `(agent, cases) -> EvalSuite` | Run multiple cases sequentially. |

---

### AccuracyEval

LLM judge that scores agent output against expected output on a 1-10 scale.

```python
from definable.agent.eval import AccuracyEval, EvalCase

eval = AccuracyEval(
  judge_model="openai/gpt-4o-mini",  # model string shorthand or Model instance
  threshold=7.0,  # minimum score to pass (1-10)
)

result = await eval.arun(
  agent,
  EvalCase(
    input="Explain photosynthesis in one sentence.",
    expected="Plants convert sunlight, water, and CO2 into glucose and oxygen.",
  ),
)

# AccuracyResult fields
result.score  # float, 1.0-10.0
result.success  # True if score >= threshold
result.reason  # judge's explanation
result.threshold  # the threshold used
result.expected  # the expected output
result.actual  # the agent's actual output
```

**Scoring rubric** (sent to the judge model):

| Score Range | Meaning |
|-------------|---------|
| 1-3 | Completely wrong or irrelevant |
| 4-5 | Partially correct but missing key information |
| 6-7 | Mostly correct with minor issues |
| 8-9 | Very accurate with only trivial differences |
| 10 | Perfect match in meaning (wording can differ) |

The judge prompt can be customized via the `judge_prompt` parameter. It must contain `{input}`, `{expected}`, and `{actual}` placeholders.

---

### PerformanceEval

Runtime and memory profiling using `tracemalloc`. Runs the agent multiple times and reports p95 duration and peak memory delta.

```python
from definable.agent.eval import PerformanceEval, EvalCase

eval = PerformanceEval(
  duration_threshold_ms=5000,  # max p95 execution time (None = no check)
  memory_threshold_mb=50,  # max peak memory delta (None = no check)
  runs=3,  # number of profiling runs
  warmup_runs=1,  # warmup runs excluded from results
)

result = await eval.arun(agent, EvalCase(input="Complex query"))

# PerformanceResult fields
result.duration_ms  # p95 execution time in milliseconds
result.peak_memory_mb  # peak memory delta across all runs (MB)
result.duration_threshold_ms  # the threshold used (or None)
result.memory_threshold_mb  # the threshold used (or None)
result.runs  # number of profiling runs executed
result.durations  # list of individual run durations (ms)
result.success  # True if both duration and memory are within thresholds
```

**Pass criteria:** The eval passes when both thresholds are met (or when a threshold is `None`, that dimension is skipped). If both are `None`, the eval always passes but still collects profiling data.

---

### ReliabilityEval

Verifies that the agent called the expected tools during execution.

```python
from definable.agent.eval import ReliabilityEval, EvalCase

eval = ReliabilityEval(
  expected_tools=["search_web", "summarize"],  # tools that must be called
  strict=False,  # True = fail on unexpected tools
)

result = await eval.arun(agent, EvalCase(input="Research quantum computing"))

# ReliabilityResult fields
result.expected_tools  # ["search_web", "summarize"]
result.actual_tools  # tools that were actually called
result.missing_tools  # expected but not called
result.extra_tools  # called but not expected
result.strict  # whether strict mode was used
result.success  # True if all expected tools called (and no extras in strict mode)
```

**Modes:**

| Mode | Pass Condition |
|------|----------------|
| Permissive (`strict=False`) | All expected tools were called. Extra tools are OK. |
| Strict (`strict=True`) | All expected tools were called AND no unexpected tools were called. |

**Per-case override:** Set `metadata={"expected_tools": ["tool_a"]}` on an `EvalCase` to override the eval-level `expected_tools` for that specific case.

---

### AgentAsJudgeEval

Flexible evaluation with custom criteria. Supports numeric scoring (1-10) and binary pass/fail modes.

```python
from definable.agent.eval import AgentAsJudgeEval, EvalCase

# Numeric mode (default)
eval = AgentAsJudgeEval(
  criteria="Output must be professional, concise, and include specific data points.",
  judge_model="openai/gpt-4o-mini",
  mode="numeric",
  threshold=8.0,
)
result = await eval.arun(agent, EvalCase(input="Write a quarterly report summary"))

# JudgeResult fields
result.score  # float, 1.0-10.0 (numeric) or 10.0/0.0 (binary)
result.success  # True if score >= threshold (numeric) or passed (binary)
result.reason  # judge's explanation
result.criteria  # the criteria used
result.mode  # "numeric" or "binary"
result.threshold  # threshold (numeric mode only)
```

```python
# Binary mode
eval = AgentAsJudgeEval(
  criteria="Output must NOT contain any profanity or offensive language.",
  mode="binary",
)
result = await eval.arun(agent, EvalCase(input="Respond to a complaint"))
# result.success is True (passed) or False (failed)
```

**Per-case override:** Set `metadata={"criteria": "Custom criteria for this case"}` to override the eval-level criteria.

---

### Result Types

All results inherit from `EvalResult` and add type-specific fields.

```python
from definable.agent.eval import (
  EvalResult,
  AccuracyResult,
  PerformanceResult,
  ReliabilityResult,
  JudgeResult,
)
```

**EvalResult** (base):

| Field | Type | Description |
|-------|------|-------------|
| `eval_name` | `str` | Name of the eval that produced this result |
| `success` | `bool` | Whether the eval passed |
| `score` | `float \| None` | Numeric score (eval-type dependent) |
| `reason` | `str \| None` | Human-readable explanation |
| `metadata` | `dict` | Arbitrary metadata |

All result types implement `to_dict()` for serialization.

**AccuracyResult** adds: `threshold`, `expected`, `actual`

**PerformanceResult** adds: `duration_ms`, `peak_memory_mb`, `duration_threshold_ms`, `memory_threshold_mb`, `runs`, `durations`

**ReliabilityResult** adds: `expected_tools`, `actual_tools`, `missing_tools`, `extra_tools`, `strict`

**JudgeResult** adds: `criteria`, `mode`, `threshold`

---

## Patterns

### Batch Evaluation with Reporting

```python
from definable.agent.eval import AccuracyEval, EvalCase

eval = AccuracyEval(judge_model="openai/gpt-4o-mini", threshold=7.0)

cases = [
  EvalCase(input="What is 2+2?", expected="4", name="arithmetic"),
  EvalCase(input="Capital of Japan?", expected="Tokyo", name="geography"),
  EvalCase(input="Who wrote Hamlet?", expected="Shakespeare", name="literature"),
]

suite = await eval.arun_batch(agent, cases)

print(f"Results: {suite.passed}/{suite.total} passed ({suite.pass_rate:.0%})")
for r in suite.results:
  status = "PASS" if r.success else "FAIL"
  print(f"  [{status}] score={r.score:.1f} — {r.reason}")

# Serialize for CI/CD
import json

print(json.dumps(suite.to_dict(), indent=2))
```

### Multi-Dimensional Evaluation

Combine multiple eval types for a comprehensive assessment:

```python
from definable.agent.eval import (
  AccuracyEval,
  PerformanceEval,
  ReliabilityEval,
  AgentAsJudgeEval,
  EvalCase,
)

case = EvalCase(input="Search for AI news and summarize", expected="A summary of recent AI news")

# Run all evals
accuracy = await AccuracyEval(threshold=7.0).arun(agent, case)
performance = await PerformanceEval(duration_threshold_ms=10000).arun(agent, case)
reliability = await ReliabilityEval(expected_tools=["search_web"]).arun(agent, case)
tone = await AgentAsJudgeEval(criteria="Output must be neutral and factual", mode="binary").arun(agent, case)

print(f"Accuracy:    {'PASS' if accuracy.success else 'FAIL'} (score={accuracy.score})")
print(f"Performance: {'PASS' if performance.success else 'FAIL'} ({performance.duration_ms:.0f}ms)")
print(f"Reliability: {'PASS' if reliability.success else 'FAIL'} (missing={reliability.missing_tools})")
print(f"Tone:        {'PASS' if tone.success else 'FAIL'} ({tone.reason})")
```

### Custom Eval

Extend `BaseEval` for domain-specific evaluation logic:

```python
from definable.agent.eval import BaseEval, EvalCase
from definable.agent.eval import EvalResult


class JSONFormatEval(BaseEval):
  """Verify the agent returns valid JSON."""

  name = "json_format"

  async def evaluate(self, agent, case: EvalCase) -> EvalResult:
    import json

    output = await agent.arun(case.input)
    content = output.content or ""
    try:
      json.loads(content)
      return EvalResult(eval_name=self.name, success=True, score=10.0, reason="Valid JSON")
    except json.JSONDecodeError as e:
      return EvalResult(eval_name=self.name, success=False, score=0.0, reason=f"Invalid JSON: {e}")


# Use it like any other eval
eval = JSONFormatEval()
result = await eval.arun(agent, EvalCase(input="Return a JSON object with name and age"))
suite = await eval.arun_batch(agent, [case1, case2, case3])
```

---

## Gotchas

| Pitfall | Correct Approach |
|---------|------------------|
| `AccuracyEval` without `expected` | Returns `success=False` with reason "No expected output provided." Always set `EvalCase.expected` for accuracy evals. |
| `ReliabilityEval` with empty `expected_tools` | Returns `success=False`. Must specify at least one expected tool. |
| `PerformanceEval` with both thresholds as `None` | Always passes. Useful for profiling-only, but set at least one threshold for a meaningful pass/fail signal. |
| `AgentAsJudgeEval` with empty `criteria` | Returns `success=False`. Must provide criteria at the eval level or per-case via `metadata={"criteria": "..."}`. |
| Judge model API errors | All judge-based evals (Accuracy, AgentAsJudge) return `score=0.0, success=False` with an error reason if the judge model call fails. They do not raise exceptions. |
| `PerformanceEval` and `tracemalloc` | Starts/stops `tracemalloc` on each profiling run. If your code also uses `tracemalloc`, results may conflict. |
| `arun_batch` runs sequentially | Cases execute one at a time. For parallel execution, use `asyncio.gather` with individual `arun()` calls. |
