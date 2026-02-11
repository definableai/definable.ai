# reasoning

Structured reasoning step types for chain-of-thought traces.

## Overview

This module defines the data models used to represent structured reasoning workflows. Each `ReasoningStep` captures a step in a chain-of-thought process with an action, result, reasoning, and a decision on what to do next. This is a single-file module (`step.py`) with no `__init__.py`.

## API Reference

### ReasoningStep

```python
from definable.reasoning.step import ReasoningStep
```

| Field | Type | Description |
|-------|------|-------------|
| `title` | `Optional[str]` | Step title |
| `action` | `Optional[str]` | Action taken |
| `result` | `Optional[str]` | Result of the action |
| `reasoning` | `Optional[str]` | Reasoning behind the action |
| `next_action` | `Optional[NextAction]` | What to do next |
| `confidence` | `Optional[float]` | Confidence score |

### ReasoningSteps

```python
from definable.reasoning.step import ReasoningSteps
```

A container for a list of `ReasoningStep` objects:

| Field | Type | Description |
|-------|------|-------------|
| `reasoning_steps` | `List[ReasoningStep]` | Ordered list of reasoning steps |

### NextAction

```python
from definable.reasoning.step import NextAction
```

Enum controlling the reasoning flow:

| Value | Description |
|-------|-------------|
| `CONTINUE` | Continue to the next step |
| `VALIDATE` | Validate the current result |
| `FINAL_ANSWER` | The reasoning is complete |
| `RESET` | Restart the reasoning process |

## See Also

- `agents/` — Agent streaming emits `ReasoningStepEvent` and `ReasoningContentDeltaEvent`
- `run/` — `RunOutput.reasoning_steps` and `RunOutput.reasoning_content`
