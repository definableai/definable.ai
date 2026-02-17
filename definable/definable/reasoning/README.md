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

### ThinkingOutput

```python
from definable.reasoning.step import ThinkingOutput
```

Compact output from the context-aware thinking phase, used when the agent's thinking layer is enabled.

| Field | Type | Description |
|-------|------|-------------|
| `analysis` | `str` | 1-2 sentence analysis of what the user needs |
| `approach` | `str` | 1-2 sentence plan for how to respond |
| `tool_plan` | `Optional[List[str]]` | Ordered tool names to use (null if no tools needed) |

### thinking_output_to_reasoning_steps()

```python
from definable.reasoning.step import thinking_output_to_reasoning_steps
```

Maps a `ThinkingOutput` to a `List[ReasoningStep]` for backward compatibility. The thinking layer uses `ThinkingOutput` internally but exposes results as `ReasoningStep` objects on `RunOutput.reasoning_steps`.

## See Also

- `agents/` — Agent streaming emits `ReasoningStepEvent` and `ReasoningContentDeltaEvent`
- `run/` — `RunOutput.reasoning_steps` and `RunOutput.reasoning_content`
