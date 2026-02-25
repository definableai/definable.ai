# agent/reasoning — Structured Thinking Layer

The reasoning module gives an agent an inner monologue before it produces its final answer. When thinking is active, the agent runs a lightweight reasoning pass — analysing what the user needs and sketching a tool plan — then uses that analysis to compose a better response. This is the prefrontal cortex: deliberate, structured thought before speech.

## Module Structure

```
reasoning/
├── __init__.py   # Public API
├── thinking.py   # Thinking dataclass
└── step.py       # ReasoningStep, ReasoningSteps, ThinkingOutput, NextAction,
                  # thinking_output_to_reasoning_steps
```

## Quick Start

```python
from definable.agent import Agent

# Enable with defaults — runs before every model call
agent = Agent(model="openai/gpt-4o-mini", thinking=True)

# Fine-grained control
from definable.agent.reasoning import Thinking

agent = Agent(
  model="openai/gpt-4o-mini",
  thinking=Thinking(trigger="auto"),   # only think when the query seems complex
)

result = await agent.arun("Explain the tradeoffs between B-trees and LSM trees.")
print(result.content)
```

## API Reference

### `Thinking`

Dataclass. The block you pass to `Agent(thinking=...)`.

```python
@dataclass
class Thinking:
  enabled: bool = True
  model: Model | None = None
  instructions: str | None = None
  trigger: Literal["always", "auto", "never"] = "always"
  description: str | None = None
```

| Field | Purpose |
|-------|---------|
| `enabled` | Master switch. `False` skips the reasoning phase entirely. |
| `model` | Model to use for the thinking pass. Defaults to the agent's own model when `None`. |
| `instructions` | Custom system prompt for the thinking model. Overrides the built-in reasoning prompt. |
| `trigger` | When to activate — see trigger modes below. |
| `description` | Free-text description injected into the agent's layer guide (system prompt) so the model understands its own reasoning capability. |

**Trigger modes:**

| Value | Behaviour |
|-------|-----------|
| `"always"` | Thinking runs on every `arun()` call regardless of query content. |
| `"auto"` | The pipeline performs a lightweight pre-check and runs thinking only when the query appears to need it. |
| `"never"` | Thinking is disabled even when the `Thinking` block is present. Useful for temporary suppression without removing the block. |

**Shorthand — `Agent(thinking=True)`:**

Equivalent to `Agent(thinking=Thinking())`, which uses all defaults: enabled, same model as the agent, default instructions, `trigger="always"`.

### `ThinkingOutput`

Pydantic model. The structured result produced by the thinking phase. The agent's pipeline converts this to context injected into the main model call.

```python
class ThinkingOutput(BaseModel):
  analysis: str                   # 1-2 sentence read of what the user needs
  approach: str                   # 1-2 sentence plan for how to respond
  tool_plan: list[str] | None     # ordered tool names to call; None if no tools needed
```

### `ReasoningStep`

Pydantic model. One discrete step in an explicit chain-of-thought trace. Used by the legacy reasoning path and surfaced on `RunOutput.reasoning_steps`.

```python
class ReasoningStep(BaseModel):
  title: str | None            # concise label for the step
  action: str | None           # what was done ("I will ..." / "I did ...")
  result: str | None           # what came out of the action
  reasoning: str | None        # the thought process behind the step
  next_action: NextAction | None   # what happens next
  confidence: float | None     # 0.0–1.0 confidence in this step
```

### `ReasoningSteps`

Thin wrapper around a list of `ReasoningStep` objects.

```python
class ReasoningSteps(BaseModel):
  reasoning_steps: list[ReasoningStep]
```

### `NextAction`

Enum controlling step-to-step flow in explicit reasoning chains.

```python
class NextAction(str, Enum):
  CONTINUE     = "continue"       # keep reasoning
  VALIDATE     = "validate"       # check the current result
  FINAL_ANSWER = "final_answer"   # commit and return
  RESET        = "reset"          # start over
```

### `thinking_output_to_reasoning_steps`

Utility that converts a `ThinkingOutput` to a `list[ReasoningStep]` for backward compatibility with consumers that expect the older step format.

```python
def thinking_output_to_reasoning_steps(output: ThinkingOutput) -> list[ReasoningStep]:
  ...
```

Returns one or two steps: an "Analysis" step always, plus a "Tool Plan" step when `output.tool_plan` is non-empty.

## Integration with Agent

```python
from definable.agent import Agent
from definable.agent.reasoning import Thinking
from definable.model.openai import OpenAIChat

# Use a separate, cheaper model for thinking
agent = Agent(
  model=OpenAIChat(id="gpt-4o"),
  thinking=Thinking(
    model=OpenAIChat(id="gpt-4o-mini"),   # thinking is cheap, output is powerful
    trigger="always",
    instructions="Be concise. Identify the single most important constraint.",
  ),
)
```

Accessing reasoning output after a run:

```python
result = await agent.arun("Design a rate limiter for 10k req/s.")

# The final answer
print(result.content)

# The reasoning trace (when available)
if result.reasoning_steps:
  for step in result.reasoning_steps:
    print(f"[{step.title}] {step.reasoning}")
```

Streaming reasoning content as it arrives:

```python
async for event in agent.arun_stream("Explain CAP theorem."):
  if event.event == "ReasoningContentDelta":
    print(event.reasoning_content, end="", flush=True)
  elif event.event == "RunCompleted":
    print("\n---\n", event.content)
```

## Gotchas

- `trigger="auto"` runs a pre-check model call, which adds latency and cost. Reserve it for agents where most queries are simple — otherwise use `"always"`.
- `Thinking(model=None)` is intentional. The pipeline substitutes the agent's model at runtime. Pass an explicit model only when you want a different one for the thinking pass.
- `Thinking(enabled=False)` and `Thinking(trigger="never")` both disable the phase, but for different audiences: `enabled=False` is a permanent config flag; `trigger="never"` is for runtime suppression while leaving the block in place.
- `ThinkingOutput.tool_plan` is `None`, not an empty list, when no tools are needed. Check `if output.tool_plan:` rather than `if output.tool_plan is not None:`.
- `reasoning_steps` on `RunOutput` contains `ReasoningStep` Pydantic model instances, not raw dicts. Call `.model_dump()` to serialise.
- `ThinkingConfig` (old name) is **deprecated** — use `Thinking` directly.

## Related Modules

- `agent/` — `Agent` accepts `thinking=Thinking(...)` or the boolean shorthand `thinking=True`
- `agent/run/` — `RunOutput.reasoning_steps` and `RunOutput.reasoning_content` carry the thinking output
- `agent/run/agent.py` — `ReasoningStartedEvent`, `ReasoningStepEvent`, `ReasoningContentDeltaEvent`, `ReasoningCompletedEvent` stream the reasoning phase to consumers
