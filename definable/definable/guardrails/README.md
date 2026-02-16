# guardrails

Content policy enforcement at input, output, and tool checkpoints.

## Quick Start

```python
from definable.agents import Agent
from definable.guardrails import Guardrails, max_tokens, pii_filter, tool_blocklist

agent = Agent(
    model=model,
    guardrails=Guardrails(
        input=[max_tokens(500)],
        output=[pii_filter()],
        tool=[tool_blocklist({"dangerous_tool"})],
    ),
)

# Guardrails run automatically on every arun() / arun_stream() call.
output = agent.run("Hello, what's my account balance?")
```

## Module Structure

```
guardrails/
├── __init__.py        # Public API exports
├── base.py            # GuardrailResult, Protocol definitions, Guardrails container
├── decorators.py      # @input_guardrail, @output_guardrail, @tool_guardrail
├── composable.py      # ALL, ANY, NOT, when combinators
├── events.py          # GuardrailCheckedEvent, GuardrailBlockedEvent
└── builtin/
    ├── __init__.py    # Built-in factory exports
    ├── input.py       # max_tokens, block_topics, regex_filter
    ├── output.py      # pii_filter, max_output_tokens
    └── tool.py        # tool_allowlist, tool_blocklist
```

## API Reference

### Guardrails

```python
from definable.guardrails import Guardrails
```

Container that holds all guardrails and runs them at the appropriate checkpoints.

```python
guardrails = Guardrails(
    input=[max_tokens(500)],           # Before LLM call
    output=[pii_filter()],            # After LLM call
    tool=[tool_blocklist({"rm"})],    # Before each tool execution
    mode="fail_fast",                  # "fail_fast" | "run_all"
    on_block="raise",                  # "raise" | "return_message"
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | `List[InputGuardrail]` | `[]` | Guardrails run on user input before the LLM call |
| `output` | `List[OutputGuardrail]` | `[]` | Guardrails run on model output after the LLM call |
| `tool` | `List[ToolGuardrail]` | `[]` | Guardrails run on each tool call before execution |
| `mode` | `Literal["fail_fast", "run_all"]` | `"fail_fast"` | Stop at first block vs. run all and collect results |
| `on_block` | `Literal["raise", "return_message"]` | `"raise"` | Raise exception vs. return `RunOutput(status=blocked)` |

Runner methods (called internally by the agent):

| Method | Description |
|--------|-------------|
| `run_input_checks(text, context)` | Run all input guardrails on user text |
| `run_output_checks(text, context)` | Run all output guardrails on model response |
| `run_tool_checks(tool_name, tool_args, context)` | Run all tool guardrails before execution |

### GuardrailResult

```python
from definable.guardrails import GuardrailResult
```

Return value from every guardrail check. Use factory methods to create:

| Factory Method | Description |
|----------------|-------------|
| `GuardrailResult.allow()` | Allow the content through |
| `GuardrailResult.block(reason)` | Block the content with a reason string |
| `GuardrailResult.modify(new_text, reason="")` | Replace the content with `new_text` |
| `GuardrailResult.warn(message)` | Allow but emit a warning |

Fields: `action` (`"allow"` / `"block"` / `"modify"` / `"warn"`), `message`, `modified_text`, `metadata`.

### Protocols

```python
from definable.guardrails import InputGuardrail, OutputGuardrail, ToolGuardrail
```

All three are `@runtime_checkable` protocols.

| Protocol | `check()` Signature |
|----------|---------------------|
| `InputGuardrail` | `async def check(self, text: str, context: RunContext) -> GuardrailResult` |
| `OutputGuardrail` | `async def check(self, text: str, context: RunContext) -> GuardrailResult` |
| `ToolGuardrail` | `async def check(self, tool_name: str, tool_args: Dict[str, Any], context: RunContext) -> GuardrailResult` |

### Built-in Guardrails

**Input:**

| Factory | Description |
|---------|-------------|
| `max_tokens(n, model_id="gpt-4o")` | Block input exceeding `n` tokens |
| `block_topics(topics)` | Block input containing any keyword from `topics` (case-insensitive) |
| `regex_filter(patterns, action="block")` | Block or redact input matching regex patterns |

**Output:**

| Factory | Description |
|---------|-------------|
| `pii_filter(action="modify")` | Detect and redact (or block) PII — credit cards, SSN, email, phone |
| `max_output_tokens(n, model_id="gpt-4o")` | Block output exceeding `n` tokens |

**Tool:**

| Factory | Description |
|---------|-------------|
| `tool_allowlist(allowed)` | Only allow tools in the given `Set[str]` |
| `tool_blocklist(blocked)` | Block tools in the given `Set[str]` |

### Decorators

Convert plain async functions into guardrail objects:

```python
from definable.guardrails import input_guardrail, GuardrailResult

@input_guardrail
async def no_profanity(text: str, context) -> GuardrailResult:
    if "badword" in text.lower():
        return GuardrailResult.block("Profanity detected")
    return GuardrailResult.allow()

# With custom name
@input_guardrail(name="custom_name")
async def my_guard(text: str, context) -> GuardrailResult:
    ...
```

Also available: `@output_guardrail`, `@tool_guardrail`.

### Composable Combinators

| Combinator | Description |
|------------|-------------|
| `ALL(*guardrails)` | All must allow; first block wins |
| `ANY(*guardrails)` | At least one must allow; block only if all block |
| `NOT(guardrail)` | Invert: allow ↔ block (modify/warn pass through) |
| `when(condition, guardrail)` | Run guardrail only if `condition(context)` is `True` |

### Tracing Events

| Event | Emitted When |
|-------|-------------|
| `GuardrailCheckedEvent` | After each check completes (any action) |
| `GuardrailBlockedEvent` | When a guardrail blocks execution |

Fields: `guardrail_name`, `guardrail_type` (`"input"` / `"output"` / `"tool"`), `action` / `reason`, `duration_ms`.

## Usage with Agent

```python
agent = Agent(model=model, guardrails=guardrails)

# Guardrails are automatically executed in arun() and arun_stream():
# 1. Input guardrails — after memory recall, before LLM call
# 2. Tool guardrails — inside tool loop, before each tool execution
# 3. Output guardrails — after LLM response, before memory store
#
# on_block="raise" → InputCheckError / OutputCheckError
# on_block="return_message" → RunOutput(status=RunStatus.blocked)
```

## See Also

- `agents/` — Agent integration via `guardrails=` parameter
- `run/` — `RunStatus.blocked`, `RunOutput`
- `exceptions.py` — `InputCheckError`, `OutputCheckError`
