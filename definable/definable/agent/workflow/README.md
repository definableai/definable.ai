# Workflow -- Multi-Step Agent Orchestration

The Workflow module composes agents, teams, and callables into executable pipelines with support for sequential, parallel, looping, conditional, and routed execution.

A `Workflow` takes an input string, pipes it through a graph of `Step` nodes, and returns a `WorkflowOutput` with the final content and every intermediate result.

```python
from definable.agent import Agent
from definable.agent.workflow import Workflow, Step

researcher = Agent(model="openai/gpt-4o", instructions="You are a research specialist.")
writer = Agent(model="openai/gpt-4o", instructions="You are a technical writer.")

workflow = Workflow(
  name="research-pipeline",
  steps=[
    Step(name="researcher", agent=researcher),
    Step(name="writer", agent=writer),
  ],
)
result = await workflow.arun("Write about quantum computing")
print(result.content)  # Final writer output
print(result.success)  # True
print(result.duration_ms)  # Total time in milliseconds
```

---

## Architecture

```
                              Workflow.arun(input)
                                      |
                                      v
                            +-------------------+
                            |    StepInput      |
                            | (input, state,    |
                            |  additional_data) |
                            +-------------------+
                                      |
                      +---------------+----------------+
                      |               |                |
                      v               v                v
                  +-------+     +-----------+    +-----------+
                  | Step  |     |  Steps    |    | Parallel  |
                  | (leaf)|     |(sequence) |    |(concurrent|
                  +-------+     +-----------+    +-----------+
                      |               |                |
                      |          +---------+     +---------+
                      |          | Step A  |     | Step X  |
                      |          | Step B  |     | Step Y  |
                      |          | Step C  |     | Step Z  |
                      |          +---------+     +---------+
                      |               |                |
                      v               v                v
                  +-------+     +-----------+    +-----------+
                  | Loop  |     | Condition |    |  Router   |
                  |(iter) |     | (if/else) |    | (N-way)   |
                  +-------+     +-----------+    +-----------+
                      |           /      \         / | \
                      |     true_steps  false   route route
                      |                 steps    A    B
                      v
              +-------------------+
              |   WorkflowOutput  |
              | (content, steps,  |
              |  session_state)   |
              +-------------------+
```

### Data Flow Between Steps

```
StepInput                  Step                    StepOutput
+-----------------------+  +-----------------+     +-------------------+
| input: str            |->| agent.arun()    |---->| content: str      |
| previous_step_content |  | OR team.arun()  |     | success: bool     |
| previous_step_outputs |  | OR executor(si) |     | status: StepStatus|
| additional_data       |  +-----------------+     | run_output        |
| session_state         |                          | steps (nested)    |
+-----------------------+                          +-------------------+
                                                          |
                                                          v
                                                   Next StepInput
                                               (previous_step_content
                                                = this output.content)
```

---

## Exports

```python
from definable.agent.workflow import (
  # Core
  Workflow,
  Step,
  Steps,
  BaseStep,
  # Control flow
  Parallel,
  Loop,
  Condition,
  Router,
  # Context types
  StepInput,
  StepOutput,
  StepStatus,
  WorkflowOutput,
  # Events
  BaseWorkflowEvent,
  WorkflowRunStartedEvent,
  WorkflowRunCompletedEvent,
  WorkflowRunErrorEvent,
  StepStartedEvent,
  StepCompletedEvent,
  StepErrorEvent,
  StepSkippedEvent,
  LoopIterationEvent,
)
```

---

## Step Types

### Step -- Single Execution Unit

Wraps an `Agent`, `Team`, or callable. Exactly one of `agent`, `team`, or `executor` must be set.

```python
from definable.agent.workflow import Step

# With an Agent
step = Step(name="researcher", agent=my_agent)

# With a Team
step = Step(name="content-team", team=my_team)


# With an async callable (receives StepInput, returns str or any)
async def process(step_input):
  return f"Processed: {step_input.input}"


step = Step(name="processor", executor=process)

# With retries and timeout
step = Step(name="flaky-api", agent=api_agent, retries=3, timeout=30.0)
```

**Constructor:**
```python
Step(
  name: str,
  agent: Agent = None,        # Exactly one of
  team: Team = None,           # these three
  executor: Callable = None,   # is required
  input_builder: Callable[[StepInput], str] = None,  # Custom prompt builder
  timeout: float = None,       # Seconds before asyncio.TimeoutError
  retries: int = 0,            # Retry count on failure (total attempts = retries + 1)
)
```

**Custom input builder:**

By default, `Step` combines the original input with the previous step's content. Override this with `input_builder`:

```python
def custom_builder(step_input):
  research = step_input.get_step_content("researcher")
  return f"Rewrite this research as a blog post:\n\n{research}"


step = Step(name="writer", agent=writer, input_builder=custom_builder)
```

---

### Steps -- Sequential Composition

Executes steps in order, chaining each output as context to the next. Stops early on failure or if a step sets `stop=True`.

```python
from definable.agent.workflow import Steps, Step

seq = Steps(
  name="draft-review",
  steps=[
    Step(name="draft", agent=drafter),
    Step(name="review", agent=reviewer),
    Step(name="finalize", agent=finalizer),
  ],
)
```

The `review` step automatically receives the `draft` step's content as `previous_step_content`. The `finalize` step receives `review`'s content and can access `draft`'s output via `step_input.get_step_content("draft")`.

**Note:** When you pass a list of steps to `Workflow(steps=[...])`, they are automatically wrapped in a `Steps` node. You only need `Steps` explicitly when nesting sequences inside other control flow (e.g., inside `Parallel` or `Loop`).

---

### Parallel -- Concurrent Execution

Runs all steps concurrently using `asyncio.gather`. Each step receives the same input (no chaining). Results are combined with `[step_name]: content` format.

```python
from definable.agent.workflow import Parallel, Step

par = Parallel(
  name="analysis",
  steps=[
    Step(name="technical", agent=tech_agent),
    Step(name="business", agent=biz_agent),
    Step(name="legal", agent=legal_agent),
  ],
  max_concurrency=2,  # Optional: limit concurrent steps
)
```

**Constructor:**
```python
Parallel(
  name: str,
  steps: list = [],
  max_concurrency: int = None,  # None = unlimited
)
```

The combined output content format:
```
[technical]: <technical analysis content>

[business]: <business analysis content>

[legal]: <legal analysis content>
```

---

### Loop -- Iterative Execution

Runs steps repeatedly until the `end_condition` returns `True` or `max_iterations` is reached. Each iteration receives the previous iteration's output as context.

```python
from definable.agent.workflow import Loop, Step

loop = Loop(
  name="improve",
  steps=[
    Step(name="generate", agent=generator),
    Step(name="evaluate", agent=evaluator),
  ],
  end_condition=lambda outputs: any("APPROVED" in (o.content or "") for o in outputs),
  max_iterations=5,
)
```

**Constructor:**
```python
Loop(
  name: str,
  steps: list = [],
  end_condition: Callable[[list[StepOutput]], bool] = None,  # Sync or async
  max_iterations: int = 3,
)
```

The `end_condition` receives the list of all iteration outputs (each is a `StepOutput` containing the sequence result for that iteration). Return `True` to stop looping.

Loop terminates when any of these occur:
1. `end_condition` returns `True`
2. `max_iterations` is reached
3. A step fails (`success=False`)
4. A step sets `stop=True`

---

### Condition -- If/Else Branching

Evaluates a condition on the `StepInput` and executes the matching branch.

```python
from definable.agent.workflow import Condition, Step

cond = Condition(
  name="quality-gate",
  condition=lambda ctx: "PASS" in (ctx.get_last_step_content() or ""),
  true_steps=Step(name="publish", agent=publisher),
  false_steps=Step(name="rewrite", agent=writer),
)
```

**Constructor:**
```python
Condition(
  name: str,
  condition: Callable[[StepInput], bool] = None,  # Sync or async
  true_steps: Step | list = None,   # Executed if condition is True
  false_steps: Step | list = None,  # Executed if condition is False
)
```

If a branch is `None`, the condition step is marked `skipped` with `success=True`.

Both `true_steps` and `false_steps` accept a single step or a list (auto-wrapped in `Steps`).

---

### Router -- N-Way Dynamic Routing

Selects one or more routes based on a `selector` function. The selector returns a route name (or list of names) that maps into the `routes` dict.

```python
from definable.agent.workflow import Router, Step

router = Router(
  name="support",
  selector=lambda ctx: "technical" if "bug" in (ctx.input or "") else "general",
  routes={
    "technical": Step(name="tech-support", agent=tech_agent),
    "general": Step(name="general-support", agent=general_agent),
    "billing": Step(name="billing-support", agent=billing_agent),
  },
)
```

**Constructor:**
```python
Router(
  name: str,
  selector: Callable[[StepInput], str | list[str]] = None,  # Sync or async
  routes: dict[str, Step] = {},
)
```

The selector can return multiple route names to execute several routes:

```python
router = Router(
  name="multi-analysis",
  selector=lambda ctx: ["technical", "business"],
  routes={
    "technical": Step(name="tech", agent=tech_agent),
    "business": Step(name="biz", agent=biz_agent),
  },
)
```

If the selector returns a route name not in the `routes` dict, that route is marked `failed` with an error message.

---

## Context Types

### StepStatus

```python
from definable.agent.workflow import StepStatus

StepStatus.pending  # "pending"   -- not yet started
StepStatus.running  # "running"   -- currently executing
StepStatus.completed  # "completed" -- finished successfully
StepStatus.failed  # "failed"    -- encountered an error
StepStatus.skipped  # "skipped"   -- branch not taken (Condition)
```

### StepInput

The context object passed to every step during execution. Carries the original input, all previous step outputs, and shared state.

```python
from definable.agent.workflow import StepInput

si = StepInput(
  input="original user prompt",
  previous_step_content="output from previous step",
  previous_step_outputs={},  # name -> StepOutput
  additional_data={},  # extra data from Workflow.arun()
  session_state={},  # shared state across all steps
)

# Access methods
si.get_step_output("researcher")  # -> StepOutput | None
si.get_step_content("researcher")  # -> str | None
si.get_last_step_content()  # -> str | None (most recent)
si.get_all_previous_content()  # -> dict[str, str | None]
```

### StepOutput

The result from executing a single step. Supports nested outputs for composite steps (Steps, Parallel, Loop).

```python
from definable.agent.workflow import StepOutput

output = StepOutput(
  step_name="researcher",
  step_id="a1b2c3d4",  # Auto-generated
  step_type="step",  # "step", "steps", "parallel", "loop", "condition", "router"
  content="The research findings...",
  status=StepStatus.completed,
  success=True,
  error=None,
  stop=False,  # If True, halts the parent sequence
  metrics=None,  # Optional dict
  duration_ms=1234.5,
  run_output=None,  # The full RunOutput if agent/team was used
  steps=[],  # Nested StepOutputs for composite steps
)

output.to_dict()  # Serialize to dict (excludes run_output)
```

### WorkflowOutput

The top-level result from `Workflow.arun()`.

```python
from definable.agent.workflow import WorkflowOutput

result = await workflow.arun("some input")

result.workflow_id  # UUID of the workflow instance
result.workflow_name  # "research-pipeline"
result.run_id  # UUID of this specific run
result.content  # Final content from the last step
result.success  # True if all steps succeeded
result.error  # Error message if failed
result.step_outputs  # list[StepOutput] -- all top-level step results
result.duration_ms  # Total execution time
result.session_state  # Final session state dict

# Access specific step results
result.get_step_output("researcher")  # -> StepOutput | None (searches nested)
result.get_step_content("researcher")  # -> str | None
```

---

## Workflow

The top-level orchestrator. Composes steps into a runnable pipeline.

**Constructor:**
```python
Workflow(
  name: str = "",
  description: str = None,
  instructions: str = None,
  steps: list | BaseStep = [],     # List of steps, or a single composite step
  session_state: dict = {},         # Initial shared state
  debug: bool = False,
)
```

**Properties:**
- `id` -- UUID of the workflow instance
- `events` -- `EventBus` for subscribing to workflow events

**Methods:**
```python
result = await workflow.arun(
  input="user prompt",
  session_state={"key": "override"},  # Merges with workflow session_state
  additional_data={"context": "extra"},  # Passed to all steps via StepInput
)
# Returns: WorkflowOutput
```

---

## Composing Step Types

All step types are composable. You can nest any step type inside any other.

### Sequential with Parallel Analysis

```python
workflow = Workflow(
  name="research-and-analyze",
  steps=[
    Step(name="researcher", agent=researcher),
    Parallel(
      name="analysis",
      steps=[
        Step(name="technical", agent=tech_analyst),
        Step(name="business", agent=biz_analyst),
      ],
    ),
    Step(name="synthesizer", agent=synthesizer),
  ],
)
```

### Loop with Condition Gate

```python
workflow = Workflow(
  name="iterative-improvement",
  steps=[
    Loop(
      name="refine",
      steps=[
        Step(name="generate", agent=generator),
        Condition(
          name="quality-check",
          condition=lambda ctx: "PASS" in (ctx.get_last_step_content() or ""),
          true_steps=Step(name="done", executor=lambda si: si.previous_step_content),
          false_steps=Step(name="feedback", agent=critic),
        ),
      ],
      end_condition=lambda outputs: any("PASS" in (o.content or "") for o in outputs),
      max_iterations=5,
    ),
  ],
)
```

### Router into Parallel

```python
workflow = Workflow(
  name="smart-pipeline",
  steps=[
    Router(
      name="classifier",
      selector=classify_input,
      routes={
        "simple": Step(name="fast-path", agent=simple_agent),
        "complex": Parallel(
          name="deep-analysis",
          steps=[
            Step(name="a", agent=agent_a),
            Step(name="b", agent=agent_b),
          ],
        ),
      },
    ),
  ],
)
```

---

## Using Callables as Steps

Any async or sync callable can be used as a step executor. The callable receives a `StepInput` and returns a string (or any object with a `.content` attribute).

```python
async def double(step_input):
  return f"doubled: {step_input.input or step_input.previous_step_content}"


workflow = Workflow(
  name="test-wf",
  steps=[Step(name="step1", executor=double)],
)
result = await workflow.arun("hello")
result.content  # "doubled: hello"
result.success  # True
result.get_step_content("step1")  # "doubled: hello"
```

Sync callables are also supported (run in a thread executor):

```python
def transform(step_input):
  return f"transformed: {step_input.input}"


step = Step(name="sync-step", executor=transform)
```

---

## Session State

Workflows support shared state that persists across steps. Steps can read state from `StepInput.session_state`. State is returned in `WorkflowOutput.session_state`.

```python
workflow = Workflow(
  name="stateful",
  session_state={"counter": 0, "mode": "draft"},
  steps=[...],
)

# Override at runtime
result = await workflow.arun("input", session_state={"counter": 5})
result.session_state  # {"counter": 5, "mode": "draft"}
```

---

## Events

Subscribe to workflow lifecycle events via the `events` property (an `EventBus`).

```python
from definable.agent.workflow import (
  Workflow,
  WorkflowRunStartedEvent,
  WorkflowRunCompletedEvent,
  StepStartedEvent,
  StepCompletedEvent,
  StepErrorEvent,
  StepSkippedEvent,
  LoopIterationEvent,
)

workflow = Workflow(name="my-wf", steps=[...])


@workflow.events.on(StepCompletedEvent)
async def on_step_done(event):
  print(f"Step '{event.step_name}' completed in {event.duration_ms:.0f}ms")


@workflow.events.on(LoopIterationEvent)
async def on_loop_iter(event):
  print(f"Loop '{event.step_name}' iteration {event.iteration}/{event.max_iterations}")


result = await workflow.arun("input")
```

### Event Types

| Event | Emitted When | Key Fields |
|-------|-------------|------------|
| `WorkflowRunStartedEvent` | Workflow begins | `step_count`, `step_names` |
| `WorkflowRunCompletedEvent` | Workflow finishes | `content`, `success`, `duration_ms` |
| `WorkflowRunErrorEvent` | Workflow fails with unrecoverable error | `error` |
| `StepStartedEvent` | Individual step begins | `step_id`, `step_name`, `step_type`, `step_index` |
| `StepCompletedEvent` | Individual step finishes | `step_id`, `step_name`, `content`, `success`, `duration_ms` |
| `StepErrorEvent` | Individual step fails | `step_id`, `step_name`, `error` |
| `StepSkippedEvent` | Step skipped (e.g., Condition branch) | `step_id`, `step_name`, `reason` |
| `LoopIterationEvent` | Each loop iteration starts | `step_name`, `iteration`, `max_iterations`, `should_continue` |

All events inherit from `BaseWorkflowEvent` and carry `run_id`, `workflow_id`, and `workflow_name`.

---

## Gotchas

| Pitfall | Details |
|---------|---------|
| `Workflow.arun()` returns `WorkflowOutput` | Different from `Agent.arun()` which returns `RunOutput`. Use `result.content`, `result.get_step_output(name)`. |
| `Step` needs exactly one executor | Set `agent=`, `team=`, or `executor=`. Not multiple, not none. Raises `ValueError` at execution time. |
| Executor callables receive `StepInput` | Not a raw string. Access the prompt via `step_input.input` or `step_input.previous_step_content`. |
| `Steps` vs `Workflow(steps=[...])` | A list passed to `Workflow` is auto-wrapped in `Steps`. Use `Steps` explicitly only when nesting inside other composites. |
| Loop `end_condition` receives all outputs | The callback gets `list[StepOutput]` (all iterations so far), not just the latest. |
| Condition `condition` receives `StepInput` | Not `StepOutput`. Use `ctx.get_last_step_content()` to inspect the previous step's output. |
| Parallel steps share input | No chaining between parallel steps. Each gets the same `StepInput`. |
| `retries` on `Step` | Total attempts = `retries + 1`. A step with `retries=3` runs up to 4 times. |
| Nested `StepOutput.steps` | Composite steps (Steps, Parallel, Loop, Condition, Router) nest their child outputs in `StepOutput.steps`. Use `WorkflowOutput.get_step_output(name)` to search recursively. |
