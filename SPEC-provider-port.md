# SPEC: Provider port — eliminate `definable.run` legacy module

**Status**: draft, awaiting confirmation
**Author**: hash
**Date**: 2026-05-13
**Branch**: TBD (likely `chore/run-purge` or extend `feat/harness-v2`)
**Related**: `/SPEC.md` (Channel rename — sibling, runs separately), memory `2026-05-08_01-00_harness-v2-shipped.md`, memory `2026-05-13_00-00_channel-rename-spec-plan.md`

---

## 1. Objective

`definable.run/` is a 1,717-LOC legacy module that survived the harness v2 cleanup. The new self-contained `definable.agent/core/` already ships duplicate IO contracts (`RunResult`, `EventBus`, `Event` hierarchy), but model providers (`definable.model/*`) and the tool decorator (`definable.agent.toolkit.function`) still import from `definable.run`. The result is two parallel truths:

- **Old**: `RunOutput` (40+ fields), `RunOutputEvent` union (40+ event dataclasses), `RunStatus` enum, `RunContext` ambient ContextVar, `RunRequirement` HITL.
- **New**: `RunResult` (5 fields, frozen), `Event` base + ~10 typed events, `EventBus` pub/sub.

Providers carry the old types as defaulted-None `run_response: Optional[RunOutput] = None` kwargs threaded through ~12 internal methods per provider. The new agent loop never passes that kwarg — pure dead weight.

**Goal**: port every live consumer of `definable.run` onto the new `agent/core/` types, then delete `definable.run/` in one commit. Hard rename, no compat alias. Match harness-v2 + channel-rename precedent.

**Target users**: SDK consumers (Anandesh + downstream users of E-Garuda, clinic-receptionist, linkedin-outreach agents). They write `@tool def foo(ctx: RunContext, ...)` today; after the port they write `@tool def foo(ctx: ToolContext, ...)` or skip the param entirely.

**Non-goals**:
- No new public functionality.
- No HITL revival — `RunRequirement` is deleted, not ported. HITL was a pre-v2 feature already dropped from the new loop.
- No structural change to `agent/core/` types. They are the canonical surface.
- Does not touch the Channel rename. Sequencing decided in §6.

---

## 2. Commands (developer workflow)

| Command | Purpose |
|---|---|
| `.venv/bin/ruff check definable/definable/ definable/tests/` | Lint gate |
| `.venv/bin/ruff format --check definable/definable/ definable/tests/` | Format gate |
| `.venv/bin/python -m mypy definable/definable/ definable/tests/` | Type gate |
| `.venv/bin/python -m pytest definable/tests/ -x` | Unit test gate (894 expected green) |
| `.venv/bin/python -m pytest smoke/ -x` | Smoke gate (33/33 expected green) |
| `.venv/bin/python -c "import definable; import definable.agent; import definable.model"` | Import smoke |
| `rg 'from definable.run\|definable\.run\.' definable/` | Verify legacy import sweep complete (must return zero hits at end) |

All 4 quality gates + smoke must be green before merge.

---

## 3. Project structure

### Files deleted (entire directory)

```
definable/definable/run/
  __init__.py
  base.py            (RunContext, RunStatus, BaseRunOutputEvent, set/get_current_run_context — 284 LOC)
  agent.py           (RunOutput, RunInput, RunEvent, RunOutputEvent union, ~40 event dataclasses — 1,118 LOC)
  reasoning_step.py  (ReasoningStep — 136 LOC)
  requirement.py     (RunRequirement — 179 LOC)
  README.md          (stale — describes pre-v2 phase pipeline + DeepResearch + knowledge)
```

### Files created in `agent/core/`

```
definable/definable/agent/core/
  context.py         NEW — ToolContext dataclass + ambient ContextVar (replaces RunContext)
                          fields: run_id, session_id, user_id, metadata, session_state, dependencies, memory_context
                          dead fields dropped: knowledge_*, research_*, readers_*, output_schema, active_layers, knowledge_filters
                          get_current_tool_context() + set_current_tool_context()
  reasoning.py       NEW — ReasoningStep dataclass + small helpers (moved from run/reasoning_step.py)
                          ReasoningStep is still used by model providers for thinking content; keep.
```

### Files modified

```
definable/definable/agent/toolkit/function.py
  - drop `from definable.run import RunContext`
  - add `from definable.agent.core.context import ToolContext`
  - rename param-type detection from RunContext to ToolContext

definable/definable/agent/agent.py
  - wrap arun() body in `set_current_tool_context(ToolContext(...))` context manager
  - already constructs run_id, session_id — straight extension

definable/definable/model/base.py (~2,758 LOC)
  - drop imports: `from definable.run.agent import CustomEvent, RunContentEvent, RunOutput, RunOutputEvent`
                  `from definable.run.requirement import RunRequirement`
  - delete `run_response: Optional[RunOutput] = None` kwarg from every method (~12 signatures)
  - delete `run_response.metrics.set_time_to_first_token()` lines (move metric capture into ModelResponse — already partially there)
  - delete `run_response.requirements.append(RunRequirement(...))` HITL branches (dead in v2)
  - delete `Iterator[Union[ModelResponse, RunOutputEvent]]` return signatures → `Iterator[ModelResponse]` only
  - delete `if isinstance(item, tuple(get_args(RunOutputEvent)))` branches
  - drop `CompressionManager` TYPE_CHECKING import (dead — compression module gone)

definable/definable/model/{openai,anthropic,google,mistral,ollama,perplexity,openrouter,claude_code}/{chat,gemini,claude,mistral,perplexity,openrouter}.py
  - drop `from definable.run.agent import RunOutput`
  - delete `run_response: Optional[RunOutput] = None` kwarg from every signature (~18 hits/provider)
  - move time-to-first-token capture from run_response onto ModelResponse.metrics directly (one-line change per provider)

definable/definable/utils/reasoning.py
  - update import from `definable.run.reasoning_step` → `definable.agent.core.reasoning`
  - update TYPE_CHECKING import from `definable.run.agent` → drop (RunOutput no longer used)

definable/definable/model/{anthropic,google,openai,mistral,ollama,perplexity,openrouter,claude_code}/* — ReasoningStep imports
  - update import paths

tests/  (those that touch run/)
  - any test importing from `definable.run` rewritten to use `agent/core/` equivalents OR deleted if testing dead-code paths (HITL, RunOutput.parsed, RunStatus state machine).
  - `RunStatus` enum has no successor — tests asserting status transitions are deleted (the new loop has only RunResult.exit_reason = "natural" | "max_turns" | "error", which is already covered)
```

### Files NOT touched

- `definable/agent/core/result.py` — `RunResult` already canonical, untouched
- `definable/agent/core/events.py` — `Event` hierarchy already canonical, untouched
- `definable/agent/core/loop.py` — already free of run/ imports
- `definable/agent/memory/`, `toolkit/`, `mcp/`, `skill/` — already clean
- Channel rename SPEC + plan + todo at repo root — sibling work, no interaction

---

## 4. Code style

Inherits framework-patterns skill rules. Specifics:

- **Naming**: `RunContext` → `ToolContext`. Reason: post-channel-rename, the "run" verb belongs to `agent.run()`; `RunContext` is misleading. The ambient context exists for tools, hence `ToolContext`.
- **Frozen dataclasses** for `ToolContext`, `ReasoningStep`. No mutation. If a tool wants to record state, it writes to `ctx.dependencies` (which is a `dict[str, Any]`, mutable in place — but the holder dataclass is frozen).
- **No backwards-compat aliases**. No `RunContext = ToolContext` re-export. Hard rename. Match channel-rename precedent (memory `2026-05-13_00-00`).
- **No defaulted-None plumbing kwargs** in provider methods. The pre-v2 `run_response=None` pattern is the exact thing being deleted; do not reintroduce.
- **mypy strict-clean** at the end. mypy must pass on the 8 modified provider files and on `model/base.py` (currently has a few `# type: ignore` lines around `RunOutput` — they go away with the import).
- **Imports sorted by ruff** as usual. `from __future__ import annotations` retained where present.

---

## 5. Testing strategy

### Tiers

1. **Unit tests** (`definable/tests/`): all 894 existing must stay green. Tests importing `definable.run` are rewritten to use `agent/core/` types or deleted if asserting on dead-code paths. Specifically expect to delete: HITL pause/resume tests, `RunStatus` state-machine tests, `RunOutput.parsed` tests (already known stale per the `run/README.md` "open bug #6" note).
2. **Smoke tests** (`smoke/`): all 33 must stay green. Specifically the power-user composite (Agent + 2 tools + 1 Toolkit + 2 Skills + MCPfs + FileMemory + Observability + 3 subscribers, 3-turn dialogue) — this is the regression net for end-to-end provider call paths.
3. **Behavioral parity tests** (NEW, small — 4 to 6 tests in `definable/tests/model/test_post_port_parity.py`):
   - OpenAI: `ainvoke` returns `ModelResponse` with metrics populated and `time_to_first_token > 0` on a streamed call.
   - Anthropic: same, plus reasoning_content captured.
   - Tool decorator: `@tool def foo(ctx: ToolContext, q: str)` correctly receives a populated `ToolContext` during a run.
   - Ambient context: `get_current_tool_context()` returns the correct context inside a tool and `None` outside any run.
   - Import smoke: `from definable.run` raises `ImportError` (post-delete).
4. **Example agents** (live): E-Garuda + clinic-receptionist + linkedin-outreach run through one happy-path turn end-to-end. These were the validation harness for harness-v2 and stay the validation harness here.

### Verification order at each step

`ruff check → ruff format → mypy → pytest unit → pytest smoke → example run`. Same gate ordering as every other recent change (LinkedIn refactor, harness v2, feed commenter).

### Removed test surface

Tests that assert on pre-v2 features (HITL `confirm()`/`reject()`, `RunStatus.blocked`, multi-event union dispatch via `get_args(RunOutputEvent)`, `RunOutput.workflow_step_id`) are deleted, not rewritten. Reason: the underlying features were already deleted in harness v2; only the type definitions survived. The tests assert on absent behavior.

---

## 6. Boundaries

### Always

- Run all 4 quality gates + smoke before declaring a phase done.
- Delete `definable.run/` in a single commit at the end of the port. No partial sub-directory deletes.
- Run E-Garuda + clinic-receptionist + linkedin-outreach happy path before merge.
- Update memory + wiki + `/SPEC-provider-port.md` after each phase.
- Cross-reference with `/SPEC.md` (channel rename) — flag any divergence.

### Ask first

- Sequencing vs channel rename. Two options:
  - **(A) Channel rename first, provider port second**. Channel rename SPEC is locked, plan + todo exist, build paused on the dirty tree. Resume that, ship, then start the port. Lower risk — one structural change at a time.
  - **(B) Provider port first, channel rename second**. Provider port has wider blast radius but unlocks ambient `ToolContext` for the channel rename's `session_id` + `user_id` kwarg move, which is awkward without an updated `RunContext`.
  - **(C) Interleave on the same branch**. Risky — two structural renames in one diff.
  - **Recommendation**: (A). Channel rename is closer to done. Land it, then this. Confirm before I write `/tasks/provider-port-plan.md`.
- New ambient context type name. `ToolContext` is my proposal. Alternatives: `CallContext`, `RunContext` (keep name, change module path), `AgentContext`. Pick one before plan write.
- Whether `RunRequirement` HITL gets a successor in `agent/core/` or is deleted outright. Recommendation: delete. HITL can return as a separate workstream if a real customer asks. Confirm.
- Whether `definable.run` deletion ships in one mega-commit or split into provider-by-provider chunks. Recommendation: one provider per commit (8 commits) + one final "delete run/" commit. Reviewable. Confirm.

### Never

- Do not introduce a `definable.run.__init__` compat alias re-exporting from `agent.core`. Hard rename per project convention.
- Do not preserve the `run_response: Optional[RunOutput] = None` kwarg on provider methods. The kwarg is the disease.
- Do not port `RunStatus` enum forward. The new loop has `RunResult.exit_reason: Literal["natural", "max_turns", "error"]` and that is sufficient.
- Do not block on the Channel rename's `agent.run()` verb naming. Channel rename's `agent.run()` (lowercase verb, callable on agent) is unrelated to the `Run*` (PascalCase, type-name prefix) types being deleted here. No collision.
- Do not commit until all gates green on each phase. Match `feedback_never_skip_record.md` rule.
- Do not delete `model/response.py` `ModelResponse` — it is the new contract. Only the `RunOutput` reference type goes.

### Concerns / risks

- **Risk 1**: `RunOutput.metrics.set_time_to_first_token()` is a side-effect that providers call. Replacement must populate the same metric on `ModelResponse.metrics`. If the metric path lost coverage during harness v2, smoke would still pass (it doesn't assert on TTFT). Add one explicit assertion in the parity test suite.
- **Risk 2**: `from definable.run import RunContext` is the documented tool-decorator contract in the (now stale) `run/README.md`. Downstream user code in customer agents (E-Garuda etc.) may import it. Grep all agents at repo root + Anandesh side projects before the final delete. Acceptance: zero hits across all known consumers.
- **Risk 3**: `ReasoningStep` is consumed by 8 model providers. Moving it from `run/reasoning_step.py` to `agent/core/reasoning.py` is a wide rename. Plan should staple it to a single commit so the diff is reviewable.
- **Risk 4**: 18 `run_response` removals × 8 providers = 144 callsites. Mechanical but error-prone. Suggest one provider per commit; smoke runs between each.

---

## Acceptance criteria (summary)

A merge of this work is acceptable iff:

1. `rg 'from definable.run|definable\.run\.' definable/` returns zero hits.
2. `definable/definable/run/` directory does not exist.
3. `ruff check`, `ruff format --check`, `mypy`, all 894 unit tests, all 33 smoke tests pass.
4. E-Garuda + clinic-receptionist + linkedin-outreach agents complete one happy-path conversation each, end to end, against live model providers.
5. `agent/core/context.py::ToolContext` is the documented context type in `agent/README.md` and `agent/core/` docs.
6. Memory file written. Wiki updated. `/SPEC-provider-port.md` marked `Status: shipped`.

---

## Open questions awaiting answer

1. Sequencing vs Channel rename — recommend (A) channel rename first. Confirm?
2. Ambient context type name — recommend `ToolContext`. Confirm or pick alt?
3. `RunRequirement` HITL — recommend delete, not port. Confirm?
4. Commit shape — recommend 8 per-provider commits + 1 delete commit. Confirm?
5. Branch name — `chore/run-purge`? `refactor/provider-port`? Or extend `feat/harness-v2`?
6. `/SPEC-provider-port.md` path — keep distinct from `/SPEC.md` (channel rename), or merge sections after channel rename ships?
