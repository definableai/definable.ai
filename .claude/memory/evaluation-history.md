# Evaluation History

> Append-only log of evaluation runs.

## Run #1 — 2026-02-17 (first eval)

| Metric | Value |
|--------|-------|
| Version | 0.2.8 |
| E2E Tests | 975 collected, 745 passed, 1 failed, 138 skipped |
| Eval Scripts | 38 written, 34 passed (89%) |
| mypy | 0 errors |
| ruff | 0 warnings |
| Issues Filed | #6 (structured output parsed), #7 (TracingConfig export), #8 (hook signature), #9 (ddgs deprecation) |
| Scores | Robustness 8/10, Reliability 8/10, Scalability 7/10, Extensibility 7/10 |

## Run #2 — 2026-02-17 (second eval, same day)

| Metric | Value |
|--------|-------|
| Version | 0.2.8 |
| E2E Tests | 975 collected, 830 passed, 3 failed, 142 deselected |
| E2E Failures | test_tool_with_pre_post_hooks (#8), TestXAI x2 (grok-beta deprecated #14) |
| Eval Scripts | 18 written, 256 checks total, 176 passed (69%) |
| mypy | 0 errors (215 files) |
| ruff check | 0 warnings |
| ruff format | 0 issues |
| New Issues Filed | #10 (embedder re-exports), #11 (meta_data naming), #12 (chunker exports), #13 (output_schema naming), #14 (xAI grok-beta deprecated) |
| Total Open Issues | 9 (#6-14) |
| Scores | Robustness 8/10, Reliability 8/10, Scalability 7/10, Extensibility 7/10, DX 6/10 |

### Changes from Run #1
- More tests passing (830 vs 745) — more API keys available
- 2 new xAI failures — grok-beta model deprecated between runs
- 5 new issues filed focusing on DX and export consistency
- Test authoring improved — better understanding of actual APIs

## Run #3 — 2026-02-17 (third eval, same day)

| Metric | Value |
|--------|-------|
| Version | 0.2.8 |
| E2E Tests | 975 collected, 804 passed, 3 failed, 168 deselected |
| E2E Failures | test_tool_with_pre_post_hooks (#8), TestXAI x2 (#14) — all pre-existing |
| Eval Scripts | 15 written, 381 checks total, 381 passed (100%) |
| mypy | 0 errors (215 files) |
| ruff check | 0 warnings |
| ruff format | 0 issues |
| New Issues Filed | 0 (no new bugs found) |
| Total Open Issues | 9 (#6-14) |
| Scores | Robustness 9/10, Reliability 9/10, Scalability 8/10, Extensibility 9/10 |

### Changes from Run #2
- Eval pass rate: 69% → 100% (all scripts use verified correct API signatures)
- All 15 scripts cover: imports, agent construction, mock run, middleware, tools, skills, knowledge, memory, readers, guardrails, auth, testing utils, DX, real API calls, streaming, real-world scenarios
- No new bugs found — library is stable at v0.2.8
- 5 DX observations noted (not bugs): Agent accepts string model, @tool on class, Knowledge() without vector_db, Guardrails bad input types, session_id doesn't maintain history
