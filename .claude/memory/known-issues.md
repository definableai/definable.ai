# Known Issues

> Tracks all filed issues to prevent duplicates.

## Open Issues

| # | Title | Filed | Labels | Priority |
|---|-------|-------|--------|----------|
| #6 | ModelResponse.parsed never populated for structured output | 2026-02-17 run #1 | bug | P0 |
| #7 | TracingConfig not re-exported from definable.agent.tracing | 2026-02-17 run #1 | bug | P2 |
| #8 | test_tool_with_pre_post_hooks fails: hook signature mismatch | 2026-02-17 run #1 | bug | P2 |
| #9 | Deprecation warning: duckduckgo_search renamed to ddgs | 2026-02-17 run #1 | enhancement | P3 |
| #10 | knowledge: Embedder and Reranker implementations not re-exported | 2026-02-17 run #2 | bug | P1 |
| #11 | knowledge: Document uses meta_data instead of metadata | 2026-02-17 run #2 | bug | P2 |
| #12 | knowledge: Chunker implementations not exported | 2026-02-17 run #2 | bug | P1 |
| #13 | agents: Agent.arun() uses output_schema but docs reference response_model | 2026-02-17 run #2 | bug | P0 |
| #14 | tests: xAI tests fail — grok-beta deprecated, use grok-3 | 2026-02-17 run #2 | bug | P1 |

| #18 | Agent(model=None) silently accepts None, fails at runtime with unhelpful error | 2026-02-19 run #4 | bug | P1 — **VERIFIED FIXED** (eval run #5 confirms TypeError at init) |
| #19 | Sync run() breaks on sequential multi-turn calls (Event loop is closed) | 2026-02-19 run #4 | bug | P0 |

## Closed Issues

(none yet)

## Issue Categories

- **Structured output**: #6, #13 (P0 -- broken end-to-end)
- **Export consistency**: #7, #10, #12 (P1-P2 -- likely fixed by DX overhaul, need verification)
- **Naming/DX**: #11, #13, #18 (P1-P2 -- confusing API / missing validation; #18 FIXED)
- **Tests**: #8, #14 (P1-P2 -- broken tests)
- **Dependencies**: #9 (P3 -- deprecated package name)
- **Runtime**: #19 (P0 -- sync run() multi-turn breaks)

## Notes (Run #7, 2026-02-25)
- **No new issues found** — all 146 eval checks + 3625 unit tests pass
- DX issues #10, #11, #12 likely resolved by the DX overhaul (re-exports added to definable.embedder, definable.chunker, definable.reranker; Document still uses meta_data by design)
- #18 confirmed FIXED since run #5
- Only open GitHub issue: #18 (can be closed)
