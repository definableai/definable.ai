# Known Issues

> Tracks all filed issues to prevent duplicates.

## Open Issues

| # | Title | Filed | Labels | Priority |
|---|-------|-------|--------|----------|
| #6 | ModelResponse.parsed never populated for structured output | 2026-02-17 run #1 | bug | P0 |
| #7 | TracingConfig not re-exported from definable.agents.tracing | 2026-02-17 run #1 | bug | P2 |
| #8 | test_tool_with_pre_post_hooks fails: hook signature mismatch | 2026-02-17 run #1 | bug | P2 |
| #9 | Deprecation warning: duckduckgo_search renamed to ddgs | 2026-02-17 run #1 | enhancement | P3 |
| #10 | knowledge: Embedder and Reranker implementations not re-exported | 2026-02-17 run #2 | bug | P1 |
| #11 | knowledge: Document uses meta_data instead of metadata | 2026-02-17 run #2 | bug | P2 |
| #12 | knowledge: Chunker implementations not exported | 2026-02-17 run #2 | bug | P1 |
| #13 | agents: Agent.arun() uses output_schema but docs reference response_model | 2026-02-17 run #2 | bug | P0 |
| #14 | tests: xAI tests fail — grok-beta deprecated, use grok-3 | 2026-02-17 run #2 | bug | P1 |

## Closed Issues

(none yet)

## Issue Categories

- **Structured output**: #6, #13 (P0 — broken end-to-end)
- **Export consistency**: #7, #10, #12 (P1-P2 — missing re-exports)
- **Naming/DX**: #11, #13 (P2 — confusing API names)
- **Tests**: #8, #14 (P1-P2 — broken tests)
- **Dependencies**: #9 (P3 — deprecated package name)
