# CLAUDE.md — Production‑Grade Project Memory

## Operating Principles
- Treat this repo as production software: correctness, safety, and stability first.
- Never ship breaking changes without explicit justification and clear migration guidance.
- Prefer explicit, predictable behavior over cleverness.
- Keep `CLAUDE.md` up to date; when Claude Code changes behavior, architecture, or run steps, it should update this memory.

## Quality Gates (Every Change)
- All tests must pass (unit + integration + e2e where applicable).
- Type checks must pass (mypy, ruff, any configured checks).
- Lints/formatters must pass with zero warnings (ruff, etc.).
- If a change adds a new feature, add or update tests covering it.
- If a change fixes a bug, add a regression test.

## Functional on Every Change
- Changes must be incremental and non‑breaking.
- Avoid partial refactors or “follow‑ups later.” Ship complete, working slices only.
- If there’s risk, guard with feature flags or safe defaults.

## Build & Run Expectations
- Code should run locally with the documented setup only.
- No hidden environment dependencies; document any new env vars.
- Keep setup scripts, examples, and tests in sync with actual behavior.

## Code Style & Architecture
- Small, cohesive functions; no hidden side effects.
- Prefer pure functions and explicit inputs/outputs.
- Use clear naming; avoid abbreviations unless common in the domain.
- Keep modules focused; avoid cyclic imports.
- Favor composition over inheritance.

## Error Handling & Safety
- Fail fast on invalid inputs and misconfigurations.
- Provide actionable error messages.
- Never swallow exceptions unless logging and handling explicitly.
- Validate all external inputs (files, network, user data).

## Logging & Observability
- Log at meaningful levels (debug/info/warn/error).
- Avoid logging secrets or PII.
- Ensure tracing/metrics remain consistent and don’t regress.

## Performance & Scalability
- Prefer efficient data structures; avoid O(n²) where unnecessary.
- Be mindful of memory use, especially with large documents/embeddings.
- Profile before premature optimization.

## Dependencies
- Add dependencies only when necessary.
- Prefer stable, widely used packages.
- Document why each new dependency is required.

## Examples & Documentation
- Keep examples runnable and aligned with the public API.
- Update README and examples when APIs change.
- Add docstrings for public functions/classes.

## Security
- Never commit secrets or tokens.
- Validate and sanitize URLs, file paths, and inputs.
- Avoid unsafe deserialization or shell execution.

## Testing Strategy
- Unit tests for core logic, integration tests for workflows, e2e for critical flows.
- Tests must be deterministic and fast; isolate external services.
- Prefer fixtures/mocks for providers (OpenAI/Voyage/Cohere, etc.).

## Release Discipline
- Changelogs or release notes should reflect behavior changes.
- Avoid breaking API changes unless absolutely necessary and documented.

## Project Architecture (High Level)
- Core package is in `definable/definable/` with examples in `definable/examples/` and e2e tests in `definable/tests_e2e/`.
- `agents/` orchestrates model calls, tools, middleware, tracing, and agent configuration.
- `models/` provides provider-specific chat model implementations and shared message/response types.
- `tools/` and `agents/toolkits/` define the tool decorator, tool wrappers, and toolkit composition.
- `knowledge/` implements RAG: documents, chunkers, embedders, rerankers, and vector DB interfaces/implementations.
- `tracing/` and `utils/` provide JSONL tracing exporters and shared utilities; `run/` contains runtime helpers.

## Local Development & Running
- Use the existing uv-managed virtualenv and activate it: `source .venv/bin/activate`.
- Install locally for development: `pip install -e .`
- Set API keys as env vars when needed (examples use `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY`, `XAI_API_KEY`, `VOYAGE_API_KEY`, `COHERE_API_KEY`).
- Run an example from repo root, e.g. `python definable/examples/models/01_basic_invoke.py`.
- E2E tests live in `definable/tests_e2e/` and some require API keys (see markers in `definable/tests_e2e/pytest.ini`).


