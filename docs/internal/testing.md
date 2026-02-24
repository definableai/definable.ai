# Testing Conventions

> Load this doc when writing or modifying tests.

## Structure
```
tests/
  unit/          — Fast, no API calls, mock everything external
  integration/   — Real API calls (needs .env.test sourced)
  regression/    — Bug-specific tests tied to GitHub issues
  conftest.py    — Shared fixtures
  data/          — Test fixtures, sample files
```

## Running Tests
```bash
# All tests
.venv/bin/python -m pytest definable/tests/

# Specific category
.venv/bin/python -m pytest definable/tests/unit/
.venv/bin/python -m pytest definable/tests/integration/
.venv/bin/python -m pytest definable/tests/regression/

# Single file
.venv/bin/python -m pytest definable/tests/unit/test_agent.py -v

# With coverage (if needed)
.venv/bin/python -m pytest definable/tests/unit/ --cov=definable
```

## MockModel Usage
```python
from definable.agent import MockModel, create_test_agent

# Basic
mock = MockModel(responses=["response1", "response2"])
agent = create_test_agent(model=mock, tools=[...])

# GOTCHA: call_count is NOT incremented with side_effect
# Always use: len(mock.call_history)
```

## Test Writing Rules
- Every new feature gets unit tests IN THE SAME Claude Code session (better quality)
- Every bug fix gets a regression test in `tests/regression/`
- Tests must be deterministic — no random, no time-dependent assertions
- Mock all external APIs in unit tests
- Integration tests require `source .env.test` first
- Name test files to match source: `agent.py` → `test_agent.py`
- Use descriptive test names: `test_agent_raises_on_none_model`

## Quality Gates (must pass before commit)
```bash
.venv/bin/python -m pytest definable/tests/unit/      # tests pass
.venv/bin/ruff check definable/definable/              # no lint errors
.venv/bin/ruff format definable/definable/             # formatted
.venv/bin/python -m mypy definable/definable/          # type clean
```
