# Definable — Claude Code Agent System

> Context for all Claude Code agents operating on this project.

## Project Quick Reference

- **Package**: `definable` v0.2.8
- **Library source**: `definable/definable/`
- **Tests**: `definable/tests_e2e/`
- **Examples**: `definable/examples/`
- **Python**: >=3.12, managed with .venv
- **Credentials**: `.env.test` (export KEY="value" format)
- **Activate**: `source .venv/bin/activate`
- **Install**: `pip install -e ".[readers,serve,jwt,cron,runtime]"`
- **Repo root**: Use `git rev-parse --show-toplevel` (never hardcode paths)

## Agents

### 1. Evaluator (`/evaluate`, `/smoke-test`)
- **Purpose**: Stress-test the library, find bugs, file issues
- **Library access**: READ ONLY — never writes to `definable/`
- **Writes to**: `.workspace/`, `.claude/reports/`, `.claude/memory/`
- **Permissions**: Enforced by `settings.json`
- **Trigger**: Manual via `/evaluate` or `/smoke-test`

### 2. Issue Fixer (`/fix-issue <N>`)
- **Purpose**: Automatically fix GitHub issues, raise PRs
- **Library access**: READ/WRITE — creates branches, edits source, adds tests
- **Writes to**: `definable/`, `definable/tests_e2e/`, `.workspace/`, `.claude/memory/`
- **Permissions**: Runs with `--dangerously-skip-permissions` (bypasses `settings.json`)
- **Trigger**: GitHub webhook, Actions on `claude-fix` label, or manual `/fix-issue <N>`

### 3. Supporting Agents
- **api-explorer**: Read-only codebase mapper → writes to `.claude/memory/project-profile.md`
- **issue-filer**: Files GitHub issues with developer-grade reproduction steps
- **memory-keeper**: Manages 6 persistent memory files (never overwrites, always merges)

## Directory Rules

| Directory | Evaluator | Fixer | Purpose |
|-----------|-----------|-------|---------|
| `definable/definable/**` | ❌ READ ONLY | ✅ READ/WRITE | Library source |
| `definable/tests_e2e/**` | ❌ READ ONLY | ✅ READ/WRITE | Tests (fixer adds new ones) |
| `definable/examples/**` | ❌ READ ONLY | ❌ READ ONLY | Usage examples |
| `.workspace/` | ✅ READ/WRITE | ✅ READ/WRITE | Temp files — gitignored |
| `.claude/reports/` | ✅ READ/WRITE | ❌ N/A | Eval reports — gitignored |
| `.claude/memory/` | ✅ READ/WRITE | ✅ READ/WRITE | Persistent memory |
| `.env.test` | ✅ READ/WRITE | ✅ READ | API credentials — gitignored |

**Note**: `settings.json` denies writes to `definable/**`. The issue-fixer bypasses this via `--dangerously-skip-permissions`. The evaluator must ALWAYS respect these deny rules.

## Memory System — 6 Files

| File | Content | Written By |
|------|---------|------------|
| `credentials.md` | API key availability (NOT the keys themselves) | Evaluator, Setup |
| `project-profile.md` | Library structure, exports, version | API Explorer |
| `evaluation-history.md` | Past eval results (APPEND-ONLY) | Evaluator |
| `known-issues.md` | Filed issues tracker (merge, don't duplicate) | Evaluator, Issue Filer |
| `fix-history.md` | Past fix attempts and results (APPEND-ONLY) | Issue Fixer |
| `user-preferences.md` | Timeout, skip providers, labels | Setup, User |

**Rule**: Never overwrite memory files. Always read → merge → write.

## Common Commands

```bash
# Find repo root (use this, never hardcode paths)
cd "$(git rev-parse --show-toplevel)"

# Install with all extras
pip install -e ".[readers,serve,jwt,cron,runtime]"

# Test — offline only (same filter as CI)
pytest definable/tests_e2e/ \
  -m "not openai and not deepseek and not moonshot and not xai and not telegram and not discord and not signal and not postgres and not redis and not qdrant and not chroma and not mongodb and not pinecone and not mistral and not mem0" \
  -v --tb=short

# Lint + format
ruff check definable/definable/
ruff format definable/definable/

# Type check
mypy definable/definable/ --ignore-missing-imports
```

## Code Conventions (ALL agents must follow)

1. **Protocol-based extensibility** — `@runtime_checkable Protocol`
2. **Frozen dataclasses for config** — `AgentConfig`, `MemoryConfig`, etc.
3. **`to_dict()` / `from_dict()`** serialization on all types
4. **Lazy imports** — `TYPE_CHECKING` + `__getattr__` in `__init__.py`
5. **Explicit `__all__`** in every `__init__.py`
6. **Google-style docstrings** with full type annotations
7. **Sync + async dual API** — `run()` wraps `arun()`
8. **No new required deps** without strong justification
9. **Non-fatal errors** where possible — log and continue
10. **`ruff` for linting**, `mypy` for types, `pytest` for tests

## Module Map

| Module | Purpose | Key Exports | API Key? |
|--------|---------|-------------|----------|
| `agents/` | Agent orchestration | `Agent`, `AgentConfig`, `Toolkit`, `KnowledgeToolkit`, `Middleware` | Yes (model) |
| `models/` | LLM providers | `OpenAIChat`, `OpenAILike`, `DeepSeekChat`, `MoonshotChat`, `xAI` | Yes |
| `tools/` | Tool system | `@tool` decorator → `Function` | No |
| `skills/` | Domain expertise | `Skill`, `Calculator`, `WebSearch`, `DateTime`, `Shell`, etc. | No |
| `knowledge/` | RAG pipeline | `Knowledge`, `Document`, `InMemoryVectorDB`, `VoyageAIEmbedder` | Yes (embedder) |
| `memory/` | Cognitive memory | `CognitiveMemory`, `SQLiteMemoryStore`, 9+ backends | Varies |
| `guardrails/` | Validation layer | `Guardrails`, `max_tokens`, `pii_filter`, `tool_blocklist` | No |
| `readers/` | File parsing | `BaseReader`, parsers for PDF/DOCX/PPTX/XLSX/ODS/RTF/HTML/audio | No |
| `mcp/` | Model Context Protocol | `MCPToolkit`, `MCPConfig`, transports | Depends |
| `interfaces/` | Chat platforms | Discord, Telegram, Signal | Yes (tokens) |
| `research/` | Deep web research | `DeepResearch` engine, search providers | Yes |
| `replay/` | Execution replay | `Replay`, `ReplayComparison` | No |
| `runtime/` | Serve agents | FastAPI server, runner | No |
| `auth/` | Authentication | JWT, API key, allowlist, composite | No |
| `compression/` | Context compression | Compression manager | No |
| `triggers/` | Event triggers | Cron, webhook, event | No |
| `run/` | Execution context | `RunContext`, `RunStatus` | No |
| `filters.py` | Query filtering | `FilterExpr` | No |
| `media.py` | Media types | `Image`, `Audio`, `Video`, `File` | No |
| `exceptions.py` | Error types | `AgentRunException`, `StopAgentRun`, `ModelAuthenticationError`, etc. | No |

## Agent Constructor (exact signature)

```python
Agent(
    *,
    model: Model,                    # REQUIRED — OpenAIChat, MockModel, etc.
    tools: Optional[List[Function]] = None,
    toolkits: Optional[List[Toolkit]] = None,
    skills: Optional[List[Skill]] = None,
    skill_registry: Optional[Any] = None,
    instructions: Optional[str] = None,
    memory: Optional[Any] = None,
    readers: Optional[Any] = None,
    guardrails: Optional[Any] = None,
    thinking: Union[bool, ThinkingConfig, None] = None,
    deep_research: Union[bool, DeepResearchConfig, DeepResearch, None] = None,
    name: Optional[str] = None,
    session_id: Optional[str] = None,
    config: Optional[AgentConfig] = None,
)
```

**Key methods**: `.run()`, `.arun()`, `.run_stream()`, `.arun_stream()`, `.use()` (middleware), `.serve()`

## MockModel (for tests without API keys)

```python
MockModel(
    responses: Optional[List[str]] = None,
    tool_calls: Optional[List[Dict]] = None,
    side_effect: Optional[Callable] = None,
    reasoning_content: Optional[str] = None,
    structured_responses: Optional[List[str]] = None,
)
```

**Methods**: `.ainvoke()`, `.invoke()`, `.ainvoke_stream()`, `.call_count`, `.call_history`, `.reset()`, `.assert_called()`, `.assert_called_times(n)`
