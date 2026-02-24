# Architecture — Definable AI Framework

## Package Layout
```
definable/definable/     — core library (215+ .py files)
definable/examples/      — runnable examples per module
definable/tests/         — unit/, integration/, regression/
definable/docs/          — Mintlify public documentation
docs/internal/           — agent context docs (this folder)
```

## Core Design Philosophy
- **Lego-style composition**: Everything snaps into Agent via constructor params
- **No hidden magic**: Explicit > implicit. No global state, no singletons
- **Fail fast**: TypeError/ValueError at init, not at runtime
- **Composition over inheritance**: Mixins and protocols, not deep class hierarchies

## Dependency Graph
```
Agent ──┬── Model (lazy client, global HTTP pool) — or string shorthand "gpt-4o"
        ├── Thinking (trigger: always|auto|never)
        ├── Memory (session history, auto-summarization, store: SQLite/File/InMemory)
        ├── Knowledge (top_k, trigger, context_format — wraps VectorDB)
        ├── DeepResearch → DeepResearchConfig
        ├── Tracing (exporters: JSONLExporter, etc.)
        ├── AudioTranscriber (voice→text, runs in arun before pipeline)
        ├── Toolkits[] → MCPToolkit | BrowserToolkit
        ├── Tools[] → Function (decorator-based)
        ├── Skills[] → Skill (instructions + tools)
        ├── Guardrails → input/output/tool checks
        ├── Middleware[] → chain (skipped in streaming)
        └── Interfaces[] → Telegram, Discord, Desktop
              └── Auth → APIKeyAuth, JWTAuth, AllowlistAuth
```

## Module Boundaries (strict)
| Module | Owns | Never Touches |
|--------|------|---------------|
| `agent/` | Orchestration, run loop, config | Model internals, VectorDB storage |
| `model/` | LLM API calls, message format | Agent state, tool execution |
| `tool/` | @tool decorator, Function schema | Knowledge, Memory |
| `knowledge/` | RAG pipeline, embedders, chunkers | Agent run loop |
| `vectordb/` | Storage backends, search | Embeddings (receives pre-computed) |
| `memory/` | Session stores, summarization | Knowledge retrieval |
| `mcp/` | MCP protocol, server configs | Agent internals |

## Key Patterns
- **String shorthand**: `model="openai/gpt-4o-mini"` resolves at Agent init
- **Boolean shortcuts**: `memory=True` → InMemoryStore, `tracing=True` → default exporters, `audio_transcriber=True` → OpenAITranscriber
- **Exception**: `knowledge=True` raises ValueError (requires vector_db)
- **Middleware chain**: `__call__(context, next_handler)` protocol — NOT before/after hooks
- **Document metadata**: Always `meta_data` (NOT `metadata`) — this is a known quirk, don't "fix" it

## Naming Conventions
- Modules: singular (`agent/`, `model/`, `tool/`)
- Classes: PascalCase (`OpenAIChat`, `InMemoryVectorDB`)
- Decorators: lowercase (`@tool`)
- Config classes: `*Config` suffix (`AgentConfig`, `MCPConfig`)
- Store classes: `*Store` suffix (`SQLiteStore`, `FileStore`)
- Test files: `test_*.py` matching source file names
