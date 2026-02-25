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
        │     ├── FTSIndex (SQLite FTS5, hybrid search)
        │     ├── HybridSearchConfig (vector + text merge: rrf|weighted)
        │     ├── TemporalDecay (exponential score decay by document age)
        │     ├── MMRConfig (diversity reranking: lambda_param balance)
        │     └── FallbackEmbedder (multi-provider failover)
        ├── DeepResearch → DeepResearchConfig
        ├── Tracing (exporters: JSONLExporter, DebugExporter)
        ├── AudioTranscriber (voice→text, runs in arun before pipeline)
        ├── Security → SecurityConfig (all optional)
        │     ├── ToolPolicy → ToolPolicyGuardrail (deny/allowlist/full)
        │     ├── RateLimitConfig → RateLimitHook (sliding window)
        │     ├── ContentDefenseConfig → ContentDefenseGuardrail (injection detection)
        │     ├── SSRFGuardConfig → SSRFGuard (private IP blocking)
        │     └── EnvSanitizeConfig → sanitize_env (dangerous var stripping)
        ├── UsageTracker (token/cost tracking per run and session)
        ├── Eval → agent/eval (4 eval types: Accuracy, Performance, Reliability, Judge)
        ├── Toolkits[] → MCPToolkit | BrowserToolkit
        ├── Tools[] → Function (decorator-based)
        ├── Skills[] → Skill (instructions + tools)
        ├── Guardrails → input/output/tool checks
        ├── Pipeline → 8 phases (Prepare→Recall→Think→GuardInput→Compose→InvokeLoop→GuardOutput→Store)
        ├── Middleware[] → chain (skipped in streaming)
        ├── Team → agent/team (coordinate/route/collaborate/tasks)
        ├── Workflow → agent/workflow (Step, Steps, Parallel, Loop, Condition, Router)
        └── Interfaces[] → Telegram, Discord, Slack, Call, Desktop, CLI
              ├── Auth → APIKeyAuth, JWTAuth, AllowlistAuth
              ├── CLI → CLIInterface (auto-detects TUI vs REPL)
              │     └── TUI → Textual-based terminal UI (streaming, widgets, metrics)
              └── Call → CallInterface (Twilio/Plivo, managed/cascading/realtime)
                    ├── Telephony → TwilioProvider, PlivoProvider
                    ├── STT → DeepgramSTT
                    ├── TTS → CartesiaTTS
                    └── Realtime → OpenAIRealtimeProvider
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
- **Boolean shortcuts**: `memory=True` → InMemoryStore, `tracing=True` → default exporters, `audio_transcriber=True` → OpenAITranscriber, `security=True` → default SecurityConfig, `usage=True` → UsageTracker
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
