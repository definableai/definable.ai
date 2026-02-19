# CLAUDE.md — Definable AI Framework

## STOP — Read Before Every Action

### Code Style (ENFORCED BY HOOKS — ruff auto-formats on every save)
- **2-space indentation** (NOT 4-space) — ruff.toml is the authority
- 150 character line length
- Double quotes for strings
- Python: `.venv/bin/python` (3.12.10) — never use system `python`
- Run `.venv/bin/ruff format <file>` on every file you touch
- Logging: `from definable.utils.log import log_debug, log_info, log_warning, log_error`

### Git Rules
- NEVER add "Co-Authored-By" lines to commits
- NEVER amend without explicit user request
- NEVER push without explicit user request
- NEVER force-push to main

### Strict Important Rules
- EVERYTIME you do a deep-reasearch, you solutions muste be backed by solid research
- EVERYTIME you do any type of research store in memory folder.
- ALWAYS check in the memory folder if related memory is present.
- ALWAYS validate and invalidate the memory
- ALWYAS do a in-depth research while planning
- ALWAYS give new innovative ideas to make this lib more good while planning.

### API Surface — Use These Exact Signatures

**Models** — instantiate or use string shorthand:
```python
from definable.model.openai import OpenAIChat
model = OpenAIChat(id="gpt-4o-mini")
# invoke: model.invoke(messages=[Message(...)], assistant_message=Message(role="assistant", content=""))
# assistant_message is REQUIRED second positional arg
```

**Agents** — lego-style composition:
```python
from definable.agent import Agent
agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), tools=[...], instructions="...")
# Or string model shorthand (format: "provider/model-id"):
agent = Agent(model="openai/gpt-4o-mini", instructions="...")
# Supported providers: openai, deepseek, moonshot, xai
# e.g. "deepseek/deepseek-chat", "xai/grok-3", "moonshot/kimi-k2-turbo-preview"
result = await agent.arun("prompt")  # result.content has the text
# Structured output: await agent.arun("prompt", output_schema=MyModel) — NOT response_model
```

**Tools** — decorator-based:
```python
from definable.tool.decorator import tool
@tool
def my_tool(arg: str) -> str:
  """Tool description."""
  return result
```

**Knowledge** — Document uses `meta_data` (NOT `metadata`):
```python
from definable.knowledge import Document, Knowledge
doc = Document(content="...", meta_data={"source": "file.pdf"})
```

**VectorDB** — import from `definable.vectordb` (NOT from knowledge):
```python
from definable.vectordb import InMemoryVectorDB  # or PgVector, Qdrant, ChromaDb, etc.
db = InMemoryVectorDB()
db.insert(docs)  # or db.insert(content_hash, docs)
results = db.search("query", limit=5)
```

**Memory** — snap directly into Agent (no config wrapper needed):
```python
from definable.memory import Memory, SQLiteStore
agent = Agent(model=model, memory=Memory(store=SQLiteStore("./memory.db")))
# Or for quick testing:
agent = Agent(model=model, memory=True)  # uses InMemoryStore
```

**Knowledge** — snap directly into Agent (no config wrapper needed):
```python
from definable.knowledge import Knowledge
from definable.vectordb import InMemoryVectorDB
agent = Agent(model=model, knowledge=Knowledge(vector_db=InMemoryVectorDB(), top_k=5))
```

**Embedders** — import from top-level or deep path:
```python
from definable.embedder import OpenAIEmbedder, VoyageAIEmbedder
# Or deep: from definable.knowledge.embedder.voyageai import VoyageAIEmbedder
```

**Auth** — use correct param names:
```python
from definable.agent.auth import APIKeyAuth, AllowlistAuth
auth = APIKeyAuth(keys={"key1", "key2"})  # NOT api_keys
auth = AllowlistAuth(user_ids={"user1"})   # NOT allowed_ids
```

**MCPToolkit** — config object, not individual params:
```python
from definable.mcp import MCPToolkit, MCPConfig
toolkit = MCPToolkit(config=MCPConfig(...))
```

**Middleware** — `__call__` protocol:
```python
class MyMiddleware:
  async def __call__(self, context, next_handler):  # NOT before_run/after_run
    result = await next_handler(context)
    return result
```

**Multi-turn** — `session_id` alone does NOT maintain history:
```python
# Need messages=r1.messages OR Memory for history
r2 = await agent.arun("follow-up", messages=r1.messages)
```

---

## Project Architecture

### Layout
```
definable/definable/     — core library package
definable/examples/      — runnable examples per module
definable/tests/         — test suites (unit/, integration/, regression/)
definable/docs/          — Mintlify documentation
```

### Module Map (post-restructure — singular names, agent-scoped nesting)
| Module | Purpose | Key Types |
|--------|---------|-----------|
| `agent/` | Orchestration + agent-scoped features | Agent, AgentConfig, RunOutput |
| `agent/tracing/` | Tracing | Tracing, JSONLExporter |
| `agent/guardrail/` | Input/output/tool checks | Guardrails |
| `agent/interface/` | Chat platforms | TelegramInterface, DiscordInterface |
| `agent/research/` | Deep research | DeepResearch |
| `agent/reasoning/` | Thinking layer | Thinking |
| `agent/replay/` | Replay system | Replay |
| `agent/auth/` | Authentication | APIKeyAuth, JWTAuth, AllowlistAuth |
| `agent/run/` | Run events/output | RunOutput, RunContext |
| `agent/trigger/` | Triggers | cron, interval |
| `model/` | LLM providers | OpenAIChat, DeepSeek, Moonshot, xAI |
| `tool/` | Tool system | `@tool` decorator → Function |
| `toolkit/` | Toolkit base class | Toolkit |
| `skill/` | Skill registry | Skill, SkillRegistry |
| `knowledge/` | RAG pipeline | Knowledge, Document |
| `knowledge/embedder/` | Embedders | OpenAIEmbedder, VoyageAIEmbedder |
| `knowledge/chunker/` | Chunkers | RecursiveChunker |
| `knowledge/reranker/` | Rerankers | CohereReranker |
| `knowledge/reader/` | Knowledge readers | PDFReader, TextReader |
| `vectordb/` | Vector storage | InMemoryVectorDB, PgVector, Qdrant, etc. |
| `memory/` | Conversation memory | Memory, SQLiteStore |
| `mcp/` | MCP protocol | MCPToolkit, MCPClient, MCPConfig |
| `browser/` | Browser automation | BrowserToolkit |
| `reader/` | File parsers | BaseReader |

### Dependency Graph
```
Agent ──┬── Model (lazy client, global HTTP pool) — or string shorthand "gpt-4o"
        ├── Thinking (trigger: always|auto|never)
        ├── Memory (session history with auto-summarization, store: SQLite/File/InMemory)
        ├── Knowledge (top_k, trigger, context_format — wraps VectorDB)
        ├── DeepResearch → DeepResearchConfig
        ├── Tracing (exporters: JSONLExporter, etc.)
        ├── Toolkits[] → MCPToolkit | BrowserToolkit
        ├── Tools[] → Function (decorator-based)
        ├── Skills[] → Skill (instructions + tools)
        ├── Guardrails → input/output/tool checks
        ├── Middleware[] → chain (skipped in streaming)
        └── Interfaces[] → Telegram, Discord, Signal, Desktop
              └── Auth → APIKeyAuth, JWTAuth, AllowlistAuth
```

---

## Development Standards

### Quality Gates (Every Change)
- All tests must pass: `.venv/bin/python -m pytest definable/tests/<category>/`
- Lint: `.venv/bin/ruff check definable/definable/`
- Format: `.venv/bin/ruff format definable/definable/`
- Type check: `.venv/bin/python -m mypy definable/definable/`
- If adding a feature → add tests. If fixing a bug → add regression test.

### Build & Run
- Virtualenv: `source .venv/bin/activate`
- Install: `pip install -e .` (or with extras: `pip install -e ".[mem0-memory,readers,runtime,research]"`)
- API keys: in `.env.test` (gitignored). Source with `source .env.test`
- Run example: `.venv/bin/python definable/examples/models/01_basic_invoke.py`

### Code Principles
- Small, cohesive functions; no hidden side effects
- Composition over inheritance
- Fail fast on invalid inputs with actionable error messages
- Never swallow exceptions silently
- Never commit secrets or tokens
- Incremental, non-breaking changes only

### Testing Strategy
- Unit tests for core logic, integration tests for workflows, e2e for critical flows
- Tests must be deterministic and fast; isolate external services with mocks
- MockModel gotcha: `call_count` is NOT incremented with `side_effect` — use `len(mock_model.call_history)`
- `Agent(knowledge=True)` raises ValueError (unlike other bool params)

---

## Evaluator Agent System

### Commands
| Command | Purpose | Interactive? |
|---------|---------|-------------|
| `/setup` | One-time credential & preference collection | Yes (once) |
| `/evaluate` | Full autonomous evaluation | No |
| `/smoke-test` | Quick import check | No |
| `/memory` | View/manage stored memory | Yes |
| `/file-issue` | File a single bug manually | Yes |

### Autonomy Rules
- Never ask the user anything during `/evaluate` — use stored credentials or skip
- If credentials missing → skip that feature, log in report
- If unsure whether something is a bug → file with `needs-triage` label
- Always write memory files after every run (all 5 files in `.claude/memory/`)

### Persistent Memory
Stored in `.claude/memory/`: `credentials.md`, `project-profile.md`, `evaluation-history.md`, `known-issues.md`, `user-preferences.md`

### Credential Source
All API keys in `.env.test` (gitignored). Source with `source .env.test`.

---

## Snippet Validation Task

When asked to validate documentation snippets, follow this workflow:

### Tools available in `/tmp/definable-validation/`:
- `snippet_extractor.py` — Scans all .md files and examples/, outputs `snippets.json`
- `test_snippet.py` — Runs individual snippets or all snippets against the manifest

### Quick start:
```bash
mkdir -p /tmp/definable-validation
cd /tmp/definable-validation
DEFINABLE_ROOT="$(pwd)" python snippet_extractor.py
python test_snippet.py --manifest snippets.json --all --save
```

### Subagent delegation pattern:
For parallel execution, spawn subagents with:
```
claude -p "Test these snippets against the definable library. For each: run it, if it fails fix it (max 3 tries), report JSON results. PYTHONPATH=/Users/hash/work/definable.ai/definable/definable. Snippets: <json>"
```

### After validation:
1. Generate a markdown report of all results
2. For fixable failures, prepare minimal diffs
3. Ask before applying any changes to source files