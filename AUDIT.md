# Harness v2 — Dead Import Audit

**Generated**: 2026-05-07  
**Branch**: `feat/harness-v2`  
**Scope**: Complete inventory of all files importing from modules being deleted or moved  

---

## Summary

This audit maps every file consuming imports from 20 modules scheduled for deletion and 4 modules scheduled for relocation. The data feeds Phase 1 of the harness rewrite and enables developers to identify rewrite targets in Phases 12-17.

**Affected file counts (preliminary)**:
- Source library files: ~35+ (definable/definable/)
- Test files: ~65+ (tests/)
- Example files: ~8+ (examples/)
- Documentation files: ~79+ (docs/*.mdx)

---

## Section 1: DELETE-ONLY Modules (imports must be rewritten or files deleted)

### `definable.agent.pipeline` and submodules

**Submodules included**:
- `pipeline.debug`
- `pipeline.event_stream`  
- `pipeline.phase`
- `pipeline.pipeline`
- `pipeline.state`
- `pipeline.tool_retry`
- `pipeline.phases.*` (all)
- `pipeline.sub_agent`

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/__init__.py | Re-export of Pipeline | REWRITE — Phase 12 |
| definable/agent/agent.py | Core: Pipeline, DebugConfig, LoopState, SubAgentPolicy, etc. | REWRITE — Phase 12 |
| definable/agent/harness.py | Uses Pipeline, loop phases, state | REWRITE — Phase 12 |
| definable/agent/loop.py | Imports from pipeline.pipeline | REWRITE — Phase 12 |
| definable/agent/resolution.py | Pipeline imports | REWRITE — Phase 12 |
| definable/agent/run/agent.py | Pipeline imports | REWRITE — Phase 12 |
| definable/agent/pipeline/*.py | Internal self-references (18+ files) | DELETE — Phase 17 |

**Test files** (delete or rewrite):
| File | Action |
|------|--------|
| tests/unit/agent/test_pipeline_foundation.py | DELETE — Phase 17 |
| tests/unit/agent/test_pipeline_phases.py | DELETE — Phase 17 |
| tests/unit/agent/test_pipeline_dx.py | DELETE — Phase 17 |
| tests/unit/agent/test_pipeline_activation.py | DELETE — Phase 17 |
| tests/unit/agent/test_pipeline_remove_hook.py | DELETE — Phase 17 |
| tests/unit/agent/test_sub_agents.py | DELETE — Phase 17 |
| tests/unit/agent/test_sub_agent_runtime.py | REWRITE — refactor to new harness |
| tests/unit/agent/test_unified_thinking.py | REWRITE — Phase 15 |

**Example files**:
| File | Action |
|------|--------|
| examples/advanced/05_pipeline_customization.py | REWRITE — Phase 13 |
| examples/advanced/06_all_features.py | REWRITE — Phase 13 |

**Documentation**:
| File | Action |
|------|--------|
| docs/agents/pipeline.mdx | REWRITE — Phase 19 (new harness docs) |

---

### `definable.agent.middleware`

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/__init__.py | Re-export | REWRITE — Phase 12 |
| definable/agent/agent.py | Middleware class usage | REWRITE — Phase 12 |
| definable/agent/layers.py | Imports Middleware | REWRITE — Phase 12 |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/agent/test_middleware.py | DELETE — Phase 17 |
| tests/integration/agent/test_streaming.py | REWRITE |

**Example files**:
| File | Action |
|------|--------|
| examples/advanced/01_middleware.py | DELETE/REWRITE — Phase 13 |

---

### `definable.agent.lifecycle`

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/agent.py | Lifecycle imports | REWRITE — Phase 12 |
| definable/agent/layers.py | Lifecycle imports | REWRITE — Phase 12 |

---

### `definable.agent.layers`

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/agent.py | Imports Layers class | REWRITE — Phase 12 |
| definable/agent/prompt.py | Uses Layers | REWRITE — Phase 12 |
| definable/agent/harness.py | Uses Layers | REWRITE — Phase 12 |

---

### `definable.agent.resolution`

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/agent.py | Resolution logic | REWRITE — Phase 12 |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/model/test_resilient_model_integration.py | REWRITE |

---

### `definable.agent.usage`

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/resolution.py | Usage tracking | REWRITE — Phase 12 |
| definable/agent/__init__.py | Re-export | REWRITE — Phase 12 |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/agent/test_usage_tracker.py | DELETE — Phase 17 |

---

### `definable.agent.guardrail` and submodules

**Submodules**: `builtin`, `composable`, `decorators`

**Source library consumers**:
| File | Count | Action |
|------|-------|--------|
| definable/agent/run/agent.py | Guardrail imports | REWRITE |
| definable/agent/resolution.py | Guardrail resolution | REWRITE — Phase 12 |
| definable/agent/agent.py | Guardrail API | REWRITE — Phase 12 |
| definable/agent/__init__.py | Re-export guardrails | REWRITE — Phase 12 |
| definable/agent/loop.py | Guardrail usage | REWRITE — Phase 12 |
| definable/agent/layers.py | Guardrail integration | REWRITE — Phase 12 |
| definable/agent/security/tool_policy.py | Policy guardrails | KEEP (security module survives) |
| definable/agent/security/content_defense.py | Content guardrails | KEEP (security module survives) |
| definable/agent/interface/cli/renderers/guardrail.py | CLI rendering | KEEP (interface survives) |
| definable/ | **26+ source files** | Mixed |

**Example files**:
| File | Action |
|------|--------|
| examples/guardrails/01_basic_guardrails.py | REWRITE — Phase 13 |
| examples/guardrails/02_custom_guardrails.py | REWRITE — Phase 13 |
| examples/docs/agent_guardrails.py | REWRITE — Phase 13 |
| examples/advanced/06_all_features.py | REWRITE — Phase 13 |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/guardrail/test_guardrail_base.py | DELETE — Phase 17 |
| tests/unit/guardrail/test_guardrail_builtins.py | DELETE — Phase 17 |
| tests/unit/security/test_audit.py | REWRITE |
| tests/unit/agent/test_cli_interface.py | REWRITE |

---

### `definable.agent.security` and submodules

**Submodules**: `audit`, `content_defense`, `env_sanitizer`, `rate_limiter`, `ssrf`, `tool_policy`

**NOTE**: Security module itself survives, but code using specific deleted submodules needs rewriting.

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/agent.py | Security setup | REWRITE — Phase 12 |
| definable/agent/resolution.py | Security resolution | REWRITE — Phase 12 |
| definable/agent/security/*.py | 6 files internal | KEEP (module survives) |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/security/test_*.py | 7 files | KEEP (security tests survive) |

**Example files**:
| File | Action |
|------|--------|
| examples/docs/agent_security.py | KEEP or REWRITE (module survives) |

---

### `definable.agent.research` and submodules

**Submodules**: `engine`, `planner`, `reader`, `search`, `synthesis`, etc.

**Source library consumers**:
| File | Count | Action |
|------|-------|--------|
| definable/agent/agent.py | Research API | REWRITE — Phase 12 |
| definable/agent/resolution.py | Research resolution | REWRITE — Phase 12 |
| definable/agent/__init__.py | Re-export | REWRITE — Phase 12 |
| definable/agent/layers.py | Research layers | REWRITE — Phase 12 |
| definable/agent/research/*.py | 13 files internal | DELETE — Phase 17 |

**Example files**:
| File | Action |
|------|--------|
| examples/research/01_basic_research.py | REWRITE — Phase 13 |
| examples/research/02_agent_with_research.py | REWRITE — Phase 13 |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/agent/test_research.py | DELETE — Phase 17 |

---

### `definable.agent.replay` and submodules

**Submodules**: `compare`, `replay`, `types`

**Source library consumers**:
| File | Count | Action |
|------|-------|--------|
| definable/agent/agent.py | Replay API | REWRITE — Phase 12 |
| definable/agent/__init__.py | Re-export | REWRITE — Phase 12 |
| definable/agent/observability/api.py | Replay integration | KEEP or REWRITE |
| definable/agent/observability/trace_browser.py | Trace replay | KEEP or REWRITE |

**Example files**:
| File | Action |
|------|--------|
| examples/docs/agent_replay.py | REWRITE — Phase 13 |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/agent/test_replay.py | DELETE — Phase 17 |

---

### `definable.agent.tracing` and submodules

**Submodules**: `base`, `debug`, `jsonl`

**Source library consumers**:
| File | Count | Action |
|------|-------|--------|
| definable/agent/agent.py | Tracing API | REWRITE — Phase 12 |
| definable/agent/resolution.py | Tracing resolution | REWRITE — Phase 12 |
| definable/agent/__init__.py | Re-export | REWRITE — Phase 12 |
| definable/agent/config.py | Tracing config | KEEP or REWRITE |
| definable/agent/tracing/*.py | 4 files internal | DELETE — Phase 17 |

**Example files** (42+ files affected):
| File | Action |
|------|--------|
| examples/advanced/02_tracing.py | REWRITE — Phase 13 |
| examples/observability/02_custom_config.py | REWRITE — Phase 13 |
| examples/docs/agent_tracing.py | REWRITE — Phase 13 |

**Test files**:
| File | Count | Action |
|------|--------|
| tests/unit/tracing/test_*.py | 2 files | DELETE — Phase 17 |
| tests/unit/observability/test_*.py | Multiple | REWRITE |
| tests/regression/test_loop_backward_compat.py | DELETE or REWRITE |

---

### `definable.agent.eval` and submodules

**Submodules**: `accuracy`, `base`, `judge`, `performance`, `reliability`

**Source library consumers**:
| File | Count | Action |
|------|-------|--------|
| definable/agent/eval/*.py | 5 files internal | DELETE — Phase 17 |

**Example files**:
| File | Action |
|------|--------|
| examples/docs/evaluation_basics.py | REWRITE — Phase 13 |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/agent/test_eval.py | DELETE — Phase 17 |

---

### `definable.agent.runtime` and submodules

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/agent.py | Runtime API | REWRITE — Phase 12 |
| definable/agent/runtime/*.py | 2 files internal | DELETE or MOVE — Phase 17 |
| definable/agent/interface/call/__init__.py | Runtime integration | REWRITE — Phase 14 |

**Example files**:
| File | Action |
|------|--------|
| examples/docs/agent_runtime.py | REWRITE — Phase 13 |
| examples/docs/agent_auth.py | REWRITE — Phase 13 |
| examples/slack/02_slack_webhook.py | REWRITE |
| examples/observability/_test_dashboard.py | DELETE or REWRITE |

---

### `definable.agent.scheduler` and submodules

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/agent.py | Scheduler API | REWRITE — Phase 12 |
| definable/agent/scheduler/*.py | 3 files internal | DELETE or MOVE — Phase 17 |
| definable/agent/runtime/runner.py | Scheduler integration | REWRITE |

**Example files**:
| File | Action |
|------|--------|
| examples/docs/agent_scheduling.py | REWRITE — Phase 13 |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/agent/test_scheduler.py | DELETE — Phase 17 |
| tests/unit/agent/test_scheduler_integration.py | DELETE — Phase 17 |

---

### `definable.agent.auth` and submodules

**Submodules**: `allowlist`, `api_key`, `composite`, `jwt`

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/agent.py | Auth API | REWRITE — Phase 12 |
| definable/agent/auth/*.py | 4 files internal | DELETE or MOVE |
| definable/agent/runtime/server.py | Auth integration | REWRITE |
| definable/agent/interface/base.py | Auth hooks | REWRITE — Phase 14 |

**Example files**:
| File | Action |
|------|--------|
| examples/auth/01_unified_auth.py | REWRITE — Phase 13 |
| examples/runtime/03_unified.py | REWRITE — Phase 13 |
| examples/docs/agent_auth.py | REWRITE — Phase 13 |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/agent/test_auth.py | DELETE — Phase 17 |

---

### `definable.agent.trigger` and submodules

**Submodules**: `cron`, `event`, `executor`, `interval`, `oneshot`, `webhook`

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/agent.py | Trigger API | REWRITE — Phase 12 |
| definable/agent/trigger/*.py | 6 files internal | DELETE or MOVE |
| definable/agent/runtime/server.py | Trigger integration | REWRITE |
| definable/agent/scheduler/job.py | Trigger/job link | REWRITE |
| agents/school-agent/school_agent/main.py | Trigger usage | REWRITE |

**Example files**:
| File | Action |
|------|--------|
| examples/runtime/01_webhook_basic.py | REWRITE — Phase 13 |
| examples/runtime/02_cron_basic.py | REWRITE — Phase 13 |
| examples/runtime/03_unified.py | REWRITE — Phase 13 |
| examples/auth/01_unified_auth.py | REWRITE — Phase 13 |
| examples/docs/agent_scheduling.py | REWRITE — Phase 13 |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/agent/test_cron_trigger.py | DELETE — Phase 17 |
| tests/unit/agent/test_trigger_types.py | DELETE — Phase 17 |
| tests/unit/agent/test_scheduler_integration.py | DELETE — Phase 17 |
| tests/unit/trigger/test_trigger.py | DELETE — Phase 17 |

---

### `definable.knowledge` (entire module)

**Scope**: All submodules including `chunker`, `embedder`, `reader`, `reranker`, `fts`, `scoring`, `document`

**Source library consumers** (99+ files):
| Category | Count | Action |
|----------|-------|--------|
| definable/knowledge/*.py | 19 files | DELETE — Phase 17 |
| definable/vectordb/*.py | Uses knowledge | REWRITE or KEEP |
| definable/agent/toolkits/knowledge.py | Knowledge toolkit | REWRITE — Phase 12 |
| definable/agent/middleware.py | Knowledge in middleware | REWRITE — Phase 12 |
| definable/agent/pipeline/sub_agent.py | Sub-agent knowledge | REWRITE — Phase 12 |
| definable/agent/research/reader.py | Research uses knowledge | REWRITE/DELETE |
| definable/ top-level imports | Re-exports | REWRITE — Phase 12 |

**Example files** (11+ files):
| File | Action |
|------|--------|
| examples/knowledge/01_basic_rag.py | REWRITE or DELETE |
| examples/knowledge/02_document_management.py | REWRITE or DELETE |
| examples/knowledge/03_chunking_strategies.py | REWRITE or DELETE |
| examples/knowledge/04_custom_embedder.py | REWRITE or DELETE |
| examples/knowledge/05_vector_databases.py | REWRITE or DELETE |
| examples/knowledge/06_agent_with_knowledge.py | REWRITE or DELETE |
| examples/knowledge/07_reranking.py | REWRITE or DELETE |
| examples/toolkits/03_knowledge_toolkit.py | REWRITE — Phase 13 |
| examples/advanced/06_all_features.py | REWRITE — Phase 13 |

**Test files** (20+ files):
| File | Action |
|------|--------|
| tests/unit/knowledge/test_*.py | 15 files | DELETE — Phase 17 |
| tests/integration/knowledge/test_*.py | 3+ files | DELETE — Phase 17 |
| tests/unit/vectordb/test_*.py | Uses knowledge | REWRITE or DELETE |

**Documentation**:
| File | Count | Action |
|------|--------|
| docs/knowledge/*.mdx | Multiple | REWRITE or DELETE — Phase 19 |

---

### `definable.memory.cortex` (submodule only — rest of memory survives)

**NOTE**: Only the cortex submodule is deleted; `definable.memory` and `definable.memory.v2` survive.

**Submodules being deleted**: `cortex/`, including `ingestion`, `retrieval`, `update`, `learning`, `index`

**Source library consumers**:
| File | Count | Action |
|------|-------|--------|
| definable/memory/__init__.py | Re-export | REWRITE or KEEP |
| definable/memory/cortex/*.py | 11 files internal | DELETE — Phase 17 |
| definable/memory/manager.py | Cortex integration | REWRITE — Phase 12 |

**Example files**:
| File | Action |
|------|--------|
| examples/memory/04_cortex_memory.py | DELETE or REWRITE — Phase 13 |

**Test files** (12+ files):
| File | Action |
|------|--------|
| tests/unit/memory/cortex/test_*.py | 11 files | DELETE — Phase 17 |
| tests/validation/cortex_evolution_test.py | DELETE — Phase 17 |

---

### `definable.agent.loop` (top-level loop.py being replaced by agent.core.loop)

**NOTE**: The standalone loop.py at `definable/agent/loop.py` is deleted. A new loop implementation will appear at `definable/agent/core/loop.py`.

**Source library consumers**:
| File | Usage | Action |
|------|-------|--------|
| definable/agent/agent.py | Loop imports | REWRITE — Phase 12 |
| definable/agent/harness.py | Loop usage | REWRITE — Phase 12 |
| definable/agent/pipeline/pipeline.py | Loop integration | DELETE with pipeline |
| definable/agent/pipeline/state.py | Loop state | DELETE with pipeline |
| definable/agent/pipeline/phases/invoke.py | Loop invocation | DELETE with pipeline |
| definable/agent/loop.py | Self (file deleted) | DELETE — Phase 17 |

**Test files**:
| File | Action |
|------|--------|
| tests/unit/model/test_structured_output_parsed.py | REWRITE |
| tests/unit/compression/test_compression.py | REWRITE |
| tests/unit/agent/test_unified_thinking.py | REWRITE |
| tests/unit/agent/test_streaming_tools.py | REWRITE |
| tests/regression/test_loop_backward_compat.py | DELETE — Phase 17 |

---

## Section 2: MOVE (Path changes — imports need rewriting, files survive)

### `definable.tool` → `definable.agent.toolkit`

**Consumer count**: 126+ files across all categories

**Source library consumers**:
| File | Old → New | Action |
|------|-----------|--------|
| definable/tool/__init__.py | Rename to agent/toolkit/tool.py | MOVE — Phase 8 |
| definable/tool/decorator.py | Rename | MOVE — Phase 8 |
| Model integration files | Update imports | REWRITE — Phase 9 |
| definable/__init__.py | Update re-exports | REWRITE — Phase 12 |

**Example files** (24+ files):
| File | Action |
|------|--------|
| examples/tools/*.py | Update imports | REWRITE — Phase 13 |
| examples/agents/02_agent_with_tools.py | Update imports | REWRITE — Phase 13 |
| examples/agents/03_agent_with_toolkit.py | Update imports | REWRITE — Phase 13 |
| examples/docs/tools_*.py | Update imports | REWRITE — Phase 13 |

**Test files** (6+ files):
| File | Action |
|------|--------|
| tests/unit/tool/test_decorator.py | Update imports | MOVE — Phase 9 |
| tests/unit/tool/test_function.py | Update imports | MOVE — Phase 9 |

---

### `definable.toolkit` → `definable.agent.toolkit`

**Consumer count**: 3+ files

**Source library consumers**:
| File | Old → New | Action |
|------|-----------|--------|
| definable/toolkit/__init__.py | Merge into agent/toolkit | MOVE — Phase 8 |
| definable/__init__.py | Update re-exports | REWRITE — Phase 12 |

**Example files**:
| File | Action |
|------|--------|
| examples/docs/toolkits_basics.py | Update imports | REWRITE — Phase 13 |

---

### `definable.skill` → `definable.agent.skill`

**Consumer count**: 46+ files

**Source library consumers**:
| File | Old → New | Action |
|------|-----------|--------|
| definable/skill/*.py | Move to agent/skill/ | MOVE — Phase 8 |
| definable/__init__.py | Update re-exports | REWRITE — Phase 12 |

**Example files** (8+ files):
| File | Action |
|------|--------|
| examples/skills/01_markdown_skills.py | Update imports | REWRITE — Phase 13 |
| examples/skills/02_macos_basic.py | Update imports | REWRITE — Phase 13 |
| examples/skills/02_coding_agent_skills.py | Update imports | REWRITE — Phase 13 |
| examples/skills/03_library_skill_discovery.py | Update imports | REWRITE — Phase 13 |
| examples/docs/skills_basics.py | Update imports | REWRITE — Phase 13 |
| examples/desktop/*.py | Update imports | REWRITE — Phase 13 |

**Test files** (5+ files):
| File | Action |
|------|--------|
| tests/unit/skill/test_*.py | Update imports | MOVE — Phase 9 |

---

### `definable.memory` → `definable.agent.memory` (partial — FileMemory survives only)

**NOTE**: Most memory submodules are deleted. Only `memory.v2` (renamed to `agent.memory.v2`) and `memory.FileMemory` survive.

**Submodules being deleted**: `memory.store` (except file store), `memory.strategies`, `memory.consolidation`

**Submodules being moved**: `memory.v2` → `agent.memory.v2`

**Source library consumers** (87+ files):
| Category | Action |
|----------|--------|
| definable/memory/v2/*.py | MOVE to agent/memory/v2/ — Phase 8 |
| definable/memory/store/file.py | MOVE to agent/memory/file.py — Phase 8 |
| definable/memory/manager.py | Rewrite for new structure — Phase 12 |
| definable/memory/__init__.py | Update re-exports — Phase 12 |
| definable/__init__.py | Update re-exports — Phase 12 |

**Example files** (8+ files):
| File | Action |
|------|--------|
| examples/memory/01_basic_memory.py | Update imports | REWRITE — Phase 13 |
| examples/memory/02_store_protocol.py | Update imports | REWRITE — Phase 13 |
| examples/memory/03_store_backends.py | Update imports | REWRITE — Phase 13 |
| examples/memory/05_memory_v2.py | Update imports | REWRITE — Phase 13 |

**Test files** (20+ files):
| File | Action |
|------|--------|
| tests/unit/memory/test_*.py (not cortex) | Update imports | REWRITE — Phase 9 |
| tests/integration/memory/*.py | Update imports | REWRITE — Phase 9 |

**Documentation** (multiple files):
| File | Action |
|------|--------|
| docs/memory/*.mdx | Update imports | REWRITE — Phase 19 |

---

## Section 3: Interface Adapters (need event-bus port in Phase 14)

**Location**: `definable/agent/interface/`

**Adapters affected**:
- CLI: `/cli/` (67+ files)
- Discord: `/discord/` (3+ files)
- Slack: `/slack/` (3+ files)
- Telegram: `/telegram/` (6+ files)
- WebSocket: `/websocket/` (3+ files)
- Desktop: `/desktop/` (6+ files)
- Email: `/email/` (3+ files)
- WhatsApp: `/whatsapp/` (8+ files)
- Call/VoIP: `/call/` (20+ files)

**Current dependencies** (these must be ported to event bus):
| Module | Used in adapters | Phase 14 action |
|--------|------------------|-----------------|
| `interface.hooks` | All adapters | Rewrite as event subscribers |
| `interface.gateway` | Many | Rewrite as event coordination |
| `interface.session` | All adapters | Rewrite session lifecycle |
| `interface.message` | All adapters | Keep or rewrite |
| `interface.base` | All adapters | Base interface protocol |

**Key files to port**:
| File | Current role | Phase 14 action |
|------|--------------|-----------------|
| interface/hooks.py | Hook protocol | PORT — convert to event subs |
| interface/gateway.py | Gateway coordination | PORT — convert to event bus |
| interface/session.py | Session management | KEEP or PORT |
| interface/base.py | Base interface | KEEP (core protocol) |

**Test impact**:
| File | Count | Action |
|------|-------|--------|
| tests/integration/agent/test_*.py | 6+ | REWRITE — Phase 15 |
| tests/unit/agent/test_desktop_events.py | 1 | REWRITE |
| tests/unit/agent/test_cli_interface.py | 1 | REWRITE |

---

## Section 4: Surprise Findings & Non-Obvious Dependencies

### 1. Guardrails ARE used by security module

**Discovery**: `definable/agent/security/tool_policy.py` and `content_defense.py` both import from `guardrail` module. Since security module survives but guardrails are deleted, these files must either:
- Be rewritten to not use guardrails, OR
- Guardrails must be moved (not deleted)

**Recommendation**: Clarify whether guardrails are truly being deleted or moved to a new module.

### 2. Knowledge is widely distributed across vectordb

**Discovery**: All 7 vectordb implementations (Chroma, MongoDB, Pinecone, PGVector, QDrant, Redis, Qdrant) import from `definable.knowledge`. If knowledge is deleted, vectordb survives but will be non-functional.

**Recommendation**: Either:
- Vectordb needs rewriting to not use knowledge
- Knowledge needs survival as internal api (not public)
- Vectordb deletion is also needed

### 3. Self-references within pipeline

**Discovery**: Pipeline phases import from each other:
- `phase.py` → pipeline imports
- `phases/*.py` → parent imports
- `state.py` → pipeline imports

**Impact**: ~18 internal files all import from the same module set. When pipeline is deleted, these all disappear together (no cross-contamination risk).

### 4. Memory.cortex has a manager dependency

**Discovery**: `definable/memory/manager.py` (which survives) imports from `memory.cortex`. If cortex is deleted, manager needs rewriting.

**Recommendation**: Update manager.py to not depend on cortex.

### 5. Runtime and scheduler are linked

**Discovery**: `trigger/executor.py` imports from both `runtime` and `scheduler`. These are co-deleted modules with internal coupling.

### 6. Interface session management is non-obvious

**Discovery**: All interface adapters use `InterfaceSession` from `interface.session`. This is NOT in the delete list but interfaces are required for Phase 14. Session management pattern is critical — ensure port plan covers this.

### 7. Config classes scattered across modules

**Discovery**: Submodules define their own Config classes (e.g., `pipeline.debug.DebugConfig`, `interface.cli.config.CLIConfig`). When parent is deleted, config is deleted with it. Consumer files (e.g., agent.py) need new config source.

### 8. CLI uses virtually every deleted module

**Discovery**: CLI interface (`interface/cli/`) has renderers and commands that depend on almost every deleted module:
- `renderers/guardrail.py` → guardrail
- `renderers/research.py` → research
- `renderers/sub_agent.py` → pipeline.sub_agent
- `renderers/reasoning.py` → tracing
- Etc.

**Impact**: CLI rewrite is substantial (Phase 14). Consider if CLI adapter survives or is deprecated.

### 9. Re-export chains in __init__.py

**Discovery**: `definable/__init__.py` re-exports 25+ classes from deleted modules. This is the public API surface. Removing without replacement will break all downstream packages.

**Recommendation**: 
- Keep re-exports until replacement harness is ready
- Consider short deprecation window
- Update docs before deletion

### 10. Test file naming patterns

**Discovery**: Pipeline tests are all named `test_pipeline_*.py`, making batch deletion easy but requires confirmation all use OLD pipeline. No cross-references to new harness detected.

---

## Action Codes

- **DELETE** — File is removed entirely (usually internal implementation or legacy tests)
- **MOVE** — File path changes; imports need updating but code largely survives
- **REWRITE** — File survives but all/most imports need rewriting (substantial refactor expected)
- **KEEP** — No change to file or imports
- **PORT** — Interface adapter; rewrite to use event bus instead of hooks/gateway

---

## Phase Mapping (from harness-v2 plan)

| Phase | Scope | Affects audit |
|-------|-------|---|
| 8 | Move tool/toolkit/skill/memory modules | Section 2 (MOVE modules) |
| 9 | Update all moved-module consumers | Section 2 (test updates) |
| 12 | Rewrite agent.py, harness.py, resolution.py | Section 1 (main library) |
| 13 | Update all examples | Sections 1-2 (example files) |
| 14 | Port interface adapters to event bus | Section 3 |
| 15 | Rewrite integration tests | Sections 1-3 |
| 17 | Delete dead module files | Sections 1-2 (DELETE files) |
| 19 | Rewrite documentation | Sections 1-2 (docs) |

---

## Files to Be Deleted (Summary)

**Total deletion set**: ~100+ files across modules

| Module | File count | Notes |
|--------|-----------|-------|
| pipeline/ | 18 | All internal phases, state, helpers |
| middleware.py | 1 | Top-level module |
| lifecycle.py | 1 | Top-level module |
| layers.py | 1 | Top-level module |
| resolution.py | 1 | Top-level module |
| usage.py | 1 | Top-level module |
| guardrail/ | 9 | All subdirs and files |
| research/ | 13 | All subdirs and files |
| replay/ | 3 | All files |
| eval/ | 5 | All files |
| runtime/ | 2 | Runner and server (to be replaced) |
| scheduler/ | 3 | Scheduler and job (to be replaced) |
| auth/ | 4 | All auth implementations |
| trigger/ | 6 | All trigger types |
| knowledge/ | 19 | All subdirs |
| memory/cortex/ | 11 | Cortex only (not full memory) |
| agent/loop.py | 1 | Standalone loop file |
| Tests | 65+ | All pipeline, middleware, research, etc. tests |

---

## Recommendations

### 1. Verification
- [ ] Confirm guardrail deletion (security module depends on it)
- [ ] Confirm knowledge deletion (vectordb depends on it)
- [ ] Confirm runtime/scheduler deletion (interface adapters depend on them)

### 2. Re-export strategy
- [ ] Plan deprecation period for definable.__init__ re-exports
- [ ] Document migration path for downstream users
- [ ] Consider feature flags or shims during transition

### 3. Testing strategy
- [ ] Create new harness tests early (don't wait for Phase 15)
- [ ] Keep old pipeline tests until new ones pass
- [ ] Integration test pack must cover all interface adapters

### 4. Documentation
- [ ] Start migration guide early (Phase 1 deliverable)
- [ ] Update architecture diagrams before deletion phase
- [ ] Maintain v0.7 docs as legacy reference

### 5. Interface adapter decision
- [ ] Decide on CLI/adapter survival before Phase 14
- [ ] If kept, prioritize event-bus port
- [ ] If deprecated, decide deprecation timeline

---

## Generated metadata

- **Audit date**: 2026-05-07
- **Branch**: feat/harness-v2
- **Repository**: definable.ai
- **Python version**: 3.10+
- **Scope**: All Python source, tests, examples, documentation
- **Excluded**: `.egg-info`, `__pycache__`, `.workspace`, `.junk`

