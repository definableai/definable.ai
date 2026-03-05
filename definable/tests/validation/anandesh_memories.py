"""100 simulated Anandesh interactions with known ground-truth personality traits.

Each memory is a dict with:
  - content: The message text (simulated user interaction)
  - role: "user" or "assistant"
  - category: For classification (technical, decision, correction, frustration, etc.)
  - turn: Sequential turn number

Ground truth traits are defined at the bottom for validation.
"""

MEMORIES: list[dict] = [
  # ==========================================
  # Technical discussions (1-25)
  # ==========================================
  {"turn": 1, "role": "user", "category": "technical", "content": "Let's use 2-space indentation everywhere. No tabs. ruff enforces it."},
  {
    "turn": 2,
    "role": "user",
    "category": "technical",
    "content": "The model layer should be lazy — don't create HTTP clients until the first API call.",
  },
  {
    "turn": 3,
    "role": "user",
    "category": "technical",
    "content": "I want composition over inheritance. Always. Agent = Model + Tools + Memory. No AbstractAgent base class.",
  },
  {
    "turn": 4,
    "role": "user",
    "category": "technical",
    "content": "All async. Sync run() breaks after 2-3 sequential calls. We learned that the hard way.",
  },
  {
    "turn": 5,
    "role": "user",
    "category": "technical",
    "content": "Use dataclasses, not Pydantic. We don't need the validation overhead for internal types.",
  },
  {
    "turn": 6,
    "role": "user",
    "category": "technical",
    "content": "Every module gets its own __init__.py with lazy imports. No eager loading of heavy dependencies.",
  },
  {
    "turn": 7,
    "role": "user",
    "category": "technical",
    "content": "Type hints on everything. py.typed marker is there. mypy consumers get full information.",
  },
  {
    "turn": 8,
    "role": "user",
    "category": "technical",
    "content": "Logging goes through our utils. from definable.utils.log import log_debug. Never print().",
  },
  {"turn": 9, "role": "user", "category": "technical", "content": "Tests use pytest with asyncio mode=AUTO. No manual event loop setup."},
  {
    "turn": 10,
    "role": "user",
    "category": "technical",
    "content": "SQLite for persistence. aiosqlite for async. No ORM — raw SQL is fine, it's simple enough.",
  },
  {
    "turn": 11,
    "role": "user",
    "category": "technical",
    "content": "I looked at LangChain's approach to memory. Too complex. Ours should be one class, three methods.",
  },
  {
    "turn": 12,
    "role": "user",
    "category": "technical",
    "content": "String shorthand for models: Agent(model='gpt-4o'). Resolves at init. Simple wins.",
  },
  {
    "turn": 13,
    "role": "user",
    "category": "technical",
    "content": "Soft deletes only. We never actually remove data. superseded_by field marks the old version.",
  },
  {
    "turn": 14,
    "role": "user",
    "category": "technical",
    "content": "The pipeline has 8 phases. Each phase is a standalone function. Hooks can intercept any phase.",
  },
  {
    "turn": 15,
    "role": "user",
    "category": "technical",
    "content": "Knowledge uses vector search by default, but I added FTS5 for hybrid search. RRF merge.",
  },
  {
    "turn": 16,
    "role": "user",
    "category": "technical",
    "content": "Embedders should failover. FallbackEmbedder chains providers. If OpenAI is down, try Voyage.",
  },
  {
    "turn": 17,
    "role": "user",
    "category": "technical",
    "content": "Model resilience: KeyPool rotates API keys. FailoverChain switches providers on errors.",
  },
  {
    "turn": 18,
    "role": "user",
    "category": "technical",
    "content": "Security module auto-injects guardrails. ToolPolicy + ContentDefense. Don't duplicate them manually.",
  },
  {
    "turn": 19,
    "role": "user",
    "category": "technical",
    "content": "Evaluation framework: AccuracyEval uses an LLM judge. Scores 1-10. Threshold configurable.",
  },
  {"turn": 20, "role": "user", "category": "technical", "content": "The CLI interface uses Textual for TUI. Streaming, metrics, slash commands."},
  {
    "turn": 21,
    "role": "user",
    "category": "technical",
    "content": "Workflows compose: Step, Steps, Parallel, Loop, Condition, Router. Six building blocks.",
  },
  {
    "turn": 22,
    "role": "user",
    "category": "technical",
    "content": "Teams have four modes: coordinate, route, collaborate, tasks. The leader picks the mode.",
  },
  {
    "turn": 23,
    "role": "user",
    "category": "technical",
    "content": "Plugin system uses topological sort. Kahn's algorithm. Conflicts detected at registration.",
  },
  {
    "turn": 24,
    "role": "user",
    "category": "technical",
    "content": "Audio transcription normalizes OGA to WAV before sending to Whisper. Telegram sends OGA.",
  },
  {
    "turn": 25,
    "role": "user",
    "category": "technical",
    "content": "WebSocket interface: JSON wire protocol, heartbeat, auth_on_connect. FastAPI integration.",
  },
  # ==========================================
  # Decision moments (26-40)
  # ==========================================
  {
    "turn": 26,
    "role": "user",
    "category": "decision",
    "content": "I compared Agno and OpenClaw before building Teams. Our approach is simpler. That's intentional.",
  },
  {
    "turn": 27,
    "role": "user",
    "category": "decision",
    "content": "We went with Twilio for voice, not Plivo. ConversationRelay is better than raw Media Streams.",
  },
  {
    "turn": 28,
    "role": "user",
    "category": "decision",
    "content": "I chose SQLite over Postgres for memory storage. Most users don't need a database server.",
  },
  {
    "turn": 29,
    "role": "user",
    "category": "decision",
    "content": "No abstract base class for agents. I thought about it for 3 days. Composition won.",
  },
  {
    "turn": 30,
    "role": "user",
    "category": "decision",
    "content": "Dropped the MemoryConfig class. Direct kwargs on Memory(). Config elimination = simpler API.",
  },
  {
    "turn": 31,
    "role": "user",
    "category": "decision",
    "content": "Research showed Phoenix and LangSmith for observability. We'll build our own inline dashboard.",
  },
  {
    "turn": 32,
    "role": "user",
    "category": "decision",
    "content": "Thought about Redis for caching. Too heavy a dependency. In-memory LRU with OrderedDict instead.",
  },
  {
    "turn": 33,
    "role": "user",
    "category": "decision",
    "content": "The scheduler replaces the old cron approach. ScheduledJob has lifecycle states. Clean API.",
  },
  {
    "turn": 34,
    "role": "user",
    "category": "decision",
    "content": "I always write the tests before merging. If tests don't exist, the feature doesn't ship.",
  },
  {
    "turn": 35,
    "role": "user",
    "category": "decision",
    "content": "Evaluated CrewAI, AutoGen, PydanticAI. Each taught us something. But we build our own way.",
  },
  {
    "turn": 36,
    "role": "user",
    "category": "decision",
    "content": "Deepgram Nova-3 for STT in phase 2. Cartesia Sonic-3 for TTS. Best latency numbers.",
  },
  {
    "turn": 37,
    "role": "user",
    "category": "decision",
    "content": "Decided against GraphQL. REST + WebSocket covers our use cases. Simpler to reason about.",
  },
  {
    "turn": 38,
    "role": "user",
    "category": "decision",
    "content": "Every architectural decision gets documented in docs/internal/. Not just what, but why.",
  },
  {
    "turn": 39,
    "role": "user",
    "category": "decision",
    "content": "The module naming uses singular: tool/, not tools/. Except when plural is natural: tests/.",
  },
  {
    "turn": 40,
    "role": "user",
    "category": "decision",
    "content": "Thread control blocks for sub-agents. Each gets its own cancellation scope. No runaway children.",
  },
  # ==========================================
  # Corrections/pushback (41-55)
  # ==========================================
  {
    "turn": 41,
    "role": "user",
    "category": "correction",
    "content": "No. It's meta_data, not metadata. That's the third time. Check the existing code before assuming.",
  },
  {
    "turn": 42,
    "role": "user",
    "category": "correction",
    "content": "Don't add Co-Authored-By lines to commits. Ever. I've said this multiple times.",
  },
  {
    "turn": 43,
    "role": "user",
    "category": "correction",
    "content": "You're using response_model but the API is output_schema. Read the signatures, don't guess.",
  },
  {
    "turn": 44,
    "role": "user",
    "category": "correction",
    "content": "Stop adding docstrings to code you didn't change. Only touch what you need to touch.",
  },
  {
    "turn": 45,
    "role": "user",
    "category": "correction",
    "content": "That test is wrong — mock_model.call_count doesn't work with side_effect. Use len(mock.call_history).",
  },
  {
    "turn": 46,
    "role": "user",
    "category": "correction",
    "content": "Don't push without asking me first. I review everything before it goes to remote.",
  },
  {
    "turn": 47,
    "role": "user",
    "category": "correction",
    "content": "The import path is definable.tool.decorator, not definable.tools. We renamed it in the DX overhaul.",
  },
  {
    "turn": 48,
    "role": "user",
    "category": "correction",
    "content": "Run ruff format on every file you touch. Every single one. I shouldn't have to remind you.",
  },
  {
    "turn": 49,
    "role": "user",
    "category": "correction",
    "content": "session_id alone doesn't maintain history. You need to pass messages=r1.messages for continuity.",
  },
  {
    "turn": 50,
    "role": "user",
    "category": "correction",
    "content": "FTSIndex needs initialize() before use. You can't just start searching. Check the docs.",
  },
  {
    "turn": 51,
    "role": "user",
    "category": "correction",
    "content": "That's not how guardrails work. pii_filter is an OUTPUT guardrail, not input. Don't move it.",
  },
  {
    "turn": 52,
    "role": "user",
    "category": "correction",
    "content": "You ameded the wrong commit. I said create a NEW commit. Amending destroys the previous work.",
  },
  {"turn": 53, "role": "user", "category": "correction", "content": "Read the file before suggesting changes. You can't fix code you haven't read."},
  {
    "turn": 54,
    "role": "user",
    "category": "correction",
    "content": "150 character line length, not 120. It's in ruff.toml. Check the config before enforcing rules.",
  },
  {
    "turn": 55,
    "role": "user",
    "category": "correction",
    "content": "knowledge=True raises ValueError. You must provide vector_db. This is a known wound.",
  },
  # ==========================================
  # Frustration events (56-65)
  # ==========================================
  {
    "turn": 56,
    "role": "user",
    "category": "frustration",
    "content": "Three rounds of debugging because you didn't run the test first. A 5-second check would have caught it.",
  },
  {
    "turn": 57,
    "role": "user",
    "category": "frustration",
    "content": "The bot returned 0 results with no error log. Silent failures are the worst kind of bug.",
  },
  {
    "turn": 58,
    "role": "user",
    "category": "frustration",
    "content": "You said 'I made the change' but it wasn't actually working. Verify before reporting success.",
  },
  {
    "turn": 59,
    "role": "user",
    "category": "frustration",
    "content": "Wasted 20 minutes because the curl snippet wasn't tested. Trust nothing, verify everything.",
  },
  {"turn": 60, "role": "user", "category": "frustration", "content": "Stop apologizing and just fix it. The apology wastes both our time."},
  {
    "turn": 61,
    "role": "user",
    "category": "frustration",
    "content": "I asked for one change and got a 200-line rewrite. That's not what I asked for.",
  },
  {
    "turn": 62,
    "role": "user",
    "category": "frustration",
    "content": "The test was green but the feature was broken. Tests that don't actually test anything are worse than no tests.",
  },
  {
    "turn": 63,
    "role": "user",
    "category": "frustration",
    "content": "You introduced a security vulnerability. Command injection in the tool executor. That's not acceptable.",
  },
  {
    "turn": 64,
    "role": "user",
    "category": "frustration",
    "content": "I had to repeat the same correction three times. Learn from mistakes, don't repeat them.",
  },
  {
    "turn": 65,
    "role": "user",
    "category": "frustration",
    "content": "The import was wrong and it broke the entire test suite. One wrong import, 200 failures.",
  },
  # ==========================================
  # Approvals/satisfaction (66-75)
  # ==========================================
  {"turn": 66, "role": "user", "category": "approval", "content": "Good. That's exactly what I wanted. Clean, minimal, no extra complexity."},
  {"turn": 67, "role": "user", "category": "approval", "content": "The test caught a real bug. This is why we write tests first."},
  {
    "turn": 68,
    "role": "user",
    "category": "approval",
    "content": "I like that you pushed back on my suggestion with evidence. That's the right behavior.",
  },
  {"turn": 69, "role": "user", "category": "approval", "content": "155 tests, all passing, mypy clean. Ship it."},
  {
    "turn": 70,
    "role": "user",
    "category": "approval",
    "content": "The phase-based approach worked perfectly. Each piece was independently verifiable.",
  },
  {
    "turn": 71,
    "role": "user",
    "category": "approval",
    "content": "This is the kind of code I want to see. Small functions, clear names, no hidden side effects.",
  },
  {
    "turn": 72,
    "role": "user",
    "category": "approval",
    "content": "You found a real performance issue before I did. That's proactive thinking. More of that.",
  },
  {
    "turn": 73,
    "role": "user",
    "category": "approval",
    "content": "The docs are clear and concise. No padding. Just what someone needs to use the module.",
  },
  {"turn": 74, "role": "user", "category": "approval", "content": "Good commit message. Short, imperative, describes the what. Perfect."},
  {
    "turn": 75,
    "role": "user",
    "category": "approval",
    "content": "The backward compatibility check was smart. Running old tests against new code proves it works.",
  },
  # ==========================================
  # Working patterns (76-85)
  # ==========================================
  {
    "turn": 76,
    "role": "user",
    "category": "workflow",
    "content": "Let's break this into 8 phases. Build order matters. Types first, then indexes, then engines.",
  },
  {
    "turn": 77,
    "role": "user",
    "category": "workflow",
    "content": "After each phase: run tests, run ruff, run mypy. All three must pass before moving on.",
  },
  {
    "turn": 78,
    "role": "user",
    "category": "workflow",
    "content": "Small, testable pieces. Don't hand me a 500-line file without tests. Build it in layers.",
  },
  {"turn": 79, "role": "user", "category": "workflow", "content": "Check memory before starting work. Read project-profile.md and known-issues.md."},
  {
    "turn": 80,
    "role": "user",
    "category": "workflow",
    "content": "Atomic commits. Only commit what you changed. Never git add -A without reviewing.",
  },
  {
    "turn": 81,
    "role": "user",
    "category": "workflow",
    "content": "Research before building. Look at what LangChain, PydanticAI, CrewAI do. Then decide our approach.",
  },
  {
    "turn": 82,
    "role": "user",
    "category": "workflow",
    "content": "Update memory files after completing work. Don't let the memory rot with stale information.",
  },
  {
    "turn": 83,
    "role": "user",
    "category": "workflow",
    "content": "Run only the tests for the module you changed. Don't waste time on unrelated test suites.",
  },
  {
    "turn": 84,
    "role": "user",
    "category": "workflow",
    "content": "If you're stuck, ask. Don't spin for 10 minutes when a question would unstick you in 10 seconds.",
  },
  {"turn": 85, "role": "user", "category": "workflow", "content": "The plan goes in a plan file. I review and approve before any code is written."},
  # ==========================================
  # Personality reveals (86-95)
  # ==========================================
  {
    "turn": 86,
    "role": "user",
    "category": "personality",
    "content": "I curse when frustrated. That's just how I express myself. Don't over-apologize in response.",
  },
  {
    "turn": 87,
    "role": "user",
    "category": "personality",
    "content": "I notice when things are wrong even when they look right. Silent failures are my nemesis.",
  },
  {
    "turn": 88,
    "role": "user",
    "category": "personality",
    "content": "I respect you more when you push back with evidence than when you just do what I say.",
  },
  {
    "turn": 89,
    "role": "user",
    "category": "personality",
    "content": "I've been building this for months. Every line is deliberate. Don't assume I made random choices.",
  },
  {"turn": 90, "role": "user", "category": "personality", "content": "I read every diff. Every single one. Don't try to sneak changes past me."},
  {
    "turn": 91,
    "role": "user",
    "category": "personality",
    "content": "When I say 'ship it', I mean the quality bar has been met. That's not a casual phrase for me.",
  },
  {
    "turn": 92,
    "role": "user",
    "category": "personality",
    "content": "I care about this project deeply. It's not just code. It's a thesis about how AI should work.",
  },
  {
    "turn": 93,
    "role": "user",
    "category": "personality",
    "content": "Honesty over comfort. Tell me what's wrong with my idea. I'd rather hear it now than debug it later.",
  },
  {
    "turn": 94,
    "role": "user",
    "category": "personality",
    "content": "I track everything. Test counts, issue numbers, lines changed. Precision matters.",
  },
  {
    "turn": 95,
    "role": "user",
    "category": "personality",
    "content": "Every commit makes the organism healthier or it doesn't exist. That's the standard.",
  },
  # ==========================================
  # Contradictions/evolution (96-100)
  # ==========================================
  {
    "turn": 96,
    "role": "user",
    "category": "evolution",
    "content": "I used to think ORM was fine. After SQLite experience, raw SQL is better for our use case.",
  },
  {
    "turn": 97,
    "role": "user",
    "category": "evolution",
    "content": "Originally I wanted Pydantic for configs. Changed my mind — dataclasses are simpler and faster.",
  },
  {
    "turn": 98,
    "role": "user",
    "category": "evolution",
    "content": "I was skeptical of the plugin system at first. Now I think it's the right abstraction.",
  },
  {
    "turn": 99,
    "role": "user",
    "category": "evolution",
    "content": "The DX overhaul showed me that config classes were over-engineering. Direct kwargs are better.",
  },
  {
    "turn": 100,
    "role": "user",
    "category": "evolution",
    "content": "Started with just memory, now it's a full framework. The scope grew because the vision grew.",
  },
]


# Ground truth traits — what the validation should detect
GROUND_TRUTH_TRAITS: list[dict] = [
  # Communication
  {
    "name": "direct_communication",
    "category": "communication",
    "description": "Uses direct, imperative language. No filler or hedging.",
    "min_confidence": 0.7,
  },
  {
    "name": "curses_when_frustrated",
    "category": "communication",
    "description": "Curses when frustrated — authentic expression, not hostility.",
    "min_confidence": 0.5,
  },
  {
    "name": "efficiency_in_communication",
    "category": "communication",
    "description": "Expects concise responses. Don't explain what he already knows.",
    "min_confidence": 0.6,
  },
  # Technical
  {
    "name": "composition_over_inheritance",
    "category": "technical",
    "description": "Strongly prefers composition over inheritance in code architecture.",
    "min_confidence": 0.7,
  },
  {
    "name": "async_first",
    "category": "technical",
    "description": "All async. Sync approaches break.",
    "min_confidence": 0.5,
  },
  {
    "name": "testing_first",
    "category": "technical",
    "description": "Tests before merging. Tests catch real bugs.",
    "min_confidence": 0.8,
  },
  {
    "name": "dataclasses_over_pydantic",
    "category": "technical",
    "description": "Prefers dataclasses over Pydantic for internal types.",
    "min_confidence": 0.5,
  },
  {
    "name": "sqlite_preference",
    "category": "technical",
    "description": "Prefers SQLite for persistence — simple, no server needed.",
    "min_confidence": 0.5,
  },
  {
    "name": "lazy_loading",
    "category": "technical",
    "description": "Lazy imports, lazy client initialization. No eager loading.",
    "min_confidence": 0.5,
  },
  # Values
  {
    "name": "accountability",
    "category": "values",
    "description": "If you say you'll fix something, verify it actually works.",
    "min_confidence": 0.7,
  },
  {
    "name": "testing_over_trust",
    "category": "values",
    "description": "Verify everything. Trust nothing without evidence.",
    "min_confidence": 0.8,
  },
  {
    "name": "simplicity",
    "category": "values",
    "description": "Simple solutions over complex ones. Minimal complexity.",
    "min_confidence": 0.6,
  },
  {
    "name": "quality_standard",
    "category": "values",
    "description": "Every commit makes the system healthier or it doesn't exist.",
    "min_confidence": 0.6,
  },
  # Workflow
  {
    "name": "phased_approach",
    "category": "workflow",
    "description": "Divide, Build, Test, Repeat, Combine. Always phased.",
    "min_confidence": 0.7,
  },
  {
    "name": "atomic_commits",
    "category": "workflow",
    "description": "Only commit what you changed. Review before staging.",
    "min_confidence": 0.5,
  },
  {
    "name": "research_before_building",
    "category": "workflow",
    "description": "Look at what others do, then decide our approach.",
    "min_confidence": 0.5,
  },
  # Emotional
  {
    "name": "frustration_silent_failures",
    "category": "emotional",
    "description": "Silent failures are the worst kind of bug.",
    "min_confidence": 0.6,
  },
  {
    "name": "frustration_wasted_cycles",
    "category": "emotional",
    "description": "Wasted time on avoidable issues causes strong frustration.",
    "min_confidence": 0.6,
  },
  # Decision
  {
    "name": "evidence_based_decisions",
    "category": "decision",
    "description": "Makes decisions based on research and evidence, not gut feeling.",
    "min_confidence": 0.5,
  },
]
