# skills

Higher-level abstractions that bundle instructions, tools, and dependencies into reusable capabilities.

## Quick Start

```python
from definable.agents import Agent
from definable.skills import Calculator, DateTime, Skill
from definable.tools.decorator import tool

# Built-in skills
agent = Agent(model=model, skills=[Calculator(), DateTime()])

# Inline skill with custom tools
@tool
def search_docs(query: str) -> str:
  """Search documentation."""
  return docs.search(query)

support = Skill(
  name="support",
  instructions="You are a support specialist. Always cite sources.",
  tools=[search_docs],
)
agent = Agent(model=model, skills=[support])

# Markdown skills via registry
from definable.skills import SkillRegistry

registry = SkillRegistry()  # loads built-in library
agent = Agent(model=model, skill_registry=registry)
```

## Module Structure

```
skills/
├── __init__.py        # Public API exports
├── base.py            # Skill base class
├── markdown.py        # MarkdownSkill, SkillLoader
├── registry.py        # SkillRegistry (eager/lazy)
├── builtin/           # 8 built-in skills
│   ├── calculator.py
│   ├── datetime_skill.py
│   ├── file_ops.py
│   ├── http_requests.py
│   ├── json_ops.py
│   ├── shell.py
│   ├── text_processing.py
│   └── web_search.py
└── library/           # 8 markdown skill files
    ├── code-review.md
    ├── data-analysis.md
    ├── debug-code.md
    ├── explain-concept.md
    ├── plan-project.md
    ├── summarize-document.md
    ├── web-research.md
    └── write-report.md
```

## API Reference

### Skill

```python
from definable.skills import Skill

skill = Skill(
  name="my_skill",                  # Identifier (defaults to class name)
  instructions="Domain expertise",  # Injected into system prompt
  tools=[my_tool],                  # Tool functions
  dependencies={"api_key": "..."},  # Shared config for tools
)
```

| Method | Description |
|--------|-------------|
| `tools` (property) | Returns explicit tools or auto-discovers `@tool` methods |
| `get_instructions()` | Returns instructions for system prompt injection |
| `setup()` | Lifecycle hook called once at agent init (non-fatal) |
| `teardown()` | Cleanup hook called on agent shutdown |

Class-based skills can define `@tool`-decorated methods and override `get_instructions()` for dynamic content.

### Built-in Skills

| Skill | Tools | Description | Optional Deps |
|-------|-------|-------------|---------------|
| `Calculator` | `calculate` | Safe math via AST (`sqrt`, `log`, `pi`, etc.) | — |
| `DateTime` | `get_current_time`, `date_difference` | Current time with timezone, date diffs | — |
| `TextProcessing` | `regex_search`, `regex_replace`, `text_stats`, `text_transform`, `extract_patterns` | Regex, stats, extraction (emails, URLs, etc.) | — |
| `FileOperations` | `read_file`, `list_files`, `write_file`, `append_to_file` | Sandboxed file I/O (`base_dir=`, `allow_write=`) | — |
| `HTTPRequests` | `http_get`, `http_post`, `http_put`, `http_patch`, `http_delete` | HTTP calls (`allowed_domains=`, `timeout=`) | — |
| `JSONOperations` | `parse_json`, `query_json`, `transform_json`, `compare_json` | Parse, query (dot notation), transform, diff | — |
| `Shell` | `run_command` | Shell execution (`blocked_commands=`, `timeout=`) | — |
| `WebSearch` | `search_web`, `fetch_url` | DuckDuckGo search + URL fetching | `duckduckgo-search` |

### MarkdownSkill

```python
from definable.skills import MarkdownSkill, SkillLoader

# Load from file
skill = SkillLoader.load_file(Path("my-skill.md"))

# Load from string
skill = SkillLoader.parse("---\nname: my-skill\n---\n## Steps\n...")
```

A `Skill` backed by markdown content with YAML frontmatter. Provides instructions only (no tools). `get_instructions()` wraps content in `<skill name="...">` tags.

**MarkdownSkillMeta** fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Unique identifier |
| `description` | `str` | `""` | Short description |
| `version` | `str` | `"1.0.0"` | Semver |
| `requires_tools` | `List[str]` | `[]` | Expected tool names |
| `tags` | `List[str]` | `[]` | Searchable tags |
| `author` | `str` | `""` | Skill author |

**SkillLoader** methods:

| Method | Description |
|--------|-------------|
| `parse(text, source_path=)` | Parse markdown string into `MarkdownSkill` |
| `load_file(path)` | Load a single `.md` file |
| `load_directory(directory)` | Recursively load all `.md` files (skips failures) |

### SkillRegistry

```python
from definable.skills import SkillRegistry

registry = SkillRegistry(
  skills=None,              # Explicit MarkdownSkill list
  directories=None,         # Custom directories to load from
  include_library=True,     # Load built-in library skills
)
```

| Method | Description |
|--------|-------------|
| `list_skills()` | Return metadata for all skills, sorted by name |
| `get_skill(name)` | Look up skill by name |
| `search_skills(query)` | Search by keyword/tag (scored: name=3, tag=2, desc=1) |
| `as_eager()` | Return all skills as a list (all injected into prompt) |
| `as_lazy()` | Return single wrapper skill with catalog + `read_skill` tool |

**Eager vs. Lazy mode:**

- **Eager** (`as_eager()`): All skill instructions injected into the system prompt. Best for small collections (<15 skills).
- **Lazy** (`as_lazy()`): Injects a skill catalog table and a `read_skill` tool. The LLM loads skills on demand. Best for large collections (15+ skills).

When using `skill_registry=` on Agent, the mode is chosen automatically based on `len(registry)`.

### Creating Custom Markdown Skills

```markdown
---
name: my-custom-skill
description: One-line description
version: 1.0.0
tags: [tag1, tag2]
requires_tools: [search_web]
---

## When to Use
Describe when the LLM should apply this skill.

## Steps
1. First step
2. Second step

## Output Format
Describe expected output structure.
```

### Built-in Library Skills

| Name | Description | Tags |
|------|-------------|------|
| `code-review` | Systematic code review with severity-ranked findings | code, review, quality |
| `data-analysis` | Structured data analysis with statistical reasoning | data, analysis, statistics |
| `debug-code` | Methodical debugging with hypothesis testing | debug, code, troubleshooting |
| `explain-concept` | Clear explanation of technical concepts | explain, concept, education |
| `plan-project` | Project planning with scope/timeline/risks | plan, project, organization |
| `summarize-document` | Document summarization with key insights | summarize, document, writing |
| `web-research` | Deep research using web search + source synthesis | research, web, search |
| `write-report` | Report writing with structure/clarity/evidence | write, report, documentation |

## See Also

- `agents/` — Agent integration via `skills=` and `skill_registry=` parameters
- `tools/` — `@tool` decorator used by built-in skills
- `examples/skills/` — Runnable examples
