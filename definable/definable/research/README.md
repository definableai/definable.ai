# research

Multi-wave deep research pipeline with Compressed Knowledge Unit (CKU) extraction, knowledge graph accumulation, and gap analysis.

## Quick Start

### Standalone

```python
from definable.models.openai import OpenAIChat
from definable.research import DeepResearch, DeepResearchConfig
from definable.research.search import create_search_provider

model = OpenAIChat(id="gpt-4o-mini")
researcher = DeepResearch(
  model=model,
  search_provider=create_search_provider("duckduckgo"),
  config=DeepResearchConfig(depth="standard"),
)

result = await researcher.arun("What are the latest developments in quantum computing?")
print(result.context)   # Formatted context for system prompt injection
print(result.metrics)   # Performance metrics
print(result.sources)   # Sources consulted
```

### Agent Integration

```python
from definable.agents import Agent
from definable.models.openai import OpenAIChat

# Simple enable — standard depth, DuckDuckGo search
agent = Agent(
  model=OpenAIChat(id="gpt-4o"),
  deep_research=True,
)
output = await agent.arun("Compare React and Vue frameworks")

# Custom configuration
from definable.research import DeepResearchConfig

agent = Agent(
  model=OpenAIChat(id="gpt-4o"),
  deep_research=DeepResearchConfig(
    depth="deep",
    max_sources=30,
    max_waves=5,
    include_citations=True,
  ),
)
```

## Module Structure

```
research/
├── __init__.py           # Public exports: DeepResearch, DeepResearchConfig, data models
├── config.py             # DeepResearchConfig dataclass with depth presets
├── engine.py             # DeepResearch orchestrator — multi-wave pipeline
├── models.py             # Data models: ResearchResult, CKU, Fact, SourceInfo, etc.
├── planner.py            # Query decomposition into sub-questions
├── compressor.py         # Page content → CKU extraction via cheap model
├── reader.py             # Async page reading with rate limiting
├── knowledge_graph.py    # Knowledge accumulation, dedup, contradiction detection
├── gap_analyzer.py       # Gap assessment between waves
├── synthesizer.py        # Final context/report generation
├── prompts.py            # LLM prompts for planning, compression, synthesis
└── search/               # Search provider backends
    ├── base.py           # SearchProvider protocol, SearchResult, create_search_provider
    ├── duckduckgo.py     # DuckDuckGo (free, default)
    ├── google.py         # Google Custom Search Engine (httpx)
    └── serpapi.py        # SerpAPI (httpx)
```

## How It Works

The pipeline follows this flow for each wave:

1. **Decompose** — Break the query into sub-questions (`planner.py`)
2. **Search** — Run parallel web searches for each sub-question (`search/`)
3. **Read** — Fetch and extract content from result pages (`reader.py`)
4. **Compress** — Extract CKUs from each page via a cheap model (`compressor.py`)
5. **Accumulate** — Add facts to knowledge graph with dedup and contradiction detection (`knowledge_graph.py`)
6. **Gap Analysis** — Assess coverage, identify remaining gaps (`gap_analyzer.py`)
7. **Repeat** — If gaps remain and waves budget allows, generate follow-up queries and repeat
8. **Synthesize** — Produce formatted context and report (`synthesizer.py`)

## API Reference

### DeepResearch

```python
from definable.research import DeepResearch
```

The main orchestrator class.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Model` | — | Model for planning and synthesis |
| `search_provider` | `SearchProvider` | — | Search backend |
| `compression_model` | `Optional[Model]` | `None` | Cheap model for CKU extraction (defaults to `model`) |
| `config` | `Optional[DeepResearchConfig]` | `None` | Pipeline configuration |

**Methods:**

| Method | Description |
|--------|-------------|
| `arun(query, on_progress=None)` | Execute the full research pipeline. Returns `ResearchResult`. |

### DeepResearchConfig

```python
from definable.research import DeepResearchConfig
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `True` | Whether deep research is active |
| `depth` | `"quick" \| "standard" \| "deep"` | `"standard"` | Research depth preset |
| `search_provider` | `str` | `"duckduckgo"` | Backend: `"duckduckgo"`, `"google"`, `"serpapi"` |
| `search_provider_config` | `Optional[Dict]` | `None` | Backend-specific config (API keys, CSE ID) |
| `search_fn` | `Optional[Callable]` | `None` | Custom search callable (overrides `search_provider`) |
| `compression_model` | `Optional[Model]` | `None` | Model for CKU extraction |
| `max_sources` | `int` | `15` | Max unique sources across all waves |
| `max_waves` | `int` | `3` | Max research waves |
| `parallel_searches` | `int` | `5` | Concurrent search queries per wave |
| `parallel_reads` | `int` | `10` | Concurrent page reads |
| `min_relevance` | `float` | `0.3` | Min relevance score for CKU inclusion |
| `include_citations` | `bool` | `True` | Include source citations in context |
| `include_contradictions` | `bool` | `True` | Surface contradictions between sources |
| `context_format` | `"xml" \| "markdown"` | `"xml"` | Format for the injected context |
| `max_context_tokens` | `int` | `4000` | Approx token budget for context block |
| `early_termination_threshold` | `float` | `0.2` | Stop when novelty drops below this |
| `trigger` | `"always" \| "auto" \| "tool"` | `"always"` | When to run research |

**Depth presets:**

| Preset | Waves | Max Sources | Parallel Searches |
|--------|-------|-------------|-------------------|
| `"quick"` | 1 | 8 | 3 |
| `"standard"` | 3 | 15 | 5 |
| `"deep"` | 5 | 30 | 8 |

### Output Types

#### ResearchResult

```python
from definable.research import ResearchResult
```

| Field | Type | Description |
|-------|------|-------------|
| `context` | `str` | Formatted context for system prompt injection |
| `report` | `str` | Standalone research report |
| `sources` | `List[SourceInfo]` | Sources consulted |
| `facts` | `List[Fact]` | All extracted facts |
| `gaps` | `List[TopicGap]` | Remaining knowledge gaps |
| `contradictions` | `List[Contradiction]` | Contradictions between sources |
| `sub_questions` | `List[str]` | Decomposed sub-questions |
| `metrics` | `ResearchMetrics` | Performance metrics |

#### ResearchMetrics

| Field | Type | Description |
|-------|------|-------------|
| `total_time_ms` | `float` | Total research time |
| `total_sources_found` | `int` | Total search results found |
| `total_sources_read` | `int` | Pages actually read |
| `total_facts_extracted` | `int` | Raw facts extracted |
| `unique_facts` | `int` | After deduplication |
| `contradictions_found` | `int` | Contradictions detected |
| `gaps_identified` | `int` | Knowledge gaps found |
| `gaps_filled` | `int` | Gaps resolved across waves |
| `waves_executed` | `int` | Research waves completed |
| `compression_ratio_avg` | `float` | Average CKU compression ratio |

#### CKU

Compressed Knowledge Unit — structured extraction from a single page.

| Field | Type | Description |
|-------|------|-------------|
| `source_url` | `str` | Page URL |
| `source_title` | `str` | Page title |
| `query_context` | `str` | Sub-question that led to this page |
| `facts` | `List[Fact]` | Extracted facts |
| `relevance_score` | `float` | Relevance to the query |
| `raw_token_count` | `int` | Original page tokens |
| `compressed_token_count` | `int` | CKU tokens |
| `compression_ratio` | `float` | Compression ratio |

#### Fact, SourceInfo, TopicGap, Contradiction

See `research/models.py` for full field definitions.

### Events

When used with an agent, the research pipeline emits events for streaming:

| Event | Fields | Description |
|-------|--------|-------------|
| `DeepResearchStartedEvent` | `query` | Research pipeline started |
| `DeepResearchProgressEvent` | `wave`, `sources_read`, `facts_extracted`, `gaps_remaining`, `message` | Per-wave progress |
| `DeepResearchCompletedEvent` | `total_sources`, `total_facts`, `total_time_ms` | Pipeline finished |

### Search Providers

```python
from definable.research.search import create_search_provider
```

| Provider | Config | Notes |
|----------|--------|-------|
| `"duckduckgo"` | None | Free, no API key, default |
| `"google"` | `{"api_key": "...", "cse_id": "..."}` | Google Custom Search Engine |
| `"serpapi"` | `{"api_key": "..."}` | SerpAPI |
| Custom | `search_fn=my_search` | Any `async (query, max_results) -> List[SearchResult]` |

## See Also

- `agents/` — Agent integration via `deep_research` param
- `run/base.py` — `RunContext.research_context` and `RunContext.research_result`
- `knowledge/` — RAG pipeline (complementary to research)
