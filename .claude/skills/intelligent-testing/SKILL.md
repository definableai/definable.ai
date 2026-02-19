---
name: intelligent-testing
description: Enforces the project's testing strategy whenever writing, editing, or reviewing tests. Triggers on any test-related work including writing new tests, fixing failing tests, adding test coverage, testing new features, testing integrations, verifying agent behavior, reviewing test quality, or creating test fixtures. Ensures no mocks in integration tests, behavioral assertions for agent tests, contract tests for ABC implementations, and regression snapshots for silent breakage detection. Applies to all files in tests/ directory.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# Intelligent Testing Strategy

## Core Philosophy

This is an agentic AI library. Tests must verify real behavior, not mock fantasies.

- **Mocked embeddings prove nothing** — a real OpenAI call proves everything
- **Checking `.assert_called_once()` on memory is useless** — checking the agent's response contains the right answer is useful
- **A test that always passes is worse than no test** — it gives false confidence

## Five Test Layers

Every piece of code maps to one or more of these layers. Pick the right one.

### Layer 1: Unit Tests (`tests/unit/`)

**For:** Pure logic with ZERO external dependencies. Functions that take input and return output without calling any API, database, or service.

**Examples:** Chunking algorithms, document parsing, config validation, prompt building, signal-based routing keyword matching, metadata extraction.

**Marker:** `@pytest.mark.unit`

```python
# ✅ GOOD unit test — pure logic
def test_recursive_chunker_respects_max_tokens():
    chunker = RecursiveChunker(max_tokens=100, overlap=20)
    doc = Document(content="word " * 500)
    chunks = chunker.chunk(doc)
    assert all(len(c.content) <= 100 for c in chunks)
    assert len(chunks) > 1

# ❌ BAD — this calls an API, it's not a unit test
def test_embedder_returns_vector():
    embedder = OpenAIEmbedder(config=EmbedConfig(model="text-embedding-3-small"))
    result = embedder.embed("hello")  # This is an integration test
```

**Rules:**
- No API keys, no network calls, no database connections
- No mocks either — if you need a mock, it's not a unit test
- Must run in < 1 second
- Test edge cases: empty input, huge input, unicode, malformed data

### Layer 2: Contract Tests (`tests/contract/`)

**For:** Verifying that every implementation of an ABC/base class fulfills the same contract. When someone adds a new `BaseVectorStore` implementation, it must pass the SAME test suite as all others.

**Marker:** `@pytest.mark.contract`

**Pattern — shared test class that implementations inherit:**

```python
# tests/contract/test_vectorstore_contracts.py
"""
Contract tests for BaseVectorStore.
Every implementation MUST inherit this class and pass all tests.
When adding a new vector store, create a new class that inherits
VectorStoreContractTests and provide the store fixture.
"""

class VectorStoreContractTests:
    """Shared contract — all implementations must pass every test here."""

    @pytest.fixture
    def store(self):
        raise NotImplementedError("Subclass must provide a real store fixture")

    @pytest.fixture(autouse=True)
    def cleanup_after_test(self, store):
        yield
        # Always clean up test data
        store.delete_collection()

    def test_upsert_stores_and_search_retrieves(self, store):
        store.upsert("doc1", embedding=[0.1] * 128, metadata={"text": "hello world"})
        results = store.search(embedding=[0.1] * 128, top_k=1)
        assert len(results) >= 1
        assert results[0]["id"] == "doc1"

    def test_delete_actually_removes_document(self, store):
        store.upsert("doc2", embedding=[0.2] * 128, metadata={"text": "delete me"})
        store.delete(["doc2"])
        results = store.search(embedding=[0.2] * 128, top_k=10)
        assert not any(r["id"] == "doc2" for r in results)

    def test_search_respects_metadata_filters(self, store):
        store.upsert("pdf1", embedding=[0.3] * 128, metadata={"source": "pdf"})
        store.upsert("url1", embedding=[0.3] * 128, metadata={"source": "url"})
        results = store.search(embedding=[0.3] * 128, top_k=10, filters={"source": "pdf"})
        assert all(r["metadata"]["source"] == "pdf" for r in results)

    def test_upsert_same_id_updates_not_duplicates(self, store):
        store.upsert("doc3", embedding=[0.5] * 128, metadata={"version": 1})
        store.upsert("doc3", embedding=[0.5] * 128, metadata={"version": 2})
        results = store.search(embedding=[0.5] * 128, top_k=10)
        matching = [r for r in results if r["id"] == "doc3"]
        assert len(matching) == 1
        assert matching[0]["metadata"]["version"] == 2

    def test_search_returns_results_ordered_by_similarity(self, store):
        store.upsert("close", embedding=[0.9] * 128, metadata={"text": "close"})
        store.upsert("far", embedding=[0.1] * 128, metadata={"text": "far"})
        results = store.search(embedding=[0.9] * 128, top_k=2)
        assert results[0]["id"] == "close"


# --- Each implementation inherits the full suite ---

class TestQdrantStore(VectorStoreContractTests):
    @pytest.fixture
    def store(self):
        return QdrantStore(config=QdrantConfig(
            collection_name="test_contract",
            url=os.environ.get("QDRANT_URL", "http://localhost:6333")
        ))

class TestPineconeStore(VectorStoreContractTests):
    @pytest.fixture
    def store(self):
        return PineconeStore(config=PineconeConfig(
            index_name="test-contract",
            api_key=os.environ["PINECONE_API_KEY"]
        ))
```

**Rules:**
- NEVER duplicate contract tests per implementation — one shared class, many inheritors
- Uses real APIs and real databases — these are NOT mocked
- Every ABC in the library (`BaseLLM`, `BaseEmbedder`, `BaseReranker`, `BaseVectorStore`, `BaseChunker`) must have a contract test class
- When someone adds a new implementation, they MUST add a new inheriting test class
- Clean up after every test — don't pollute shared databases

### Layer 3: Integration Tests (`tests/integration/`)

**For:** Testing real pipelines end-to-end with real external services. Embedding actually works. Vector store actually persists. Retrieval pipeline actually returns relevant results.

**Marker:** `@pytest.mark.integration`

```python
# ✅ GOOD — real API, real pipeline
@pytest.mark.integration
async def test_knowledge_ingest_and_retrieve_end_to_end(knowledge_base):
    """Ingest a real document, then verify retrieval returns relevant chunks."""
    await knowledge_base.ingest(Document(
        content="Python was created by Guido van Rossum and first released in 1991. "
                "It emphasizes code readability and supports multiple programming paradigms.",
        source="python_facts.txt"
    ))

    results = await knowledge_base.retrieve("Who created Python?")

    assert len(results) >= 1
    assert "guido" in results[0].content.lower()


# ❌ BAD — mocked, proves nothing about real behavior
@pytest.mark.integration
async def test_knowledge_ingest(mock_embedder, mock_store):
    mock_embedder.embed.return_value = [0.1] * 1536
    mock_store.search.return_value = [{"content": "fake result"}]
    # This test will ALWAYS pass. It tests nothing.
```

**Rules:**
- ABSOLUTELY NO MOCKS — the entire point is testing real connections
- Use session-scoped fixtures for expensive clients (embedders, LLM clients, DB connections)
- Always clean up test data in fixtures (delete test collections, test entries)
- Skip gracefully when API keys are missing — don't fail the whole suite
- Mark slow tests with `@pytest.mark.slow` if they take > 10 seconds

### Layer 4: Behavioral Tests (`tests/behavioral/`)

**For:** Testing agent-level decisions. Does the agent route to memory vs knowledge correctly? Does it call the right tool? Does it assemble the right context? Tests assert on OUTCOMES — the content of the response — not on which internal methods were called.

**Marker:** `@pytest.mark.behavioral`

```python
# ✅ GOOD — tests what the agent DOES
@pytest.mark.behavioral
async def test_agent_answers_personal_question_from_memory(agent):
    """When asked a personal question, agent should use memory, not knowledge base."""
    await agent.memory.store(Episode(
        content="User's name is Anandesh Sharma",
        type=MemoryType.SEMANTIC
    ))

    response = await agent.run("What's my name?")

    assert "anandesh" in response.content.lower()


@pytest.mark.behavioral
async def test_agent_answers_factual_question_from_knowledge(agent):
    """When asked a factual question, agent should use knowledge base."""
    await agent.knowledge.ingest(Document(
        content="Definable.ai was founded in 2024 by Anandesh Sharma.",
        source="about.txt"
    ))

    response = await agent.run("When was the company founded?")

    assert "2024" in response.content


@pytest.mark.behavioral
async def test_agent_handles_multi_intent_query(agent):
    """Agent should pull from multiple sources for multi-intent queries."""
    await agent.memory.store(Episode(content="User's name is Anandesh"))
    await agent.knowledge.ingest(Document(
        content="Definable.ai was founded in 2024.",
        source="about.txt"
    ))

    response = await agent.run(
        "What's my name and when was the company founded? "
        "Send this info to anandesh.sharma@definable.ai"
    )

    assert "anandesh" in response.content.lower()
    assert "2024" in response.content
    # Tool assertion IS valid here — verifying the action was attempted
    assert any(t.name == "send_email" for t in response.tool_calls)


# ❌ BAD — tests internal mechanics, not behavior
@pytest.mark.behavioral
async def test_agent_calls_memory():
    agent.run("what's my name?")
    agent.memory.recall.assert_called_once()  # WHO CARES if it was called?
    # The only thing that matters is: did the response contain the name?
```

**Rules:**
- NEVER assert on internal method calls (`.assert_called_once()`, `.assert_called_with()`)
- ALWAYS assert on response content or tool calls (the observable output)
- Set up real state (store memories, ingest documents) before each test
- Each test should represent a real user scenario
- Test edge cases: ambiguous queries, empty memory, conflicting information
- The agent can change HOW it gets the answer — the test only cares THAT it did

### Layer 5: Regression Tests (`tests/regression/`)

**For:** Locking in known-good behavior so changes are intentional. If someone modifies the system prompt builder, chunking algorithm, or routing logic, these tests FORCE them to review and approve the change.

**Marker:** `@pytest.mark.regression`

```python
@pytest.mark.regression
def test_system_prompt_with_all_capabilities_enabled(snapshot):
    """System prompt structure should not change without explicit review."""
    agent = Agent(
        model=model,
        memory=True,
        knowledge=True,
        thinking=True,
        instructions="You are a helpful assistant."
    )
    prompt = agent._build_system_prompt()
    snapshot.assert_match(prompt, "system_prompt_all_enabled.txt")


@pytest.mark.regression
def test_chunking_is_deterministic(snapshot):
    """Same input document must always produce the same chunks."""
    chunker = RecursiveChunker(max_tokens=200, overlap=50)
    doc = Document(content=SAMPLE_DOCUMENT)
    chunks = [c.content for c in chunker.chunk(doc)]
    snapshot.assert_match(str(chunks), "chunking_sample_doc.txt")


@pytest.mark.regression
def test_signal_router_classifications(snapshot):
    """Known queries should always route to the same sources."""
    router = SignalRouter()
    test_cases = {
        "what's my name": router.route("what's my name"),
        "company refund policy": router.route("company refund policy"),
        "send email to john": router.route("send email to john"),
        "remember I like Python": router.route("remember I like Python"),
    }
    snapshot.assert_match(str(test_cases), "routing_known_queries.txt")
```

**Rules:**
- Use `pytest-snapshot` or `syrupy` for snapshot management
- When a snapshot test fails, the developer MUST review the diff and explicitly update the snapshot
- Only snapshot deterministic outputs — don't snapshot LLM responses
- Good candidates: system prompts, chunking output, config serialization, routing decisions

## Test Infrastructure

### conftest.py Pattern

```python
"""
Shared test configuration and fixtures.

Real API clients are session-scoped to avoid creating new connections per test.
Integration and behavioral tests are auto-skipped when API keys are missing.
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv(".env.test")


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Pure logic, no external calls")
    config.addinivalue_line("markers", "integration: Real APIs, real databases")
    config.addinivalue_line("markers", "behavioral: Agent-level behavior tests")
    config.addinivalue_line("markers", "contract: ABC contract compliance")
    config.addinivalue_line("markers", "regression: Snapshot-based regression")
    config.addinivalue_line("markers", "slow: Tests taking >10 seconds")


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration/behavioral tests when API keys are missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        skip = pytest.mark.skip(reason="OPENAI_API_KEY not set — skipping real API tests")
        for item in items:
            if "integration" in item.keywords or "behavioral" in item.keywords:
                item.add_marker(skip)


# --- Session-scoped fixtures for expensive clients ---

@pytest.fixture(scope="session")
def openai_embedder():
    from definable.libs.embedders import OpenAIEmbedder, EmbedConfig
    return OpenAIEmbedder(
        config=EmbedConfig(model="text-embedding-3-small"),
        api_key=os.environ["OPENAI_API_KEY"]
    )

@pytest.fixture(scope="session")
def vector_store():
    from definable.libs.vectorstores import QdrantStore, QdrantConfig
    store = QdrantStore(config=QdrantConfig(
        collection_name="test_definable",
        url=os.environ.get("QDRANT_URL", "http://localhost:6333")
    ))
    yield store
    store.delete_collection("test_definable")

@pytest.fixture(scope="session")
def retrieval_service(openai_embedder, vector_store):
    from definable.libs.retrieval import RetrievalService
    return RetrievalService(embedder=openai_embedder, store=vector_store)

@pytest.fixture
def knowledge_base(retrieval_service):
    from definable.libs.knowledge import KnowledgeBase
    kb = KnowledgeBase(retrieval=retrieval_service, collection="test_kb")
    yield kb
    kb.delete_collection()

@pytest.fixture
def agent(retrieval_service):
    from definable.agent import Agent
    from definable.libs.llms import OpenAILLM, LLMConfig
    a = Agent(
        model=OpenAILLM(
            config=LLMConfig(model="gpt-4o-mini"),
            api_key=os.environ["OPENAI_API_KEY"]
        ),
        memory=True,
        knowledge=True
    )
    yield a
    # Clean up agent's memory and knowledge after each test
    a.cleanup()
```

### pytest.ini Pattern

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
markers =
    unit: Pure logic tests (no external calls)
    integration: Real API and database tests
    behavioral: Agent-level behavior tests
    contract: ABC contract compliance tests
    regression: Snapshot-based regression tests
    slow: Tests taking more than 10 seconds

# Default: run fast tests only
addopts = -m "not slow" --tb=short -q
```

### Running Tests

```bash
# Developer workflow — fast feedback
pytest -m unit                          # < 5 seconds

# Before committing — verify contracts
pytest -m "unit or contract"            # < 30 seconds

# Before PR — full integration check
pytest -m "unit or contract or integration"

# Before release — everything including behavioral
pytest                                  # Full suite

# Specific module
pytest tests/integration/test_knowledge_ingest.py -v

# Only regression snapshots
pytest -m regression

# Update snapshots after intentional changes
pytest -m regression --snapshot-update
```

## Decision Guide: What Tests Does My Code Need?

When writing tests for any new or modified code, follow this checklist:

```
New pure function (chunker, parser, validator, builder)?
  → Write unit tests in tests/unit/

New ABC implementation (new embedder, new vector store, new LLM)?
  → Inherit from the contract test class in tests/contract/
  → If contract test class doesn't exist yet, create it first

New pipeline or service that calls external APIs?
  → Write integration tests in tests/integration/
  → Use real clients from conftest.py fixtures
  → Add cleanup in fixture teardown

New agent capability (routing, tool selection, context assembly)?
  → Write behavioral tests in tests/behavioral/
  → Assert on response content, not method calls

Changed system prompt, chunking logic, or routing rules?
  → Add or update regression snapshots in tests/regression/
```

Most features need tests in 2-3 layers. A new embedder implementation needs:
1. Contract test (inherits `EmbedderContractTests`)
2. Integration test (real embed call, verify dimensions and type)
3. Regression test (embedding dimensions haven't changed)

## Anti-Patterns — NEVER Do These

| Anti-Pattern | Why It's Bad | Do This Instead |
|---|---|---|
| `mock_embedder.embed.return_value = [0.1]*1536` | Proves nothing about real embedding | Use real embedder with real API key |
| `agent.memory.recall.assert_called_once()` | Tests mechanics, not behavior | Assert response contains expected content |
| `def test_search(): assert True` | Placeholder that gives false confidence | Delete it or write a real assertion |
| Duplicating contract tests per implementation | Maintenance nightmare, tests drift apart | One shared class, implementations inherit |
| Testing LLM response word-for-word | LLM responses are non-deterministic | Assert on key facts: `"2024" in response` |
| No cleanup after integration tests | Pollutes shared databases, causes flaky tests | Always use fixture teardown with cleanup |
| Snapshot testing LLM responses | Non-deterministic, will always fail | Only snapshot deterministic outputs |
| `@pytest.mark.skip("TODO")` | Dead tests that never get written | Write it now or delete the placeholder |