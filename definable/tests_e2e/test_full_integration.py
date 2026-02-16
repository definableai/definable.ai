"""E2E tests — Complete Agent Workflows.

Scenario: "I want an agent with tools, knowledge, memory, and readers
all working together."

All tests require OPENAI_API_KEY.
"""

from unittest.mock import AsyncMock, patch

import pytest

from definable.agents.agent import Agent
from definable.agents.config import AgentConfig, KnowledgeConfig, TracingConfig
from definable.knowledge import Knowledge
from definable.knowledge.document import Document
from definable.knowledge.vector_dbs.memory import InMemoryVectorDB
from definable.media import File
from definable.memory import CognitiveMemory
from definable.memory.store.in_memory import InMemoryStore
from definable.run.agent import RunContentEvent
from definable.tools.decorator import tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def knowledge_base():
  """Knowledge base with pre-embedded docs about a fictional company."""
  vector_db = InMemoryVectorDB(dimensions=3)
  docs = [
    Document(
      id="company",
      content="NovaTech was founded in 2021 by Dr. Elena Park. It builds quantum computing software.",
      embedding=[0.9, 0.1, 0.0],
    ),
    Document(
      id="product",
      content="NovaTech's product QBit is a quantum circuit simulator released in 2024.",
      embedding=[0.8, 0.2, 0.0],
    ),
    Document(
      id="pricing",
      content="QBit costs $99/month for the starter plan and $499/month for enterprise.",
      embedding=[0.7, 0.3, 0.0],
    ),
  ]
  vector_db.add(docs)
  return Knowledge(vector_db=vector_db)


@pytest.fixture
async def memory_store():
  """Fresh InMemoryStore."""
  store = InMemoryStore()
  await store.initialize()
  yield store
  await store.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.openai
class TestFullAgent:
  """Agent with multiple capabilities working together."""

  @pytest.mark.asyncio
  async def test_agent_with_tools_and_knowledge(self, openai_model, knowledge_base):
    """Agent uses both tools and knowledge context."""
    call_log = []

    @tool
    def calculate_discount(price: float, percent: float) -> str:
      """Calculate a discount on a price."""
      call_log.append("discount")
      discounted = price * (1 - percent / 100)
      return f"${discounted:.2f}"

    mock_docs = [
      Document(
        id="pricing",
        content="QBit costs $99/month for the starter plan and $499/month for enterprise.",
        reranking_score=0.9,
      ),
    ]

    with patch.object(knowledge_base, "asearch", new_callable=AsyncMock, return_value=mock_docs):
      agent = Agent(
        model=openai_model,
        tools=[calculate_discount],
        instructions=("Answer questions about NovaTech using the knowledge context. Use tools for calculations."),
        config=AgentConfig(
          tracing=TracingConfig(enabled=False),
          knowledge=KnowledgeConfig(
            knowledge=knowledge_base,
            top_k=2,
            enabled=True,
          ),
        ),
      )

      output = await agent.arun("What does QBit's starter plan cost? Apply a 10% discount using the tool.")

    assert output.content is not None
    # Should reference the pricing from knowledge
    assert any(w in output.content.lower() for w in ["99", "89", "discount"])

  @pytest.mark.asyncio
  async def test_agent_with_memory_and_readers(self, openai_model, memory_store):
    """Agent with memory reads a file and stores the conversation."""
    memory = CognitiveMemory(store=memory_store, token_budget=500)

    agent = Agent(
      model=openai_model,
      memory=memory,
      readers=True,
      instructions="Analyze uploaded files. Be concise.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    output = await agent.arun(
      "What data is in this file?",
      files=[
        File(
          content=b"quarter,revenue\nQ1,$1M\nQ2,$1.5M\nQ3,$2M\nQ4,$2.5M",
          filename="revenue.csv",
          mime_type="text/csv",
        ),
      ],
      user_id="test-user",
      session_id="test-session",
    )

    assert output.content is not None
    assert any(w in output.content.lower() for w in ["revenue", "quarter", "q1", "$"])

    # Drain pending memory tasks before checking store
    await agent._drain_memory_tasks()

    # Verify memory stored the conversation
    episodes = await memory_store.get_episodes(user_id="test-user")
    assert len(episodes) >= 1

  @pytest.mark.asyncio
  async def test_complete_agent_workflow(self, openai_model, knowledge_base, memory_store):
    """Full pipeline: knowledge + memory + tools + readers."""
    memory = CognitiveMemory(store=memory_store, token_budget=500)

    @tool
    def format_currency(amount: float, currency: str = "USD") -> str:
      """Format a number as currency."""
      if currency == "USD":
        return f"${amount:,.2f}"
      return f"{amount:,.2f} {currency}"

    mock_docs = [
      Document(
        id="pricing",
        content="QBit costs $99/month for starter and $499/month for enterprise.",
        reranking_score=0.9,
      ),
    ]

    with patch.object(knowledge_base, "asearch", new_callable=AsyncMock, return_value=mock_docs):
      agent = Agent(
        model=openai_model,
        tools=[format_currency],
        memory=memory,
        readers=True,
        instructions="Use knowledge context and tools as needed. Be concise.",
        config=AgentConfig(
          tracing=TracingConfig(enabled=False),
          knowledge=KnowledgeConfig(
            knowledge=knowledge_base,
            top_k=2,
            enabled=True,
          ),
        ),
      )

      output = await agent.arun(
        "Tell me about QBit pricing.",
        user_id="full-test",
        session_id="full-session",
      )

    assert output.content is not None
    assert any(w in output.content for w in ["99", "499", "QBit", "starter", "enterprise"])

  @pytest.mark.asyncio
  async def test_streaming_full_pipeline(self, openai_model, knowledge_base):
    """Full agent pipeline via arun_stream emits events correctly."""
    mock_docs = [
      Document(
        id="company",
        content="NovaTech was founded in 2021 by Dr. Elena Park.",
        reranking_score=0.9,
      ),
    ]

    with patch.object(knowledge_base, "asearch", new_callable=AsyncMock, return_value=mock_docs):
      agent = Agent(
        model=openai_model,
        instructions="Be concise. Use knowledge context.",
        config=AgentConfig(
          tracing=TracingConfig(enabled=False),
          knowledge=KnowledgeConfig(
            knowledge=knowledge_base,
            top_k=2,
            enabled=True,
          ),
        ),
      )

      chunks = []
      async for event in agent.arun_stream("Who founded NovaTech?"):
        if isinstance(event, RunContentEvent) and event.content:
          chunks.append(event.content)

    assert len(chunks) > 0
    full = "".join(chunks)
    assert any(w in full.lower() for w in ["elena", "park", "2021"])

  @pytest.mark.asyncio
  async def test_agent_multi_turn_with_everything(self, openai_model, knowledge_base, memory_store):
    """3-turn conversation with tools + knowledge + memory stays coherent."""
    memory = CognitiveMemory(store=memory_store, token_budget=500)

    @tool
    def lookup_status(service: str) -> str:
      """Check the status of a service."""
      return f"{service} is operational (99.9% uptime)"

    mock_docs = [
      Document(
        id="company",
        content="NovaTech was founded in 2021 by Dr. Elena Park.",
        reranking_score=0.9,
      ),
    ]

    with patch.object(knowledge_base, "asearch", new_callable=AsyncMock, return_value=mock_docs):
      agent = Agent(
        model=openai_model,
        tools=[lookup_status],
        memory=memory,
        instructions="Answer questions about NovaTech. Use tools when needed. Be concise.",
        config=AgentConfig(
          tracing=TracingConfig(enabled=False),
          knowledge=KnowledgeConfig(
            knowledge=knowledge_base,
            top_k=2,
            enabled=True,
          ),
        ),
      )

      # Turn 1
      output1 = await agent.arun(
        "Who founded NovaTech?",
        user_id="multi-turn",
        session_id="multi-sess",
      )
      assert output1.content is not None
      assert any(w in output1.content.lower() for w in ["elena", "park"])

      # Turn 2 with history
      output2 = await agent.arun(
        "Check the status of QBit using the lookup_status tool.",
        messages=output1.messages,
        session_id=output1.session_id,
        user_id="multi-turn",
      )
      assert output2.content is not None

      # Turn 3 referencing earlier context
      output3 = await agent.arun(
        "Summarize what we've discussed so far.",
        messages=output2.messages,
        session_id=output2.session_id,
        user_id="multi-turn",
      )
      assert output3.content is not None
      # Should reference NovaTech or the founder from turn 1
      assert any(w in output3.content.lower() for w in ["novatech", "elena", "park", "qbit", "founded"])
