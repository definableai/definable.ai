"""E2E tests — Core Agent Workflows.

Scenario: "I want a chatbot that answers questions and follows instructions."

All tests require OPENAI_API_KEY.
"""

import pytest

from definable.agents.agent import Agent
from definable.agents.config import AgentConfig, CompressionConfig, TracingConfig
from definable.run.agent import RunContentEvent
from definable.tools.decorator import tool


@pytest.mark.e2e
@pytest.mark.openai
class TestSimpleAgent:
  """Basic agent question-answering and instruction-following."""

  @pytest.mark.asyncio
  async def test_basic_question_answering(self, openai_model):
    """Agent answers a simple factual question."""
    agent = Agent(
      model=openai_model,
      instructions="You are a helpful assistant. Be concise.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("What is the capital of France? Answer in one word.")

    assert output.content is not None
    assert len(output.content) > 0
    assert "paris" in output.content.lower()

  @pytest.mark.asyncio
  async def test_system_instructions_followed(self, openai_model):
    """Agent follows system instructions (respond in French)."""
    agent = Agent(
      model=openai_model,
      instructions="Always respond in French, no matter what language the user writes in.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("What is 2+2?")

    assert output.content is not None
    # French numbers or words should appear
    content_lower = output.content.lower()
    assert any(word in content_lower for word in ["quatre", "4", "est"])

  @pytest.mark.asyncio
  async def test_multi_turn_conversation(self, openai_model):
    """Second turn references information from the first turn."""
    agent = Agent(
      model=openai_model,
      instructions="Be concise.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    # First turn: introduce a name
    output1 = await agent.arun("My name is Zephyr.")

    assert output1.content is not None
    assert output1.messages is not None

    # Second turn: ask about the name using conversation history
    output2 = await agent.arun(
      "What is my name?",
      messages=output1.messages,
      session_id=output1.session_id,
    )

    assert output2.content is not None
    assert "zephyr" in output2.content.lower()

  @pytest.mark.asyncio
  async def test_streaming_response(self, openai_model):
    """arun_stream() yields RunContentEvent chunks that form a coherent response."""
    agent = Agent(
      model=openai_model,
      instructions="Be concise.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    chunks = []
    async for event in agent.arun_stream("Count from 1 to 5."):
      if isinstance(event, RunContentEvent) and event.content:
        chunks.append(event.content)

    assert len(chunks) > 0
    full = "".join(chunks)
    assert any(n in full for n in ["1", "2", "3", "4", "5"])

  @pytest.mark.asyncio
  async def test_structured_output(self, openai_model):
    """Agent with output_schema returns data matching a Pydantic model."""
    import json

    from pydantic import BaseModel

    class MathResult(BaseModel):
      answer: int
      explanation: str

    agent = Agent(
      model=openai_model,
      instructions="Solve math problems. Return structured output.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun(
      "What is 7 * 6?",
      output_schema=MathResult,
    )

    assert output.content is not None
    # Parse the structured output
    if isinstance(output.content, MathResult):
      assert output.content.answer == 42
    else:
      data = json.loads(output.content)
      assert data["answer"] == 42

  @pytest.mark.asyncio
  async def test_structured_output_list(self, openai_model):
    """Structured output with a list field."""
    import json
    from typing import List

    from pydantic import BaseModel, Field

    class ShoppingList(BaseModel):
      items: List[str] = Field(description="List of items to buy")
      store: str

    agent = Agent(
      model=openai_model,
      instructions="Create shopping lists as requested.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun(
      "Create a grocery shopping list with exactly 3 items: milk, bread, eggs.",
      output_schema=ShoppingList,
    )

    assert output.content is not None
    if isinstance(output.content, ShoppingList):
      assert len(output.content.items) >= 3
    else:
      data = json.loads(output.content)
      assert len(data["items"]) >= 3


@pytest.mark.e2e
@pytest.mark.openai
class TestAgentWithCompression:
  """Compression integration with real agent runs."""

  @pytest.mark.asyncio
  async def test_long_tool_results_compressed(self, openai_model):
    """Agent with compression config handles long tool results without failing."""

    @tool
    def get_long_report() -> str:
      """Get a detailed report."""
      return "REPORT: " + "This is a detailed line of data. " * 50

    agent = Agent(
      model=openai_model,
      tools=[get_long_report],
      instructions="Use the get_long_report tool to answer questions about reports.",
      config=AgentConfig(
        tracing=TracingConfig(enabled=False),
        compression=CompressionConfig(
          enabled=True,
          tool_results_limit=1,
        ),
      ),
    )
    output = await agent.arun("Get me the report and summarize it.")

    assert output.content is not None
    assert len(output.content) > 0

  @pytest.mark.asyncio
  async def test_many_tool_calls_compression(self, openai_model):
    """Agent making multiple tool calls with compression still produces valid response."""
    call_count = {"n": 0}

    @tool
    def lookup_item(item_id: str) -> str:
      """Look up an item by ID."""
      call_count["n"] += 1
      return f"Item {item_id}: Widget model {item_id}, price $99, in stock"

    agent = Agent(
      model=openai_model,
      tools=[lookup_item],
      instructions=("You MUST use the lookup_item tool for each item. Look up each item individually. Never skip the tool."),
      config=AgentConfig(
        tracing=TracingConfig(enabled=False),
        compression=CompressionConfig(
          enabled=True,
          tool_results_limit=2,
        ),
      ),
    )
    output = await agent.arun("Look up items A1, A2, and A3 using the tool, then give me a summary.")

    assert output.content is not None
    assert call_count["n"] >= 2  # At least some tool calls were made
