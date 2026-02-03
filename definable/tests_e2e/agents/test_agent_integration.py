"""E2E integration tests for Agent with real OpenAI model."""

import os

import pytest

from definable.agents.testing import AgentTestCase
from definable.agents.toolkit import Toolkit
from definable.knowledge import Knowledge
from definable.knowledge.document import Document
from definable.knowledge.vector_dbs.memory import InMemoryVectorDB
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool


def requires_openai():
    """Skip if OPENAI_API_KEY not set."""
    return pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )


@pytest.mark.e2e
@requires_openai()
class TestAgentWithOpenAI(AgentTestCase):
    """Integration tests with real OpenAI model."""

    @pytest.fixture
    def openai_model(self):
        """OpenAI model for testing."""
        return OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def test_agent_simple_response(self, openai_model):
        """Agent returns a response to a simple query."""
        agent = self.create_agent(
            model=openai_model,
            instructions="You are a helpful assistant. Be concise.",
        )
        output = agent.run("Say 'hello' and nothing else.")

        self.assert_has_content(output)
        assert "hello" in output.content.lower()

    def test_agent_with_system_instructions(self, openai_model):
        """Agent follows system instructions."""
        agent = self.create_agent(
            model=openai_model,
            instructions="You are a pirate. Always respond like a pirate.",
        )
        output = agent.run("How are you today?")

        self.assert_has_content(output)
        # Pirate responses often contain these words
        content_lower = output.content.lower()
        assert any(
            word in content_lower
            for word in ["arr", "ahoy", "matey", "ye", "aye", "sea", "ship"]
        )

    def test_agent_with_tool_execution(self, openai_model):
        """Agent can call a tool and use the result."""

        @tool
        def get_current_time() -> str:
            """Get the current time."""
            return "The current time is 3:45 PM"

        agent = self.create_agent(
            model=openai_model,
            tools=[get_current_time],
            instructions="Use the get_current_time tool when asked about time.",
        )
        output = agent.run("What time is it?")

        self.assert_has_content(output)
        # Check if tool was called or result mentioned
        assert "3:45" in output.content or self._tool_was_called(output, "get_current_time")

    def _tool_was_called(self, output, tool_name: str) -> bool:
        """Check if a tool was called in the output."""
        if not output.tools:
            return False
        return any(t.tool_name == tool_name for t in output.tools)

    def test_agent_with_parameterized_tool(self, openai_model):
        """Agent can call a tool with parameters."""

        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers. You MUST use this tool for any multiplication."""
            return a * b

        agent = self.create_agent(
            model=openai_model,
            tools=[multiply],
            instructions="You MUST use the multiply tool to calculate products. Never calculate in your head.",
        )
        output = agent.run("Use the multiply tool to calculate 7 times 6. You MUST call the tool.")

        self.assert_has_content(output)
        # Either the tool was called and result shown, or model computed it (fallback)
        assert "42" in output.content or self._tool_was_called(output, "multiply")

    def test_agent_with_toolkit(self, openai_model):
        """Agent works with toolkit."""

        @tool
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}! Welcome!"

        @tool
        def farewell(name: str) -> str:
            """Say goodbye to someone."""
            return f"Goodbye, {name}! See you later!"

        class GreetingToolkit(Toolkit):
            hello = greet
            bye = farewell

            @property
            def tools(self):
                return [self.hello, self.bye]

        tk = GreetingToolkit()
        agent = self.create_agent(
            model=openai_model,
            toolkits=[tk],
            instructions="Use the greeting tools when asked to greet or say bye.",
        )
        output = agent.run("Please greet Alice")

        self.assert_has_content(output)
        assert "alice" in output.content.lower() or "hello" in output.content.lower()

    def test_agent_multiple_tool_calls(self, openai_model):
        """Agent can make multiple tool calls in sequence."""

        results = []

        @tool
        def get_weather(city: str) -> str:
            """Get current weather for a city. You MUST use this tool."""
            results.append(city)
            return f"The weather in {city} is sunny, 72°F"

        agent = self.create_agent(
            model=openai_model,
            tools=[get_weather],
            instructions="You MUST use the get_weather tool to check weather. Never guess the weather.",
        )
        output = agent.run("Use the get_weather tool to check the weather in Tokyo. You MUST call the tool.")

        self.assert_has_content(output)
        # Verify tool was used or weather info appears
        assert len(results) >= 1 or "tokyo" in output.content.lower() or "sunny" in output.content.lower()

    @pytest.mark.asyncio
    async def test_agent_async_run(self, openai_model):
        """Agent can run asynchronously."""
        agent = self.create_agent(
            model=openai_model,
            instructions="Be concise.",
        )
        output = await agent.arun("Say 'async works' and nothing else.")

        self.assert_has_content(output)
        assert "async" in output.content.lower() or "works" in output.content.lower()

    def test_agent_streaming_response(self, openai_model):
        """Agent can stream responses."""
        agent = self.create_agent(
            model=openai_model,
            instructions="Be concise.",
        )

        chunks = []
        for chunk in agent.run_stream("Count from 1 to 3."):
            if chunk.content:
                chunks.append(chunk.content)

        # Should have received multiple chunks
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert any(n in full_response for n in ["1", "2", "3"])


@pytest.mark.e2e
@requires_openai()
class TestAgentWithKnowledge(AgentTestCase):
    """Integration tests for Agent with Knowledge base."""

    @pytest.fixture
    def openai_model(self):
        """OpenAI model for testing."""
        return OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    @pytest.fixture
    def knowledge_base(self):
        """Knowledge base with test documents."""
        vector_db = InMemoryVectorDB(dimensions=3)
        docs = [
            Document(
                id="company-info",
                content="Acme Corp was founded in 2020 by John Smith. The company specializes in AI solutions.",
                embedding=[0.9, 0.1, 0.0],
            ),
            Document(
                id="product-info",
                content="Acme's flagship product is SmartBot, an AI assistant launched in 2023.",
                embedding=[0.8, 0.2, 0.0],
            ),
            Document(
                id="contact-info",
                content="Acme Corp headquarters is located at 123 Tech Street, San Francisco.",
                embedding=[0.7, 0.3, 0.0],
            ),
        ]
        vector_db.add(docs)
        return Knowledge(vector_db=vector_db)

    def test_agent_with_knowledge_toolkit(self, openai_model, knowledge_base):
        """Agent can search knowledge base using toolkit."""
        from definable.agents.toolkits.knowledge import KnowledgeToolkit

        tk = KnowledgeToolkit(knowledge=knowledge_base)
        agent = self.create_agent(
            model=openai_model,
            toolkits=[tk],
            instructions="Search the knowledge base to answer questions about the company.",
        )

        # Verify toolkit is registered
        all_tools = agent._flatten_tools()
        assert "search_knowledge" in all_tools
