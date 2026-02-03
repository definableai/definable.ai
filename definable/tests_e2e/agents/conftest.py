"""Agent E2E test fixtures."""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from definable.agents.testing import AgentTestCase, MockModel
from definable.models.metrics import Metrics


@pytest.fixture
def mock_model():
    """Basic MockModel with simple response."""
    return MockModel(responses=["Test response"])


@pytest.fixture
def mock_model_factory():
    """Factory for creating MockModels with custom responses."""

    def _create(responses: Optional[List[str]] = None, **kwargs):
        return MockModel(responses=responses or ["Mock response"], **kwargs)

    return _create


@pytest.fixture
def mock_model_with_tool_call():
    """
    MockModel that simulates a tool call then final response.

    First call returns a tool call, second call returns final answer.
    """
    call_count = {"count": 0}

    def side_effect(messages: List, tools: List, **kwargs):
        response = MagicMock()
        response.metrics = Metrics()
        response.reasoning_content = None
        response.citations = None
        response.images = None
        response.videos = None
        response.audios = None

        if call_count["count"] == 0 and tools:
            # First call: return a tool call for the first available tool
            tool_name = tools[0]["function"]["name"] if tools else "unknown"
            response.content = None
            response.tool_calls = [
                {
                    "id": "call_test_123",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ]
            response.tool_executions = []
        else:
            # Subsequent calls: return final response
            response.content = "Final response after tool call"
            response.tool_calls = []
            response.tool_executions = []

        call_count["count"] += 1
        return response

    return MockModel(side_effect=side_effect)


@pytest.fixture
def agent_test_case():
    """AgentTestCase instance for test assertions."""
    return AgentTestCase()


@pytest.fixture
def in_memory_knowledge():
    """
    In-memory Knowledge base with test documents.

    Uses mock embeddings for deterministic testing without an embedder.
    """
    from definable.knowledge import Knowledge
    from definable.knowledge.document import Document
    from definable.knowledge.vector_dbs.memory import InMemoryVectorDB

    # Create vector DB with small dimensions for testing
    vector_db = InMemoryVectorDB(dimensions=3)

    # Add test documents with pre-computed embeddings
    docs = [
        Document(
            id="doc1",
            content="Python is a programming language.",
            meta_data={"topic": "programming"},
            embedding=[0.1, 0.2, 0.3],
        ),
        Document(
            id="doc2",
            content="Machine learning uses algorithms.",
            meta_data={"topic": "ml"},
            embedding=[0.4, 0.5, 0.6],
        ),
        Document(
            id="doc3",
            content="Artificial intelligence is the future.",
            meta_data={"topic": "ai"},
            embedding=[0.7, 0.8, 0.9],
        ),
    ]

    # Add documents directly to vector DB
    vector_db.add(docs)

    # Create Knowledge instance with the populated vector DB
    kb = Knowledge(vector_db=vector_db)

    return kb
