"""
KnowledgeToolkit for explicit RAG search.

This example shows how to:
- Create a Knowledge base
- Use KnowledgeToolkit for explicit search
- Let the agent decide when to search

Requirements:
    export OPENAI_API_KEY=sk-...

Note: This example uses a mock embedder for demonstration.
"""

from typing import List

from definable.agent import Agent, KnowledgeToolkit
from definable.embedder import Embedder
from definable.knowledge import Document, Knowledge
from definable.vectordb import InMemoryVectorDB
from definable.model.openai import OpenAIChat


class MockEmbedder(Embedder):
  """A mock embedder for demonstration.

  For production, use VoyageAIEmbedder or OpenAIEmbedder.
  """

  dimensions: int = 64

  def get_embedding(self, text: str) -> List[float]:
    """Generate a mock embedding based on text hash."""
    import hashlib

    hash_bytes = hashlib.sha256(text.encode()).digest()
    embedding = [(b / 127.5) - 1.0 for b in hash_bytes[: self.dimensions]]
    return embedding

  def get_embedding_and_usage(self, text: str):
    embedding = self.get_embedding(text)
    usage = {"tokens": len(text.split())}
    return embedding, usage

  async def async_get_embedding(self, text: str) -> List[float]:
    return self.get_embedding(text)

  async def async_get_embedding_and_usage(self, text: str):
    return self.get_embedding_and_usage(text)


def create_knowledge_base():
  """Create a knowledge base with sample documents."""
  embedder = MockEmbedder()
  vector_db = InMemoryVectorDB(dimensions=embedder.dimensions)

  kb = Knowledge(vector_db=vector_db, embedder=embedder)

  # Add documents
  documents = [
    Document(
      content="The company was founded in 2020 by Jane Smith and John Doe. Our headquarters is located in San Francisco, California.",
      meta_data={"source": "about.txt", "category": "company"},
    ),
    Document(
      content="Employees receive 20 days of paid time off (PTO) per year. PTO accrues monthly at a rate of 1.67 days per month.",
      meta_data={"source": "hr_policies.txt", "category": "benefits"},
    ),
    Document(
      content="The health insurance plan covers medical, dental, and vision. The company pays 80% of the premium, employees pay 20%.",
      meta_data={"source": "hr_policies.txt", "category": "benefits"},
    ),
    Document(
      content="Remote work is allowed up to 3 days per week. Employees must be in the office on Tuesdays and Thursdays.",
      meta_data={"source": "hr_policies.txt", "category": "work_policy"},
    ),
    Document(
      content="Our main product is CloudSync, a cloud synchronization platform. CloudSync supports AWS, Azure, and Google Cloud.",
      meta_data={"source": "products.txt", "category": "products"},
    ),
  ]

  for doc in documents:
    kb.add(doc)

  print(f"Created knowledge base with {len(documents)} documents")
  return kb


def knowledge_toolkit_example():
  """Use KnowledgeToolkit for explicit search."""
  print("KnowledgeToolkit Example")
  print("=" * 50)

  # Create knowledge base
  kb = create_knowledge_base()

  # Create KnowledgeToolkit
  knowledge_toolkit = KnowledgeToolkit(
    knowledge=kb,
    top_k=3,  # Return top 3 results
  )

  model = OpenAIChat(id="gpt-4o-mini")

  # Create agent with the toolkit
  agent = Agent(
    model=model,
    toolkits=[knowledge_toolkit],
    instructions="""You are a helpful HR assistant for the company.
When users ask questions about company policies, benefits, or products,
use the search_knowledge tool to find relevant information.""",
  )

  # Test queries
  queries = [
    "How many PTO days do employees get?",
    "What is the remote work policy?",
    "When was the company founded?",
  ]

  for query in queries:
    print(f"\nQ: {query}")
    output = agent.run(query)
    print(f"A: {output.content}")

    # Show which tools were used
    if output.tools:
      print(f"   [Used: {[t.tool_name for t in output.tools]}]")


def agent_decides_when_to_search():
  """Let the agent decide when to search."""
  print("\n" + "=" * 50)
  print("Agent Decides When to Search")
  print("=" * 50)

  kb = create_knowledge_base()

  agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    toolkits=[KnowledgeToolkit(knowledge=kb, top_k=2)],
    instructions="""You are a helpful assistant.
When you need company-specific information, use the search_knowledge tool.
For general knowledge questions, answer directly.""",
  )

  # Mix of questions
  questions = [
    "What is 2 + 2?",  # No search needed
    "How many vacation days do I get?",  # Needs search
    "What is the capital of France?",  # No search needed
  ]

  for question in questions:
    print(f"\nQ: {question}")
    output = agent.run(question)
    print(f"A: {output.content}")

    if output.tools:
      print("   [Searched knowledge base]")
    else:
      print("   [Answered directly]")


if __name__ == "__main__":
  knowledge_toolkit_example()
  agent_decides_when_to_search()
