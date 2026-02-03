"""
Agent with automatic knowledge retrieval (RAG).

This example shows how to:
- Configure KnowledgeConfig for automatic RAG
- Agent automatically retrieves relevant context
- Control retrieval behavior

Requirements:
    export OPENAI_API_KEY=sk-...

Note: This example uses a mock embedder to work without additional API keys.
"""

from typing import List, Literal

from definable.agents import Agent, AgentConfig, KnowledgeConfig
from definable.knowledge import Document, Embedder, InMemoryVectorDB, Knowledge
from definable.models.openai import OpenAIChat


class MockEmbedder(Embedder):
  """Mock embedder for demonstration."""

  dimensions: int = 128

  def get_embedding(self, text: str) -> List[float]:
    import hashlib

    text = text.lower().strip()
    embedding = [0.0] * self.dimensions

    for i, word in enumerate(text.split()):
      word_hash = hashlib.md5(word.encode()).digest()
      for j, byte in enumerate(word_hash):
        idx = (i + j) % self.dimensions
        embedding[idx] += (byte / 255.0 - 0.5) * (1 / (i + 1))

    magnitude = sum(x**2 for x in embedding) ** 0.5
    if magnitude > 0:
      embedding = [x / magnitude for x in embedding]

    return embedding

  def get_embedding_and_usage(self, text: str):
    return self.get_embedding(text), {"tokens": len(text.split())}

  async def async_get_embedding(self, text: str) -> List[float]:
    return self.get_embedding(text)

  async def async_get_embedding_and_usage(self, text: str):
    return self.get_embedding_and_usage(text)


def create_company_knowledge_base():
  """Create a knowledge base with company information."""
  embedder = MockEmbedder()
  vector_db = InMemoryVectorDB(dimensions=embedder.dimensions)

  kb = Knowledge(vector_db=vector_db, embedder=embedder)

  # Add company documents
  documents = [
    # HR Policies
    Document(
      content="Vacation Policy: Full-time employees receive 20 days of paid time off (PTO) per year. "
      "PTO accrues at 1.67 days per month. Unused PTO can be carried over up to 5 days.",
      meta_data={"category": "hr", "topic": "vacation"},
    ),
    Document(
      content="Remote Work Policy: Employees may work remotely up to 3 days per week. "
      "Core office hours are Tuesday and Thursday. Remote work requires manager approval.",
      meta_data={"category": "hr", "topic": "remote_work"},
    ),
    Document(
      content="Health Insurance: The company offers comprehensive health coverage including medical, "
      "dental, and vision. The company pays 80% of premiums, employees pay 20%.",
      meta_data={"category": "hr", "topic": "benefits"},
    ),
    Document(
      content="401(k) Plan: The company matches 50% of employee contributions up to 6% of salary. "
      "Vesting is immediate for all matching contributions.",
      meta_data={"category": "hr", "topic": "benefits"},
    ),
    # Company Info
    Document(
      content="Company History: Founded in 2018 by Sarah Chen and Michael Park. "
      "Headquarters in Austin, Texas. Currently has 250 employees across 3 offices.",
      meta_data={"category": "company", "topic": "history"},
    ),
    Document(
      content="Our Mission: To make enterprise software accessible to small businesses. "
      "We believe every company deserves powerful tools, regardless of size.",
      meta_data={"category": "company", "topic": "mission"},
    ),
    # Products
    Document(
      content="Product Overview: CloudSync is our flagship product - a cloud-based data "
      "synchronization platform. It supports AWS, Azure, and Google Cloud.",
      meta_data={"category": "product", "topic": "cloudsync"},
    ),
    Document(
      content="Pricing: CloudSync starts at $99/month for small teams (up to 10 users). "
      "Enterprise pricing is custom based on usage and support requirements.",
      meta_data={"category": "product", "topic": "pricing"},
    ),
  ]

  for doc in documents:
    kb.add(doc)

  print(f"Created knowledge base with {len(documents)} documents")
  return kb


def basic_knowledge_config():
  """Basic agent with knowledge configuration."""
  print("Basic Agent with Knowledge")
  print("=" * 50)

  kb = create_company_knowledge_base()

  model = OpenAIChat(id="gpt-4o-mini")

  # Create agent with automatic knowledge retrieval
  agent = Agent(
    model=model,
    instructions="You are an HR assistant for the company. Answer questions about company policies.",
    config=AgentConfig(
      knowledge=KnowledgeConfig(
        knowledge=kb,
        top_k=3,  # Retrieve top 3 relevant documents
      ),
    ),
  )

  # Ask questions - knowledge is automatically retrieved
  questions = [
    "How many vacation days do I get?",
    "Can I work from home?",
    "What's the 401k matching?",
  ]

  for question in questions:
    print(f"\nQ: {question}")
    output = agent.run(question)
    print(f"A: {output.content}")


def advanced_knowledge_config():
  """Advanced knowledge configuration options."""
  print("\n" + "=" * 50)
  print("Advanced Knowledge Configuration")
  print("=" * 50)

  kb = create_company_knowledge_base()
  model = OpenAIChat(id="gpt-4o-mini")

  # Advanced configuration
  agent = Agent(
    model=model,
    instructions="You are a company assistant. Use the provided context to answer questions accurately.",
    config=AgentConfig(
      knowledge=KnowledgeConfig(
        knowledge=kb,
        top_k=5,  # More documents for broader context
        rerank=True,  # Rerank results by relevance (if reranker available)
        min_score=0.5,  # Minimum relevance score
        context_format="xml",  # Format: "xml", "markdown", or "json"
        context_position="system",  # Position: "system" or "before_user"
        query_from="last_user",  # Query source: "last_user" or "full_conversation"
        max_query_length=500,  # Truncate long queries
        enabled=True,  # Can be disabled dynamically
      ),
    ),
  )

  print("\nKnowledge Config:")
  print("  - top_k: 5")
  print("  - rerank: True")
  print("  - context_format: xml")

  output = agent.run("Tell me about the company history and our main product.")
  print(f"\nResponse: {output.content}")


def context_formats():
  """Compare different context formats."""
  print("\n" + "=" * 50)
  print("Context Format Comparison")
  print("=" * 50)

  kb = create_company_knowledge_base()
  model = OpenAIChat(id="gpt-4o-mini")

  formats: List[Literal["xml", "markdown", "json"]] = ["xml", "markdown", "json"]

  for fmt in formats:
    print(f"\n{fmt.upper()} Format:")
    print("-" * 30)

    agent = Agent(
      model=model,
      instructions="Answer concisely based on the provided context.",
      config=AgentConfig(
        knowledge=KnowledgeConfig(
          knowledge=kb,
          top_k=2,
          context_format=fmt,
        ),
      ),
    )

    output = agent.run("What is the vacation policy?")
    print(f"Response: {(output.content or '')[:150]}...")


def multi_turn_with_knowledge():
  """Multi-turn conversation with knowledge retrieval."""
  print("\n" + "=" * 50)
  print("Multi-turn Conversation with Knowledge")
  print("=" * 50)

  kb = create_company_knowledge_base()
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are an HR assistant. Help employees understand company policies.",
    config=AgentConfig(
      knowledge=KnowledgeConfig(
        knowledge=kb,
        top_k=3,
      ),
    ),
  )

  # Multi-turn conversation
  messages = None

  conversation = [
    "What benefits does the company offer?",
    "Tell me more about the health insurance.",
    "And what about retirement benefits?",
  ]

  for question in conversation:
    print(f"\nEmployee: {question}")
    output = agent.run(question, messages=messages)
    print(f"HR Assistant: {output.content}")
    messages = output.messages


def conditional_knowledge():
  """Enable/disable knowledge based on context."""
  print("\n" + "=" * 50)
  print("Conditional Knowledge Retrieval")
  print("=" * 50)

  kb = create_company_knowledge_base()
  model = OpenAIChat(id="gpt-4o-mini")

  # Agent with knowledge enabled
  agent_with_kb = Agent(
    model=model,
    instructions="You are a helpful assistant.",
    config=AgentConfig(
      knowledge=KnowledgeConfig(
        knowledge=kb,
        top_k=3,
        enabled=True,  # Knowledge enabled
      ),
    ),
  )

  # Agent without knowledge (for general questions)
  agent_without_kb = Agent(
    model=model,
    instructions="You are a helpful assistant.",
  )

  # Company-specific question -> use knowledge
  print("\nCompany-specific question (with knowledge):")
  output = agent_with_kb.run("What's the remote work policy?")
  print(f"  {output.content}")

  # General question -> don't need knowledge
  print("\nGeneral question (without knowledge):")
  output = agent_without_kb.run("What is 2 + 2?")
  print(f"  {output.content}")


def main():
  basic_knowledge_config()
  advanced_knowledge_config()
  context_formats()
  multi_turn_with_knowledge()
  conditional_knowledge()


if __name__ == "__main__":
  main()
