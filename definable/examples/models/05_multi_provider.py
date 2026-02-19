"""
Multi-provider model support.

This example shows how to:
- Use different model providers (OpenAI, DeepSeek, Moonshot, xAI)
- Switch between providers with the same interface
- Access provider-specific features (e.g., xAI citations)

Requirements:
    export OPENAI_API_KEY=sk-...
    export DEEPSEEK_API_KEY=sk-...  # For DeepSeek
    export MOONSHOT_API_KEY=sk-...  # For Moonshot
    export XAI_API_KEY=...          # For xAI/Grok
"""

import os

from definable.model.message import Message


def openai_example():
  """OpenAI GPT models."""
  from definable.model.openai import OpenAIChat

  print("OpenAI Example")
  print("-" * 40)

  model = OpenAIChat(
    id="gpt-4o-mini",
    # api_key=os.getenv("OPENAI_API_KEY"),  # Auto-detected from env
  )

  response = model.invoke(
    messages=[Message(role="user", content="What is Python?")],
    assistant_message=Message(role="assistant", content=""),
  )

  print(f"Model: {model.id}")
  print(f"Response: {(response.content or '')[:200]}...")
  print()


def deepseek_example():
  """DeepSeek models."""
  from definable.model.deepseek import DeepSeekChat

  print("DeepSeek Example")
  print("-" * 40)

  # Check if API key is available
  if not os.getenv("DEEPSEEK_API_KEY"):
    print("DEEPSEEK_API_KEY not set, skipping...")
    print()
    return

  model = DeepSeekChat(
    id="deepseek-chat",
    # api_key=os.getenv("DEEPSEEK_API_KEY"),  # Auto-detected from env
  )

  response = model.invoke(
    messages=[Message(role="user", content="Explain recursion in programming.")],
    assistant_message=Message(role="assistant", content=""),
  )

  print(f"Model: {model.id}")
  print(f"Response: {(response.content or '')[:200]}...")
  print()


def moonshot_example():
  """Moonshot models."""
  from definable.model.moonshot import MoonshotChat

  print("Moonshot Example")
  print("-" * 40)

  # Check if API key is available
  if not os.getenv("MOONSHOT_API_KEY"):
    print("MOONSHOT_API_KEY not set, skipping...")
    print()
    return

  model = MoonshotChat(
    id="moonshot-v1-8k",
    # api_key=os.getenv("MOONSHOT_API_KEY"),  # Auto-detected from env
  )

  response = model.invoke(
    messages=[Message(role="user", content="What is machine learning?")],
    assistant_message=Message(role="assistant", content=""),
  )

  print(f"Model: {model.id}")
  print(f"Response: {(response.content or '')[:200]}...")
  print()


def xai_example():
  """xAI/Grok models with optional web search."""
  from definable.model.xai import xAI

  print("xAI (Grok) Example")
  print("-" * 40)

  # Check if API key is available
  if not os.getenv("XAI_API_KEY"):
    print("XAI_API_KEY not set, skipping...")
    print()
    return

  model = xAI(
    id="grok-beta",
    # api_key=os.getenv("XAI_API_KEY"),  # Auto-detected from env
    # search_parameters={"enabled": True},  # Enable live web search
  )

  response = model.invoke(
    messages=[Message(role="user", content="What are the latest AI developments?")],
    assistant_message=Message(role="assistant", content=""),
  )

  print(f"Model: {model.id}")
  print(f"Response: {(response.content or '')[:200]}...")

  # Access citations if available (xAI feature with web search)
  if response.citations:
    print("\nCitations:")
    if response.citations.urls:
      for citation in response.citations.urls[:3]:
        print(f"  - {citation.title}: {citation.url}")
  print()


def provider_agnostic_function(model, prompt: str) -> str:
  """
  Demonstrate provider-agnostic code.

  All providers share the same interface, so you can write
  code that works with any model.
  """
  response = model.invoke(
    messages=[Message(role="user", content=prompt)],
    assistant_message=Message(role="assistant", content=""),
  )
  return response.content


def compare_providers():
  """Compare responses from different providers."""
  from definable.model.openai import OpenAIChat

  print("Provider-Agnostic Example")
  print("-" * 40)

  prompt = "What is 2 + 2? Answer with just the number."

  # Use OpenAI as the default
  model = OpenAIChat(id="gpt-4o-mini")
  result = provider_agnostic_function(model, prompt)
  print(f"OpenAI response: {result}")

  # You can easily swap in other providers:
  # model = DeepSeekChat(id="deepseek-chat")
  # model = MoonshotChat(id="moonshot-v1-8k")
  # model = xAI(id="grok-beta")
  print()


if __name__ == "__main__":
  openai_example()
  deepseek_example()
  moonshot_example()
  xai_example()
  compare_providers()
