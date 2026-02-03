"""
Multi-turn conversation sessions.

This example shows how to:
- Maintain conversation context across multiple turns
- Use session_id for conversation tracking
- Pass messages between agent runs

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from definable.agents import Agent
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool


@tool
def remember_fact(fact: str) -> str:
  """Store a fact that the user wants to remember."""
  return f"I'll remember: {fact}"


def basic_multi_turn():
  """Basic multi-turn conversation by passing messages."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a helpful assistant. Remember details from our conversation.",
  )

  print("Multi-turn Conversation")
  print("=" * 50)

  # First turn
  output1 = agent.run("Hi! My name is Alice and I'm a software engineer.")
  print("\nUser: Hi! My name is Alice and I'm a software engineer.")
  print(f"Agent: {output1.content}")

  # Second turn - pass previous messages to maintain context
  output2 = agent.run(
    "What's my name and profession?",
    messages=output1.messages,  # Pass previous messages
  )
  print("\nUser: What's my name and profession?")
  print(f"Agent: {output2.content}")

  # Third turn - continue the conversation
  output3 = agent.run(
    "I also enjoy hiking on weekends.",
    messages=output2.messages,  # Continue from output2
  )
  print("\nUser: I also enjoy hiking on weekends.")
  print(f"Agent: {output3.content}")

  # Fourth turn - test that context is maintained
  output4 = agent.run(
    "Summarize what you know about me.",
    messages=output3.messages,
  )
  print("\nUser: Summarize what you know about me.")
  print(f"Agent: {output4.content}")


def session_based_conversation():
  """Multi-turn with session tracking."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a helpful assistant.",
  )

  print("\n" + "=" * 50)
  print("Session-Based Conversation")
  print("=" * 50)

  # Start a new session
  output1 = agent.run("Let's start a new conversation. I'm Bob.")
  session_id = output1.session_id

  print(f"\nSession ID: {session_id}")
  print("User: Let's start a new conversation. I'm Bob.")
  print(f"Agent: {output1.content}")

  # Continue with same session
  output2 = agent.run(
    "What's my name?",
    messages=output1.messages,
    session_id=session_id,  # Keep the same session
  )
  print("\nUser: What's my name?")
  print(f"Agent: {output2.content}")
  print(f"Session ID unchanged: {output2.session_id == session_id}")


def conversation_with_tools():
  """Multi-turn conversation with tool usage."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[remember_fact],
    instructions="""You are a helpful assistant with memory capabilities.
Use the remember_fact tool to store important information the user shares.""",
  )

  print("\n" + "=" * 50)
  print("Conversation with Tools")
  print("=" * 50)

  # First turn - agent should use the remember_fact tool
  output1 = agent.run("Remember that my favorite color is blue.")
  print("\nUser: Remember that my favorite color is blue.")
  print(f"Agent: {output1.content}")
  if output1.tools:
    print(f"Tools used: {[t.tool_name for t in output1.tools]}")

  # Second turn - continue conversation
  output2 = agent.run(
    "Also remember that I have a cat named Whiskers.",
    messages=output1.messages,
  )
  print("\nUser: Also remember that I have a cat named Whiskers.")
  print(f"Agent: {output2.content}")

  # Third turn - ask about remembered facts
  output3 = agent.run(
    "What facts do you remember about me?",
    messages=output2.messages,
  )
  print("\nUser: What facts do you remember about me?")
  print(f"Agent: {output3.content}")


def interactive_conversation():
  """Interactive conversation loop (for demonstration)."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a helpful assistant. Be concise in your responses.",
  )

  print("\n" + "=" * 50)
  print("Interactive Conversation (Demo)")
  print("=" * 50)

  # Simulated user inputs for demonstration
  user_inputs = [
    "Hello!",
    "Tell me a short joke.",
    "That was funny! Tell me another one.",
  ]

  messages = None

  for user_input in user_inputs:
    print(f"\nUser: {user_input}")

    output = agent.run(
      user_input,
      messages=messages,
    )

    print(f"Agent: {output.content}")

    # Update messages for next turn
    messages = output.messages


if __name__ == "__main__":
  basic_multi_turn()
  session_based_conversation()
  conversation_with_tools()
  interactive_conversation()
