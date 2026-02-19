"""
Error handling and retry logic.

This example shows how to:
- Handle common errors gracefully
- Configure retry behavior
- Use try/except with agents
- Implement custom error handling

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import time
from typing import Optional

from definable.agent import Agent, AgentConfig
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool

# Tool that sometimes fails
fail_count = 0


@tool
def unreliable_api(query: str) -> str:
  """A mock API that fails intermittently."""
  global fail_count
  fail_count += 1

  # Fail on first 2 attempts, succeed on 3rd
  if fail_count <= 2:
    raise ConnectionError(f"Connection failed (attempt {fail_count})")

  fail_count = 0  # Reset for next test
  return f"Success! Result for: {query}"


@tool
def validated_input(number: int) -> str:
  """A tool that validates its input."""
  if number < 0:
    raise ValueError("Number must be non-negative")
  if number > 1000:
    raise ValueError("Number must be 1000 or less")
  return f"Processed number: {number}"


def basic_error_handling():
  """Basic try/except with agents."""
  print("Basic Error Handling")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(model=model, instructions="You are helpful.")

  try:
    output = agent.run("Hello!")
    print(f"Success: {output.content}")
  except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")


def retry_middleware_example():
  """Using RetryMiddleware for automatic retries."""
  print("\n" + "=" * 50)
  print("Retry Middleware")
  print("=" * 50)

  global fail_count
  fail_count = 0  # Reset

  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[unreliable_api],
    instructions="Use the unreliable_api tool when asked to fetch data.",
    config=AgentConfig(
      retry_transient_errors=True,
      max_retries=3,
      retry_backoff_base=1.0,  # Exponential backoff base
    ),
  )

  print("Making request to unreliable API...")
  print("(API will fail twice, then succeed)")

  try:
    output = agent.run("Fetch data for 'user stats'")
    print(f"\nFinal result: {output.content}")
  except Exception as e:
    print(f"Failed after retries: {e}")


def manual_retry_logic():
  """Implementing manual retry logic."""
  print("\n" + "=" * 50)
  print("Manual Retry Logic")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(model=model, instructions="Be brief.")

  max_attempts = 3
  delay = 1.0

  for attempt in range(1, max_attempts + 1):
    try:
      print(f"Attempt {attempt}/{max_attempts}...")
      output = agent.run("Say hello")
      print(f"Success: {output.content}")
      break
    except Exception as e:
      print(f"Failed: {e}")
      if attempt < max_attempts:
        print(f"Retrying in {delay}s...")
        time.sleep(delay)
        delay *= 2  # Exponential backoff
      else:
        print("All attempts failed")
        raise


def handle_tool_errors():
  """Handling errors from tools."""
  print("\n" + "=" * 50)
  print("Tool Error Handling")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[validated_input],
    instructions="""Use the validated_input tool with numbers.
If the tool returns an error, explain the constraint to the user.""",
  )

  # Valid input
  print("Testing valid input (50):")
  output = agent.run("Process the number 50")
  print(f"Response: {output.content}")

  # This will cause a validation error in the tool
  # The agent should handle it gracefully
  print("\nTesting invalid input (-5):")
  output = agent.run("Process the number -5")
  print(f"Response: {output.content}")


def error_types():
  """Different types of errors to handle."""
  print("\n" + "=" * 50)
  print("Common Error Types")
  print("=" * 50)

  print("""
Common Errors and Handling:

1. API Errors (rate limits, authentication):
   - Use RetryMiddleware with exponential backoff
   - Check API key configuration
   - Monitor usage and rate limits

2. Network Errors (timeouts, connection failures):
   - Configure appropriate timeouts
   - Implement retry logic
   - Use circuit breakers for repeated failures

3. Validation Errors (invalid inputs):
   - Validate inputs before calling agent
   - Handle gracefully in tool implementations
   - Return helpful error messages

4. Tool Execution Errors:
   - Tools should handle their own errors
   - Return error messages instead of raising exceptions
   - Log errors for debugging

5. Model Errors (content filters, context length):
   - Handle content policy violations gracefully
   - Truncate long inputs if needed
   - Consider model-specific limits
""")


def graceful_degradation():
  """Implement graceful degradation."""
  print("\n" + "=" * 50)
  print("Graceful Degradation")
  print("=" * 50)

  def run_with_fallback(
    primary_agent: Agent,
    fallback_agent: Optional[Agent],
    query: str,
  ) -> str:
    """Try primary agent, fall back if it fails."""
    try:
      output = primary_agent.run(query)
      return output.content or ""
    except Exception as e:
      print(f"Primary agent failed: {e}")

      if fallback_agent:
        print("Trying fallback agent...")
        try:
          output = fallback_agent.run(query)
          return output.content or ""
        except Exception as e2:
          print(f"Fallback also failed: {e2}")

      return "Sorry, I'm unable to process your request at this time."

  # Create primary and fallback agents
  model = OpenAIChat(id="gpt-4o-mini")

  primary = Agent(
    model=model,
    instructions="You are a detailed assistant.",
  )

  fallback = Agent(
    model=model,
    instructions="You are a simple assistant. Keep responses very brief.",
  )

  result = run_with_fallback(primary, fallback, "What is Python?")
  print(f"Result: {result}")


def error_logging():
  """Logging errors for debugging."""
  print("\n" + "=" * 50)
  print("Error Logging")
  print("=" * 50)

  import logging

  # Setup logging
  logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
  logger = logging.getLogger("agent")

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(model=model)

  try:
    output = agent.run("Hello")
    logger.info(f"Agent response: {(output.content or '')[:50]}...")
  except Exception as e:
    logger.error(f"Agent error: {type(e).__name__}: {e}", exc_info=True)
    raise


def main():
  basic_error_handling()
  retry_middleware_example()
  manual_retry_logic()
  handle_tool_errors()
  error_types()
  graceful_degradation()
  error_logging()


if __name__ == "__main__":
  main()
