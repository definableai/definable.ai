"""
Tool result caching.

This example shows how to:
- Enable caching for tool results
- Configure cache TTL (time-to-live)
- Use custom cache directories
- Understand when caching is beneficial

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import time

from definable.agents import Agent
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool

# Counter to track actual function calls
call_counter = {"expensive_lookup": 0, "api_call": 0}


# Tool with caching enabled
@tool(cache_results=True, cache_ttl=3600)  # Cache for 1 hour
def expensive_lookup(query: str) -> str:
  """Perform an expensive lookup operation.

  Results are cached for 1 hour to avoid redundant processing.

  Args:
      query: The search query
  """
  call_counter["expensive_lookup"] += 1
  print(f"  [CACHE MISS] expensive_lookup called (call #{call_counter['expensive_lookup']})")

  # Simulate expensive operation
  time.sleep(0.5)

  return f"Results for '{query}': Found 42 items after expensive processing"


# Tool with shorter cache TTL
@tool(cache_results=True, cache_ttl=60)  # Cache for 1 minute
def api_call(endpoint: str) -> str:
  """Call an external API.

  Results are cached for 1 minute to reduce API calls.

  Args:
      endpoint: The API endpoint to call
  """
  call_counter["api_call"] += 1
  print(f"  [CACHE MISS] api_call called (call #{call_counter['api_call']})")

  # Simulate API call
  time.sleep(0.2)

  return f'Response from {endpoint}: {{"status": "ok", "data": [...]}}'


# Tool without caching (for comparison)
@tool
def quick_calculation(x: int, y: int) -> int:
  """Perform a quick calculation (no caching needed).

  Args:
      x: First number
      y: Second number
  """
  print("  [NO CACHE] quick_calculation called")
  return x + y


# Tool with custom cache directory
@tool(
  cache_results=True,
  cache_ttl=7200,  # Cache for 2 hours
  cache_dir="/tmp/definable_cache",  # Custom cache directory
)
def heavy_processing(data: str) -> str:
  """Process data with heavy computation.

  Results are cached in a custom directory.

  Args:
      data: Data to process
  """
  print("  [CACHE MISS] heavy_processing called")
  time.sleep(0.3)
  return f"Processed: {data.upper()}"


def demonstrate_caching():
  """Demonstrate how caching works."""
  print("Cache Demonstration")
  print("=" * 50)

  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[expensive_lookup, api_call, quick_calculation, heavy_processing],
    instructions="""You are a helpful assistant with access to various tools.
Some tools have caching enabled to improve performance.""",
  )

  # First call - should be a cache miss
  print("\n1. First call to expensive_lookup:")
  start = time.time()
  output = agent.run("Look up information about 'python programming'")
  print(f"   Time: {time.time() - start:.2f}s")
  print(f"   Response: {(output.content or '')[:80]}...")

  # Second call with same query - should be a cache hit
  print("\n2. Second call with same query (should be cached):")
  start = time.time()
  output = agent.run("Look up information about 'python programming' again")
  print(f"   Time: {time.time() - start:.2f}s")
  print(f"   Response: {(output.content or '')[:80]}...")

  # Different query - should be a cache miss
  print("\n3. Different query (cache miss):")
  start = time.time()
  output = agent.run("Look up information about 'machine learning'")
  print(f"   Time: {time.time() - start:.2f}s")
  print(f"   Response: {(output.content or '')[:80]}...")

  # API call caching
  print("\n4. API call caching test:")
  output = agent.run("Call the /users endpoint")
  print("   First call done")

  output = agent.run("Call the /users endpoint again")
  print("   Second call done (should be cached)")

  # Non-cached function
  print("\n5. Non-cached calculation:")
  output = agent.run("Calculate 10 + 20")
  print(f"   First: {output.content}")

  output = agent.run("Calculate 10 + 20 again")
  print(f"   Second: {output.content}")

  # Summary
  print("\n" + "=" * 50)
  print("Call Summary:")
  print(f"  expensive_lookup actual calls: {call_counter['expensive_lookup']}")
  print(f"  api_call actual calls: {call_counter['api_call']}")
  print("\nNote: With caching, repeated identical calls use cached results.")


def caching_best_practices():
  """Best practices for tool caching."""
  print("\n" + "=" * 50)
  print("Caching Best Practices")
  print("=" * 50)
  print("""
When to enable caching:
  - Expensive computations that produce deterministic results
  - External API calls with rate limits
  - Database queries that don't change frequently
  - File parsing operations

When NOT to use caching:
  - Functions with side effects (sending emails, updating databases)
  - Functions that should return real-time data
  - Functions with non-deterministic outputs
  - Fast functions where caching overhead isn't worth it

Cache TTL guidelines:
  - Real-time data: Don't cache or use very short TTL (10-60 seconds)
  - Semi-static data: 5-15 minutes
  - Static reference data: 1-24 hours
  - Configuration data: 1-7 days
""")


if __name__ == "__main__":
  demonstrate_caching()
  caching_best_practices()
