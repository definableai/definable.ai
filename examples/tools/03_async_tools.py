"""
Async tool functions.

This example shows how to:
- Define async tools using the @tool decorator
- Use async tools with agents
- Handle async operations within tools

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import asyncio

from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool


# Simple async tool
@tool
async def async_greet(name: str) -> str:
  """Greet a person asynchronously."""
  await asyncio.sleep(0.1)  # Simulate async operation
  return f"Hello, {name}!"


# Async tool simulating API call
@tool
async def fetch_user_data(user_id: int) -> str:
  """Fetch user data from a mock API.

  Args:
      user_id: The ID of the user to fetch
  """
  await asyncio.sleep(0.2)  # Simulate API latency

  # Mock user data
  users = {
    1: {"name": "Alice", "email": "alice@example.com", "role": "admin"},
    2: {"name": "Bob", "email": "bob@example.com", "role": "user"},
    3: {"name": "Charlie", "email": "charlie@example.com", "role": "user"},
  }

  if user_id in users:
    user = users[user_id]
    return f"User {user_id}: {user['name']} ({user['email']}) - {user['role']}"
  return f"User {user_id} not found"


# Async tool simulating database query
@tool
async def query_database(table: str, filter_field: str, filter_value: str) -> str:
  """Query a mock database table.

  Args:
      table: The table name to query
      filter_field: The field to filter on
      filter_value: The value to filter by
  """
  await asyncio.sleep(0.15)  # Simulate DB query time

  # Mock database
  data = {
    "products": [
      {"id": 1, "name": "Laptop", "category": "electronics", "price": 999},
      {"id": 2, "name": "Phone", "category": "electronics", "price": 699},
      {"id": 3, "name": "Desk", "category": "furniture", "price": 299},
    ],
    "orders": [
      {"id": 101, "status": "shipped", "total": 1299},
      {"id": 102, "status": "pending", "total": 699},
      {"id": 103, "status": "shipped", "total": 299},
    ],
  }

  if table not in data:
    return f"Table '{table}' not found"

  results = [row for row in data[table] if str(row.get(filter_field, "")).lower() == filter_value.lower()]

  if results:
    return f"Found {len(results)} results in {table}: {results}"
  return f"No results found in {table} where {filter_field} = {filter_value}"


# Async tool with error handling
@tool
async def process_payment(amount: float, currency: str = "USD") -> str:
  """Process a mock payment.

  Args:
      amount: The payment amount
      currency: The currency code (default: USD)
  """
  await asyncio.sleep(0.3)  # Simulate payment processing

  if amount <= 0:
    raise ValueError("Payment amount must be positive")

  if amount > 10000:
    return f"Payment of {amount} {currency} requires additional verification"

  return f"Payment of {amount} {currency} processed successfully. Transaction ID: TXN-{hash(str(amount)) % 10000:04d}"


# Async tool that calls external service
@tool
async def check_service_status(service_name: str) -> str:
  """Check the status of a mock service.

  Args:
      service_name: Name of the service to check
  """
  await asyncio.sleep(0.1)  # Simulate health check

  services = {
    "api": {"status": "healthy", "latency": "45ms"},
    "database": {"status": "healthy", "latency": "12ms"},
    "cache": {"status": "degraded", "latency": "120ms"},
    "queue": {"status": "healthy", "latency": "8ms"},
  }

  service = services.get(service_name.lower())
  if service:
    return f"{service_name}: {service['status']} (latency: {service['latency']})"
  return f"Service '{service_name}' not found"


async def main():
  """Demonstrate async tools with an agent."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[
      async_greet,
      fetch_user_data,
      query_database,
      process_payment,
      check_service_status,
    ],
    instructions="""You are a helpful assistant with access to various async tools.
Use the appropriate tool to answer user questions.""",
  )

  print("Testing Async Tools")
  print("=" * 50)

  # Test basic async tool
  output = await agent.arun("Greet someone named Diana")
  print(f"\nAsync greeting: {output.content}")

  # Test API fetch tool
  output = await agent.arun("Get information about user 1")
  print(f"\nUser data: {output.content}")

  # Test database query
  output = await agent.arun("Find all products in the electronics category")
  print(f"\nDatabase query: {output.content}")

  # Test payment processing
  output = await agent.arun("Process a payment of $150")
  print(f"\nPayment: {output.content}")

  # Test service status
  output = await agent.arun("Check the status of the database and cache services")
  print(f"\nService status: {output.content}")


def sync_usage():
  """Async tools can also be used with sync agent.run()."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[async_greet, fetch_user_data],
    instructions="You are a helpful assistant.",
  )

  print("\n" + "=" * 50)
  print("Sync Usage of Async Tools")
  print("=" * 50)

  # Async tools work with sync run() as well
  output = agent.run("Greet Eve and fetch user 2's data")
  print(f"\nResponse: {output.content}")


if __name__ == "__main__":
  asyncio.run(main())
  sync_usage()
