"""
Toolkit dependencies and shared state.

This example shows how to:
- Pass dependencies to a Toolkit
- Access dependencies in tools
- Use configuration with toolkits

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from typing import Dict, List

from definable.agent import Agent, Toolkit
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool

# ========================================
# Example 1: Simple toolkit with dependencies
# ========================================


@tool
def fetch_records(table: str, limit: int = 10) -> str:
  """Fetch records from a database table.

  Args:
      table: The table name
      limit: Maximum records to fetch
  """
  # In a real implementation, this would use the injected db_client
  return f"Fetched {limit} records from {table}"


@tool
def count_records(table: str) -> int:
  """Count records in a table."""
  return 42  # Mock count


class DatabaseToolkit(Toolkit):
  """Toolkit with database configuration."""

  # Assign tool functions as class attributes
  fetch = fetch_records
  count = count_records

  def __init__(self, connection_string: str):
    # Store dependencies that tools might need
    super().__init__(dependencies={"connection_string": connection_string})
    self._connection_string = connection_string
    print(f"[DB Toolkit] Initialized with connection: {connection_string[:20]}...")

  @property
  def tools(self) -> List:
    return [self.fetch, self.count]


# ========================================
# Example 2: API toolkit with configuration
# ========================================


@tool
def get_resource(resource_type: str, resource_id: int) -> str:
  """Get a resource from the API."""
  return f"Resource: {resource_type}/{resource_id}"


@tool
def create_resource(resource_type: str, name: str) -> str:
  """Create a new resource."""
  return f"Created {resource_type}: {name}"


class APIToolkit(Toolkit):
  """Toolkit for API operations."""

  get = get_resource
  create = create_resource

  def __init__(self, base_url: str, api_key: str):
    super().__init__(dependencies={"base_url": base_url, "api_key": api_key})
    self._base_url = base_url
    print(f"[API Toolkit] Initialized for: {base_url}")

  @property
  def tools(self) -> List:
    return [self.get, self.create]


# ========================================
# Example 3: Config toolkit
# ========================================


@tool
def get_setting(key: str) -> str:
  """Get a configuration setting."""
  # Mock settings
  settings = {"app_name": "MyApp", "version": "1.0.0", "debug": "true"}
  return settings.get(key, "not found")


@tool
def list_settings() -> str:
  """List all configuration settings."""
  return "Settings: app_name, version, debug"


class ConfigToolkit(Toolkit):
  """Toolkit for configuration access."""

  get = get_setting
  list_all = list_settings

  def __init__(self, config: Dict):
    super().__init__(dependencies={"config": config})
    self._config = config
    print(f"[Config Toolkit] Loaded {len(config)} settings")

  @property
  def tools(self) -> List:
    return [self.get, self.list_all]


def main():
  """Demonstrate toolkit dependencies."""
  print("Toolkit Dependencies")
  print("=" * 50)

  # Create toolkits with different configurations
  db_toolkit = DatabaseToolkit(connection_string="postgresql://localhost/mydb")
  api_toolkit = APIToolkit(base_url="https://api.example.com", api_key="sk-123")
  config_toolkit = ConfigToolkit(config={"app_name": "Demo", "version": "2.0"})

  model = OpenAIChat(id="gpt-4o-mini")

  # Test database toolkit
  print("\n1. Database Toolkit:")
  agent = Agent(model=model, toolkits=[db_toolkit])
  output = agent.run("Fetch 5 records from the users table")
  print(f"   Result: {output.content}")

  # Test API toolkit
  print("\n2. API Toolkit:")
  agent = Agent(model=model, toolkits=[api_toolkit])
  output = agent.run("Get user resource with ID 123")
  print(f"   Result: {output.content}")

  # Test config toolkit
  print("\n3. Config Toolkit:")
  agent = Agent(model=model, toolkits=[config_toolkit])
  output = agent.run("What is the app version setting?")
  print(f"   Result: {output.content}")

  # Combined usage
  print("\n4. Multiple Toolkits:")
  agent = Agent(
    model=model,
    toolkits=[db_toolkit, api_toolkit, config_toolkit],
    instructions="You have access to database, API, and config tools.",
  )
  output = agent.run("List all config settings and get API user resource 1")
  print(f"   Result: {output.content}")


def show_dependencies():
  """Show how to access toolkit dependencies."""
  print("\n" + "=" * 50)
  print("Accessing Dependencies")
  print("=" * 50)

  db_toolkit = DatabaseToolkit(connection_string="postgresql://localhost/test")

  print(f"\nToolkit name: {db_toolkit.name}")
  print(f"Dependencies: {db_toolkit.dependencies}")

  # Dependencies can be used when tools are executed
  # In a real implementation, tools would access these via
  # the agent's dependency injection


if __name__ == "__main__":
  main()
  show_dependencies()
