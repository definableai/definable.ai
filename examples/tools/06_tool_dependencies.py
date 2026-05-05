"""
Tool dependencies and injection.

This example shows how to:
- Use Toolkit classes to manage shared dependencies
- Pass configuration to toolkits
- Share state between tools in a toolkit

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from typing import Dict, List, Optional

from definable.agent import Agent, Toolkit
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool

# ========================================
# Database Tools
# ========================================


@tool
def query_users(filter_name: Optional[str] = None) -> str:
  """Query users from the database.

  Args:
      filter_name: Optional name filter
  """
  # Mock database results
  results: List[Dict[str, object]] = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
  if filter_name:
    results = [u for u in results if filter_name.lower() in str(u["name"]).lower()]
  return f"Found {len(results)} users: {results}"


@tool
def insert_user(name: str, email: str) -> str:
  """Insert a new user into the database.

  Args:
      name: User's name
      email: User's email address
  """
  return f"Inserted user: {name} ({email})"


@tool
def get_user_count() -> int:
  """Get the total number of users in the database."""
  return 42  # Mock count


class DatabaseToolkit(Toolkit):
  """Toolkit with database client dependency."""

  # Assign tool functions
  query = query_users
  insert = insert_user
  count = get_user_count

  def __init__(self, connection_string: str):
    super().__init__(dependencies={"connection_string": connection_string})
    self._connection_string = connection_string
    print(f"[DB] Connected to {connection_string}")

  @property
  def tools(self) -> List:
    return [self.query, self.insert, self.count]


# ========================================
# API Tools
# ========================================


@tool
def fetch_resource(resource_type: str, resource_id: int) -> str:
  """Fetch a resource from the external API.

  Args:
      resource_type: Type of resource (users, products, orders)
      resource_id: ID of the resource
  """
  return f'Fetched {resource_type}/{resource_id}: {{"id": {resource_id}, "status": "ok"}}'


@tool
def create_resource(resource_type: str, name: str) -> str:
  """Create a new resource via the external API.

  Args:
      resource_type: Type of resource to create
      name: Name of the resource
  """
  return f"Created {resource_type}: {name}"


class APIToolkit(Toolkit):
  """Toolkit with HTTP client dependency."""

  fetch = fetch_resource
  create = create_resource

  def __init__(self, base_url: str, api_key: str):
    super().__init__(dependencies={"base_url": base_url, "api_key": api_key})
    self._base_url = base_url
    print(f"[API] Initialized for: {base_url}")

  @property
  def tools(self) -> List:
    return [self.fetch, self.create]


# ========================================
# Config Tools
# ========================================


@tool
def get_setting(key: str) -> str:
  """Get a configuration setting value.

  Args:
      key: The setting key to retrieve
  """
  # Mock settings
  settings = {
    "app_name": "MyApp",
    "version": "1.0.0",
    "max_retries": "3",
  }
  return f"{key} = {settings.get(key, 'not found')}"


@tool
def check_feature_flag(flag_name: str) -> bool:
  """Check if a feature flag is enabled.

  Args:
      flag_name: Name of the feature flag
  """
  # Mock feature flags
  flags = {"new_ui": True, "beta_features": False}
  return flags.get(flag_name, False)


class ConfigToolkit(Toolkit):
  """Toolkit with configuration dictionary dependency."""

  get = get_setting
  check_flag = check_feature_flag

  def __init__(self, config: Dict):
    super().__init__(dependencies={"config": config})
    self._config = config
    print(f"[Config] Loaded {len(config)} settings")

  @property
  def tools(self) -> List:
    return [self.get, self.check_flag]


def main():
  """Demonstrate tool dependencies."""
  model = OpenAIChat(id="gpt-4o-mini")

  # Create toolkits with dependencies
  db_toolkit = DatabaseToolkit(connection_string="postgresql://localhost/mydb")
  api_toolkit = APIToolkit(base_url="https://api.example.com", api_key="sk-test-123")
  config_toolkit = ConfigToolkit(
    config={
      "app_name": "MyApp",
      "version": "1.0.0",
      "feature_flags": {"new_ui": True},
    }
  )

  print("\nTool Dependencies Demonstration")
  print("=" * 50)

  # Test database toolkit
  print("\n1. Database Toolkit:")
  agent = Agent(model=model, toolkits=[db_toolkit])
  output = agent.run("Query all users from the database")
  print(f"   Result: {output.content}")

  # Test API toolkit
  print("\n2. API Toolkit:")
  agent = Agent(model=model, toolkits=[api_toolkit])
  output = agent.run("Fetch the user with ID 123")
  print(f"   Result: {output.content}")

  # Test config toolkit
  print("\n3. Config Toolkit:")
  agent = Agent(model=model, toolkits=[config_toolkit])
  output = agent.run("What is the app version?")
  print(f"   Result: {output.content}")

  # Combined - all toolkits
  print("\n4. All Toolkits Combined:")
  agent = Agent(
    model=model,
    toolkits=[db_toolkit, api_toolkit, config_toolkit],
    instructions="""You have access to:
1. Database tools for user management
2. API tools for external resources
3. Config tools for application settings""",
  )
  output = agent.run("Get the user count and check the new_ui feature flag")
  print(f"   Result: {output.content}")


def dependency_injection_example():
  """Show how to access toolkit dependencies."""
  print("\n" + "=" * 50)
  print("Dependency Injection Pattern")
  print("=" * 50)

  db_toolkit = DatabaseToolkit(connection_string="postgresql://localhost/test")

  print(f"\nToolkit: {db_toolkit.name}")
  print(f"Dependencies: {db_toolkit.dependencies}")
  print(f"Tools available: {len(db_toolkit.tools)}")

  for t in db_toolkit.tools:
    print(f"  - {t.name}: {t.description}")


if __name__ == "__main__":
  main()
  dependency_injection_example()
