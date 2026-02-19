"""Testing utilities for agents - MockModel and AgentTestCase."""

from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

from definable.agent.config import AgentConfig
from definable.agent.tracing import Tracing
from definable.model.metrics import Metrics
from definable.agent.events import RunStatus


class MockModel:
  """
  Mock Model for unit testing agents without API calls.

  MockModel simulates model responses for testing, allowing you to
  verify agent behavior without making actual API calls.

  Example:
      # Simple mock with canned responses
      model = MockModel(responses=["Hello!", "How can I help?"])

      # Mock with custom side effect
      def custom_response(messages, tools, **kwargs):
          response = MagicMock()
          response.content = f"Got {len(messages)} messages"
          response.tool_executions = []
          response.metrics = Metrics()
          return response

      model = MockModel(side_effect=custom_response)
  """

  def __init__(
    self,
    responses: Optional[List[str]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    side_effect: Optional[Callable] = None,
    reasoning_content: Optional[str] = None,
    structured_responses: Optional[List[str]] = None,
  ):
    """
    Initialize the mock model.

    Args:
        responses: List of canned responses to return in sequence.
        tool_calls: List of tool call dicts to simulate.
        side_effect: Custom function to generate responses.
        reasoning_content: Optional reasoning content to include.
        structured_responses: Canned JSON string responses for calls with response_format.
            Used when the agent requests structured output (e.g., thinking layer).
    """
    self.responses = responses or ["Mock response"]
    self.tool_calls = tool_calls or []
    self.side_effect = side_effect
    self.reasoning_content = reasoning_content
    self.structured_responses = structured_responses or []

    self._call_count = 0
    self._structured_call_count = 0
    self._call_history: List[Dict[str, Any]] = []

    # Model identity
    self.id = "mock-model"
    self.provider = "mock"
    self.supports_native_structured_outputs = True

  async def ainvoke(
    self,
    messages: Optional[List] = None,
    tools: Optional[List] = None,
    system_message: Optional[str] = None,
    output_schema=None,
    **kwargs,
  ) -> MagicMock:
    """
    Async invoke - returns mock response.

    Args:
        messages: Input messages.
        tools: Available tools.
        system_message: System message.
        output_schema: Output schema.
        **kwargs: Additional arguments.

    Returns:
        MagicMock with response attributes.
    """
    # Record the call
    self._call_history.append({
      "messages": messages,
      "tools": tools,
      "system_message": system_message,
      "output_schema": output_schema,
      **kwargs,
    })

    # Use side effect if provided
    if self.side_effect:
      return self.side_effect(messages, tools, **kwargs)

    # Check if this is a structured output request with canned structured responses
    response_format = kwargs.get("response_format") or output_schema
    if response_format is not None and self.structured_responses:
      response = MagicMock()
      response.content = self.structured_responses[min(self._structured_call_count, len(self.structured_responses) - 1)]
      response.tool_executions = []
      response.tool_calls = []
      response.response_usage = Metrics()
      response.reasoning_content = None
      response.citations = None
      response.images = None
      response.videos = None
      response.audios = None
      self._structured_call_count += 1
      self._call_count += 1
      return response

    # Build mock response
    response = MagicMock()
    response.content = self.responses[min(self._call_count, len(self.responses) - 1)]
    response.tool_executions = []
    response.tool_calls = []
    response.response_usage = Metrics()
    response.reasoning_content = self.reasoning_content
    response.citations = None
    response.images = None
    response.videos = None
    response.audios = None

    self._call_count += 1
    return response

  def invoke(self, *args, **kwargs) -> MagicMock:
    """
    Sync invoke - wraps async version.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        MagicMock with response attributes.
    """
    import asyncio

    return asyncio.run(self.ainvoke(*args, **kwargs))

  async def ainvoke_stream(
    self,
    messages: Optional[List] = None,
    tools: Optional[List] = None,
    **kwargs,
  ):
    """
    Async streaming invoke - yields mock chunks.

    Args:
        messages: Input messages.
        tools: Available tools.
        **kwargs: Additional arguments (response_format, assistant_message, etc.).

    Yields:
        MagicMock chunks with content.
    """
    # Check for structured output request
    response_format = kwargs.get("response_format")
    if response_format is not None and self.structured_responses:
      content = self.structured_responses[min(self._structured_call_count, len(self.structured_responses) - 1)]
      self._structured_call_count += 1
    else:
      content = self.responses[min(self._call_count, len(self.responses) - 1)]

    self._call_count += 1

    # Yield content in chunks (character-level)
    for char in content:
      chunk = MagicMock()
      chunk.content = char
      chunk.tool_calls = None
      chunk.response_usage = None
      yield chunk

  @property
  def call_count(self) -> int:
    """Return number of times the model was called."""
    return self._call_count

  @property
  def call_history(self) -> List[Dict[str, Any]]:
    """Return history of all calls made to the model."""
    return self._call_history

  def reset(self) -> None:
    """Reset call count and history."""
    self._call_count = 0
    self._call_history.clear()

  def assert_called(self) -> None:
    """Assert that the model was called at least once."""
    assert self._call_count > 0, "MockModel was not called"

  def assert_called_times(self, n: int) -> None:
    """Assert that the model was called exactly n times."""
    assert self._call_count == n, f"MockModel was called {self._call_count} times, expected {n}"


class AgentTestCase:
  """
  Base class for agent integration tests with utilities.

  Provides helper methods for creating test agents and making
  assertions about run outputs.

  Example:
      class TestMyAgent(AgentTestCase):
          def test_simple_response(self):
              agent = self.create_agent(
                  model=MockModel(responses=["Hello!"]),
                  instructions="Be helpful.",
              )
              output = agent.run("Say hello")
              assert output.content == "Hello!"
              self.assert_no_errors(output)

          def test_with_tools(self):
              @tool
              def greet(name: str) -> str:
                  return f"Hello, {name}!"

              agent = self.create_agent(tools=[greet])
              output = agent.run("Greet Alice")
              self.assert_tool_called(output, "greet")
  """

  def create_agent(
    self,
    model: Optional[MockModel] = None,
    tools: Optional[List] = None,
    toolkits: Optional[List] = None,
    instructions: Optional[str] = None,
    config_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
  ):
    """
    Create an agent with default test configuration.

    Args:
        model: Model to use (defaults to MockModel).
        tools: List of tools.
        toolkits: List of toolkits.
        instructions: System instructions.
        config_kwargs: Additional AgentConfig parameters.
        **kwargs: Additional Agent parameters.

    Returns:
        Agent instance configured for testing.
    """
    from definable.agent.agent import Agent

    config_kwargs = config_kwargs or {}

    return Agent(
      model=model or MockModel(),  # type: ignore[arg-type]
      tools=tools,
      toolkits=toolkits,
      instructions=instructions,
      config=AgentConfig(
        tracing=Tracing(enabled=False),  # Disable tracing in tests
        **config_kwargs,
      ),
      **kwargs,
    )

  def assert_tool_called(
    self,
    output,
    tool_name: str,
    msg: Optional[str] = None,
  ) -> None:
    """
    Assert a specific tool was called during the run.

    Args:
        output: RunOutput from agent.run().
        tool_name: Name of the tool expected to be called.
        msg: Optional message for assertion failure.
    """
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert tool_name in tool_names, msg or f"Tool '{tool_name}' not found in called tools: {tool_names}"

  def assert_tool_not_called(
    self,
    output,
    tool_name: str,
    msg: Optional[str] = None,
  ) -> None:
    """
    Assert a specific tool was NOT called during the run.

    Args:
        output: RunOutput from agent.run().
        tool_name: Name of the tool expected NOT to be called.
        msg: Optional message for assertion failure.
    """
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert tool_name not in tool_names, msg or f"Tool '{tool_name}' was unexpectedly called"

  def assert_no_errors(
    self,
    output,
    msg: Optional[str] = None,
  ) -> None:
    """
    Assert the run completed without errors.

    Args:
        output: RunOutput from agent.run().
        msg: Optional message for assertion failure.
    """
    assert output.status == RunStatus.completed, msg or f"Run did not complete successfully: status={output.status}"

  def assert_has_content(
    self,
    output,
    msg: Optional[str] = None,
  ) -> None:
    """
    Assert the output has non-empty content.

    Args:
        output: RunOutput from agent.run().
        msg: Optional message for assertion failure.
    """
    assert output.content, msg or "Output has no content"

  def assert_content_contains(
    self,
    output,
    substring: str,
    msg: Optional[str] = None,
  ) -> None:
    """
    Assert the output content contains a substring.

    Args:
        output: RunOutput from agent.run().
        substring: String expected to be in content.
        msg: Optional message for assertion failure.
    """
    content = str(output.content or "")
    assert substring in content, msg or f"Content does not contain '{substring}': {content[:100]}..."

  def assert_message_count(
    self,
    output,
    count: int,
    msg: Optional[str] = None,
  ) -> None:
    """
    Assert the output has a specific number of messages.

    Args:
        output: RunOutput from agent.run().
        count: Expected message count.
        msg: Optional message for assertion failure.
    """
    actual = len(output.messages or [])
    assert actual == count, msg or f"Expected {count} messages, got {actual}"


# Convenience function for creating test agents
def create_test_agent(
  responses: Optional[List[str]] = None,
  tools: Optional[List] = None,
  **kwargs,
):
  """
  Quick helper to create a test agent with MockModel.

  Args:
      responses: Canned responses for the mock model.
      tools: Tools to provide to the agent.
      **kwargs: Additional Agent parameters.

  Returns:
      Agent configured with MockModel.

  Example:
      agent = create_test_agent(
          responses=["Hello!", "Goodbye!"],
          tools=[my_tool],
      )
      output = agent.run("Hi")
      assert output.content == "Hello!"
  """
  from definable.agent.agent import Agent

  return Agent(
    model=MockModel(responses=responses),  # type: ignore[arg-type]
    tools=tools,
    config=AgentConfig(
      tracing=Tracing(enabled=False),
    ),
    **kwargs,
  )
