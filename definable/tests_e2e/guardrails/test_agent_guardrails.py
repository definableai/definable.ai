"""Integration tests: guardrails with Agent and MockModel."""

import pytest

from definable.agents.agent import Agent
from definable.agents.config import AgentConfig, TracingConfig
from definable.agents.testing import MockModel
from definable.exceptions import InputCheckError, OutputCheckError
from definable.guardrails.base import Guardrails, GuardrailResult
from definable.guardrails.builtin.input import block_topics
from definable.guardrails.builtin.output import pii_filter
from definable.guardrails.builtin.tool import tool_blocklist
from definable.guardrails.decorators import input_guardrail
from definable.run.base import RunStatus


def _agent(
  responses=None,
  guardrails=None,
  tools=None,
  **kwargs,
) -> Agent:
  return Agent(
    model=MockModel(responses=responses or ["Mock response"]),
    guardrails=guardrails,
    tools=tools,
    config=AgentConfig(tracing=TracingConfig(enabled=False)),
    **kwargs,
  )


# ------------------------------------------------------------------
# Input guardrails
# ------------------------------------------------------------------


class TestInputGuardrails:
  def test_input_block_prevents_model_call(self):
    """When input is blocked, the model should never be called."""
    model = MockModel(responses=["should not see this"])
    agent = Agent(
      model=model,
      guardrails=Guardrails(
        input=[block_topics(["forbidden"])],
        on_block="return_message",
      ),
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    output = agent.run("Tell me about forbidden things")
    assert output.status == RunStatus.blocked
    assert model.call_count == 0

  def test_input_block_raises_by_default(self):
    """Default on_block='raise' should raise InputCheckError."""
    agent = _agent(
      guardrails=Guardrails(input=[block_topics(["forbidden"])]),
    )

    with pytest.raises(InputCheckError) as exc_info:
      agent.run("Tell me about forbidden things")
    assert "forbidden" in str(exc_info.value).lower()

  def test_input_allow_passes_through(self):
    """Allowed input should reach the model normally."""
    model = MockModel(responses=["Hello!"])
    agent = Agent(
      model=model,
      guardrails=Guardrails(input=[block_topics(["forbidden"])]),
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    output = agent.run("Tell me about cooking")
    assert output.content == "Hello!"
    assert model.call_count == 1

  def test_input_modify_changes_text(self):
    """Input 'modify' should alter the message text sent to the model."""

    @input_guardrail
    async def clean_input(text, context):
      if "badword" in text:
        return GuardrailResult.modify(text.replace("badword", "***"))
      return GuardrailResult.allow()

    model = MockModel(responses=["Got it"])
    agent = Agent(
      model=model,
      guardrails=Guardrails(input=[clean_input]),
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    output = agent.run("Hello badword world")
    assert output.content == "Got it"
    assert model.call_count == 1
    # The model should have received the modified text
    last_call = model.call_history[0]
    messages = last_call["messages"]
    user_msgs = [m for m in messages if hasattr(m, "role") and m.role == "user"]
    assert any("***" in (m.content or "") for m in user_msgs)


# ------------------------------------------------------------------
# Output guardrails
# ------------------------------------------------------------------


class TestOutputGuardrails:
  def test_output_block_with_return_message(self):
    """Output block with on_block='return_message' returns blocked status."""
    agent = _agent(
      responses=["My email is john@example.com"],
      guardrails=Guardrails(
        output=[pii_filter(action="block")],
        on_block="return_message",
      ),
    )

    output = agent.run("What is your email?")
    assert output.status == RunStatus.blocked

  def test_output_block_raises(self):
    """Output block with on_block='raise' should raise OutputCheckError."""
    agent = _agent(
      responses=["My email is john@example.com"],
      guardrails=Guardrails(
        output=[pii_filter(action="block")],
        on_block="raise",
      ),
    )

    with pytest.raises(OutputCheckError):
      agent.run("What is your email?")

  def test_output_modify_redacts(self):
    """Output 'modify' should change the returned content."""
    agent = _agent(
      responses=["Contact john@example.com for help"],
      guardrails=Guardrails(
        output=[pii_filter(action="modify")],
      ),
    )

    output = agent.run("How do I contact support?")
    assert output.content is not None
    assert "[EMAIL]" in output.content
    assert "john@example.com" not in output.content

  def test_output_allow_passes_through(self):
    """Clean output passes through unchanged."""
    agent = _agent(
      responses=["Hello, how can I help?"],
      guardrails=Guardrails(output=[pii_filter()]),
    )

    output = agent.run("Say hello")
    assert output.content == "Hello, how can I help?"


# ------------------------------------------------------------------
# Tool guardrails
# ------------------------------------------------------------------


class TestToolGuardrails:
  def test_tool_block_prevents_execution(self):
    """A blocked tool should not be executed."""
    from definable.tools.decorator import tool

    @tool
    def dangerous_tool(x: str) -> str:
      """A dangerous tool."""
      return f"Executed: {x}"

    # MockModel needs to return tool calls then a final response
    from unittest.mock import MagicMock

    call_count = 0

    def side_effect(messages, tools, **kwargs):
      nonlocal call_count
      response = MagicMock()
      response.reasoning_content = None
      response.citations = None
      response.images = None
      response.videos = None
      response.audios = None
      response.response_usage = None

      if call_count == 0:
        # First call — model wants to use the tool
        call_count += 1
        response.content = None
        response.tool_calls = [
          {
            "id": "tc-1",
            "type": "function",
            "function": {
              "name": "dangerous_tool",
              "arguments": '{"x": "test"}',
            },
          }
        ]
      else:
        # Second call — model gives final answer after seeing block message
        call_count += 1
        response.content = "Tool was blocked, I cannot do that."
        response.tool_calls = []
      return response

    model = MockModel(side_effect=side_effect)
    agent = Agent(
      model=model,
      tools=[dangerous_tool],
      guardrails=Guardrails(
        tool=[tool_blocklist({"dangerous_tool"})],
      ),
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    output = agent.run("Use the dangerous tool on test")
    assert output.content == "Tool was blocked, I cannot do that."
    # The tool should have been blocked — check that the block message was in tool results
    blocked_tools = [t for t in (output.tools or []) if t.tool_call_error]
    assert len(blocked_tools) == 1
    assert "Blocked by guardrail" in (blocked_tools[0].result or "")


# ------------------------------------------------------------------
# on_block modes
# ------------------------------------------------------------------


class TestOnBlockModes:
  def test_return_message_mode(self):
    agent = _agent(
      guardrails=Guardrails(
        input=[block_topics(["forbidden"])],
        on_block="return_message",
      ),
    )
    output = agent.run("forbidden content")
    assert output.status == RunStatus.blocked
    assert output.content is not None

  def test_raise_mode(self):
    agent = _agent(
      guardrails=Guardrails(
        input=[block_topics(["forbidden"])],
        on_block="raise",
      ),
    )
    with pytest.raises(InputCheckError):
      agent.run("forbidden content")


# ------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------


class TestStreamingGuardrails:
  def test_input_block_stops_stream(self):
    """Input block during streaming should yield completed event and stop."""
    agent = _agent(
      guardrails=Guardrails(
        input=[block_topics(["forbidden"])],
        on_block="return_message",
      ),
    )

    events = list(agent.run_stream("Tell me about forbidden stuff"))
    # Should have a completed event with the block message
    assert len(events) >= 1
    completed = [e for e in events if hasattr(e, "event") and "Completed" in e.event]
    assert len(completed) >= 1


# ------------------------------------------------------------------
# Lazy exports
# ------------------------------------------------------------------


class TestLazyExports:
  def test_guardrails_from_agents(self):
    from definable.agents import Guardrails as G

    assert G is Guardrails

  def test_guardrail_result_from_agents(self):
    from definable.agents import GuardrailResult as GR

    assert GR is GuardrailResult
