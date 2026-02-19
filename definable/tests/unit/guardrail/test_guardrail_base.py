"""Unit tests for the guardrail base types.

Tests cover GuardrailResult factories, the Guardrails container
instantiation, and the protocol-level check signatures.
No real LLM or agent interaction is needed.
"""

import pytest

from definable.agent.guardrail.base import (
  GuardrailResult,
  Guardrails,
  InputGuardrail,
  OutputGuardrail,
  ToolGuardrail,
)


@pytest.mark.unit
class TestGuardrailResult:
  """Tests for GuardrailResult creation and factory helpers."""

  def test_create_allow(self):
    """GuardrailResult.allow() creates an allow result."""
    r = GuardrailResult.allow()
    assert r.action == "allow"
    assert r.message is None

  def test_create_block(self):
    """GuardrailResult.block(reason) creates a block result with message."""
    r = GuardrailResult.block("not allowed")
    assert r.action == "block"
    assert r.message == "not allowed"

  def test_create_modify(self):
    """GuardrailResult.modify() creates a modify result with new text."""
    r = GuardrailResult.modify("sanitized text", reason="PII removed")
    assert r.action == "modify"
    assert r.modified_text == "sanitized text"
    assert r.message == "PII removed"

  def test_create_modify_no_reason(self):
    """GuardrailResult.modify() with no reason sets message to None."""
    r = GuardrailResult.modify("clean")
    assert r.action == "modify"
    assert r.message is None

  def test_create_warn(self):
    """GuardrailResult.warn(message) creates a warn result."""
    r = GuardrailResult.warn("heads up")
    assert r.action == "warn"
    assert r.message == "heads up"

  def test_metadata_defaults_to_none(self):
    """GuardrailResult metadata defaults to None."""
    r = GuardrailResult(action="allow")
    assert r.metadata is None

  def test_metadata_can_be_set(self):
    """GuardrailResult metadata can be initialized."""
    r = GuardrailResult(action="allow", metadata={"latency_ms": 5.0})
    assert r.metadata["latency_ms"] == 5.0


@pytest.mark.unit
class TestGuardrailsContainer:
  """Tests for the Guardrails dataclass container."""

  def test_empty_guardrails(self):
    """Guardrails with no lists defaults to empty lists."""
    g = Guardrails()
    assert g.input == []
    assert g.output == []
    assert g.tool == []

  def test_default_mode_is_fail_fast(self):
    """Default mode is fail_fast."""
    g = Guardrails()
    assert g.mode == "fail_fast"

  def test_default_on_block_is_raise(self):
    """Default on_block is raise."""
    g = Guardrails()
    assert g.on_block == "raise"

  def test_mode_run_all(self):
    """Mode can be set to run_all."""
    g = Guardrails(mode="run_all")
    assert g.mode == "run_all"

  def test_on_block_return_message(self):
    """on_block can be set to return_message."""
    g = Guardrails(on_block="return_message")
    assert g.on_block == "return_message"


@pytest.mark.unit
class TestGuardrailProtocols:
  """Tests verifying that concrete classes satisfy the guardrail protocols."""

  def test_input_guardrail_protocol(self):
    """A class with name and async check(text, context) satisfies InputGuardrail."""

    class MyInput:
      name = "my_input"

      async def check(self, text, context):
        return GuardrailResult.allow()

    assert isinstance(MyInput(), InputGuardrail)

  def test_output_guardrail_protocol(self):
    """A class with name and async check(text, context) satisfies OutputGuardrail."""

    class MyOutput:
      name = "my_output"

      async def check(self, text, context):
        return GuardrailResult.allow()

    assert isinstance(MyOutput(), OutputGuardrail)

  def test_tool_guardrail_protocol(self):
    """A class with name and async check(tool_name, tool_args, context) satisfies ToolGuardrail."""

    class MyTool:
      name = "my_tool"

      async def check(self, tool_name, tool_args, context):
        return GuardrailResult.allow()

    assert isinstance(MyTool(), ToolGuardrail)


@pytest.mark.unit
class TestGuardrailsRunChecks:
  """Tests for Guardrails.run_input_checks / run_output_checks / run_tool_checks."""

  @pytest.mark.asyncio
  async def test_run_input_checks_empty(self):
    """run_input_checks with no guardrails returns empty list."""
    g = Guardrails()
    results = await g.run_input_checks("hello", context=None)
    assert results == []

  @pytest.mark.asyncio
  async def test_run_input_checks_allow(self):
    """run_input_checks returns allow results from the guardrail."""

    class AllowAll:
      name = "allow_all"

      async def check(self, text, context):
        return GuardrailResult.allow()

    g = Guardrails(input=[AllowAll()])
    results = await g.run_input_checks("hello", context=None)
    assert len(results) == 1
    assert results[0].action == "allow"

  @pytest.mark.asyncio
  async def test_run_input_checks_fail_fast_stops_on_block(self):
    """fail_fast mode stops after the first block."""

    class Blocker:
      name = "blocker"

      async def check(self, text, context):
        return GuardrailResult.block("blocked")

    class NeverReached:
      name = "never"

      async def check(self, text, context):
        raise AssertionError("Should not be reached")

    g = Guardrails(input=[Blocker(), NeverReached()], mode="fail_fast")
    results = await g.run_input_checks("test", context=None)
    assert len(results) == 1
    assert results[0].action == "block"

  @pytest.mark.asyncio
  async def test_run_output_checks_empty(self):
    """run_output_checks with no guardrails returns empty list."""
    g = Guardrails()
    results = await g.run_output_checks("response", context=None)
    assert results == []

  @pytest.mark.asyncio
  async def test_run_tool_checks_empty(self):
    """run_tool_checks with no guardrails returns empty list."""
    g = Guardrails()
    results = await g.run_tool_checks("my_tool", {"arg": "val"}, context=None)
    assert results == []

  @pytest.mark.asyncio
  async def test_run_tool_checks_block(self):
    """run_tool_checks returns block results from the guardrail."""

    class ToolBlocker:
      name = "tool_blocker"

      async def check(self, tool_name, tool_args, context):
        if tool_name == "dangerous":
          return GuardrailResult.block("tool blocked")
        return GuardrailResult.allow()

    g = Guardrails(tool=[ToolBlocker()])
    results = await g.run_tool_checks("dangerous", {}, context=None)
    assert len(results) == 1
    assert results[0].action == "block"

  @pytest.mark.asyncio
  async def test_guardrail_exception_becomes_block(self):
    """If a guardrail raises, it is caught and converted to a block result."""

    class RaisesError:
      name = "broken"

      async def check(self, text, context):
        raise RuntimeError("kaboom")

    g = Guardrails(input=[RaisesError()])
    results = await g.run_input_checks("test", context=None)
    assert len(results) == 1
    assert results[0].action == "block"
    assert "kaboom" in results[0].message
