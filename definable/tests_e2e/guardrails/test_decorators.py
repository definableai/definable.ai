"""Tests for guardrail decorators."""

import pytest

from definable.guardrails.base import GuardrailResult, InputGuardrail, OutputGuardrail, ToolGuardrail
from definable.guardrails.decorators import input_guardrail, output_guardrail, tool_guardrail
from definable.run.base import RunContext


def _ctx() -> RunContext:
  return RunContext(run_id="test-run", session_id="test-session")


class TestInputGuardrailDecorator:
  def test_creates_valid_protocol_object(self):
    @input_guardrail
    async def my_guard(text, context):
      return GuardrailResult.allow()

    assert isinstance(my_guard, InputGuardrail)
    assert my_guard.name == "my_guard"

  @pytest.mark.asyncio
  async def test_decorated_function_executes(self):
    @input_guardrail
    async def block_all(text, context):
      return GuardrailResult.block("nope")

    result = await block_all.check("hello", _ctx())
    assert result.action == "block"
    assert result.message == "nope"

  def test_name_override(self):
    @input_guardrail(name="custom_name")
    async def my_guard(text, context):
      return GuardrailResult.allow()

    assert my_guard.name == "custom_name"

  @pytest.mark.asyncio
  async def test_with_name_override_executes(self):
    @input_guardrail(name="my_custom")
    async def check_input(text, context):
      if "bad" in text:
        return GuardrailResult.block("bad word")
      return GuardrailResult.allow()

    r1 = await check_input.check("hello", _ctx())
    assert r1.action == "allow"

    r2 = await check_input.check("this is bad", _ctx())
    assert r2.action == "block"

  def test_repr(self):
    @input_guardrail
    async def my_guard(text, context):
      return GuardrailResult.allow()

    assert "InputGuardrail" in repr(my_guard)
    assert "my_guard" in repr(my_guard)


class TestOutputGuardrailDecorator:
  def test_creates_valid_protocol_object(self):
    @output_guardrail
    async def my_guard(text, context):
      return GuardrailResult.allow()

    assert isinstance(my_guard, OutputGuardrail)

  @pytest.mark.asyncio
  async def test_executes(self):
    @output_guardrail
    async def redact(text, context):
      return GuardrailResult.modify("redacted")

    result = await redact.check("hello", _ctx())
    assert result.action == "modify"
    assert result.modified_text == "redacted"


class TestToolGuardrailDecorator:
  def test_creates_valid_protocol_object(self):
    @tool_guardrail
    async def my_guard(tool_name, tool_args, context):
      return GuardrailResult.allow()

    assert isinstance(my_guard, ToolGuardrail)

  @pytest.mark.asyncio
  async def test_executes(self):
    @tool_guardrail
    async def no_delete(tool_name, tool_args, context):
      if tool_name == "delete":
        return GuardrailResult.block("no deleting")
      return GuardrailResult.allow()

    r1 = await no_delete.check("search", {}, _ctx())
    assert r1.action == "allow"

    r2 = await no_delete.check("delete", {}, _ctx())
    assert r2.action == "block"
