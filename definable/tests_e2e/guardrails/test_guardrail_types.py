"""Tests for core guardrail types: GuardrailResult, protocols, Guardrails container."""

import pytest

from definable.guardrails.base import (
  Guardrails,
  GuardrailResult,
  InputGuardrail,
  OutputGuardrail,
  ToolGuardrail,
)
from definable.run.base import RunContext


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _ctx() -> RunContext:
  return RunContext(run_id="test-run", session_id="test-session")


class _AllowInput:
  name = "allow_input"

  async def check(self, text, context):
    return GuardrailResult.allow()


class _BlockInput:
  name = "block_input"

  async def check(self, text, context):
    return GuardrailResult.block("blocked")


class _ModifyInput:
  name = "modify_input"

  async def check(self, text, context):
    return GuardrailResult.modify("modified text", reason="cleaned")


class _WarnInput:
  name = "warn_input"

  async def check(self, text, context):
    return GuardrailResult.warn("heads up")


class _AllowOutput:
  name = "allow_output"

  async def check(self, text, context):
    return GuardrailResult.allow()


class _BlockOutput:
  name = "block_output"

  async def check(self, text, context):
    return GuardrailResult.block("output blocked")


class _AllowTool:
  name = "allow_tool"

  async def check(self, tool_name, tool_args, context):
    return GuardrailResult.allow()


class _BlockTool:
  name = "block_tool"

  async def check(self, tool_name, tool_args, context):
    return GuardrailResult.block("tool blocked")


# ------------------------------------------------------------------
# GuardrailResult tests
# ------------------------------------------------------------------


class TestGuardrailResult:
  def test_allow(self):
    r = GuardrailResult.allow()
    assert r.action == "allow"
    assert r.message is None
    assert r.modified_text is None

  def test_block(self):
    r = GuardrailResult.block("bad input")
    assert r.action == "block"
    assert r.message == "bad input"

  def test_modify(self):
    r = GuardrailResult.modify("new text", reason="cleaned")
    assert r.action == "modify"
    assert r.modified_text == "new text"
    assert r.message == "cleaned"

  def test_modify_no_reason(self):
    r = GuardrailResult.modify("new text")
    assert r.action == "modify"
    assert r.message is None

  def test_warn(self):
    r = GuardrailResult.warn("watch out")
    assert r.action == "warn"
    assert r.message == "watch out"


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------


class TestProtocolConformance:
  def test_input_protocol(self):
    assert isinstance(_AllowInput(), InputGuardrail)
    assert isinstance(_BlockInput(), InputGuardrail)

  def test_output_protocol(self):
    assert isinstance(_AllowOutput(), OutputGuardrail)

  def test_tool_protocol(self):
    assert isinstance(_AllowTool(), ToolGuardrail)
    assert isinstance(_BlockTool(), ToolGuardrail)


# ------------------------------------------------------------------
# Guardrails container tests
# ------------------------------------------------------------------


class TestGuardrailsContainer:
  @pytest.mark.asyncio
  async def test_input_allow(self):
    g = Guardrails(input=[_AllowInput()])
    results = await g.run_input_checks("hello", _ctx())
    assert len(results) == 1
    assert results[0].action == "allow"

  @pytest.mark.asyncio
  async def test_input_block(self):
    g = Guardrails(input=[_BlockInput()])
    results = await g.run_input_checks("hello", _ctx())
    assert len(results) == 1
    assert results[0].action == "block"

  @pytest.mark.asyncio
  async def test_input_modify(self):
    g = Guardrails(input=[_ModifyInput()])
    results = await g.run_input_checks("hello", _ctx())
    assert len(results) == 1
    assert results[0].action == "modify"
    assert results[0].modified_text == "modified text"

  @pytest.mark.asyncio
  async def test_input_warn(self):
    g = Guardrails(input=[_WarnInput()])
    results = await g.run_input_checks("hello", _ctx())
    assert len(results) == 1
    assert results[0].action == "warn"

  @pytest.mark.asyncio
  async def test_fail_fast_stops_at_first_block(self):
    g = Guardrails(input=[_BlockInput(), _AllowInput()], mode="fail_fast")
    results = await g.run_input_checks("hello", _ctx())
    assert len(results) == 1
    assert results[0].action == "block"

  @pytest.mark.asyncio
  async def test_run_all_continues_after_block(self):
    g = Guardrails(input=[_BlockInput(), _AllowInput()], mode="run_all")
    results = await g.run_input_checks("hello", _ctx())
    assert len(results) == 2
    assert results[0].action == "block"
    assert results[1].action == "allow"

  @pytest.mark.asyncio
  async def test_output_checks(self):
    g = Guardrails(output=[_AllowOutput(), _BlockOutput()])
    results = await g.run_output_checks("hello", _ctx())
    # fail_fast default: stops at first block — but first is allow, second is block
    assert len(results) == 2
    assert results[0].action == "allow"
    assert results[1].action == "block"

  @pytest.mark.asyncio
  async def test_tool_checks(self):
    g = Guardrails(tool=[_AllowTool()])
    results = await g.run_tool_checks("search", {"q": "test"}, _ctx())
    assert len(results) == 1
    assert results[0].action == "allow"

  @pytest.mark.asyncio
  async def test_tool_block(self):
    g = Guardrails(tool=[_BlockTool()], mode="fail_fast")
    results = await g.run_tool_checks("search", {}, _ctx())
    assert len(results) == 1
    assert results[0].action == "block"

  @pytest.mark.asyncio
  async def test_empty_guardrails(self):
    g = Guardrails()
    results = await g.run_input_checks("hello", _ctx())
    assert results == []

  @pytest.mark.asyncio
  async def test_guardrail_exception_becomes_block(self):
    class _ErrorGuardrail:
      name = "error_guardrail"

      async def check(self, text, context):
        raise RuntimeError("boom")

    g = Guardrails(input=[_ErrorGuardrail()])
    results = await g.run_input_checks("hello", _ctx())
    assert len(results) == 1
    assert results[0].action == "block"
    assert "boom" in (results[0].message or "")
