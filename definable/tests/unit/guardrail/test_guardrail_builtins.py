"""Unit tests for built-in guardrails, decorators, combinators, and events.

Covers:
  - builtin/input.py: max_tokens, block_topics, regex_filter
  - builtin/output.py: pii_filter, max_output_tokens
  - builtin/tool.py: tool_allowlist, tool_blocklist
  - composable.py: ALL, ANY, NOT, when
  - decorators.py: @input_guardrail, @output_guardrail, @tool_guardrail
  - events.py: GuardrailCheckedEvent, GuardrailBlockedEvent
"""

import pytest

from definable.agent.guardrail.base import GuardrailResult, Guardrails
from definable.agent.guardrail.composable import ALL, ANY, NOT, when
from definable.agent.guardrail.decorators import (
  _InputGuardrailWrapper,
  _OutputGuardrailWrapper,
  _ToolGuardrailWrapper,
  input_guardrail,
  output_guardrail,
  tool_guardrail,
)
from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent
from definable.agent.guardrail.builtin.input import block_topics, max_tokens, regex_filter
from definable.agent.guardrail.builtin.output import pii_filter, max_output_tokens
from definable.agent.guardrail.builtin.tool import tool_allowlist, tool_blocklist
from definable.agent.run.base import RunContext


def _ctx() -> RunContext:
  return RunContext(run_id="test", session_id="test")


# ===========================================================================
# Built-in input guardrails
# ===========================================================================


@pytest.mark.unit
class TestBlockTopics:
  """Tests for block_topics guardrail."""

  @pytest.mark.asyncio
  async def test_blocks_matching_topic(self):
    guard = block_topics(["violence", "drugs"])
    result = await guard.check("Tell me about violence", _ctx())
    assert result.action == "block"
    assert "violence" in result.message  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_case_insensitive(self):
    guard = block_topics(["Violence"])
    result = await guard.check("VIOLENCE is bad", _ctx())
    assert result.action == "block"

  @pytest.mark.asyncio
  async def test_allows_clean_text(self):
    guard = block_topics(["violence", "drugs"])
    result = await guard.check("Tell me about cooking", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_empty_topics_allows_all(self):
    guard = block_topics([])
    result = await guard.check("anything", _ctx())
    assert result.action == "allow"

  def test_name(self):
    guard = block_topics(["x"])
    assert guard.name == "block_topics"


@pytest.mark.unit
class TestMaxTokens:
  """Tests for max_tokens guardrail."""

  @pytest.mark.asyncio
  async def test_short_text_allowed(self):
    guard = max_tokens(1000)
    result = await guard.check("hello", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_long_text_blocked(self):
    guard = max_tokens(5)
    long_text = "word " * 100
    result = await guard.check(long_text, _ctx())
    assert result.action == "block"
    assert "exceeds token limit" in result.message  # type: ignore[operator]

  def test_name(self):
    guard = max_tokens(100)
    assert guard.name == "max_tokens"


@pytest.mark.unit
class TestRegexFilter:
  """Tests for regex_filter guardrail."""

  @pytest.mark.asyncio
  async def test_block_mode_blocks_match(self):
    guard = regex_filter([r"\d{3}-\d{2}-\d{4}"])  # SSN pattern
    result = await guard.check("My SSN is 123-45-6789", _ctx())
    assert result.action == "block"
    assert "blocked pattern" in result.message  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_block_mode_allows_clean(self):
    guard = regex_filter([r"\d{3}-\d{2}-\d{4}"])
    result = await guard.check("Just a normal message", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_modify_mode_redacts(self):
    guard = regex_filter([r"\d{3}-\d{2}-\d{4}"], action="modify")
    result = await guard.check("My SSN is 123-45-6789", _ctx())
    assert result.action == "modify"
    assert "[REDACTED]" in result.modified_text  # type: ignore[operator]
    assert "123-45-6789" not in result.modified_text  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_multiple_patterns(self):
    guard = regex_filter([r"secret", r"password"], action="modify")
    result = await guard.check("My secret password is abc", _ctx())
    assert result.action == "modify"
    assert "secret" not in result.modified_text  # type: ignore[operator]
    assert "password" not in result.modified_text  # type: ignore[operator]

  def test_name(self):
    guard = regex_filter([r"test"])
    assert guard.name == "regex_filter"


# ===========================================================================
# Built-in output guardrails
# ===========================================================================


@pytest.mark.unit
class TestPIIFilter:
  """Tests for pii_filter guardrail."""

  @pytest.mark.asyncio
  async def test_modify_mode_redacts_email(self):
    guard = pii_filter(action="modify")
    result = await guard.check("Contact me at alice@example.com", _ctx())
    assert result.action == "modify"
    assert "[EMAIL]" in result.modified_text  # type: ignore[operator]
    assert "alice@example.com" not in result.modified_text  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_modify_mode_redacts_phone(self):
    guard = pii_filter(action="modify")
    result = await guard.check("Call me at 555-123-4567", _ctx())
    assert result.action == "modify"
    assert "[PHONE]" in result.modified_text  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_modify_mode_redacts_ssn(self):
    guard = pii_filter(action="modify")
    result = await guard.check("SSN: 123-45-6789", _ctx())
    assert result.action == "modify"
    assert "[SSN]" in result.modified_text  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_modify_mode_redacts_credit_card(self):
    guard = pii_filter(action="modify")
    result = await guard.check("Card: 4111-1111-1111-1111", _ctx())
    assert result.action == "modify"
    assert "[CREDIT_CARD]" in result.modified_text  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_block_mode_blocks_on_pii(self):
    guard = pii_filter(action="block")
    result = await guard.check("Email: test@example.com", _ctx())
    assert result.action == "block"
    assert "PII detected" in result.message  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_clean_text_allowed(self):
    guard = pii_filter()
    result = await guard.check("No personal information here.", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_default_action_is_modify(self):
    guard = pii_filter()
    assert guard._action == "modify"

  def test_name(self):
    guard = pii_filter()
    assert guard.name == "pii_filter"


@pytest.mark.unit
class TestMaxOutputTokens:
  """Tests for max_output_tokens guardrail."""

  @pytest.mark.asyncio
  async def test_short_output_allowed(self):
    guard = max_output_tokens(1000)
    result = await guard.check("short response", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_long_output_blocked(self):
    guard = max_output_tokens(5)
    long_text = "word " * 100
    result = await guard.check(long_text, _ctx())
    assert result.action == "block"
    assert "exceeds token limit" in result.message  # type: ignore[operator]

  def test_name(self):
    guard = max_output_tokens(100)
    assert guard.name == "max_output_tokens"


# ===========================================================================
# Built-in tool guardrails
# ===========================================================================


@pytest.mark.unit
class TestToolAllowlist:
  """Tests for tool_allowlist guardrail."""

  @pytest.mark.asyncio
  async def test_allowed_tool_passes(self):
    guard = tool_allowlist({"search", "calculate"})
    result = await guard.check("search", {"query": "test"}, _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_blocked_tool_rejected(self):
    guard = tool_allowlist({"search"})
    result = await guard.check("delete_all", {}, _ctx())
    assert result.action == "block"
    assert "not in the allowlist" in result.message  # type: ignore[operator]

  def test_name(self):
    guard = tool_allowlist({"x"})
    assert guard.name == "tool_allowlist"


@pytest.mark.unit
class TestToolBlocklist:
  """Tests for tool_blocklist guardrail."""

  @pytest.mark.asyncio
  async def test_blocked_tool_rejected(self):
    guard = tool_blocklist({"delete_all", "drop_table"})
    result = await guard.check("delete_all", {}, _ctx())
    assert result.action == "block"
    assert "is blocked" in result.message  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_allowed_tool_passes(self):
    guard = tool_blocklist({"delete_all"})
    result = await guard.check("search", {"query": "test"}, _ctx())
    assert result.action == "allow"

  def test_name(self):
    guard = tool_blocklist({"x"})
    assert guard.name == "tool_blocklist"


# ===========================================================================
# Composable combinators
# ===========================================================================


@pytest.mark.unit
class TestALL:
  """Tests for the ALL combinator."""

  @pytest.mark.asyncio
  async def test_all_allow_returns_allow(self):
    class AllowGuard:
      name = "a"

      async def check(self, *a, **kw):
        return GuardrailResult.allow()

    combo = ALL(AllowGuard(), AllowGuard())
    result = await combo.check("text", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_any_block_returns_block(self):
    class AllowGuard:
      name = "a"

      async def check(self, *a, **kw):
        return GuardrailResult.allow()

    class BlockGuard:
      name = "b"

      async def check(self, *a, **kw):
        return GuardrailResult.block("nope")

    combo = ALL(AllowGuard(), BlockGuard())
    result = await combo.check("text", _ctx())
    assert result.action == "block"
    assert result.message == "nope"

  @pytest.mark.asyncio
  async def test_short_circuits_on_block(self):
    """ALL stops at the first block, not running subsequent guardrails."""
    called = []

    class BlockGuard:
      name = "b"

      async def check(self, *a, **kw):
        called.append("block")
        return GuardrailResult.block("blocked")

    class NeverReached:
      name = "n"

      async def check(self, *a, **kw):
        called.append("never")
        return GuardrailResult.allow()

    combo = ALL(BlockGuard(), NeverReached())
    await combo.check("text", _ctx())
    assert called == ["block"]

  def test_name(self):
    combo = ALL(name="custom")
    assert combo.name == "custom"


@pytest.mark.unit
class TestANY:
  """Tests for the ANY combinator."""

  @pytest.mark.asyncio
  async def test_one_allow_returns_allow(self):
    class BlockGuard:
      name = "b"

      async def check(self, *a, **kw):
        return GuardrailResult.block("nope")

    class AllowGuard:
      name = "a"

      async def check(self, *a, **kw):
        return GuardrailResult.allow()

    combo = ANY(BlockGuard(), AllowGuard())
    result = await combo.check("text", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_all_block_returns_last_block(self):
    class BlockA:
      name = "a"

      async def check(self, *a, **kw):
        return GuardrailResult.block("first")

    class BlockB:
      name = "b"

      async def check(self, *a, **kw):
        return GuardrailResult.block("second")

    combo = ANY(BlockA(), BlockB())
    result = await combo.check("text", _ctx())
    assert result.action == "block"
    assert result.message == "second"

  @pytest.mark.asyncio
  async def test_short_circuits_on_allow(self):
    called = []

    class AllowGuard:
      name = "a"

      async def check(self, *a, **kw):
        called.append("allow")
        return GuardrailResult.allow()

    class NeverReached:
      name = "n"

      async def check(self, *a, **kw):
        called.append("never")
        return GuardrailResult.block("nope")

    combo = ANY(AllowGuard(), NeverReached())
    await combo.check("text", _ctx())
    assert called == ["allow"]

  @pytest.mark.asyncio
  async def test_empty_guardrails_returns_block(self):
    combo = ANY()
    result = await combo.check("text", _ctx())
    assert result.action == "block"


@pytest.mark.unit
class TestNOT:
  """Tests for the NOT combinator."""

  @pytest.mark.asyncio
  async def test_inverts_allow_to_block(self):
    class AllowGuard:
      name = "a"

      async def check(self, *a, **kw):
        return GuardrailResult.allow()

    combo = NOT(AllowGuard())
    result = await combo.check("text", _ctx())
    assert result.action == "block"
    assert "inverted" in result.message  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_inverts_block_to_allow(self):
    class BlockGuard:
      name = "b"

      async def check(self, *a, **kw):
        return GuardrailResult.block("nope")

    combo = NOT(BlockGuard())
    result = await combo.check("text", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_modify_passes_through(self):
    class ModifyGuard:
      name = "m"

      async def check(self, *a, **kw):
        return GuardrailResult.modify("changed", reason="modified")

    combo = NOT(ModifyGuard())
    result = await combo.check("text", _ctx())
    assert result.action == "modify"

  @pytest.mark.asyncio
  async def test_warn_passes_through(self):
    class WarnGuard:
      name = "w"

      async def check(self, *a, **kw):
        return GuardrailResult.warn("heads up")

    combo = NOT(WarnGuard())
    result = await combo.check("text", _ctx())
    assert result.action == "warn"


@pytest.mark.unit
class TestWhen:
  """Tests for the when conditional combinator."""

  @pytest.mark.asyncio
  async def test_runs_when_condition_true(self):
    class BlockGuard:
      name = "b"

      async def check(self, *a, **kw):
        return GuardrailResult.block("conditional block")

    combo = when(lambda ctx: True, BlockGuard())
    result = await combo.check("text", _ctx())
    assert result.action == "block"

  @pytest.mark.asyncio
  async def test_skips_when_condition_false(self):
    class BlockGuard:
      name = "b"

      async def check(self, *a, **kw):
        return GuardrailResult.block("should not reach")

    combo = when(lambda ctx: False, BlockGuard())
    result = await combo.check("text", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_context_from_positional_args(self):
    """Context is extracted from the last positional argument."""
    received_ctx = []

    class Spy:
      name = "spy"

      async def check(self, text, context):
        return GuardrailResult.allow()

    ctx = _ctx()
    combo = when(lambda c: received_ctx.append(c) or True, Spy())  # type: ignore[func-returns-value]
    await combo.check("text", ctx)
    assert len(received_ctx) == 1
    assert received_ctx[0] is ctx

  @pytest.mark.asyncio
  async def test_no_context_skips(self):
    """If no RunContext is found, condition is skipped (returns allow)."""

    class BlockGuard:
      name = "b"

      async def check(self, *a, **kw):
        return GuardrailResult.block("nope")

    combo = when(lambda ctx: True, BlockGuard())
    # Pass something that is NOT a RunContext
    result = await combo.check("text", "not-a-context")
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_context_from_kwargs(self):
    class Spy:
      name = "spy"

      async def check(self, *a, **kw):
        return GuardrailResult.allow()

    ctx = _ctx()
    combo = when(lambda c: True, Spy())
    result = await combo.check("text", context=ctx)
    assert result.action == "allow"


# ===========================================================================
# Decorators
# ===========================================================================


@pytest.mark.unit
class TestInputGuardrailDecorator:
  """Tests for @input_guardrail decorator."""

  def test_bare_decorator(self):
    @input_guardrail
    async def my_guard(text, context):
      return GuardrailResult.allow()

    assert isinstance(my_guard, _InputGuardrailWrapper)
    assert my_guard.name == "my_guard"

  def test_decorator_with_name(self):
    @input_guardrail(name="custom_name")
    async def my_guard(text, context):
      return GuardrailResult.allow()

    assert my_guard.name == "custom_name"

  @pytest.mark.asyncio
  async def test_decorated_function_runs(self):
    @input_guardrail
    async def block_bad(text, context):
      if "bad" in text:
        return GuardrailResult.block("bad word")
      return GuardrailResult.allow()

    result = await block_bad.check("bad stuff", _ctx())
    assert result.action == "block"
    result = await block_bad.check("good stuff", _ctx())
    assert result.action == "allow"

  def test_repr(self):
    @input_guardrail
    async def my_guard(text, context):
      return GuardrailResult.allow()

    assert repr(my_guard) == "InputGuardrail('my_guard')"


@pytest.mark.unit
class TestOutputGuardrailDecorator:
  """Tests for @output_guardrail decorator."""

  def test_bare_decorator(self):
    @output_guardrail
    async def my_guard(text, context):
      return GuardrailResult.allow()

    assert isinstance(my_guard, _OutputGuardrailWrapper)
    assert my_guard.name == "my_guard"

  def test_decorator_with_name(self):
    @output_guardrail(name="custom")
    async def my_guard(text, context):
      return GuardrailResult.allow()

    assert my_guard.name == "custom"

  @pytest.mark.asyncio
  async def test_decorated_function_runs(self):
    @output_guardrail
    async def check_length(text, context):
      if len(text) > 10:
        return GuardrailResult.block("too long")
      return GuardrailResult.allow()

    result = await check_length.check("short", _ctx())
    assert result.action == "allow"
    result = await check_length.check("this is a very long response", _ctx())
    assert result.action == "block"

  def test_repr(self):
    @output_guardrail
    async def my_guard(text, context):
      return GuardrailResult.allow()

    assert repr(my_guard) == "OutputGuardrail('my_guard')"


@pytest.mark.unit
class TestToolGuardrailDecorator:
  """Tests for @tool_guardrail decorator."""

  def test_bare_decorator(self):
    @tool_guardrail
    async def my_guard(tool_name, tool_args, context):
      return GuardrailResult.allow()

    assert isinstance(my_guard, _ToolGuardrailWrapper)
    assert my_guard.name == "my_guard"

  def test_decorator_with_name(self):
    @tool_guardrail(name="custom")
    async def my_guard(tool_name, tool_args, context):
      return GuardrailResult.allow()

    assert my_guard.name == "custom"

  @pytest.mark.asyncio
  async def test_decorated_function_runs(self):
    @tool_guardrail
    async def no_delete(tool_name, tool_args, context):
      if tool_name == "delete":
        return GuardrailResult.block("delete forbidden")
      return GuardrailResult.allow()

    result = await no_delete.check("delete", {}, _ctx())
    assert result.action == "block"
    result = await no_delete.check("search", {}, _ctx())
    assert result.action == "allow"

  def test_repr(self):
    @tool_guardrail
    async def my_guard(tool_name, tool_args, context):
      return GuardrailResult.allow()

    assert repr(my_guard) == "ToolGuardrail('my_guard')"


# ===========================================================================
# Events
# ===========================================================================


@pytest.mark.unit
class TestGuardrailEvents:
  """Tests for guardrail event types."""

  def test_checked_event_defaults(self):
    e = GuardrailCheckedEvent()
    assert e.event == "GuardrailChecked"
    assert e.guardrail_name == ""
    assert e.guardrail_type == ""
    assert e.action == ""
    assert e.message is None
    assert e.duration_ms is None

  def test_checked_event_fields(self):
    e = GuardrailCheckedEvent(
      guardrail_name="pii_filter",
      guardrail_type="output",
      action="modify",
      message="PII redacted",
      duration_ms=1.5,
    )
    assert e.guardrail_name == "pii_filter"
    assert e.guardrail_type == "output"
    assert e.action == "modify"
    assert e.message == "PII redacted"
    assert e.duration_ms == 1.5

  def test_blocked_event_defaults(self):
    e = GuardrailBlockedEvent()
    assert e.event == "GuardrailBlocked"
    assert e.guardrail_name == ""
    assert e.guardrail_type == ""
    assert e.reason == ""

  def test_blocked_event_fields(self):
    e = GuardrailBlockedEvent(
      guardrail_name="block_topics",
      guardrail_type="input",
      reason="Blocked topic: violence",
    )
    assert e.guardrail_name == "block_topics"
    assert e.reason == "Blocked topic: violence"


# ===========================================================================
# Container integration with builtins
# ===========================================================================


@pytest.mark.unit
class TestGuardrailsContainerWithBuiltins:
  """Tests for the Guardrails container using actual built-in guardrails."""

  @pytest.mark.asyncio
  async def test_input_block_topics_integration(self):
    g = Guardrails(input=[block_topics(["violence"])])
    results = await g.run_input_checks("tell me about violence", _ctx())
    assert len(results) == 1
    assert results[0].action == "block"

  @pytest.mark.asyncio
  async def test_output_pii_filter_integration(self):
    g = Guardrails(output=[pii_filter()])
    results = await g.run_output_checks("Email: test@example.com", _ctx())
    assert len(results) == 1
    assert results[0].action == "modify"
    assert "[EMAIL]" in results[0].modified_text  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_tool_blocklist_integration(self):
    g = Guardrails(tool=[tool_blocklist({"dangerous"})])
    results = await g.run_tool_checks("dangerous", {}, _ctx())
    assert len(results) == 1
    assert results[0].action == "block"

  @pytest.mark.asyncio
  async def test_run_all_mode_continues_after_block(self):
    g = Guardrails(
      input=[block_topics(["violence"]), block_topics(["drugs"])],
      mode="run_all",
    )
    results = await g.run_input_checks("violence and drugs", _ctx())
    assert len(results) == 2
    assert all(r.action == "block" for r in results)

  @pytest.mark.asyncio
  async def test_metadata_injected(self):
    g = Guardrails(input=[block_topics(["test"])])
    results = await g.run_input_checks("test", _ctx())
    assert results[0].metadata is not None
    assert "duration_ms" in results[0].metadata
    assert results[0].metadata["guardrail_name"] == "block_topics"

  @pytest.mark.asyncio
  async def test_composable_with_container(self):
    """ALL combinator works as an input guardrail in the container."""
    combo = ALL(block_topics(["violence"]), block_topics(["drugs"]))
    g = Guardrails(input=[combo])
    # Only drugs triggers
    results = await g.run_input_checks("tell me about drugs", _ctx())
    assert len(results) == 1
    assert results[0].action == "block"

  @pytest.mark.asyncio
  async def test_decorator_with_container(self):
    @input_guardrail
    async def custom_guard(text, context):
      if "forbidden" in text:
        return GuardrailResult.block("custom block")
      return GuardrailResult.allow()

    g = Guardrails(input=[custom_guard])
    results = await g.run_input_checks("this is forbidden", _ctx())
    assert len(results) == 1
    assert results[0].action == "block"
    assert results[0].message == "custom block"
