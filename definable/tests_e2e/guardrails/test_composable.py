"""Tests for composable guardrail combinators: ALL, ANY, NOT, when."""

import pytest

from definable.guardrails.base import GuardrailResult
from definable.guardrails.composable import ALL, ANY, NOT, when
from definable.run.base import RunContext


def _ctx(**kwargs) -> RunContext:
  return RunContext(run_id="test-run", session_id="test-session", **kwargs)


class _Allow:
  name = "allow"

  async def check(self, *args, **kwargs):
    return GuardrailResult.allow()


class _Block:
  name = "block"

  async def check(self, *args, **kwargs):
    return GuardrailResult.block("blocked")


class _Warn:
  name = "warn"

  async def check(self, *args, **kwargs):
    return GuardrailResult.warn("warning")


class TestALL:
  @pytest.mark.asyncio
  async def test_all_allow(self):
    g = ALL(_Allow(), _Allow())
    result = await g.check("text", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_blocks_if_any_blocks(self):
    g = ALL(_Allow(), _Block(), _Allow())
    result = await g.check("text", _ctx())
    assert result.action == "block"

  @pytest.mark.asyncio
  async def test_all_block(self):
    g = ALL(_Block(), _Block())
    result = await g.check("text", _ctx())
    assert result.action == "block"


class TestANY:
  @pytest.mark.asyncio
  async def test_allows_if_any_allows(self):
    g = ANY(_Block(), _Allow(), _Block())
    result = await g.check("text", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_blocks_if_all_block(self):
    g = ANY(_Block(), _Block())
    result = await g.check("text", _ctx())
    assert result.action == "block"

  @pytest.mark.asyncio
  async def test_all_allow(self):
    g = ANY(_Allow(), _Allow())
    result = await g.check("text", _ctx())
    assert result.action == "allow"


class TestNOT:
  @pytest.mark.asyncio
  async def test_inverts_allow_to_block(self):
    g = NOT(_Allow())
    result = await g.check("text", _ctx())
    assert result.action == "block"

  @pytest.mark.asyncio
  async def test_inverts_block_to_allow(self):
    g = NOT(_Block())
    result = await g.check("text", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_passes_through_warn(self):
    g = NOT(_Warn())
    result = await g.check("text", _ctx())
    assert result.action == "warn"


class TestWhen:
  @pytest.mark.asyncio
  async def test_runs_when_condition_true(self):
    g = when(lambda ctx: True, _Block())
    result = await g.check("text", _ctx())
    assert result.action == "block"

  @pytest.mark.asyncio
  async def test_skips_when_condition_false(self):
    g = when(lambda ctx: False, _Block())
    result = await g.check("text", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_condition_receives_context(self):
    def has_user(ctx):
      return ctx.user_id is not None

    g = when(has_user, _Block())

    # Without user_id — should skip
    result1 = await g.check("text", _ctx())
    assert result1.action == "allow"

    # With user_id — should block
    result2 = await g.check("text", _ctx(user_id="user-1"))
    assert result2.action == "block"


class TestNesting:
  @pytest.mark.asyncio
  async def test_nested_all_any(self):
    # ANY(ALL(allow, allow), block) -> allow (first branch)
    g = ANY(ALL(_Allow(), _Allow()), _Block())
    result = await g.check("text", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_nested_any_all(self):
    # ALL(ANY(block, allow), allow) -> allow
    g = ALL(ANY(_Block(), _Allow()), _Allow())
    result = await g.check("text", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_not_in_all(self):
    # ALL(NOT(block), allow) -> allow (NOT turns block → allow)
    g = ALL(NOT(_Block()), _Allow())
    result = await g.check("text", _ctx())
    assert result.action == "allow"
