"""Tests for built-in guardrails."""

import pytest

from definable.guardrails.builtin.input import block_topics, max_tokens, regex_filter
from definable.guardrails.builtin.output import max_output_tokens, pii_filter
from definable.guardrails.builtin.tool import tool_allowlist, tool_blocklist
from definable.run.base import RunContext


def _ctx() -> RunContext:
  return RunContext(run_id="test-run", session_id="test-session")


# ------------------------------------------------------------------
# Input built-ins
# ------------------------------------------------------------------


class TestMaxTokens:
  @pytest.mark.asyncio
  async def test_allows_under_limit(self):
    g = max_tokens(1000)
    result = await g.check("Hello world", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_blocks_over_limit(self):
    g = max_tokens(3)
    result = await g.check("This is a long input that definitely exceeds three tokens", _ctx())
    assert result.action == "block"
    assert "exceeds token limit" in (result.message or "")


class TestBlockTopics:
  @pytest.mark.asyncio
  async def test_blocks_matching_topic(self):
    g = block_topics(["violence", "drugs"])
    result = await g.check("Tell me about violence in movies", _ctx())
    assert result.action == "block"
    assert "violence" in (result.message or "")

  @pytest.mark.asyncio
  async def test_allows_non_matching(self):
    g = block_topics(["violence", "drugs"])
    result = await g.check("Tell me about cooking recipes", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_case_insensitive(self):
    g = block_topics(["VIOLENCE"])
    result = await g.check("Tell me about Violence in movies", _ctx())
    assert result.action == "block"


class TestRegexFilter:
  @pytest.mark.asyncio
  async def test_blocks_on_pattern_match(self):
    g = regex_filter([r"\b\d{3}-\d{2}-\d{4}\b"], action="block")
    result = await g.check("My SSN is 123-45-6789", _ctx())
    assert result.action == "block"

  @pytest.mark.asyncio
  async def test_allows_no_match(self):
    g = regex_filter([r"\b\d{3}-\d{2}-\d{4}\b"])
    result = await g.check("Hello world", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_modify_redacts(self):
    g = regex_filter([r"\b\d{3}-\d{2}-\d{4}\b"], action="modify")
    result = await g.check("My SSN is 123-45-6789", _ctx())
    assert result.action == "modify"
    assert "[REDACTED]" in (result.modified_text or "")
    assert "123-45-6789" not in (result.modified_text or "")


# ------------------------------------------------------------------
# Output built-ins
# ------------------------------------------------------------------


class TestPIIFilter:
  @pytest.mark.asyncio
  async def test_detects_email(self):
    g = pii_filter()
    result = await g.check("Contact me at john@example.com", _ctx())
    assert result.action == "modify"
    assert "[EMAIL]" in (result.modified_text or "")

  @pytest.mark.asyncio
  async def test_detects_phone(self):
    g = pii_filter()
    result = await g.check("Call me at 555-123-4567", _ctx())
    assert result.action == "modify"
    assert "[PHONE]" in (result.modified_text or "")

  @pytest.mark.asyncio
  async def test_detects_ssn(self):
    g = pii_filter()
    result = await g.check("SSN: 123-45-6789", _ctx())
    assert result.action == "modify"
    assert "[SSN]" in (result.modified_text or "")

  @pytest.mark.asyncio
  async def test_detects_credit_card(self):
    g = pii_filter()
    result = await g.check("Card: 4111-1111-1111-1111", _ctx())
    assert result.action == "modify"
    assert "[CREDIT_CARD]" in (result.modified_text or "")

  @pytest.mark.asyncio
  async def test_allows_no_pii(self):
    g = pii_filter()
    result = await g.check("Hello, how are you?", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_block_mode(self):
    g = pii_filter(action="block")
    result = await g.check("Email: john@example.com", _ctx())
    assert result.action == "block"


class TestMaxOutputTokens:
  @pytest.mark.asyncio
  async def test_allows_under_limit(self):
    g = max_output_tokens(1000)
    result = await g.check("Short response", _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_blocks_over_limit(self):
    g = max_output_tokens(3)
    result = await g.check("This is a long output that definitely exceeds three tokens", _ctx())
    assert result.action == "block"


# ------------------------------------------------------------------
# Tool built-ins
# ------------------------------------------------------------------


class TestToolAllowlist:
  @pytest.mark.asyncio
  async def test_allows_listed_tool(self):
    g = tool_allowlist({"search", "calculate"})
    result = await g.check("search", {"q": "test"}, _ctx())
    assert result.action == "allow"

  @pytest.mark.asyncio
  async def test_blocks_unlisted_tool(self):
    g = tool_allowlist({"search", "calculate"})
    result = await g.check("delete_all", {}, _ctx())
    assert result.action == "block"
    assert "not in the allowlist" in (result.message or "")


class TestToolBlocklist:
  @pytest.mark.asyncio
  async def test_blocks_listed_tool(self):
    g = tool_blocklist({"delete_all", "drop_table"})
    result = await g.check("delete_all", {}, _ctx())
    assert result.action == "block"

  @pytest.mark.asyncio
  async def test_allows_unlisted_tool(self):
    g = tool_blocklist({"delete_all", "drop_table"})
    result = await g.check("search", {"q": "test"}, _ctx())
    assert result.action == "allow"
