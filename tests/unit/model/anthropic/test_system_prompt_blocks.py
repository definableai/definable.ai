"""SystemPromptBlock + per-block cache control + TTL ordering."""

import pytest

from definable.model.anthropic.claude import Claude, SystemPromptBlock


def test_system_prompt_block_defaults():
  block = SystemPromptBlock(text="static rules")
  assert block.text == "static rules"
  assert block.cache is True
  assert block.ttl is None


def test_build_blocks_attaches_cache_control_when_cache_true():
  model = Claude(
    id="claude-sonnet-4-5-20250929",
    api_key="test",
    system_prompt_blocks=[SystemPromptBlock(text="static", cache=True)],
  )
  blocks = model._build_system_prompt_blocks(system_message=None)
  assert len(blocks) == 1
  assert blocks[0]["text"] == "static"
  assert blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_build_blocks_omits_cache_control_when_cache_false():
  model = Claude(
    id="claude-sonnet-4-5-20250929",
    api_key="test",
    system_prompt_blocks=[SystemPromptBlock(text="dynamic", cache=False)],
  )
  blocks = model._build_system_prompt_blocks(system_message=None)
  assert "cache_control" not in blocks[0]


def test_build_blocks_uses_block_ttl_over_extended_cache_time():
  model = Claude(
    id="claude-sonnet-4-5-20250929",
    api_key="test",
    extended_cache_time=True,
    system_prompt_blocks=[SystemPromptBlock(text="x", cache=True, ttl="5m")],
  )
  blocks = model._build_system_prompt_blocks(system_message=None)
  assert blocks[0]["cache_control"]["ttl"] == "5m"


def test_build_blocks_appends_agent_system_message_uncached():
  model = Claude(
    id="claude-sonnet-4-5-20250929",
    api_key="test",
    system_prompt_blocks=[SystemPromptBlock(text="static", cache=True)],
  )
  blocks = model._build_system_prompt_blocks(system_message="agent injected")
  assert len(blocks) == 2
  assert blocks[1]["text"] == "agent injected"
  assert "cache_control" not in blocks[1]


def test_validate_cache_ttl_order_rejects_5m_after_1h():
  blocks = [
    SystemPromptBlock(text="a", cache=True, ttl="1h"),
    SystemPromptBlock(text="b", cache=True, ttl="5m"),
  ]
  with pytest.raises(ValueError, match="5m.*after a 1h block"):
    Claude._validate_cache_ttl_order(blocks)


def test_validate_cache_ttl_order_rejects_default_after_1h():
  blocks = [
    SystemPromptBlock(text="a", cache=True, ttl="1h"),
    SystemPromptBlock(text="b", cache=True, ttl=None),
  ]
  with pytest.raises(ValueError):
    Claude._validate_cache_ttl_order(blocks)


def test_validate_cache_ttl_order_accepts_5m_then_1h():
  blocks = [
    SystemPromptBlock(text="a", cache=True, ttl="5m"),
    SystemPromptBlock(text="b", cache=True, ttl="1h"),
  ]
  # Should not raise
  Claude._validate_cache_ttl_order(blocks)


def test_validate_cache_ttl_order_skips_uncached_blocks():
  blocks = [
    SystemPromptBlock(text="a", cache=False, ttl="1h"),
    SystemPromptBlock(text="b", cache=True, ttl="5m"),
  ]
  # Uncached 1h block doesn't count as a 1h cache constraint
  Claude._validate_cache_ttl_order(blocks)
