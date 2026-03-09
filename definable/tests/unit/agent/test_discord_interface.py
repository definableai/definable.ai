"""Unit tests for DiscordInterface.

Covers the _split_text static method and DiscordConfig validation.
Network-dependent logic (bot connection, message handling) is not tested
here — that would require a real Discord gateway or extensive mocking.
"""

import pytest

from definable.agent.interface.discord.config import DiscordConfig


# ===========================================================================
# DiscordConfig
# ===========================================================================


@pytest.mark.unit
class TestDiscordConfig:
  """Tests for DiscordConfig dataclass."""

  def test_valid_config(self):
    cfg = DiscordConfig(bot_token="tok-123")
    assert cfg.bot_token == "tok-123"
    assert cfg.platform == "discord"

  def test_default_intents(self):
    cfg = DiscordConfig(bot_token="tok")
    assert cfg.intents_message_content is True

  def test_default_respond_to_bots(self):
    cfg = DiscordConfig(bot_token="tok")
    assert cfg.respond_to_bots is False

  def test_default_max_message_length(self):
    cfg = DiscordConfig(bot_token="tok")
    assert cfg.max_message_length == 2000

  def test_default_connect_timeout(self):
    cfg = DiscordConfig(bot_token="tok")
    assert cfg.connect_timeout == 30.0

  def test_allowed_guild_ids(self):
    cfg = DiscordConfig(bot_token="tok", allowed_guild_ids=[123, 456])
    assert cfg.allowed_guild_ids == [123, 456]

  def test_allowed_channel_ids(self):
    cfg = DiscordConfig(bot_token="tok", allowed_channel_ids=[789])
    assert cfg.allowed_channel_ids == [789]

  def test_command_prefix(self):
    cfg = DiscordConfig(bot_token="tok", command_prefix="!")
    assert cfg.command_prefix == "!"

  def test_empty_bot_token_raises(self):
    from definable.agent.interface.errors import InterfaceError

    with pytest.raises(InterfaceError, match="bot_token"):
      DiscordConfig(bot_token="")


# ===========================================================================
# _split_text
# ===========================================================================


@pytest.mark.unit
class TestDiscordSplitText:
  """Tests for DiscordInterface._split_text static method."""

  @pytest.fixture
  def split_text(self):
    from definable.agent.interface.discord.interface import DiscordInterface

    return DiscordInterface._split_text

  def test_short_text_not_split(self, split_text):
    result = split_text("Hello!", 2000)
    assert result == ["Hello!"]

  def test_exact_max_length(self, split_text):
    text = "a" * 2000
    result = split_text(text, 2000)
    assert result == [text]

  def test_splits_at_newline(self, split_text):
    text = "Line one\nLine two"
    result = split_text(text, 10)
    assert len(result) == 2
    assert result[0] == "Line one"
    assert result[1] == "Line two"

  def test_splits_at_space(self, split_text):
    text = "Word1 Word2 Word3"
    result = split_text(text, 10)
    assert len(result) >= 2
    # Each chunk should be <= 10
    for chunk in result:
      assert len(chunk) <= 10

  def test_hard_split_no_space(self, split_text):
    text = "a" * 20
    result = split_text(text, 10)
    assert len(result) == 2
    assert result[0] == "a" * 10
    assert result[1] == "a" * 10

  def test_multiple_chunks(self, split_text):
    text = "Hello\nWorld\nFoo\nBar\nBaz"
    result = split_text(text, 12)
    # Should produce multiple chunks each <= 12
    for chunk in result:
      assert len(chunk) <= 12
    # Recombined should contain all original words
    combined = " ".join(result)
    assert "Hello" in combined
    assert "Baz" in combined

  def test_empty_text(self, split_text):
    result = split_text("", 2000)
    assert result == [""]

  def test_newline_preferred_over_space(self, split_text):
    text = "A B C\nD E F"
    result = split_text(text, 8)
    # Should split at newline (pos 5) not space
    assert result[0] == "A B C"
