"""Tests for the Slack interface implementation."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from definable.agent.interface.slack.config import SlackConfig
from definable.agent.interface.slack.formatter import markdown_to_mrkdwn, split_text


# ============================================================================
# Config tests
# ============================================================================


class TestSlackConfig:
  """Tests for SlackConfig validation."""

  def test_valid_socket_config(self):
    config = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
    assert config.platform == "slack"
    assert config.mode == "socket"
    assert config.bot_token == "xoxb-test"
    assert config.app_token == "xapp-test"

  def test_valid_http_config(self):
    config = SlackConfig(bot_token="xoxb-test", signing_secret="secret", mode="http")
    assert config.mode == "http"
    assert config.signing_secret == "secret"

  def test_missing_bot_token_raises(self):
    from definable.agent.interface.errors import InterfaceError

    with pytest.raises(InterfaceError, match="bot_token is required"):
      SlackConfig()

  def test_socket_mode_missing_app_token_raises(self):
    from definable.agent.interface.errors import InterfaceError

    with pytest.raises(InterfaceError, match="app_token is required"):
      SlackConfig(bot_token="xoxb-test", mode="socket")

  def test_http_mode_missing_signing_secret_raises(self):
    from definable.agent.interface.errors import InterfaceError

    with pytest.raises(InterfaceError, match="signing_secret is required"):
      SlackConfig(bot_token="xoxb-test", mode="http")

  def test_defaults(self):
    config = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
    assert config.respond_to_mentions is True
    assert config.respond_to_dms is True
    assert config.respond_to_thread_replies is True
    assert config.thread_replies_in_channel is True
    assert config.thread_replies_in_dm is False
    assert config.typing_reaction == "hourglass_flowing_sand"
    assert config.done_reaction == ""
    assert config.convert_markdown is True
    assert config.max_message_length == 40000
    assert config.max_retries == 3
    assert config.allowed_user_ids is None
    assert config.allowed_channel_ids is None

  def test_custom_config(self):
    config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      respond_to_dms=False,
      thread_replies_in_dm=True,
      typing_reaction="thinking_face",
      done_reaction="white_check_mark",
      allowed_user_ids=["U001", "U002"],
      allowed_channel_ids=["C001"],
      max_session_history=100,
      session_ttl_seconds=7200,
    )
    assert config.respond_to_dms is False
    assert config.thread_replies_in_dm is True
    assert config.typing_reaction == "thinking_face"
    assert config.done_reaction == "white_check_mark"
    assert config.allowed_user_ids == ["U001", "U002"]
    assert config.allowed_channel_ids == ["C001"]
    assert config.max_session_history == 100
    assert config.session_ttl_seconds == 7200

  def test_frozen(self):
    config = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
    with pytest.raises(AttributeError):
      config.bot_token = "changed"  # type: ignore[misc]

  def test_with_updates(self):
    config = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
    updated = config.with_updates(typing_reaction="brain")
    assert updated.typing_reaction == "brain"  # type: ignore[attr-defined]
    assert config.typing_reaction == "hourglass_flowing_sand"  # type: ignore[attr-defined]  # original unchanged

  def test_events_path_default(self):
    config = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
    assert config.events_path == "/slack/events"
    assert config.interactions_path == "/slack/interactions"


# ============================================================================
# Formatter tests
# ============================================================================


class TestMarkdownToMrkdwn:
  """Tests for the Markdown to mrkdwn converter."""

  def test_bold(self):
    assert markdown_to_mrkdwn("**hello**") == "*hello*"

  def test_italic_asterisk(self):
    assert markdown_to_mrkdwn("*hello*") == "_hello_"

  def test_strikethrough(self):
    assert markdown_to_mrkdwn("~~hello~~") == "~hello~"

  def test_links(self):
    assert markdown_to_mrkdwn("[Google](https://google.com)") == "<https://google.com|Google>"

  def test_headings(self):
    assert markdown_to_mrkdwn("# Title") == "*Title*"
    assert markdown_to_mrkdwn("## Subtitle") == "*Subtitle*"
    assert markdown_to_mrkdwn("### H3") == "*H3*"

  def test_inline_code_preserved(self):
    assert markdown_to_mrkdwn("`code`") == "`code`"
    # Bold inside code should not be converted
    assert markdown_to_mrkdwn("`**not bold**`") == "`**not bold**`"

  def test_code_block_language_stripped(self):
    input_text = "```python\nprint('hello')\n```"
    result = markdown_to_mrkdwn(input_text)
    assert result.startswith("```\n")
    assert "python" not in result
    assert "print('hello')" in result

  def test_code_block_content_preserved(self):
    input_text = "```\n**not bold**\n```"
    result = markdown_to_mrkdwn(input_text)
    assert "**not bold**" in result

  def test_mixed_formatting(self):
    input_text = "**Bold** and *italic* with [link](https://example.com)"
    result = markdown_to_mrkdwn(input_text)
    assert "*Bold*" in result
    assert "_italic_" in result
    assert "<https://example.com|link>" in result

  def test_plain_text_unchanged(self):
    text = "Hello, how are you?"
    assert markdown_to_mrkdwn(text) == text

  def test_multiline_headings(self):
    input_text = "# First\nSome text\n## Second"
    result = markdown_to_mrkdwn(input_text)
    assert "*First*" in result
    assert "Some text" in result
    assert "*Second*" in result

  def test_blockquote_preserved(self):
    assert markdown_to_mrkdwn("> quote") == "> quote"

  def test_empty_string(self):
    assert markdown_to_mrkdwn("") == ""


class TestSplitText:
  """Tests for the text splitting utility."""

  def test_short_text_no_split(self):
    assert split_text("hello", 100) == ["hello"]

  def test_exact_length(self):
    text = "a" * 100
    assert split_text(text, 100) == [text]

  def test_split_at_newline(self):
    text = "line1\nline2\nline3"
    chunks = split_text(text, 12)
    assert len(chunks) == 2
    assert chunks[0] == "line1\nline2"
    assert chunks[1] == "line3"

  def test_split_at_space(self):
    text = "word1 word2 word3"
    chunks = split_text(text, 12)
    assert len(chunks) == 2
    assert chunks[0] == "word1 word2"
    assert chunks[1] == "word3"

  def test_hard_split(self):
    text = "abcdefghij"
    chunks = split_text(text, 5)
    assert chunks == ["abcde", "fghij"]

  def test_empty_text(self):
    assert split_text("", 100) == [""]

  def test_long_text_many_chunks(self):
    text = "word " * 100  # 500 chars
    chunks = split_text(text.strip(), 50)
    assert all(len(c) <= 50 for c in chunks)
    # Ensure no content lost
    assert "".join(c.strip() for c in chunks).replace(" ", "") == "word" * 100


# ============================================================================
# Interface construction tests
# ============================================================================


class TestSlackInterfaceConstruction:
  """Tests for SlackInterface construction and configuration."""

  def test_constructor_kwargs(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      interface = SlackInterface(
        bot_token="xoxb-test",
        app_token="xapp-test",
      )
      assert interface._slack_config.bot_token == "xoxb-test"
      assert interface._slack_config.app_token == "xapp-test"
      assert interface._slack_config.mode == "socket"

  def test_constructor_http_mode(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      interface = SlackInterface(
        bot_token="xoxb-test",
        signing_secret="secret",
        mode="http",
      )
      assert interface._slack_config.mode == "http"
      assert interface._slack_config.signing_secret == "secret"

  def test_constructor_custom_params(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      interface = SlackInterface(
        bot_token="xoxb-test",
        app_token="xapp-test",
        respond_to_dms=False,
        typing_reaction="brain",
        done_reaction="white_check_mark",
        max_session_history=100,
        allowed_user_ids=["U001"],
      )
      assert interface._slack_config.respond_to_dms is False
      assert interface._slack_config.typing_reaction == "brain"
      assert interface._slack_config.done_reaction == "white_check_mark"
      assert interface.config.max_session_history == 100
      assert interface._slack_config.allowed_user_ids == ["U001"]

  def test_bot_thread_parents_initialized(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      interface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      assert interface._bot_thread_parents == set()
      assert interface._bot_user_id is None


# ============================================================================
# Inbound conversion tests
# ============================================================================


class TestConvertInbound:
  """Tests for _convert_inbound message conversion."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      iface._bot_user_id = "U_BOT"
      iface._client = MagicMock()
      # Mock users_info
      iface._client.users_info = AsyncMock(return_value={"user": {"real_name": "Test User", "name": "testuser"}})
      return iface

  @pytest.mark.asyncio
  async def test_basic_dm(self, interface):
    event = {
      "user": "U123",
      "channel": "D456",
      "ts": "1234567890.123456",
      "text": "Hello bot",
      "channel_type": "im",
    }
    msg = await interface._convert_inbound(event)
    assert msg is not None
    assert msg.platform == "slack"
    assert msg.platform_user_id == "U123"
    assert msg.text == "Hello bot"
    assert msg.platform_chat_id == "D456"  # DM channel as session
    assert msg.username == "Test User"

  @pytest.mark.asyncio
  async def test_mention_strips_bot_id(self, interface):
    event = {
      "user": "U123",
      "channel": "C789",
      "ts": "1234567890.123456",
      "text": "<@U_BOT> what's the weather?",
      "channel_type": "channel",
    }
    msg = await interface._convert_inbound(event)
    assert msg is not None
    assert msg.text == "what's the weather?"

  @pytest.mark.asyncio
  async def test_thread_reply_uses_thread_ts_as_chat_id(self, interface):
    event = {
      "user": "U123",
      "channel": "C789",
      "ts": "1234567891.000000",
      "thread_ts": "1234567890.123456",
      "text": "follow up",
      "channel_type": "channel",
    }
    msg = await interface._convert_inbound(event)
    assert msg is not None
    assert msg.platform_chat_id == "1234567890.123456"
    assert msg.reply_to_message_id == "1234567890.123456"

  @pytest.mark.asyncio
  async def test_access_control_user(self, interface):
    interface._slack_config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      allowed_user_ids=["U999"],
    )
    event = {
      "user": "U123",
      "channel": "D456",
      "ts": "123.456",
      "text": "hello",
      "channel_type": "im",
    }
    msg = await interface._convert_inbound(event)
    assert msg is None

  @pytest.mark.asyncio
  async def test_access_control_channel(self, interface):
    interface._slack_config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      allowed_channel_ids=["C999"],
    )
    event = {
      "user": "U123",
      "channel": "C123",
      "ts": "123.456",
      "text": "hello",
      "channel_type": "channel",
    }
    msg = await interface._convert_inbound(event)
    assert msg is None

  @pytest.mark.asyncio
  async def test_access_control_allowed(self, interface):
    interface._slack_config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      allowed_user_ids=["U123"],
    )
    event = {
      "user": "U123",
      "channel": "D456",
      "ts": "123.456",
      "text": "hello",
      "channel_type": "im",
    }
    msg = await interface._convert_inbound(event)
    assert msg is not None
    assert msg.text == "hello"

  @pytest.mark.asyncio
  async def test_metadata_populated(self, interface):
    event = {
      "user": "U123",
      "channel": "C789",
      "ts": "1234567890.123456",
      "thread_ts": "1234567889.000000",
      "text": "hello",
      "channel_type": "channel",
    }
    msg = await interface._convert_inbound(event)
    assert msg is not None
    assert msg.metadata["channel"] == "C789"
    assert msg.metadata["thread_ts"] == "1234567889.000000"
    assert msg.metadata["channel_type"] == "channel"
    assert msg.metadata["ts"] == "1234567890.123456"

  @pytest.mark.asyncio
  async def test_empty_text(self, interface):
    event = {
      "user": "U123",
      "channel": "D456",
      "ts": "123.456",
      "text": "",
      "channel_type": "im",
    }
    msg = await interface._convert_inbound(event)
    assert msg is not None
    assert msg.text is None  # Empty string → None

  @pytest.mark.asyncio
  async def test_new_channel_message_uses_ts_as_chat_id(self, interface):
    event = {
      "user": "U123",
      "channel": "C789",
      "ts": "1234567890.123456",
      "text": "<@U_BOT> hello",
      "channel_type": "channel",
    }
    msg = await interface._convert_inbound(event)
    assert msg is not None
    assert msg.platform_chat_id == "1234567890.123456"


# ============================================================================
# Event routing tests
# ============================================================================


class TestEventRouting:
  """Tests for message and mention event routing."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      iface._bot_user_id = "U_BOT"
      iface._client = MagicMock()
      iface.handle_platform_message = AsyncMock()  # type: ignore[method-assign]
      return iface

  @pytest.mark.asyncio
  async def test_skip_bot_messages(self, interface):
    event = {"bot_id": "B123", "user": "U123", "channel": "C789", "ts": "1.0"}
    await interface._on_message_event(event)
    interface.handle_platform_message.assert_not_called()

  @pytest.mark.asyncio
  async def test_skip_own_messages(self, interface):
    event = {"user": "U_BOT", "channel": "C789", "ts": "1.0"}
    await interface._on_message_event(event)
    interface.handle_platform_message.assert_not_called()

  @pytest.mark.asyncio
  async def test_skip_subtype_messages(self, interface):
    event = {"user": "U123", "channel": "C789", "ts": "1.0", "subtype": "channel_join"}
    await interface._on_message_event(event)
    interface.handle_platform_message.assert_not_called()

  @pytest.mark.asyncio
  async def test_file_share_subtype_allowed(self, interface):
    event = {
      "user": "U123",
      "channel": "D456",
      "ts": "1.0",
      "subtype": "file_share",
      "channel_type": "im",
      "text": "",
    }
    await interface._on_message_event(event)
    interface.handle_platform_message.assert_called_once()

  @pytest.mark.asyncio
  async def test_dm_handled(self, interface):
    event = {"user": "U123", "channel": "D456", "ts": "1.0", "channel_type": "im", "text": "hi"}
    await interface._on_message_event(event)
    interface.handle_platform_message.assert_called_once_with(event)

  @pytest.mark.asyncio
  async def test_dm_disabled(self, interface):
    interface._slack_config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      respond_to_dms=False,
    )
    event = {"user": "U123", "channel": "D456", "ts": "1.0", "channel_type": "im", "text": "hi"}
    await interface._on_message_event(event)
    interface.handle_platform_message.assert_not_called()

  @pytest.mark.asyncio
  async def test_thread_reply_in_bot_thread(self, interface):
    interface._bot_thread_parents.add("parent.ts")
    event = {
      "user": "U123",
      "channel": "C789",
      "ts": "1.1",
      "thread_ts": "parent.ts",
      "channel_type": "channel",
      "text": "follow up",
    }
    await interface._on_message_event(event)
    interface.handle_platform_message.assert_called_once()

  @pytest.mark.asyncio
  async def test_thread_reply_not_in_bot_thread(self, interface):
    event = {
      "user": "U123",
      "channel": "C789",
      "ts": "1.1",
      "thread_ts": "other.ts",
      "channel_type": "channel",
      "text": "follow up",
    }
    await interface._on_message_event(event)
    interface.handle_platform_message.assert_not_called()

  @pytest.mark.asyncio
  async def test_channel_message_without_mention_ignored(self, interface):
    event = {"user": "U123", "channel": "C789", "ts": "1.0", "channel_type": "channel", "text": "random chat"}
    await interface._on_message_event(event)
    interface.handle_platform_message.assert_not_called()

  @pytest.mark.asyncio
  async def test_mention_tracks_thread(self, interface):
    event = {"user": "U123", "channel": "C789", "ts": "1.0", "text": "<@U_BOT> hello"}
    await interface._on_mention_event(event)
    assert "1.0" in interface._bot_thread_parents
    interface.handle_platform_message.assert_called_once()

  @pytest.mark.asyncio
  async def test_mention_with_existing_thread(self, interface):
    event = {
      "user": "U123",
      "channel": "C789",
      "ts": "1.1",
      "thread_ts": "1.0",
      "text": "<@U_BOT> hello",
    }
    await interface._on_mention_event(event)
    assert "1.0" in interface._bot_thread_parents

  @pytest.mark.asyncio
  async def test_mention_disabled(self, interface):
    interface._slack_config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      respond_to_mentions=False,
    )
    event = {"user": "U123", "channel": "C789", "ts": "1.0", "text": "<@U_BOT> hello"}
    await interface._on_mention_event(event)
    interface.handle_platform_message.assert_not_called()


# ============================================================================
# Response sending tests
# ============================================================================


class TestSendResponse:
  """Tests for _send_response behavior."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      iface._bot_user_id = "U_BOT"
      iface._client = MagicMock()
      iface._client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "resp.ts"})
      iface._client.reactions_add = AsyncMock()
      iface._client.reactions_remove = AsyncMock()
      iface._client.files_upload_v2 = AsyncMock()
      return iface

  def _make_msg(self, **overrides):
    from definable.agent.interface.message import InterfaceMessage

    defaults: dict = {
      "platform": "slack",
      "platform_user_id": "U123",
      "platform_chat_id": "1.0",
      "platform_message_id": "1.0",
      "text": "hello",
      "metadata": {
        "channel": "C789",
        "thread_ts": None,
        "channel_type": "channel",
        "ts": "1.0",
      },
    }
    defaults.update(overrides)
    return InterfaceMessage(**defaults)  # type: ignore[arg-type]

  @pytest.mark.asyncio
  async def test_send_text_response(self, interface):
    from definable.agent.interface.message import InterfaceResponse

    msg = self._make_msg()
    response = InterfaceResponse(content="Hello back!")
    raw = {"channel": "C789", "ts": "1.0"}

    await interface._send_response(msg, response, raw)

    interface._client.chat_postMessage.assert_called_once()
    call_kwargs = interface._client.chat_postMessage.call_args
    assert call_kwargs.kwargs["channel"] == "C789"
    assert "Hello back!" in call_kwargs.kwargs["text"]

  @pytest.mark.asyncio
  async def test_thread_reply_in_channel(self, interface):
    from definable.agent.interface.message import InterfaceResponse

    msg = self._make_msg()
    response = InterfaceResponse(content="reply")
    raw = {"channel": "C789", "ts": "1.0"}

    await interface._send_response(msg, response, raw)

    call_kwargs = interface._client.chat_postMessage.call_args
    assert call_kwargs.kwargs.get("thread_ts") == "1.0"

  @pytest.mark.asyncio
  async def test_no_thread_in_dm(self, interface):
    from definable.agent.interface.message import InterfaceResponse

    msg = self._make_msg(
      metadata={"channel": "D456", "thread_ts": None, "channel_type": "im", "ts": "1.0"},
    )
    response = InterfaceResponse(content="reply")
    raw = {"channel": "D456", "ts": "1.0"}

    await interface._send_response(msg, response, raw)

    call_kwargs = interface._client.chat_postMessage.call_args
    assert call_kwargs.kwargs.get("thread_ts") is None

  @pytest.mark.asyncio
  async def test_existing_thread_continued(self, interface):
    from definable.agent.interface.message import InterfaceResponse

    msg = self._make_msg(
      metadata={"channel": "C789", "thread_ts": "parent.ts", "channel_type": "channel", "ts": "1.1"},
    )
    response = InterfaceResponse(content="reply")
    raw = {"channel": "C789", "ts": "1.1"}

    await interface._send_response(msg, response, raw)

    call_kwargs = interface._client.chat_postMessage.call_args
    assert call_kwargs.kwargs.get("thread_ts") == "parent.ts"

  @pytest.mark.asyncio
  async def test_markdown_conversion(self, interface):
    from definable.agent.interface.message import InterfaceResponse

    msg = self._make_msg()
    response = InterfaceResponse(content="**bold** and *italic*")
    raw = {"channel": "C789", "ts": "1.0"}

    await interface._send_response(msg, response, raw)

    call_kwargs = interface._client.chat_postMessage.call_args
    text = call_kwargs.kwargs["text"]
    assert "*bold*" in text
    assert "_italic_" in text

  @pytest.mark.asyncio
  async def test_done_reaction_added(self, interface):
    from definable.agent.interface.message import InterfaceResponse

    interface._slack_config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      done_reaction="white_check_mark",
    )
    msg = self._make_msg()
    response = InterfaceResponse(content="done")
    raw = {"channel": "C789", "ts": "1.0"}

    await interface._send_response(msg, response, raw)

    interface._client.reactions_add.assert_called_with(channel="C789", timestamp="1.0", name="white_check_mark")

  @pytest.mark.asyncio
  async def test_typing_reaction_removed(self, interface):
    from definable.agent.interface.message import InterfaceResponse

    msg = self._make_msg()
    response = InterfaceResponse(content="hi")
    raw = {"channel": "C789", "ts": "1.0"}

    await interface._send_response(msg, response, raw)

    interface._client.reactions_remove.assert_called_with(channel="C789", timestamp="1.0", name="hourglass_flowing_sand")

  @pytest.mark.asyncio
  async def test_send_image_url(self, interface):
    from definable.agent.interface.message import InterfaceResponse
    from definable.media import Image

    msg = self._make_msg()
    response = InterfaceResponse(images=[Image(url="https://example.com/img.png")])
    raw = {"channel": "C789", "ts": "1.0"}

    await interface._send_response(msg, response, raw)

    call_kwargs = interface._client.chat_postMessage.call_args
    blocks = call_kwargs.kwargs.get("blocks", [])
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0]["image_url"] == "https://example.com/img.png"

  @pytest.mark.asyncio
  async def test_send_image_bytes(self, interface):
    from definable.agent.interface.message import InterfaceResponse
    from definable.media import Image

    msg = self._make_msg()
    response = InterfaceResponse(images=[Image(content=b"\x89PNG...")])
    raw = {"channel": "C789", "ts": "1.0"}

    await interface._send_response(msg, response, raw)

    interface._client.files_upload_v2.assert_called_once()

  @pytest.mark.asyncio
  async def test_send_file(self, interface):
    from definable.agent.interface.message import InterfaceResponse
    from definable.media import File

    msg = self._make_msg()
    response = InterfaceResponse(files=[File(content=b"data", filename="test.txt")])
    raw = {"channel": "C789", "ts": "1.0"}

    await interface._send_response(msg, response, raw)

    interface._client.files_upload_v2.assert_called_once()

  @pytest.mark.asyncio
  async def test_new_thread_tracked(self, interface):
    from definable.agent.interface.message import InterfaceResponse

    interface._client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "new_thread.ts"})

    msg = self._make_msg()
    response = InterfaceResponse(content="starting a thread")
    raw = {"channel": "C789", "ts": "1.0"}

    await interface._send_response(msg, response, raw)

    # The new message ts should be tracked as a bot thread
    assert "new_thread.ts" in interface._bot_thread_parents


# ============================================================================
# Thread resolution tests
# ============================================================================


class TestThreadResolution:
  """Tests for _resolve_thread_ts logic."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      return SlackInterface(bot_token="xoxb-test", app_token="xapp-test")

  def _make_msg(self, thread_ts=None, ts="1.0", channel_type="channel"):
    from definable.agent.interface.message import InterfaceMessage

    return InterfaceMessage(
      platform="slack",
      platform_user_id="U123",
      platform_chat_id="1.0",
      platform_message_id=ts,
      text="test",
      metadata={"channel": "C789", "thread_ts": thread_ts, "channel_type": channel_type, "ts": ts},
    )

  def test_existing_thread_returned(self, interface):
    msg = self._make_msg(thread_ts="parent.ts")
    assert interface._resolve_thread_ts(msg, "channel") == "parent.ts"

  def test_channel_message_threaded(self, interface):
    msg = self._make_msg(ts="1.0")
    assert interface._resolve_thread_ts(msg, "channel") == "1.0"

  def test_dm_not_threaded_by_default(self, interface):
    msg = self._make_msg(ts="1.0")
    assert interface._resolve_thread_ts(msg, "im") is None

  def test_dm_threaded_when_enabled(self, interface):
    interface._slack_config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      thread_replies_in_dm=True,
    )
    msg = self._make_msg(ts="1.0")
    assert interface._resolve_thread_ts(msg, "im") == "1.0"

  def test_channel_threading_disabled(self, interface):
    interface._slack_config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      thread_replies_in_channel=False,
    )
    msg = self._make_msg(ts="1.0")
    assert interface._resolve_thread_ts(msg, "channel") is None


# ============================================================================
# Media extraction tests
# ============================================================================


class TestMediaExtraction:
  """Tests for _extract_media file categorization."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      iface._download_file = AsyncMock(return_value=b"filedata")  # type: ignore[method-assign]
      return iface

  @pytest.mark.asyncio
  async def test_image_extraction(self, interface):
    files = [{"mimetype": "image/png", "url_private": "https://slack.com/file", "name": "photo.png"}]
    images, audio, video, other = await interface._extract_media(files)
    assert images is not None
    assert len(images) == 1
    assert images[0].mime_type == "image/png"
    assert audio is None
    assert video is None
    assert other is None

  @pytest.mark.asyncio
  async def test_audio_extraction(self, interface):
    files = [
      {
        "mimetype": "audio/mp4",
        "url_private": "https://slack.com/file",
        "name": "clip.m4a",
        "duration_ms": 5000,
      }
    ]
    images, audio, video, other = await interface._extract_media(files)
    assert audio is not None
    assert len(audio) == 1
    assert audio[0].duration == 5.0
    assert images is None

  @pytest.mark.asyncio
  async def test_video_extraction(self, interface):
    files = [{"mimetype": "video/mp4", "url_private": "https://slack.com/file", "name": "clip.mp4"}]
    images, audio, video, other = await interface._extract_media(files)
    assert video is not None
    assert len(video) == 1
    assert images is None

  @pytest.mark.asyncio
  async def test_document_extraction(self, interface):
    files = [
      {
        "mimetype": "application/pdf",
        "url_private": "https://slack.com/file",
        "name": "doc.pdf",
        "size": 1024,
      }
    ]
    images, audio, video, other = await interface._extract_media(files)
    assert other is not None
    assert len(other) == 1
    assert other[0].filename == "doc.pdf"
    assert other[0].size == 1024

  @pytest.mark.asyncio
  async def test_unknown_mimetype_no_mime_set(self, interface):
    files = [
      {
        "mimetype": "application/x-unknown-type",
        "url_private": "https://slack.com/file",
        "name": "data.bin",
      }
    ]
    images, audio, video, other = await interface._extract_media(files)
    assert other is not None
    assert other[0].mime_type is None  # Not in valid_mime_types

  @pytest.mark.asyncio
  async def test_multiple_files(self, interface):
    files = [
      {"mimetype": "image/jpeg", "url_private": "https://slack.com/1", "name": "a.jpg"},
      {"mimetype": "image/png", "url_private": "https://slack.com/2", "name": "b.png"},
      {"mimetype": "application/pdf", "url_private": "https://slack.com/3", "name": "c.pdf"},
    ]
    images, audio, video, other = await interface._extract_media(files)
    assert images is not None
    assert len(images) == 2
    assert other is not None
    assert len(other) == 1

  @pytest.mark.asyncio
  async def test_no_url_private_skipped(self, interface):
    files = [{"mimetype": "image/png", "name": "broken.png"}]
    images, audio, video, other = await interface._extract_media(files)
    assert images is None

  @pytest.mark.asyncio
  async def test_download_failure_skipped(self, interface):
    interface._download_file = AsyncMock(return_value=None)  # type: ignore[method-assign]
    files = [{"mimetype": "image/png", "url_private": "https://slack.com/file", "name": "photo.png"}]
    images, audio, video, other = await interface._extract_media(files)
    assert images is None


# ============================================================================
# Error handling tests
# ============================================================================


class TestErrorHandling:
  """Tests for Slack API error mapping."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      return SlackInterface(bot_token="xoxb-test", app_token="xapp-test")

  def test_generic_error_logged(self, interface):
    """Non-Slack errors should be logged without raising."""
    interface._handle_api_error(RuntimeError("something broke"), "test_method")
    # Should not raise — just logs

  def test_auth_error_raised(self, interface):
    from definable.agent.interface.errors import InterfaceAuthenticationError

    try:
      from slack_sdk.errors import SlackApiError

      mock_resp = MagicMock()
      mock_resp.status_code = 401
      mock_resp.get = MagicMock(return_value="invalid_auth")
      error = SlackApiError("auth failed", mock_resp)

      with pytest.raises(InterfaceAuthenticationError):
        interface._handle_api_error(error, "auth.test")
    except ImportError:
      pytest.skip("slack_sdk not installed")

  def test_rate_limit_error_raised(self, interface):
    from definable.agent.interface.errors import InterfaceRateLimitError

    try:
      from slack_sdk.errors import SlackApiError

      mock_resp = MagicMock()
      mock_resp.status_code = 429
      mock_resp.get = MagicMock(return_value="rate_limited")
      mock_resp.headers = {"Retry-After": "30"}
      error = SlackApiError("rate limited", mock_resp)

      with pytest.raises(InterfaceRateLimitError) as exc_info:
        interface._handle_api_error(error, "chat.postMessage")
      assert exc_info.value.retry_after == 30.0
    except ImportError:
      pytest.skip("slack_sdk not installed")


# ============================================================================
# Lazy import / __init__.py tests
# ============================================================================


class TestLazyImports:
  """Tests for lazy import registration in interface __init__.py."""

  def test_slack_interface_importable(self):
    from definable.agent.interface import SlackInterface

    assert SlackInterface is not None

  def test_slack_config_importable(self):
    from definable.agent.interface import SlackConfig

    assert SlackConfig is not None

  def test_slack_subpackage_importable(self):
    from definable.agent.interface.slack import SlackInterface, SlackConfig

    assert SlackInterface is not None
    assert SlackConfig is not None


# ============================================================================
# Lifecycle tests
# ============================================================================


class TestLifecycle:
  """Tests for interface start/stop lifecycle."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      return SlackInterface(bot_token="xoxb-test", app_token="xapp-test")

  def test_initial_state(self, interface):
    assert interface._bolt_app is None
    assert interface._socket_handler is None
    assert interface._client is None
    assert interface._bot_user_id is None
    assert not interface._running

  def test_get_bolt_app_before_start_raises(self, interface):
    with pytest.raises(RuntimeError, match="not started"):
      interface.get_bolt_app()

  def test_bind_agent(self, interface):
    mock_agent = MagicMock()
    result = interface.bind(mock_agent)
    assert result is interface  # Returns self for chaining
    assert interface.agent is mock_agent

  def test_add_hook(self, interface):
    hook = MagicMock()
    result = interface.add_hook(hook)
    assert result is interface
    assert hook in interface._hooks

  @pytest.mark.asyncio
  async def test_stop_idempotent(self, interface):
    """Calling stop when not running should not error."""
    await interface._stop_receiver()  # Should not raise


# ============================================================================
# Phase 2: Config — slash commands
# ============================================================================


class TestSlackConfigSlashCommands:
  """Tests for slash command config fields."""

  def test_slash_commands_default_none(self):
    config = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
    assert config.slash_commands is None
    assert config.route_commands_to_agent is True

  def test_slash_commands_configured(self):
    config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      slash_commands={"/ask": "Ask the agent a question", "/help": "Show help"},
    )
    assert config.slash_commands == {"/ask": "Ask the agent a question", "/help": "Show help"}

  def test_route_commands_disabled(self):
    config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      route_commands_to_agent=False,
    )
    assert config.route_commands_to_agent is False


# ============================================================================
# Phase 2: Callback registration
# ============================================================================


class TestCallbackRegistration:
  """Tests for on_command, on_action, on_view registration."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      return SlackInterface(bot_token="xoxb-test", app_token="xapp-test")

  def test_on_command_returns_self(self, interface):
    async def handler(cmd):
      pass

    result = interface.on_command("/ask", handler)
    assert result is interface

  def test_on_command_stores_callback(self, interface):
    async def handler(cmd):
      pass

    interface.on_command("/ask", handler)
    assert "/ask" in interface._command_callbacks
    assert interface._command_callbacks["/ask"] is handler

  def test_on_command_normalizes_slash(self, interface):
    async def handler(cmd):
      pass

    interface.on_command("ask", handler)
    assert "/ask" in interface._command_callbacks

  def test_on_action_returns_self(self, interface):
    async def handler(action, body):
      pass

    result = interface.on_action("btn_click", handler)
    assert result is interface

  def test_on_action_stores_callback(self, interface):
    async def handler(action, body):
      pass

    interface.on_action("btn_click", handler)
    assert "btn_click" in interface._action_callbacks

  def test_on_view_returns_self(self, interface):
    async def handler(view, body):
      pass

    result = interface.on_view("feedback_form", handler)
    assert result is interface

  def test_on_view_stores_callback(self, interface):
    async def handler(view, body):
      pass

    interface.on_view("feedback_form", handler)
    assert "feedback_form" in interface._view_callbacks

  def test_chaining(self, interface):
    async def h1(cmd):
      pass

    async def h2(action, body):
      pass

    async def h3(view, body):
      pass

    result = interface.on_command("/ask", h1).on_action("btn", h2).on_view("form", h3)
    assert result is interface
    assert len(interface._command_callbacks) == 1
    assert len(interface._action_callbacks) == 1
    assert len(interface._view_callbacks) == 1

  def test_constructor_with_slash_commands(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      interface = SlackInterface(
        bot_token="xoxb-test",
        app_token="xapp-test",
        slash_commands={"/ask": "Ask the agent"},
        route_commands_to_agent=False,
      )
      assert interface._slack_config.slash_commands == {"/ask": "Ask the agent"}
      assert interface._slack_config.route_commands_to_agent is False


# ============================================================================
# Phase 2: Command routing
# ============================================================================


class TestCommandRouting:
  """Tests for slash command dispatch."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(
        bot_token="xoxb-test",
        app_token="xapp-test",
        slash_commands={"/ask": "Ask the agent"},
      )
      iface._bot_user_id = "U_BOT"
      iface._client = MagicMock()
      iface.handle_platform_message = AsyncMock()  # type: ignore[method-assign]
      return iface

  @pytest.mark.asyncio
  async def test_custom_callback_priority(self, interface):
    """Custom callbacks take priority over agent routing."""
    received = {}

    async def handler(command):
      received.update(command)

    interface.on_command("/ask", handler)

    command = {"command": "/ask", "text": "what is AI?", "user_id": "U123", "channel_id": "C789"}
    await interface._on_command(command)

    assert received["text"] == "what is AI?"
    interface.handle_platform_message.assert_not_called()

  @pytest.mark.asyncio
  async def test_agent_routing(self, interface):
    """Commands without custom callbacks route to agent pipeline."""
    command = {
      "command": "/ask",
      "text": "what is AI?",
      "user_id": "U123",
      "channel_id": "C789",
      "trigger_id": "trigger.123",
    }
    await interface._on_command(command)

    interface.handle_platform_message.assert_called_once()
    synthetic = interface.handle_platform_message.call_args[0][0]
    assert synthetic["user"] == "U123"
    assert synthetic["channel"] == "C789"
    assert synthetic["text"] == "what is AI?"
    assert synthetic["ts"] == "trigger.123"

  @pytest.mark.asyncio
  async def test_routing_disabled_no_callback(self, interface):
    """When routing is disabled and no callback, logs a warning."""
    interface._slack_config = SlackConfig(
      bot_token="xoxb-test",
      app_token="xapp-test",
      route_commands_to_agent=False,
    )
    command = {"command": "/ask", "text": "hello", "user_id": "U123", "channel_id": "C789"}
    await interface._on_command(command)
    interface.handle_platform_message.assert_not_called()

  @pytest.mark.asyncio
  async def test_callback_error_handled(self, interface):
    """Errors in command callbacks are logged, not propagated."""

    async def bad_handler(command):
      raise RuntimeError("boom")

    interface.on_command("/ask", bad_handler)

    command = {"command": "/ask", "text": "hello", "user_id": "U123", "channel_id": "C789"}
    # Should not raise
    await interface._on_command(command)


# ============================================================================
# Phase 2: Action dispatch
# ============================================================================


class TestActionDispatch:
  """Tests for Block Kit action callback dispatch."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      return SlackInterface(bot_token="xoxb-test", app_token="xapp-test")

  @pytest.mark.asyncio
  async def test_registered_action_called(self, interface):
    received_action = {}
    received_body = {}

    async def handler(action, body):
      received_action.update(action)
      received_body.update(body)

    interface.on_action("approve_btn", handler)

    action = {"action_id": "approve_btn", "value": "yes"}
    body = {"user": {"id": "U123"}, "channel": {"id": "C789"}}
    await interface._on_action(body, action)

    assert received_action["value"] == "yes"
    assert received_body["user"]["id"] == "U123"

  @pytest.mark.asyncio
  async def test_unhandled_action_no_error(self, interface):
    """Unhandled actions log debug, don't raise."""
    action = {"action_id": "unknown_btn", "value": "x"}
    body = {"user": {"id": "U123"}}
    # Should not raise
    await interface._on_action(body, action)

  @pytest.mark.asyncio
  async def test_action_callback_error_handled(self, interface):
    async def bad_handler(action, body):
      raise ValueError("action error")

    interface.on_action("bad_btn", bad_handler)

    action = {"action_id": "bad_btn"}
    body = {"user": {"id": "U123"}}
    # Should not raise
    await interface._on_action(body, action)


# ============================================================================
# Phase 2: View submission dispatch
# ============================================================================


class TestViewSubmissionDispatch:
  """Tests for modal view submission callback dispatch."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      return SlackInterface(bot_token="xoxb-test", app_token="xapp-test")

  @pytest.mark.asyncio
  async def test_registered_view_called(self, interface):
    received_view = {}
    received_body = {}

    async def handler(view, body):
      received_view.update(view)
      received_body.update(body)

    interface.on_view("feedback_form", handler)

    view = {"callback_id": "feedback_form", "state": {"values": {"input": {"val": "hello"}}}}
    body = {"user": {"id": "U123"}}
    await interface._on_view_submission(body, view)

    assert received_view["callback_id"] == "feedback_form"
    assert received_body["user"]["id"] == "U123"

  @pytest.mark.asyncio
  async def test_unhandled_view_no_error(self, interface):
    view = {"callback_id": "unknown_form"}
    body = {"user": {"id": "U123"}}
    await interface._on_view_submission(body, view)

  @pytest.mark.asyncio
  async def test_view_callback_error_handled(self, interface):
    async def bad_handler(view, body):
      raise RuntimeError("view error")

    interface.on_view("bad_form", bad_handler)

    view = {"callback_id": "bad_form"}
    body = {"user": {"id": "U123"}}
    await interface._on_view_submission(body, view)


# ============================================================================
# Phase 2: Message editing, ephemeral, blocks
# ============================================================================


class TestMessageAPIs:
  """Tests for update_message, send_ephemeral, send_blocks."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      iface._client = MagicMock()
      iface._client.chat_update = AsyncMock(return_value={"ok": True, "ts": "1.0"})
      iface._client.chat_postEphemeral = AsyncMock(return_value={"ok": True})
      iface._client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1.0"})
      return iface

  @pytest.mark.asyncio
  async def test_update_message(self, interface):
    await interface.update_message("C789", "1.0", text="updated text")
    interface._client.chat_update.assert_called_once()
    kwargs = interface._client.chat_update.call_args.kwargs
    assert kwargs["channel"] == "C789"
    assert kwargs["ts"] == "1.0"
    assert kwargs["text"] == "updated text"

  @pytest.mark.asyncio
  async def test_update_message_with_blocks(self, interface):
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "hello"}}]
    await interface.update_message("C789", "1.0", blocks=blocks)
    kwargs = interface._client.chat_update.call_args.kwargs
    assert kwargs["blocks"] == blocks

  @pytest.mark.asyncio
  async def test_update_message_error(self, interface):
    interface._client.chat_update = AsyncMock(side_effect=RuntimeError("fail"))
    result = await interface.update_message("C789", "1.0", text="x")
    assert result is None

  @pytest.mark.asyncio
  async def test_send_ephemeral(self, interface):
    await interface.send_ephemeral("C789", "U123", "Only you can see this")
    interface._client.chat_postEphemeral.assert_called_once()
    kwargs = interface._client.chat_postEphemeral.call_args.kwargs
    assert kwargs["channel"] == "C789"
    assert kwargs["user"] == "U123"
    assert kwargs["text"] == "Only you can see this"

  @pytest.mark.asyncio
  async def test_send_ephemeral_with_blocks_and_thread(self, interface):
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "secret"}}]
    await interface.send_ephemeral("C789", "U123", "fallback", blocks=blocks, thread_ts="t.1")
    kwargs = interface._client.chat_postEphemeral.call_args.kwargs
    assert kwargs["blocks"] == blocks
    assert kwargs["thread_ts"] == "t.1"

  @pytest.mark.asyncio
  async def test_send_ephemeral_error(self, interface):
    interface._client.chat_postEphemeral = AsyncMock(side_effect=RuntimeError("fail"))
    result = await interface.send_ephemeral("C789", "U123", "x")
    assert result is None

  @pytest.mark.asyncio
  async def test_send_blocks(self, interface):
    blocks = [{"type": "divider"}]
    await interface.send_blocks("C789", blocks, text="fallback")
    interface._client.chat_postMessage.assert_called_once()
    kwargs = interface._client.chat_postMessage.call_args.kwargs
    assert kwargs["blocks"] == blocks
    assert kwargs["text"] == "fallback"

  @pytest.mark.asyncio
  async def test_send_blocks_with_thread(self, interface):
    blocks = [{"type": "divider"}]
    await interface.send_blocks("C789", blocks, thread_ts="t.1")
    kwargs = interface._client.chat_postMessage.call_args.kwargs
    assert kwargs["thread_ts"] == "t.1"


# ============================================================================
# Phase 2: Modal operations
# ============================================================================


class TestModalOperations:
  """Tests for open_modal, update_modal, push_modal."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      iface._client = MagicMock()
      iface._client.views_open = AsyncMock(return_value={"ok": True, "view": {"id": "V123"}})
      iface._client.views_update = AsyncMock(return_value={"ok": True, "view": {"id": "V123"}})
      iface._client.views_push = AsyncMock(return_value={"ok": True, "view": {"id": "V456"}})
      return iface

  @pytest.mark.asyncio
  async def test_open_modal(self, interface):
    view = {"type": "modal", "title": {"type": "plain_text", "text": "Test"}, "blocks": []}
    await interface.open_modal("trigger.1", view)
    interface._client.views_open.assert_called_once()
    kwargs = interface._client.views_open.call_args.kwargs
    assert kwargs["trigger_id"] == "trigger.1"
    assert kwargs["view"] == view

  @pytest.mark.asyncio
  async def test_open_modal_error(self, interface):
    interface._client.views_open = AsyncMock(side_effect=RuntimeError("fail"))
    result = await interface.open_modal("trigger.1", {})
    assert result is None

  @pytest.mark.asyncio
  async def test_update_modal(self, interface):
    view = {"type": "modal", "title": {"type": "plain_text", "text": "Updated"}, "blocks": []}
    await interface.update_modal("V123", view)
    interface._client.views_update.assert_called_once()
    kwargs = interface._client.views_update.call_args.kwargs
    assert kwargs["view_id"] == "V123"
    assert kwargs["view"] == view

  @pytest.mark.asyncio
  async def test_update_modal_error(self, interface):
    interface._client.views_update = AsyncMock(side_effect=RuntimeError("fail"))
    result = await interface.update_modal("V123", {})
    assert result is None

  @pytest.mark.asyncio
  async def test_push_modal(self, interface):
    view = {"type": "modal", "title": {"type": "plain_text", "text": "Pushed"}, "blocks": []}
    await interface.push_modal("trigger.2", view)
    interface._client.views_push.assert_called_once()
    kwargs = interface._client.views_push.call_args.kwargs
    assert kwargs["trigger_id"] == "trigger.2"
    assert kwargs["view"] == view

  @pytest.mark.asyncio
  async def test_push_modal_error(self, interface):
    interface._client.views_push = AsyncMock(side_effect=RuntimeError("fail"))
    result = await interface.push_modal("trigger.2", {})
    assert result is None


# ============================================================================
# Phase 2: Block Kit builder tests
# ============================================================================


class TestBlockKitBuilders:
  """Tests for Block Kit builder functions in formatter.py."""

  def test_plain_text(self):
    from definable.agent.interface.slack.formatter import plain_text

    result = plain_text("Hello")
    assert result == {"type": "plain_text", "text": "Hello"}

  def test_mrkdwn_text(self):
    from definable.agent.interface.slack.formatter import mrkdwn_text

    result = mrkdwn_text("*bold*")
    assert result == {"type": "mrkdwn", "text": "*bold*"}

  def test_divider_block(self):
    from definable.agent.interface.slack.formatter import divider_block

    assert divider_block() == {"type": "divider"}

  def test_header_block(self):
    from definable.agent.interface.slack.formatter import header_block

    result = header_block("My Header")
    assert result["type"] == "header"
    assert result["text"]["type"] == "plain_text"
    assert result["text"]["text"] == "My Header"

  def test_section_block_text_only(self):
    from definable.agent.interface.slack.formatter import section_block

    result = section_block("Some text")
    assert result["type"] == "section"
    assert result["text"]["type"] == "mrkdwn"
    assert result["text"]["text"] == "Some text"
    assert "accessory" not in result

  def test_section_block_with_accessory(self):
    from definable.agent.interface.slack.formatter import button_element, section_block

    btn = button_element("Click me", "btn_1")
    result = section_block("Choose:", accessory=btn)
    assert result["accessory"]["type"] == "button"

  def test_actions_block(self):
    from definable.agent.interface.slack.formatter import actions_block, button_element

    btn1 = button_element("Yes", "yes_btn", value="yes", style="primary")
    btn2 = button_element("No", "no_btn", value="no", style="danger")
    result = actions_block([btn1, btn2])
    assert result["type"] == "actions"
    assert len(result["elements"]) == 2

  def test_actions_block_with_block_id(self):
    from definable.agent.interface.slack.formatter import actions_block

    result = actions_block([], block_id="my_actions")
    assert result["block_id"] == "my_actions"

  def test_context_block(self):
    from definable.agent.interface.slack.formatter import context_block, mrkdwn_text

    result = context_block([mrkdwn_text("_info_")])
    assert result["type"] == "context"
    assert len(result["elements"]) == 1

  def test_image_block(self):
    from definable.agent.interface.slack.formatter import image_block

    result = image_block("https://example.com/img.png", "A cat")
    assert result["type"] == "image"
    assert result["image_url"] == "https://example.com/img.png"
    assert result["alt_text"] == "A cat"
    assert "title" not in result

  def test_image_block_with_title(self):
    from definable.agent.interface.slack.formatter import image_block

    result = image_block("https://example.com/img.png", "A cat", title="Cat Photo")
    assert result["title"]["text"] == "Cat Photo"

  def test_input_block(self):
    from definable.agent.interface.slack.formatter import input_block, plain_text_input

    elem = plain_text_input("name_input", placeholder="Enter your name")
    result = input_block("Your Name", elem, block_id="name_block")
    assert result["type"] == "input"
    assert result["label"]["text"] == "Your Name"
    assert result["element"]["action_id"] == "name_input"
    assert result["block_id"] == "name_block"

  def test_input_block_optional(self):
    from definable.agent.interface.slack.formatter import input_block, plain_text_input

    elem = plain_text_input("opt_input")
    result = input_block("Optional Field", elem, optional=True)
    assert result["optional"] is True

  def test_button_element(self):
    from definable.agent.interface.slack.formatter import button_element

    result = button_element("Click", "btn_1", value="val", style="primary")
    assert result["type"] == "button"
    assert result["text"]["text"] == "Click"
    assert result["action_id"] == "btn_1"
    assert result["value"] == "val"
    assert result["style"] == "primary"

  def test_button_element_minimal(self):
    from definable.agent.interface.slack.formatter import button_element

    result = button_element("OK", "ok_btn")
    assert result["type"] == "button"
    assert "value" not in result
    assert "style" not in result

  def test_static_select_element(self):
    from definable.agent.interface.slack.formatter import option_object, static_select_element

    opts = [option_object("Red", "red"), option_object("Blue", "blue")]
    result = static_select_element("Pick a color", "color_select", opts)
    assert result["type"] == "static_select"
    assert result["placeholder"]["text"] == "Pick a color"
    assert result["action_id"] == "color_select"
    assert len(result["options"]) == 2
    assert result["options"][0]["value"] == "red"

  def test_plain_text_input(self):
    from definable.agent.interface.slack.formatter import plain_text_input

    result = plain_text_input("msg_input", placeholder="Type here", multiline=True)
    assert result["type"] == "plain_text_input"
    assert result["action_id"] == "msg_input"
    assert result["placeholder"]["text"] == "Type here"
    assert result["multiline"] is True

  def test_plain_text_input_with_initial_value(self):
    from definable.agent.interface.slack.formatter import plain_text_input

    result = plain_text_input("msg_input", initial_value="pre-filled")
    assert result["initial_value"] == "pre-filled"

  def test_option_object(self):
    from definable.agent.interface.slack.formatter import option_object

    result = option_object("Yes", "yes")
    assert result["text"]["type"] == "plain_text"
    assert result["text"]["text"] == "Yes"
    assert result["value"] == "yes"

  def test_modal_view_minimal(self):
    from definable.agent.interface.slack.formatter import modal_view

    result = modal_view("My Modal", [])
    assert result["type"] == "modal"
    assert result["title"]["text"] == "My Modal"
    assert result["blocks"] == []
    assert "callback_id" not in result
    assert "submit" not in result
    assert "close" not in result

  def test_modal_view_full(self):
    from definable.agent.interface.slack.formatter import divider_block, modal_view, section_block

    blocks = [section_block("Question?"), divider_block()]
    result = modal_view("Feedback", blocks, callback_id="fb_form", submit="Send", close="Cancel")
    assert result["callback_id"] == "fb_form"
    assert result["submit"]["text"] == "Send"
    assert result["close"]["text"] == "Cancel"
    assert len(result["blocks"]) == 2

  def test_composability(self):
    """Test that builders compose naturally into a full modal."""
    from definable.agent.interface.slack.formatter import (
      actions_block,
      button_element,
      divider_block,
      header_block,
      input_block,
      modal_view,
      option_object,
      plain_text_input,
      section_block,
      static_select_element,
    )

    view = modal_view(
      "Survey",
      [
        header_block("Customer Feedback"),
        section_block("Please rate your experience:"),
        actions_block([
          button_element("Good", "rate_good", value="good", style="primary"),
          button_element("Bad", "rate_bad", value="bad", style="danger"),
        ]),
        divider_block(),
        input_block(
          "Additional Comments",
          plain_text_input("comments", placeholder="Type here...", multiline=True),
          block_id="comments_block",
          optional=True,
        ),
        input_block(
          "Category",
          static_select_element(
            "Select...",
            "category",
            [
              option_object("Product", "product"),
              option_object("Service", "service"),
            ],
          ),
        ),
      ],
      callback_id="survey_form",
      submit="Submit",
      close="Cancel",
    )

    assert view["type"] == "modal"
    assert view["callback_id"] == "survey_form"
    assert len(view["blocks"]) == 6
    assert view["blocks"][0]["type"] == "header"
    assert view["blocks"][1]["type"] == "section"
    assert view["blocks"][2]["type"] == "actions"
    assert view["blocks"][3]["type"] == "divider"
    assert view["blocks"][4]["type"] == "input"
    assert view["blocks"][5]["type"] == "input"


# ============================================================================
# Phase 2: Listener registration
# ============================================================================


class TestListenerRegistration:
  """Tests for _register_listeners with commands and interactions."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(
        bot_token="xoxb-test",
        app_token="xapp-test",
        slash_commands={"/ask": "Ask a question"},
      )
      # Create a mock Bolt app
      mock_app = MagicMock()
      mock_app.event = MagicMock(return_value=lambda f: f)
      mock_app.command = MagicMock(return_value=lambda f: f)
      mock_app.action = MagicMock(return_value=lambda f: f)
      mock_app.view = MagicMock(return_value=lambda f: f)
      mock_app.shortcut = MagicMock(return_value=lambda f: f)
      iface._bolt_app = mock_app
      return iface

  def test_event_listeners_registered(self, interface):
    interface._register_listeners()
    calls = [c[0][0] for c in interface._bolt_app.event.call_args_list]
    assert "message" in calls
    assert "app_mention" in calls
    assert "reaction_added" in calls
    assert "reaction_removed" in calls
    assert "app_home_opened" in calls

  def test_command_listeners_registered(self, interface):
    interface._register_listeners()
    interface._bolt_app.command.assert_called_with("/ask")

  def test_action_catch_all_registered(self, interface):
    interface._register_listeners()
    interface._bolt_app.action.assert_called_once()

  def test_view_catch_all_registered(self, interface):
    interface._register_listeners()
    interface._bolt_app.view.assert_called_once()

  def test_shortcut_catch_all_registered(self, interface):
    interface._register_listeners()
    interface._bolt_app.shortcut.assert_called_once()

  def test_no_commands_when_not_configured(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      mock_app = MagicMock()
      mock_app.event = MagicMock(return_value=lambda f: f)
      mock_app.command = MagicMock(return_value=lambda f: f)
      mock_app.action = MagicMock(return_value=lambda f: f)
      mock_app.view = MagicMock(return_value=lambda f: f)
      mock_app.shortcut = MagicMock(return_value=lambda f: f)
      iface._bolt_app = mock_app

      iface._register_listeners()
      mock_app.command.assert_not_called()

  def test_generic_event_listeners_registered(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")

      async def handler(event):
        pass

      iface.on_event("member_joined_channel", handler)

      mock_app = MagicMock()
      mock_app.event = MagicMock(return_value=lambda f: f)
      mock_app.command = MagicMock(return_value=lambda f: f)
      mock_app.action = MagicMock(return_value=lambda f: f)
      mock_app.view = MagicMock(return_value=lambda f: f)
      mock_app.shortcut = MagicMock(return_value=lambda f: f)
      iface._bolt_app = mock_app

      iface._register_listeners()
      event_calls = [c[0][0] for c in mock_app.event.call_args_list]
      assert "member_joined_channel" in event_calls


# ============================================================================
# Phase 3: Shortcut dispatch
# ============================================================================


class TestShortcutDispatch:
  """Tests for message and global shortcut callback dispatch."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      return SlackInterface(bot_token="xoxb-test", app_token="xapp-test")

  def test_on_shortcut_returns_self(self, interface):
    async def handler(shortcut, body):
      pass

    result = interface.on_shortcut("summarize", handler)
    assert result is interface

  def test_on_shortcut_stores_callback(self, interface):
    async def handler(shortcut, body):
      pass

    interface.on_shortcut("summarize", handler)
    assert "summarize" in interface._shortcut_callbacks

  @pytest.mark.asyncio
  async def test_shortcut_callback_called(self, interface):
    received = {}

    async def handler(shortcut, body):
      received["shortcut"] = shortcut
      received["body"] = body

    interface.on_shortcut("summarize", handler)

    shortcut = {"callback_id": "summarize", "trigger_id": "t.1", "message": {"text": "hello"}}
    body = {"user": {"id": "U123"}}
    await interface._on_shortcut(shortcut, body)

    assert received["shortcut"]["trigger_id"] == "t.1"
    assert received["body"]["user"]["id"] == "U123"

  @pytest.mark.asyncio
  async def test_unhandled_shortcut_no_error(self, interface):
    shortcut = {"callback_id": "unknown"}
    body: dict[str, Any] = {}
    await interface._on_shortcut(shortcut, body)

  @pytest.mark.asyncio
  async def test_shortcut_error_handled(self, interface):
    async def bad_handler(shortcut, body):
      raise RuntimeError("shortcut error")

    interface.on_shortcut("bad", bad_handler)

    await interface._on_shortcut({"callback_id": "bad"}, {})


# ============================================================================
# Phase 3: Reaction events
# ============================================================================


class TestReactionEvents:
  """Tests for reaction_added and reaction_removed event dispatch."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      iface._bot_user_id = "U_BOT"
      return iface

  @pytest.mark.asyncio
  async def test_reaction_added_specific_emoji(self, interface):
    received = {}

    async def handler(event):
      received.update(event)

    interface.on_reaction_added("thumbsup", handler)

    event = {
      "user": "U123",
      "reaction": "thumbsup",
      "item": {"type": "message", "channel": "C789", "ts": "1.0"},
    }
    await interface._on_reaction_added(event)
    assert received["reaction"] == "thumbsup"

  @pytest.mark.asyncio
  async def test_reaction_added_catch_all(self, interface):
    received = {}

    async def handler(event):
      received.update(event)

    interface.on_reaction_added("*", handler)

    event = {"user": "U123", "reaction": "heart", "item": {"type": "message", "channel": "C789", "ts": "1.0"}}
    await interface._on_reaction_added(event)
    assert received["reaction"] == "heart"

  @pytest.mark.asyncio
  async def test_reaction_added_specific_over_catchall(self, interface):
    """Specific emoji callback takes priority over catch-all."""
    specific_called = []
    catchall_called = []

    async def specific(event):
      specific_called.append(event)

    async def catchall(event):
      catchall_called.append(event)

    interface.on_reaction_added("thumbsup", specific)
    interface.on_reaction_added("*", catchall)

    event = {"user": "U123", "reaction": "thumbsup", "item": {"channel": "C", "ts": "1"}}
    await interface._on_reaction_added(event)
    assert len(specific_called) == 1
    assert len(catchall_called) == 0

  @pytest.mark.asyncio
  async def test_bot_reaction_filtered(self, interface):
    received = []

    async def handler(event):
      received.append(event)

    interface.on_reaction_added("*", handler)

    event = {"user": "U_BOT", "reaction": "hourglass_flowing_sand", "item": {"channel": "C", "ts": "1"}}
    await interface._on_reaction_added(event)
    assert len(received) == 0

  @pytest.mark.asyncio
  async def test_reaction_removed(self, interface):
    received = {}

    async def handler(event):
      received.update(event)

    interface.on_reaction_removed("thumbsup", handler)

    event = {"user": "U123", "reaction": "thumbsup", "item": {"channel": "C", "ts": "1"}}
    await interface._on_reaction_removed(event)
    assert received["reaction"] == "thumbsup"

  @pytest.mark.asyncio
  async def test_reaction_removed_bot_filtered(self, interface):
    received = []

    async def handler(event):
      received.append(event)

    interface.on_reaction_removed("*", handler)

    event = {"user": "U_BOT", "reaction": "x", "item": {"channel": "C", "ts": "1"}}
    await interface._on_reaction_removed(event)
    assert len(received) == 0

  @pytest.mark.asyncio
  async def test_reaction_error_handled(self, interface):
    async def bad_handler(event):
      raise ValueError("reaction error")

    interface.on_reaction_added("thumbsup", bad_handler)

    event = {"user": "U123", "reaction": "thumbsup", "item": {"channel": "C", "ts": "1"}}
    await interface._on_reaction_added(event)

  @pytest.mark.asyncio
  async def test_no_callback_no_error(self, interface):
    event = {"user": "U123", "reaction": "random", "item": {"channel": "C", "ts": "1"}}
    await interface._on_reaction_added(event)
    await interface._on_reaction_removed(event)


# ============================================================================
# Phase 3: App Home tab
# ============================================================================


class TestAppHome:
  """Tests for App Home tab support."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      iface._client = MagicMock()
      iface._client.views_publish = AsyncMock(return_value={"ok": True})
      return iface

  @pytest.mark.asyncio
  async def test_home_opened_callback(self, interface):
    received = {}

    async def handler(event):
      received.update(event)

    interface.on_home_opened(handler)
    assert interface._home_opened_callback is handler

    event = {"type": "app_home_opened", "user": "U123", "tab": "home"}
    await interface._on_home_opened(event)
    assert received["user"] == "U123"

  @pytest.mark.asyncio
  async def test_home_opened_ignores_messages_tab(self, interface):
    received = []

    async def handler(event):
      received.append(event)

    interface.on_home_opened(handler)

    event = {"type": "app_home_opened", "user": "U123", "tab": "messages"}
    await interface._on_home_opened(event)
    assert len(received) == 0

  @pytest.mark.asyncio
  async def test_home_opened_no_callback(self, interface):
    event = {"type": "app_home_opened", "user": "U123", "tab": "home"}
    await interface._on_home_opened(event)  # Should not raise

  @pytest.mark.asyncio
  async def test_home_opened_error_handled(self, interface):
    async def bad_handler(event):
      raise RuntimeError("home error")

    interface.on_home_opened(bad_handler)

    event = {"type": "app_home_opened", "user": "U123", "tab": "home"}
    await interface._on_home_opened(event)

  @pytest.mark.asyncio
  async def test_publish_home(self, interface):
    view = {"type": "home", "blocks": []}
    await interface.publish_home("U123", view)
    interface._client.views_publish.assert_called_once()
    kwargs = interface._client.views_publish.call_args.kwargs
    assert kwargs["user_id"] == "U123"
    assert kwargs["view"] == view

  @pytest.mark.asyncio
  async def test_publish_home_error(self, interface):
    interface._client.views_publish = AsyncMock(side_effect=RuntimeError("fail"))
    result = await interface.publish_home("U123", {})
    assert result is None

  def test_on_home_opened_returns_self(self, interface):
    async def handler(event):
      pass

    result = interface.on_home_opened(handler)
    assert result is interface


# ============================================================================
# Phase 3: Scheduled messages
# ============================================================================


class TestScheduledMessages:
  """Tests for schedule_message and delete_scheduled_message."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      iface._client = MagicMock()
      iface._client.chat_scheduleMessage = AsyncMock(return_value={"ok": True, "scheduled_message_id": "Q123"})
      iface._client.chat_deleteScheduledMessage = AsyncMock(return_value={"ok": True})
      return iface

  @pytest.mark.asyncio
  async def test_schedule_message(self, interface):
    await interface.schedule_message("C789", "Hello future!", 1700000000)
    interface._client.chat_scheduleMessage.assert_called_once()
    kwargs = interface._client.chat_scheduleMessage.call_args.kwargs
    assert kwargs["channel"] == "C789"
    assert kwargs["text"] == "Hello future!"
    assert kwargs["post_at"] == 1700000000

  @pytest.mark.asyncio
  async def test_schedule_message_with_thread_and_blocks(self, interface):
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "scheduled"}}]
    await interface.schedule_message("C789", "fallback", 1700000000, thread_ts="t.1", blocks=blocks)
    kwargs = interface._client.chat_scheduleMessage.call_args.kwargs
    assert kwargs["thread_ts"] == "t.1"
    assert kwargs["blocks"] == blocks

  @pytest.mark.asyncio
  async def test_schedule_message_error(self, interface):
    interface._client.chat_scheduleMessage = AsyncMock(side_effect=RuntimeError("fail"))
    result = await interface.schedule_message("C789", "x", 1700000000)
    assert result is None

  @pytest.mark.asyncio
  async def test_delete_scheduled_message(self, interface):
    await interface.delete_scheduled_message("C789", "Q123")
    interface._client.chat_deleteScheduledMessage.assert_called_once()
    kwargs = interface._client.chat_deleteScheduledMessage.call_args.kwargs
    assert kwargs["channel"] == "C789"
    assert kwargs["scheduled_message_id"] == "Q123"

  @pytest.mark.asyncio
  async def test_delete_scheduled_message_error(self, interface):
    interface._client.chat_deleteScheduledMessage = AsyncMock(side_effect=RuntimeError("fail"))
    result = await interface.delete_scheduled_message("C789", "Q123")
    assert result is None


# ============================================================================
# Phase 3: Message deletion, permalink, topic, pins
# ============================================================================


class TestExtendedAPIs:
  """Tests for delete_message, get_permalink, set_topic, pin/unpin."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      iface = SlackInterface(bot_token="xoxb-test", app_token="xapp-test")
      iface._client = MagicMock()
      iface._client.chat_delete = AsyncMock(return_value={"ok": True})
      iface._client.chat_getPermalink = AsyncMock(return_value={"ok": True, "permalink": "https://slack.com/archives/C789/p123"})
      iface._client.conversations_setTopic = AsyncMock(return_value={"ok": True})
      iface._client.pins_add = AsyncMock(return_value={"ok": True})
      iface._client.pins_remove = AsyncMock(return_value={"ok": True})
      return iface

  @pytest.mark.asyncio
  async def test_delete_message(self, interface):
    await interface.delete_message("C789", "1.0")
    interface._client.chat_delete.assert_called_once()
    kwargs = interface._client.chat_delete.call_args.kwargs
    assert kwargs["channel"] == "C789"
    assert kwargs["ts"] == "1.0"

  @pytest.mark.asyncio
  async def test_delete_message_error(self, interface):
    interface._client.chat_delete = AsyncMock(side_effect=RuntimeError("fail"))
    result = await interface.delete_message("C789", "1.0")
    assert result is None

  @pytest.mark.asyncio
  async def test_get_permalink(self, interface):
    result = await interface.get_permalink("C789", "1.0")
    assert result == "https://slack.com/archives/C789/p123"
    kwargs = interface._client.chat_getPermalink.call_args.kwargs
    assert kwargs["channel"] == "C789"
    assert kwargs["message_ts"] == "1.0"

  @pytest.mark.asyncio
  async def test_get_permalink_error(self, interface):
    interface._client.chat_getPermalink = AsyncMock(side_effect=RuntimeError("fail"))
    result = await interface.get_permalink("C789", "1.0")
    assert result is None

  @pytest.mark.asyncio
  async def test_set_topic(self, interface):
    await interface.set_topic("C789", "New topic!")
    interface._client.conversations_setTopic.assert_called_once()
    kwargs = interface._client.conversations_setTopic.call_args.kwargs
    assert kwargs["channel"] == "C789"
    assert kwargs["topic"] == "New topic!"

  @pytest.mark.asyncio
  async def test_set_topic_error(self, interface):
    interface._client.conversations_setTopic = AsyncMock(side_effect=RuntimeError("fail"))
    result = await interface.set_topic("C789", "x")
    assert result is None

  @pytest.mark.asyncio
  async def test_pin_message(self, interface):
    await interface.pin_message("C789", "1.0")
    interface._client.pins_add.assert_called_once_with(channel="C789", timestamp="1.0")

  @pytest.mark.asyncio
  async def test_pin_message_error_suppressed(self, interface):
    interface._client.pins_add = AsyncMock(side_effect=RuntimeError("fail"))
    await interface.pin_message("C789", "1.0")  # Should not raise

  @pytest.mark.asyncio
  async def test_unpin_message(self, interface):
    await interface.unpin_message("C789", "1.0")
    interface._client.pins_remove.assert_called_once_with(channel="C789", timestamp="1.0")

  @pytest.mark.asyncio
  async def test_unpin_message_error_suppressed(self, interface):
    interface._client.pins_remove = AsyncMock(side_effect=RuntimeError("fail"))
    await interface.unpin_message("C789", "1.0")  # Should not raise


# ============================================================================
# Phase 3: Generic event registration
# ============================================================================


class TestGenericEvents:
  """Tests for on_event generic event handler."""

  @pytest.fixture()
  def interface(self):
    with patch("definable.agent.interface.slack.interface._ensure_slack_deps"):
      from definable.agent.interface.slack.interface import SlackInterface

      return SlackInterface(bot_token="xoxb-test", app_token="xapp-test")

  def test_on_event_returns_self(self, interface):
    async def handler(event):
      pass

    result = interface.on_event("channel_created", handler)
    assert result is interface

  def test_on_event_stores_callback(self, interface):
    async def handler(event):
      pass

    interface.on_event("member_joined_channel", handler)
    assert "member_joined_channel" in interface._event_callbacks

  def test_on_event_reserved_raises(self, interface):
    async def handler(event):
      pass

    for event_type in ["message", "app_mention", "reaction_added", "reaction_removed", "app_home_opened"]:
      with pytest.raises(ValueError, match="handled internally"):
        interface.on_event(event_type, handler)

  def test_on_event_registers_bolt_listener_if_started(self, interface):
    """If the app is already started, registers the Bolt listener immediately."""
    mock_app = MagicMock()
    mock_app.event = MagicMock(return_value=lambda f: f)
    interface._bolt_app = mock_app

    async def handler(event):
      pass

    interface.on_event("channel_created", handler)
    mock_app.event.assert_called_with("channel_created")

  def test_on_event_does_not_register_if_not_started(self, interface):
    """If the app is not started, only stores the callback."""

    async def handler(event):
      pass

    interface.on_event("channel_created", handler)
    assert "channel_created" in interface._event_callbacks
    # _bolt_app is None, so no Bolt registration


# ============================================================================
# Phase 3: Home tab view builder
# ============================================================================


class TestHomeTabViewBuilder:
  """Tests for home_tab_view formatter function."""

  def test_home_tab_minimal(self):
    from definable.agent.interface.slack.formatter import home_tab_view

    result = home_tab_view([])
    assert result["type"] == "home"
    assert result["blocks"] == []
    assert "external_id" not in result

  def test_home_tab_with_blocks(self):
    from definable.agent.interface.slack.formatter import home_tab_view, section_block

    blocks = [section_block("Welcome!")]
    result = home_tab_view(blocks)
    assert len(result["blocks"]) == 1
    assert result["blocks"][0]["type"] == "section"

  def test_home_tab_with_external_id(self):
    from definable.agent.interface.slack.formatter import home_tab_view

    result = home_tab_view([], external_id="user_123_home")
    assert result["external_id"] == "user_123_home"

  def test_home_tab_composability(self):
    """Test building a complete home tab with mixed blocks."""
    from definable.agent.interface.slack.formatter import (
      actions_block,
      button_element,
      context_block,
      divider_block,
      header_block,
      home_tab_view,
      mrkdwn_text,
      section_block,
    )

    view = home_tab_view([
      header_block("Dashboard"),
      section_block("Here's what's happening today:"),
      divider_block(),
      section_block("*Tasks*: 5 pending"),
      actions_block([
        button_element("View Tasks", "view_tasks", style="primary"),
        button_element("Settings", "settings"),
      ]),
      context_block([mrkdwn_text("Last updated: just now")]),
    ])

    assert view["type"] == "home"
    assert len(view["blocks"]) == 6
    assert view["blocks"][0]["type"] == "header"
    assert view["blocks"][2]["type"] == "divider"
    assert view["blocks"][4]["type"] == "actions"
