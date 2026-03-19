"""Comprehensive tests for TelegramInterface — all 20 phases."""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.agent.interface.telegram.config import TelegramConfig
from definable.agent.interface.telegram.formatting import markdown_to_telegram_html, split_html
from definable.agent.interface.telegram.interface import (
  TelegramInterface,
  _OutboundRateLimiter,
  _SlidingWindowRateLimiter,
  _TypingCircuitBreaker,
)
from definable.agent.interface.telegram.keyboards import InlineButton, InlineKeyboard
from definable.agent.interface.telegram.stickers import StickerCache


# ===== Phase 1: Markdown → HTML Conversion =====


class TestMarkdownToHtml:
  def test_empty_string(self):
    assert markdown_to_telegram_html("") == ""

  def test_plain_text_escaped(self):
    assert markdown_to_telegram_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"

  def test_bold(self):
    result = markdown_to_telegram_html("**hello**")
    assert "<b>hello</b>" in result

  def test_italic(self):
    result = markdown_to_telegram_html("*hello*")
    assert "<i>hello</i>" in result

  def test_bold_italic_nested(self):
    result = markdown_to_telegram_html("**bold *and italic* text**")
    assert "<b>" in result

  def test_inline_code(self):
    result = markdown_to_telegram_html("use `print(x)`")
    assert "<code>" in result
    assert "print(x)" in result

  def test_code_block(self):
    result = markdown_to_telegram_html("```python\nprint('hi')\n```")
    assert "<pre><code" in result
    assert "language-python" in result

  def test_code_block_no_lang(self):
    result = markdown_to_telegram_html("```\nsome code\n```")
    assert "<pre><code>" in result

  def test_no_formatting_inside_code(self):
    result = markdown_to_telegram_html("`**not bold**`")
    assert "<b>" not in result
    assert "<code>" in result

  def test_link(self):
    result = markdown_to_telegram_html("[click here](https://example.com)")
    assert '<a href="https://example.com">click here</a>' in result

  def test_strikethrough(self):
    result = markdown_to_telegram_html("~~deleted~~")
    assert "<s>deleted</s>" in result

  def test_spoiler(self):
    result = markdown_to_telegram_html("||secret||")
    assert "<tg-spoiler>secret</tg-spoiler>" in result

  def test_blockquote(self):
    result = markdown_to_telegram_html("> quoted text")
    assert "<blockquote>" in result
    assert "quoted text" in result

  def test_blockquote_multiline(self):
    result = markdown_to_telegram_html("> line1\n> line2")
    assert "<blockquote>" in result
    # Should be a single blockquote
    assert result.count("<blockquote>") == 1

  def test_html_escape_in_non_formatted(self):
    result = markdown_to_telegram_html("x < y && z > w")
    assert "&lt;" in result
    assert "&amp;" in result
    assert "&gt;" in result


# ===== Phase 2: Smart HTML Chunking =====


class TestSplitHtml:
  def test_short_text_single_chunk(self):
    assert split_html("hello", 100) == ["hello"]

  def test_no_html_falls_back(self):
    text = "a" * 100
    chunks = split_html(text, 50)
    assert len(chunks) >= 2
    for c in chunks:
      assert len(c) <= 50

  def test_split_preserves_tags(self):
    text = "<b>" + "x" * 100 + "</b>"
    chunks = split_html(text, 60)
    # Each chunk should have balanced tags
    for chunk in chunks:
      assert chunk.count("<b>") == chunk.count("</b>")

  def test_split_at_paragraph(self):
    text = "first paragraph\n\nsecond paragraph"
    chunks = split_html(text, 20)
    assert len(chunks) >= 2

  def test_nested_tags_preserved(self):
    text = "<b><i>" + "y" * 80 + "</i></b>"
    chunks = split_html(text, 50)
    for chunk in chunks:
      assert chunk.count("<b>") == chunk.count("</b>")
      assert chunk.count("<i>") == chunk.count("</i>")


# ===== Phase 3: Message Editing =====


class TestMessageEditing:
  @pytest.mark.asyncio
  async def test_send_message_returns_id(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"message_id": 42})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    msg_id = await iface._send_message("123", "hello")
    assert msg_id == "42"

  @pytest.mark.asyncio
  async def test_edit_message_success(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"ok": True})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    result = await iface._edit_message("123", "42", "updated")
    assert result is True

  @pytest.mark.asyncio
  async def test_edit_message_not_modified(self):
    from definable.agent.interface.errors import InterfaceMessageError

    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(  # type: ignore[method-assign]
      side_effect=InterfaceMessageError("message is not modified", platform="telegram")
    )
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    result = await iface._edit_message("123", "42", "same text")
    assert result is False

  @pytest.mark.asyncio
  async def test_send_message_parse_mode_fallback(self):
    """If HTML parse_mode fails, retry without it."""
    from definable.agent.interface.errors import InterfaceMessageError

    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    call_count = 0

    async def mock_api_call(method, data=None):
      nonlocal call_count
      call_count += 1
      if call_count == 1 and data and data.get("parse_mode"):
        raise InterfaceMessageError("Bad Request: can't parse entities", platform="telegram")
      return {"message_id": 1}

    iface._api_call = mock_api_call  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    msg_id = await iface._send_message("123", "text", parse_mode="HTML")
    assert msg_id == "1"
    assert call_count == 2


# ===== Phase 5: Callback Query Handling =====


class TestCallbackQueries:
  @pytest.mark.asyncio
  async def test_register_and_handle_callback(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    handler_called = False

    async def handler(query):
      nonlocal handler_called
      handler_called = True
      return "Done"

    iface.register_callback("confirm", handler)

    callback_query = {
      "id": "q1",
      "data": "confirm_action",
      "from": {"id": 123},
      "message": {"chat": {"id": 456}, "message_id": 1},
    }
    await iface._handle_callback_query(callback_query)
    assert handler_called

  @pytest.mark.asyncio
  async def test_callback_no_matching_handler_falls_to_agent(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface.handle_platform_message = AsyncMock()  # type: ignore[method-assign]

    callback_query = {
      "id": "q2",
      "data": "unknown",
      "from": {"id": 123},
      "message": {"chat": {"id": 456}, "message_id": 1},
    }
    await iface._handle_callback_query(callback_query)
    iface.handle_platform_message.assert_called_once()


# ===== Phase 6: Inline Keyboards =====


class TestInlineKeyboard:
  def test_button_with_callback_data(self):
    btn = InlineButton("Yes", callback_data="yes")
    assert btn.to_dict() == {"text": "Yes", "callback_data": "yes"}

  def test_button_with_url(self):
    btn = InlineButton("Help", url="https://example.com")
    assert btn.to_dict() == {"text": "Help", "url": "https://example.com"}

  def test_button_no_action_raises(self):
    with pytest.raises(ValueError, match="requires either"):
      InlineButton("Empty")

  def test_button_both_raises(self):
    with pytest.raises(ValueError, match="cannot have both"):
      InlineButton("Both", callback_data="x", url="http://x")

  def test_button_callback_data_too_long(self):
    with pytest.raises(ValueError, match="64-byte"):
      InlineButton("Big", callback_data="x" * 65)

  def test_keyboard_row(self):
    kb = InlineKeyboard()
    kb.row(InlineButton("A", callback_data="a"), InlineButton("B", callback_data="b"))
    result = kb.to_dict()
    assert len(result["inline_keyboard"]) == 1
    assert len(result["inline_keyboard"][0]) == 2

  def test_keyboard_button_creates_row(self):
    kb = InlineKeyboard()
    kb.button("Solo", callback_data="s")
    result = kb.to_dict()
    assert len(result["inline_keyboard"]) == 1
    assert len(result["inline_keyboard"][0]) == 1

  def test_keyboard_chaining(self):
    kb = InlineKeyboard()
    result = kb.button("A", callback_data="a").button("B", callback_data="b")
    assert result is kb


# ===== Phase 7: Group Chat Intelligence =====


class TestGroupChatIntelligence:
  def test_should_respond_always_mode(self):
    iface = TelegramInterface(bot_token="test:token", group_mode="always")
    assert iface._should_respond_in_group({}) is True

  def test_should_respond_disabled_mode(self):
    iface = TelegramInterface(bot_token="test:token", group_mode="disabled")
    assert iface._should_respond_in_group({}) is False

  def test_mention_mode_no_mention(self):
    iface = TelegramInterface(bot_token="test:token", group_mode="mention")
    iface._bot_username = "testbot"
    iface._bot_id = 999
    msg = {"text": "hello everyone", "entities": []}
    assert iface._should_respond_in_group(msg) is False

  def test_mention_mode_with_mention(self):
    iface = TelegramInterface(bot_token="test:token", group_mode="mention")
    iface._bot_username = "testbot"
    iface._bot_id = 999
    msg = {
      "text": "@testbot what time is it?",
      "entities": [{"type": "mention", "offset": 0, "length": 8}],
    }
    assert iface._should_respond_in_group(msg) is True

  def test_mention_mode_reply_to_bot(self):
    iface = TelegramInterface(bot_token="test:token", group_mode="mention")
    iface._bot_username = "testbot"
    iface._bot_id = 999
    msg = {
      "text": "thanks",
      "entities": [],
      "reply_to_message": {"from": {"id": 999}},
    }
    assert iface._should_respond_in_group(msg) is True

  def test_mention_mode_bot_command(self):
    iface = TelegramInterface(bot_token="test:token", group_mode="mention")
    iface._bot_username = "testbot"
    iface._bot_id = 999
    msg = {
      "text": "/help",
      "entities": [{"type": "bot_command", "offset": 0, "length": 5}],
    }
    assert iface._should_respond_in_group(msg) is True

  def test_strip_bot_mention(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._bot_username = "testbot"
    assert iface._strip_bot_mention("@testbot hello") == "hello"
    assert iface._strip_bot_mention("hello @TestBot") == "hello"

  def test_strip_bot_mention_no_username(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._bot_username = None
    assert iface._strip_bot_mention("hello") == "hello"


# ===== Phase 8: Forum/Topic Support =====


class TestForumTopicSupport:
  @pytest.mark.asyncio
  async def test_forum_topic_session_isolation(self):
    iface = TelegramInterface(bot_token="test:token", enable_forum_topics=True)
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "p"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 100, "type": "supergroup", "is_forum": True},
      "from": {"id": 42, "username": "user"},
      "message_id": 1,
      "text": "hello",
      "message_thread_id": 5,
    }

    iface._bot_username = "bot"
    iface._bot_id = 999
    # Force always mode so group filter doesn't block
    iface._tg_config = TelegramConfig(
      bot_token="test:token",
      group_mode="always",
      enable_forum_topics=True,
    )

    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.platform_chat_id == "100:topic:5"
    assert msg.metadata["thread_id"] == 5

  @pytest.mark.asyncio
  async def test_general_topic_no_isolation(self):
    iface = TelegramInterface(bot_token="test:token", enable_forum_topics=True)
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "p"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)
    iface._bot_username = "bot"
    iface._bot_id = 999
    iface._tg_config = TelegramConfig(
      bot_token="test:token",
      group_mode="always",
      enable_forum_topics=True,
    )

    raw = {
      "chat": {"id": 100, "type": "supergroup", "is_forum": True},
      "from": {"id": 42},
      "message_id": 1,
      "text": "hi",
      "message_thread_id": 1,  # General topic
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.platform_chat_id == "100"  # No topic suffix


# ===== Phase 9: Video/VideoNote/Animation =====


class TestVideoSupport:
  @pytest.mark.asyncio
  async def test_video_extraction(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "video/file.mp4"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "video": {"file_id": "vid1", "duration": 10, "width": 1920, "height": 1080},
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.videos is not None
    assert len(msg.videos) == 1
    assert msg.videos[0].duration == 10

  @pytest.mark.asyncio
  async def test_video_note_extraction(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "videonote/file.mp4"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "video_note": {"file_id": "vn1", "duration": 5},
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.videos is not None
    assert len(msg.videos) == 1

  @pytest.mark.asyncio
  async def test_animation_extraction(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "anim/file.gif"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "animation": {"file_id": "anim1", "duration": 3, "width": 320, "height": 240},
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.videos is not None
    assert len(msg.videos) == 1
    assert msg.videos[0].width == 320


# ===== Phase 10: Sticker Support =====


class TestStickerCache:
  def test_describe_with_emoji_and_set(self):
    cache = StickerCache()
    sticker = {"file_unique_id": "s1", "emoji": "😀", "set_name": "HappyPack"}
    desc = cache.describe_sticker(sticker)
    assert desc == "[Sticker: 😀 from 'HappyPack']"

  def test_describe_emoji_only(self):
    cache = StickerCache()
    sticker = {"file_unique_id": "s2", "emoji": "🎉"}
    desc = cache.describe_sticker(sticker)
    assert desc == "[Sticker: 🎉]"

  def test_describe_set_only(self):
    cache = StickerCache()
    sticker = {"file_unique_id": "s3", "set_name": "MySet"}
    desc = cache.describe_sticker(sticker)
    assert desc == "[Sticker from 'MySet']"

  def test_describe_no_info(self):
    cache = StickerCache()
    sticker = {"file_unique_id": "s4"}
    desc = cache.describe_sticker(sticker)
    assert desc == "[Sticker]"

  def test_cache_hit(self):
    cache = StickerCache()
    sticker = {"file_unique_id": "s5", "emoji": "🔥", "set_name": "Fire"}
    cache.describe_sticker(sticker)
    # Second call should hit cache
    desc = cache.describe_sticker(sticker)
    assert desc == "[Sticker: 🔥 from 'Fire']"

  def test_cache_eviction(self):
    cache = StickerCache(max_size=2)
    cache.put("a", "desc_a")
    cache.put("b", "desc_b")
    cache.put("c", "desc_c")  # Should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == "desc_b"
    assert cache.get("c") == "desc_c"

  def test_cache_len(self):
    cache = StickerCache(max_size=5)
    cache.put("x", "y")
    assert len(cache) == 1


# ===== Phase 11: Forward Context =====


class TestForwardContext:
  @pytest.mark.asyncio
  async def test_forward_from_user(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "text": "forwarded text",
      "forward_from": {"first_name": "Alice", "username": "alice"},
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text is not None
    assert "[Forwarded from @alice]" in msg.text
    assert msg.metadata["is_forward"] is True

  @pytest.mark.asyncio
  async def test_forward_from_chat(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "text": "channel post",
      "forward_from_chat": {"title": "News Channel"},
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text is not None
    assert "[Forwarded from News Channel]" in msg.text


# ===== Phase 12: Edited Message Processing =====


class TestEditedMessageProcessing:
  @pytest.mark.asyncio
  async def test_edited_message_tagged(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "text": "edited text",
      "_is_edit": True,
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.metadata["is_edit"] is True

  @pytest.mark.asyncio
  async def test_process_update_edited_message(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    dispatched = []

    async def mock_dispatch(msg, is_edit=False):
      dispatched.append(is_edit)

    iface._dispatch_message = mock_dispatch  # type: ignore[method-assign]

    update = {
      "update_id": 1,
      "edited_message": {
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 2},
        "message_id": 3,
        "text": "edited",
      },
    }
    await iface._process_update(update)
    assert dispatched == [True]


# ===== Phase 13: Typing Indicator Circuit Breaker =====


class TestTypingCircuitBreaker:
  def test_initially_allowed(self):
    cb = _TypingCircuitBreaker()
    assert cb.should_send("chat1") is True

  def test_success_resets(self):
    cb = _TypingCircuitBreaker(max_failures=2)
    cb.record_failure("chat1")
    cb.record_success("chat1")
    assert cb.should_send("chat1") is True

  def test_suspension_after_max_failures(self):
    cb = _TypingCircuitBreaker(max_failures=3, base_backoff=1.0)
    for _ in range(3):
      cb.record_failure("chat1")
    assert cb.should_send("chat1") is False

  def test_suspension_expires(self):
    cb = _TypingCircuitBreaker(max_failures=1, base_backoff=0.0)
    cb.record_failure("chat1")
    # With 0 backoff, suspension expires immediately
    assert cb.should_send("chat1") is True


# ===== Phase 14: Rate Limiting =====


class TestSlidingWindowRateLimiter:
  def test_allows_within_limit(self):
    rl = _SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    assert rl.is_allowed("user1") is True
    assert rl.is_allowed("user1") is True
    assert rl.is_allowed("user1") is True

  def test_blocks_over_limit(self):
    rl = _SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    rl.is_allowed("user1")
    rl.is_allowed("user1")
    assert rl.is_allowed("user1") is False

  def test_separate_keys(self):
    rl = _SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    assert rl.is_allowed("user1") is True
    assert rl.is_allowed("user2") is True
    assert rl.is_allowed("user1") is False


class TestOutboundRateLimiter:
  @pytest.mark.asyncio
  async def test_zero_rate_no_delay(self):
    rl = _OutboundRateLimiter(calls_per_second=0)
    await rl.acquire()  # Should not block

  @pytest.mark.asyncio
  async def test_high_rate_minimal_delay(self):
    rl = _OutboundRateLimiter(calls_per_second=1000)
    start = time.monotonic()
    await rl.acquire()
    await rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


# ===== Phase 15: Update Deduplication =====


class TestUpdateDeduplication:
  def test_first_update_not_duplicate(self):
    iface = TelegramInterface(bot_token="test:token")
    assert iface._is_duplicate_update(1) is False

  def test_second_same_update_is_duplicate(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._is_duplicate_update(1)
    assert iface._is_duplicate_update(1) is True

  def test_different_updates_not_duplicate(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._is_duplicate_update(1)
    assert iface._is_duplicate_update(2) is False

  def test_eviction_after_max(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._max_seen_updates = 3
    for i in range(5):
      iface._is_duplicate_update(i)
    # Oldest should be evicted
    assert iface._is_duplicate_update(0) is False  # Was evicted
    assert iface._is_duplicate_update(4) is True  # Still there


# ===== Phase 16: Command Menu Sync =====


class TestCommandMenuSync:
  @pytest.mark.asyncio
  async def test_sync_commands(self):
    iface = TelegramInterface(
      bot_token="test:token",
      commands={"help": "Show help", "start": "Start the bot"},
    )
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    await iface._sync_commands()
    iface._api_call.assert_called_once()
    call_args = iface._api_call.call_args
    assert call_args[0][0] == "setMyCommands"
    assert len(call_args[0][1]["commands"]) == 2

  @pytest.mark.asyncio
  async def test_sync_no_commands(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]

    await iface._sync_commands()
    iface._api_call.assert_not_called()


# ===== Phase 17: DM vs Group Policies =====


class TestDmGroupPolicies:
  def test_dm_open(self):
    iface = TelegramInterface(bot_token="test:token", dm_policy="open")
    assert iface._check_chat_policy("private", "1", "1") is True

  def test_dm_disabled(self):
    iface = TelegramInterface(bot_token="test:token", dm_policy="disabled")
    assert iface._check_chat_policy("private", "1", "1") is False

  def test_dm_allowlist_allowed(self):
    iface = TelegramInterface(bot_token="test:token", dm_policy="allowlist", dm_allowlist=[42])
    assert iface._check_chat_policy("private", "42", "42") is True

  def test_dm_allowlist_blocked(self):
    iface = TelegramInterface(bot_token="test:token", dm_policy="allowlist", dm_allowlist=[42])
    assert iface._check_chat_policy("private", "99", "99") is False

  def test_group_open(self):
    iface = TelegramInterface(bot_token="test:token", group_policy="open")
    assert iface._check_chat_policy("group", "1", "100") is True

  def test_group_disabled(self):
    iface = TelegramInterface(bot_token="test:token", group_policy="disabled")
    assert iface._check_chat_policy("group", "1", "100") is False

  def test_group_allowlist_allowed(self):
    iface = TelegramInterface(bot_token="test:token", group_policy="allowlist", group_allowlist=[100])
    assert iface._check_chat_policy("supergroup", "1", "100") is True

  def test_group_allowlist_blocked(self):
    iface = TelegramInterface(bot_token="test:token", group_policy="allowlist", group_allowlist=[100])
    assert iface._check_chat_policy("supergroup", "1", "200") is False


# ===== Phase 18: Location Messages =====


class TestLocationMessages:
  @pytest.mark.asyncio
  async def test_location_extraction(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "location": {"latitude": 37.7749, "longitude": -122.4194},
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text is not None
    assert "[Location: 37.7749, -122.4194]" in msg.text
    assert msg.metadata["location"] == {"latitude": 37.7749, "longitude": -122.4194}

  @pytest.mark.asyncio
  async def test_venue_extraction(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "venue": {
        "location": {"latitude": 40.7128, "longitude": -74.0060},
        "title": "Central Park",
        "address": "New York, NY",
      },
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text is not None
    assert "[Venue: Central Park, New York, NY" in msg.text


# ===== Phase 19: Media Groups =====


class TestMediaGroups:
  @pytest.mark.asyncio
  async def test_media_group_buffering(self):
    iface = TelegramInterface(bot_token="test:token", media_group_timeout=0.05)
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "photo/file.jpg"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    dispatched = []

    async def mock_dispatch(msg, is_edit=False):
      dispatched.append(msg)

    iface._dispatch_message = mock_dispatch  # type: ignore[method-assign]

    msg1 = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 10,
      "media_group_id": "mg1",
      "photo": [{"file_id": "p1", "file_size": 100}],
      "caption": "First",
    }
    msg2 = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 11,
      "media_group_id": "mg1",
      "photo": [{"file_id": "p2", "file_size": 200}],
    }

    await iface._buffer_media_group("mg1", msg1, False)
    await iface._buffer_media_group("mg1", msg2, False)

    # Wait for flush
    await asyncio.sleep(0.1)

    assert len(dispatched) == 1
    merged = dispatched[0]
    assert merged.get("_media_group_id") == "mg1"
    assert len(merged.get("_media_group_photos", [])) >= 1


# ===== Phase 20: Reactions =====


class TestReactions:
  @pytest.mark.asyncio
  async def test_handle_reaction(self):
    iface = TelegramInterface(bot_token="test:token", handle_reactions=True)
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    dispatched = []

    async def capture(msg):
      dispatched.append(msg)

    iface.handle_platform_message = capture  # type: ignore[method-assign]

    reaction = {
      "chat": {"id": 1, "type": "private"},
      "user": {"id": 2},
      "message_id": 5,
      "new_reaction": [{"type": "emoji", "emoji": "👍"}],
    }
    await iface._handle_reaction(reaction)
    assert len(dispatched) == 1
    assert dispatched[0]["text"] == "[Reaction: 👍 on message 5]"

  @pytest.mark.asyncio
  async def test_handle_reaction_no_emoji(self):
    iface = TelegramInterface(bot_token="test:token", handle_reactions=True)
    iface._client = MagicMock()
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface.handle_platform_message = AsyncMock()  # type: ignore[method-assign]

    reaction = {
      "chat": {"id": 1},
      "user": {"id": 2},
      "message_id": 5,
      "new_reaction": [],
    }
    await iface._handle_reaction(reaction)
    iface.handle_platform_message.assert_not_called()


# ===== Config Tests =====


class TestTelegramConfig:
  def test_new_fields_have_defaults(self):
    config = TelegramConfig(bot_token="test:token")
    assert config.auto_format is True
    assert config.streaming is True
    assert config.stream_edit_interval == 1.0
    assert config.stream_min_chars == 30
    assert config.stream_tool_indicator is True
    assert config.handle_callback_queries is True
    assert config.group_mode == "mention"
    assert config.enable_forum_topics is True
    assert config.outbound_rate_limit == 30.0
    assert config.commands is None
    assert config.sync_commands_on_startup is True
    assert config.dm_policy == "open"
    assert config.group_policy == "open"
    assert config.dm_allowlist is None
    assert config.group_allowlist is None
    assert config.media_group_timeout == 0.5
    assert config.handle_reactions is False

  def test_backward_compatible(self):
    # Existing usage should still work unchanged
    config = TelegramConfig(bot_token="test:token")
    assert config.platform == "telegram"
    assert config.parse_mode == "HTML"
    assert config.mode == "polling"


# ===== Interface Construction Tests =====


class TestTelegramInterfaceConstruction:
  def test_new_kwargs(self):
    iface = TelegramInterface(
      bot_token="test:token",
      auto_format=False,
      streaming=False,
      group_mode="always",
      enable_forum_topics=False,
      handle_reactions=True,
      dm_policy="allowlist",
      dm_allowlist=[1, 2],
      commands={"help": "Help"},
    )
    assert iface._tg_config.auto_format is False
    assert iface._tg_config.streaming is False
    assert iface._tg_config.group_mode == "always"
    assert iface._tg_config.enable_forum_topics is False
    assert iface._tg_config.handle_reactions is True
    assert iface._tg_config.dm_policy == "allowlist"
    assert iface._tg_config.dm_allowlist == [1, 2]

  def test_allowed_updates_default(self):
    iface = TelegramInterface(bot_token="test:token")
    updates = iface._get_allowed_updates()
    assert "message" in updates
    assert "edited_message" in updates
    assert "callback_query" in updates
    assert "message_reaction" not in updates  # handle_reactions=False by default

  def test_allowed_updates_with_reactions(self):
    iface = TelegramInterface(bot_token="test:token", handle_reactions=True)
    updates = iface._get_allowed_updates()
    assert "message_reaction" in updates


# ===== Process Update Routing =====


class TestProcessUpdateRouting:
  @pytest.mark.asyncio
  async def test_routes_callback_query(self):
    iface = TelegramInterface(bot_token="test:token", handle_callback_queries=True)
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    handled = []

    async def mock_handle_cb(cq):
      handled.append(cq)

    iface._handle_callback_query = mock_handle_cb  # type: ignore[method-assign]

    update = {
      "update_id": 1,
      "callback_query": {"id": "q1", "data": "test", "from": {"id": 1}},
    }
    await iface._process_update(update)
    assert len(handled) == 1

  @pytest.mark.asyncio
  async def test_routes_reaction(self):
    iface = TelegramInterface(bot_token="test:token", handle_reactions=True)
    iface._client = MagicMock()
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    handled = []

    async def mock_handle_reaction(r):
      handled.append(r)

    iface._handle_reaction = mock_handle_reaction  # type: ignore[method-assign]

    update = {
      "update_id": 2,
      "message_reaction": {
        "chat": {"id": 1},
        "user": {"id": 2},
        "message_id": 3,
        "new_reaction": [{"type": "emoji", "emoji": "😊"}],
      },
    }
    await iface._process_update(update)
    assert len(handled) == 1

  @pytest.mark.asyncio
  async def test_routes_regular_message(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    dispatched = []

    async def mock_dispatch(msg, is_edit=False):
      dispatched.append((msg, is_edit))

    iface._dispatch_message = mock_dispatch  # type: ignore[method-assign]

    update = {
      "update_id": 3,
      "message": {
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 2},
        "message_id": 3,
        "text": "hello",
      },
    }
    await iface._process_update(update)
    assert len(dispatched) == 1
    assert dispatched[0][1] is False  # not an edit


# ===== Inbound Conversion Full Pipeline =====


class TestConvertInbound:
  @pytest.mark.asyncio
  async def test_basic_text_message(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2, "username": "testuser"},
      "message_id": 3,
      "text": "hello world",
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text == "hello world"
    assert msg.platform == "telegram"
    assert msg.platform_user_id == "2"
    assert msg.platform_chat_id == "1"
    assert msg.username == "testuser"

  @pytest.mark.asyncio
  async def test_sticker_message(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "sticker/file.webp"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "sticker": {"file_id": "st1", "file_unique_id": "su1", "emoji": "😀", "set_name": "Pack"},
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text is not None
    assert "[Sticker: 😀 from 'Pack']" in msg.text
    # Static sticker should also provide an image
    assert msg.images is not None

  @pytest.mark.asyncio
  async def test_rate_limited_user_rejected(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(max_requests=1, window_seconds=60)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "text": "first",
    }
    msg1 = await iface._convert_inbound(raw)
    assert msg1 is not None

    raw2 = dict(raw)
    raw2["text"] = "second"
    raw2["message_id"] = 4
    msg2 = await iface._convert_inbound(raw2)
    assert msg2 is None  # Rate limited

  @pytest.mark.asyncio
  async def test_photo_with_caption(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "photo/file.jpg"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "caption": "Check this out!",
      "photo": [
        {"file_id": "small", "file_size": 100},
        {"file_id": "large", "file_size": 500},
      ],
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text == "Check this out!"
    assert msg.images is not None
    assert len(msg.images) == 1


# ===== Send Response Tests =====


class TestSendResponse:
  @pytest.mark.asyncio
  async def test_send_text_auto_format(self):
    iface = TelegramInterface(bot_token="test:token", auto_format=True, parse_mode="HTML")
    iface._client = MagicMock()

    sent_texts = []

    async def mock_send_msg(chat_id, text, parse_mode=None, reply_to_message_id=None, reply_markup=None, thread_id=None):
      sent_texts.append(text)
      return "1"

    iface._send_message = mock_send_msg  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    await iface._send_text("123", "**bold text**")
    assert len(sent_texts) == 1
    assert "<b>bold text</b>" in sent_texts[0]

  @pytest.mark.asyncio
  async def test_send_text_no_auto_format(self):
    iface = TelegramInterface(bot_token="test:token", auto_format=False, parse_mode="HTML")
    iface._client = MagicMock()

    sent_texts = []

    async def mock_send_msg(chat_id, text, parse_mode=None, reply_to_message_id=None, reply_markup=None, thread_id=None):
      sent_texts.append(text)
      return "1"

    iface._send_message = mock_send_msg  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    await iface._send_text("123", "**not converted**")
    assert sent_texts[0] == "**not converted**"

  @pytest.mark.asyncio
  async def test_send_text_html_chunking(self):
    iface = TelegramInterface(bot_token="test:token", auto_format=False, parse_mode="HTML", max_message_length=50)
    iface._client = MagicMock()

    sent_texts = []

    async def mock_send_msg(chat_id, text, parse_mode=None, reply_to_message_id=None, reply_markup=None, thread_id=None):
      sent_texts.append(text)
      return "1"

    iface._send_message = mock_send_msg  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    long_text = "a" * 80
    await iface._send_text("123", long_text)
    assert len(sent_texts) >= 2

  @pytest.mark.asyncio
  async def test_send_response_with_thread_id(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._typing_cb = _TypingCircuitBreaker()
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    typing_data: dict[str, Any] = {}
    sent_data: dict[str, Any] = {}

    async def mock_api(method, data=None):
      if method == "sendChatAction":
        typing_data.update(data or {})
      elif method == "sendMessage":
        sent_data.update(data or {})
      return {"message_id": 1}

    iface._api_call = mock_api  # type: ignore[method-assign]

    from definable.agent.interface.message import InterfaceMessage, InterfaceResponse

    original = InterfaceMessage(
      platform="telegram",
      platform_user_id="1",
      platform_chat_id="100:topic:5",
      platform_message_id="1",
      text="hello",
      metadata={"thread_id": 5},
    )
    response = InterfaceResponse(content="reply")

    await iface._send_response(original, response, {})
    assert typing_data.get("message_thread_id") == 5


# ===== Sticker in Convert Inbound =====


class TestStickerInConversion:
  @pytest.mark.asyncio
  async def test_animated_sticker_no_image(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "sticker/file.tgs"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "sticker": {
        "file_id": "st1",
        "file_unique_id": "su1",
        "emoji": "🔥",
        "set_name": "Animated",
        "is_animated": True,
      },
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text is not None
    assert "[Sticker: 🔥 from 'Animated']" in msg.text
    # Animated sticker should NOT have an image
    assert msg.images is None


# ===== Video Text Fallback =====


class TestVideoTextFallback:
  @pytest.mark.asyncio
  async def test_video_text_description(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "video/file.mp4"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "video": {
        "file_id": "v1",
        "duration": 10,
        "width": 1920,
        "height": 1080,
      },
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text is not None
    assert "[Video: 10s, 1920x1080]" in msg.text

  @pytest.mark.asyncio
  async def test_video_note_text_description(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "video/note.mp4"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "video_note": {"file_id": "vn1", "duration": 5},
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text is not None
    assert "[Video note: 5s]" in msg.text

  @pytest.mark.asyncio
  async def test_animation_text_description(self):
    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"file_path": "anim/file.mp4"})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    iface._inbound_rate_limiter = _SlidingWindowRateLimiter(100)

    raw = {
      "chat": {"id": 1, "type": "private"},
      "from": {"id": 2},
      "message_id": 3,
      "animation": {"file_id": "a1", "duration": 3, "width": 320, "height": 240},
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text is not None
    assert "[GIF: 3s, 320x240]" in msg.text

  def test_describe_video_static(self):
    desc = TelegramInterface._describe_video({"duration": 10, "width": 1920, "height": 1080})
    assert desc == "[Video: 10s, 1920x1080]"

  def test_describe_video_no_dimensions(self):
    desc = TelegramInterface._describe_video({"duration": 5})
    assert desc == "[Video: 5s]"

  def test_describe_video_with_filename(self):
    desc = TelegramInterface._describe_video({"duration": 10, "file_name": "cat.mp4"})
    assert desc == "[Video: 10s, cat.mp4]"

  def test_describe_video_custom_kind(self):
    desc = TelegramInterface._describe_video({"duration": 3}, kind="GIF")
    assert desc == "[GIF: 3s]"


# ===== TelegramOutputSkill =====


class TestTelegramOutputSkill:
  def test_skill_creation(self):
    from definable.agent.interface.telegram.skill import TelegramOutputSkill

    skill = TelegramOutputSkill()
    assert skill.name == "telegram_output"
    assert len(skill.tools) == 1
    assert "telegram" in skill.get_instructions().lower()

  def test_skill_without_buttons(self):
    from definable.agent.interface.telegram.skill import TelegramOutputSkill

    skill = TelegramOutputSkill(include_buttons=False)
    assert skill.tools == []

  def test_skill_custom_instructions(self):
    from definable.agent.interface.telegram.skill import TelegramOutputSkill

    skill = TelegramOutputSkill(custom_instructions="Always be polite.")
    assert "Always be polite." in skill.get_instructions()

  def test_pending_buttons_context_var(self):
    from definable.agent.interface.telegram.interface import _pending_buttons_var

    # Default is None
    assert _pending_buttons_var.get(None) is None

    # Set and read
    _pending_buttons_var.set([])
    assert _pending_buttons_var.get(None) == []

    # Reset
    _pending_buttons_var.set(None)
    assert _pending_buttons_var.get(None) is None


# ===== Streamed Response Metadata =====


class TestStreamedResponseMetadata:
  @pytest.mark.asyncio
  async def test_send_response_skips_text_when_streamed(self):
    """When _tg_streamed is set, _send_response should skip text sending."""
    from definable.agent.interface.message import InterfaceResponse

    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"message_id": 1})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    original_msg = MagicMock()
    original_msg.platform_chat_id = "123"
    original_msg.platform_message_id = "1"
    original_msg.metadata = {}

    response = InterfaceResponse(content="Hello", metadata={"_tg_streamed": True})
    await iface._send_response(original_msg, response, {})

    # _api_call should NOT have been called for sendMessage (text already sent via streaming)
    for call in iface._api_call.call_args_list:
      method = call[0][0]
      assert method != "sendMessage"

  @pytest.mark.asyncio
  async def test_send_response_sends_text_when_not_streamed(self):
    """When _tg_streamed is not set, _send_response should send text normally."""
    from definable.agent.interface.message import InterfaceResponse

    iface = TelegramInterface(bot_token="test:token")
    iface._client = MagicMock()
    iface._api_call = AsyncMock(return_value={"message_id": 1})  # type: ignore[method-assign]
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)

    original_msg = MagicMock()
    original_msg.platform_chat_id = "123"
    original_msg.platform_message_id = "1"
    original_msg.metadata = {}

    response = InterfaceResponse(content="Hello", metadata={})
    await iface._send_response(original_msg, response, {})

    # Should have called sendMessage for text
    methods = [call[0][0] for call in iface._api_call.call_args_list]
    assert "sendMessage" in methods


# ===== Streaming Double-Reply Prevention =====


class TestStreamingDoubleReply:
  """Regression tests: streaming + exception must never produce two bot messages."""

  def _make_iface(self) -> TelegramInterface:
    iface = TelegramInterface(bot_token="test:token", streaming=True)
    iface._client = MagicMock()
    iface._outbound_rate_limiter = _OutboundRateLimiter(0)
    return iface

  def _make_message(self) -> MagicMock:
    msg = MagicMock()
    msg.text = "hello"
    msg.platform_chat_id = "123"
    msg.platform_user_id = "u1"
    msg.platform_message_id = "10"
    msg.metadata = {}
    msg.images = None
    return msg

  @pytest.mark.asyncio
  async def test_no_double_send_when_stream_raises_after_partial_send(self):
    """When arun_stream raises after the first chunk is sent, only one sendMessage fires."""
    from definable.agent.interface.session import InterfaceSession
    from definable.agent.run.agent import RunContentEvent

    iface = self._make_iface()
    # stream_min_chars=0 so the first event immediately triggers sendMessage
    iface._tg_config = TelegramConfig(bot_token="test:token", streaming=True, stream_min_chars=0)

    send_calls: list[str] = []

    async def fake_api_call(method: str, data: Any = None, **kw: Any) -> Any:
      send_calls.append(method)
      if method == "sendMessage":
        return {"message_id": 99}
      return {}

    iface._api_call = fake_api_call  # type: ignore[method-assign]

    # arun_stream yields one content event then raises
    async def boom_stream(**_: Any):
      yield RunContentEvent(content="partial")
      raise RuntimeError("mid-stream error")

    iface.agent = MagicMock()
    iface.agent._thinking = None
    iface.agent._session_id_explicit = False
    iface.agent.arun_stream = boom_stream  # type: ignore[method-assign]

    session = InterfaceSession(platform="telegram", platform_user_id="u1", platform_chat_id="123")
    msg = self._make_message()

    result = await iface._run_agent_streaming(msg, session)

    # sendMessage fired exactly once (the partial send); no second message
    assert send_calls.count("sendMessage") == 1
    # Result is marked as streamed so _send_response won't resend
    assert (result.metadata or {}).get("_tg_streamed") is True

  @pytest.mark.asyncio
  async def test_fallback_fires_cleanly_when_nothing_sent_before_exception(self):
    """When stream raises before any message is sent, exception propagates for clean fallback."""
    from definable.agent.interface.session import InterfaceSession

    iface = self._make_iface()
    # stream_min_chars=9999 so buffer never reaches threshold — nothing sent before raise
    iface._tg_config = TelegramConfig(bot_token="test:token", streaming=True, stream_min_chars=9999)
    iface._api_call = AsyncMock(return_value={"message_id": 1})  # type: ignore[method-assign]

    async def boom_immediately(**_: Any):
      raise RuntimeError("instant failure")
      yield  # make it an async generator

    iface.agent = MagicMock()
    iface.agent._thinking = None
    iface.agent._session_id_explicit = False
    iface.agent.arun_stream = boom_immediately  # type: ignore[method-assign]

    session = InterfaceSession(platform="telegram", platform_user_id="u1", platform_chat_id="123")
    msg = self._make_message()

    # Should propagate so _run_agent()'s non-streaming fallback can fire
    with pytest.raises(RuntimeError, match="instant failure"):
      await iface._run_agent_streaming(msg, session)

    # Nothing was sent during the failed streaming attempt
    for call in iface._api_call.call_args_list:
      assert call[0][0] != "sendMessage"

  @pytest.mark.asyncio
  async def test_thinking_placeholder_replaced_not_duplicated(self):
    """Thinking placeholder is edited with non-streaming result, not left + new message below."""
    from definable.agent.events import RunOutput
    from definable.agent.interface.session import InterfaceSession

    iface = self._make_iface()
    iface._tg_config = TelegramConfig(bot_token="test:token", streaming=True)

    api_calls: list[tuple[str, Any]] = []

    async def tracking_api_call(method: str, data: Any = None, **kw: Any) -> Any:
      api_calls.append((method, data))
      if method == "sendMessage":
        return {"message_id": 77}
      return {}

    iface._api_call = tracking_api_call  # type: ignore[method-assign]

    # arun_stream yields nothing — triggers last-resort path
    async def empty_stream(**_: Any):
      return
      yield  # make it an async generator

    non_stream_output = RunOutput(content="Final answer", messages=[])

    iface.agent = MagicMock()
    # thinking enabled so placeholder is sent before stream starts
    thinking_mock = MagicMock()
    thinking_mock.enabled = True
    iface.agent._thinking = thinking_mock
    iface.agent._session_id_explicit = False
    iface.agent.arun_stream = empty_stream  # type: ignore[method-assign]

    # Patch super()._run_agent to return the non-streaming output
    async def fake_base_run(self_arg: Any, msg: Any, session: Any) -> RunOutput:
      return non_stream_output

    from definable.agent.interface import base as base_module

    original_run_agent = base_module.BaseInterface._run_agent
    base_module.BaseInterface._run_agent = fake_base_run  # type: ignore[method-assign]

    try:
      session = InterfaceSession(platform="telegram", platform_user_id="u1", platform_chat_id="123")
      msg = self._make_message()
      result = await iface._run_agent_streaming(msg, session)
    finally:
      base_module.BaseInterface._run_agent = original_run_agent  # type: ignore[method-assign]

    methods = [m for m, _ in api_calls]
    # Placeholder sent once
    assert methods.count("sendMessage") == 1
    # Placeholder edited with result (not a second sendMessage)
    assert "editMessageText" in methods
    assert methods.count("sendMessage") == 1
    # Result marked streamed so _send_response won't add a third message
    assert (result.metadata or {}).get("_tg_streamed") is True
    assert result.content == "Final answer"
