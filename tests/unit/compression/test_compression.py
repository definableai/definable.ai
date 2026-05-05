"""Unit tests for the CompressionManager and Compression."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from definable.agent.compression import Compression
from definable.agent.compression.manager import (
  DEFAULT_COMPRESSION_PROMPT,
  CompressionManager,
)
from definable.model.message import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GET_MODEL_PATH = "definable.agent.compression.manager.get_model"


def _tool_msg(content="tool output", tool_name="search", compressed=None, tool_calls=None):
  """Build a tool-role Message for tests."""
  return Message(
    role="tool",
    content=content,
    tool_name=tool_name,
    compressed_content=compressed,
    tool_calls=tool_calls,
  )


def _user_msg(content="hello"):
  return Message(role="user", content=content)


def _assistant_msg(content="hi"):
  return Message(role="assistant", content=content)


def _make_mock_model(response_content="compressed"):
  """Return a mock model whose .response() returns a ModelResponse-like object."""
  mock = MagicMock()
  mock_response = MagicMock()
  mock_response.content = response_content
  mock.response.return_value = mock_response
  mock.count_tokens.return_value = 100
  return mock


def _make_async_mock_model(response_content="compressed"):
  """Return a mock model whose .aresponse() is async and returns a ModelResponse-like object."""
  mock = MagicMock()
  mock_response = MagicMock()
  mock_response.content = response_content
  mock.aresponse = AsyncMock(return_value=mock_response)
  mock.acount_tokens = AsyncMock(return_value=100)
  mock.count_tokens.return_value = 100
  return mock


# ===========================================================================
# Compression (from agent/compression/__init__.py)
# ===========================================================================


@pytest.mark.unit
class TestCompressionDefaults:
  """Verify Compression default field values."""

  def test_model_defaults_none(self):
    cfg = Compression()
    assert cfg.model is None

  def test_tool_results_limit_defaults_3(self):
    cfg = Compression()
    assert cfg.tool_results_limit == 3

  def test_token_limit_defaults_none(self):
    cfg = Compression()
    assert cfg.token_limit is None

  def test_instructions_defaults_none(self):
    cfg = Compression()
    assert cfg.instructions is None


@pytest.mark.unit
class TestCompressionCustomValues:
  """Verify Compression accepts custom values."""

  def test_custom_tool_results_limit(self):
    cfg = Compression(tool_results_limit=10)
    assert cfg.tool_results_limit == 10

  def test_custom_token_limit(self):
    cfg = Compression(token_limit=4096)
    assert cfg.token_limit == 4096

  def test_custom_instructions(self):
    cfg = Compression(instructions="Be very brief.")
    assert cfg.instructions == "Be very brief."

  def test_string_model(self):
    cfg = Compression(model="gpt-4o-mini")
    assert cfg.model == "gpt-4o-mini"

  def test_object_model(self):
    sentinel = object()
    cfg = Compression(model=sentinel)  # type: ignore[arg-type]
    assert cfg.model is sentinel

  def test_string_model_resolves_via_agent(self):
    """String model in Compression is resolved to a Model via resolve_model_string in Agent."""
    cfg = Compression(model="openai/gpt-4o-mini")
    assert cfg.model == "openai/gpt-4o-mini"  # stored as-is; resolved at Agent init


# ===========================================================================
# CompressionManager construction & __post_init__
# ===========================================================================


@pytest.mark.unit
class TestCompressionManagerConstruction:
  """Verify CompressionManager dataclass construction and __post_init__ logic."""

  def test_defaults(self):
    mgr = CompressionManager()
    assert mgr.model is None
    assert mgr.compress_tool_results is True
    assert mgr.compress_tool_call_instructions is None
    assert mgr.stats == {}

  def test_post_init_sets_limit_when_both_none(self):
    mgr = CompressionManager()
    assert mgr.compress_tool_results_limit == 3
    assert mgr.compress_token_limit is None

  def test_post_init_does_not_override_explicit_results_limit(self):
    mgr = CompressionManager(compress_tool_results_limit=5)
    assert mgr.compress_tool_results_limit == 5

  def test_post_init_does_not_set_limit_when_token_limit_provided(self):
    mgr = CompressionManager(compress_token_limit=2000)
    assert mgr.compress_tool_results_limit is None
    assert mgr.compress_token_limit == 2000

  def test_post_init_does_not_set_limit_when_both_provided(self):
    mgr = CompressionManager(compress_tool_results_limit=10, compress_token_limit=5000)
    assert mgr.compress_tool_results_limit == 10
    assert mgr.compress_token_limit == 5000

  def test_compress_tool_results_disabled(self):
    mgr = CompressionManager(compress_tool_results=False)
    assert mgr.compress_tool_results is False

  def test_custom_instructions(self):
    mgr = CompressionManager(compress_tool_call_instructions="Short please.")
    assert mgr.compress_tool_call_instructions == "Short please."

  def test_stats_is_independent_per_instance(self):
    mgr1 = CompressionManager()
    mgr2 = CompressionManager()
    mgr1.stats["foo"] = 1
    assert "foo" not in mgr2.stats


# ===========================================================================
# _is_tool_result_message
# ===========================================================================


@pytest.mark.unit
class TestIsToolResultMessage:
  """Verify _is_tool_result_message identifies tool-role messages."""

  def test_tool_role_returns_true(self):
    mgr = CompressionManager()
    assert mgr._is_tool_result_message(_tool_msg()) is True

  def test_user_role_returns_false(self):
    mgr = CompressionManager()
    assert mgr._is_tool_result_message(_user_msg()) is False

  def test_assistant_role_returns_false(self):
    mgr = CompressionManager()
    assert mgr._is_tool_result_message(_assistant_msg()) is False

  def test_system_role_returns_false(self):
    mgr = CompressionManager()
    assert mgr._is_tool_result_message(Message(role="system", content="sys")) is False


# ===========================================================================
# should_compress (sync)
# ===========================================================================


@pytest.mark.unit
class TestShouldCompress:
  """Verify should_compress decides correctly based on thresholds."""

  def test_returns_false_when_disabled(self):
    mgr = CompressionManager(compress_tool_results=False)
    msgs = [_tool_msg() for _ in range(10)]
    assert mgr.should_compress(msgs) is False

  def test_returns_true_when_count_meets_limit(self):
    mgr = CompressionManager(compress_tool_results_limit=3)
    msgs = [_tool_msg() for _ in range(3)]
    assert mgr.should_compress(msgs) is True

  def test_returns_false_when_count_below_limit(self):
    mgr = CompressionManager(compress_tool_results_limit=3)
    msgs = [_tool_msg() for _ in range(2)]
    assert mgr.should_compress(msgs) is False

  def test_ignores_already_compressed_messages(self):
    mgr = CompressionManager(compress_tool_results_limit=3)
    msgs = [
      _tool_msg(compressed="already done"),
      _tool_msg(compressed="already done"),
      _tool_msg(),
      _tool_msg(),
    ]
    assert mgr.should_compress(msgs) is False

  def test_returns_true_when_token_limit_exceeded(self):
    mock_model = _make_mock_model()
    mock_model.count_tokens.return_value = 5000
    mgr = CompressionManager(compress_token_limit=4000, compress_tool_results_limit=None)
    msgs = [_user_msg()]
    assert mgr.should_compress(msgs, model=mock_model) is True

  def test_returns_false_when_token_limit_not_exceeded(self):
    mock_model = _make_mock_model()
    mock_model.count_tokens.return_value = 3000
    mgr = CompressionManager(compress_token_limit=4000, compress_tool_results_limit=None)
    msgs = [_user_msg()]
    assert mgr.should_compress(msgs, model=mock_model) is False

  def test_token_check_skipped_when_model_is_none(self):
    mgr = CompressionManager(compress_token_limit=100, compress_tool_results_limit=None)
    msgs = [_user_msg()]
    assert mgr.should_compress(msgs, model=None) is False

  def test_token_check_skipped_when_token_limit_is_none(self):
    mock_model = _make_mock_model()
    mgr = CompressionManager(compress_token_limit=None, compress_tool_results_limit=100)
    msgs = [_user_msg()]
    assert mgr.should_compress(msgs, model=mock_model) is False
    mock_model.count_tokens.assert_not_called()

  def test_non_tool_messages_ignored_for_count(self):
    mgr = CompressionManager(compress_tool_results_limit=2)
    msgs = [_user_msg(), _assistant_msg(), _tool_msg()]
    assert mgr.should_compress(msgs) is False

  def test_empty_messages_returns_false(self):
    mgr = CompressionManager()
    assert mgr.should_compress([]) is False

  def test_token_limit_at_exact_boundary_returns_true(self):
    mock_model = _make_mock_model()
    mock_model.count_tokens.return_value = 4000
    mgr = CompressionManager(compress_token_limit=4000, compress_tool_results_limit=None)
    msgs = [_user_msg()]
    assert mgr.should_compress(msgs, model=mock_model) is True


# ===========================================================================
# ashould_compress (async)
# ===========================================================================


@pytest.mark.unit
class TestAshouldCompress:
  """Verify ashould_compress mirrors should_compress logic asynchronously."""

  @pytest.mark.asyncio
  async def test_returns_false_when_disabled(self):
    mgr = CompressionManager(compress_tool_results=False)
    msgs = [_tool_msg() for _ in range(10)]
    assert await mgr.ashould_compress(msgs) is False

  @pytest.mark.asyncio
  async def test_returns_true_when_count_meets_limit(self):
    mgr = CompressionManager(compress_tool_results_limit=2)
    msgs = [_tool_msg(), _tool_msg()]
    assert await mgr.ashould_compress(msgs) is True

  @pytest.mark.asyncio
  async def test_returns_false_when_count_below_limit(self):
    mgr = CompressionManager(compress_tool_results_limit=3)
    msgs = [_tool_msg()]
    assert await mgr.ashould_compress(msgs) is False

  @pytest.mark.asyncio
  async def test_returns_true_when_token_limit_exceeded(self):
    mock_model = _make_async_mock_model()
    mock_model.acount_tokens = AsyncMock(return_value=6000)
    mgr = CompressionManager(compress_token_limit=5000, compress_tool_results_limit=None)
    msgs = [_user_msg()]
    assert await mgr.ashould_compress(msgs, model=mock_model) is True

  @pytest.mark.asyncio
  async def test_uses_acount_tokens(self):
    mock_model = _make_async_mock_model()
    mock_model.acount_tokens = AsyncMock(return_value=100)
    mgr = CompressionManager(compress_token_limit=5000, compress_tool_results_limit=None)
    msgs = [_user_msg()]
    await mgr.ashould_compress(msgs, model=mock_model)
    mock_model.acount_tokens.assert_awaited_once()


# ===========================================================================
# _compress_tool_result (sync)
# ===========================================================================


@pytest.mark.unit
class TestCompressToolResult:
  """Verify _compress_tool_result returns compressed content via model."""

  def test_returns_none_for_falsy_input(self):
    mgr = CompressionManager()
    assert mgr._compress_tool_result(None) is None  # type: ignore[arg-type]

  @patch(GET_MODEL_PATH, return_value=None)
  def test_returns_none_when_no_model(self, _mock_get_model):
    mgr = CompressionManager(model=None)
    result = mgr._compress_tool_result(_tool_msg())
    assert result is None

  @patch(GET_MODEL_PATH)
  def test_returns_model_response_content(self, mock_get_model):
    mock_model = _make_mock_model(response_content="short summary")
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    result = mgr._compress_tool_result(_tool_msg(content="long output"))
    assert result == "short summary"

  @patch(GET_MODEL_PATH)
  def test_uses_default_prompt_when_no_custom_instructions(self, mock_get_model):
    mock_model = _make_mock_model()
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    mgr._compress_tool_result(_tool_msg())
    messages_arg = mock_model.response.call_args.kwargs.get("messages")
    if messages_arg is None:
      messages_arg = mock_model.response.call_args[0][0]
    assert messages_arg[0].content == DEFAULT_COMPRESSION_PROMPT

  @patch(GET_MODEL_PATH)
  def test_uses_custom_instructions_when_provided(self, mock_get_model):
    mock_model = _make_mock_model()
    mock_get_model.return_value = mock_model
    custom = "Summarize in one word."
    mgr = CompressionManager(model=mock_model, compress_tool_call_instructions=custom)
    mgr._compress_tool_result(_tool_msg())
    messages_arg = mock_model.response.call_args.kwargs.get("messages")
    if messages_arg is None:
      messages_arg = mock_model.response.call_args[0][0]
    assert messages_arg[0].content == custom

  @patch(GET_MODEL_PATH)
  def test_formats_tool_content_with_name(self, mock_get_model):
    mock_model = _make_mock_model()
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    mgr._compress_tool_result(_tool_msg(content="data here", tool_name="web_search"))
    messages_arg = mock_model.response.call_args.kwargs.get("messages")
    if messages_arg is None:
      messages_arg = mock_model.response.call_args[0][0]
    user_content = messages_arg[1].content
    assert "Tool: web_search" in user_content
    assert "data here" in user_content

  @patch(GET_MODEL_PATH)
  def test_uses_unknown_when_tool_name_is_none(self, mock_get_model):
    mock_model = _make_mock_model()
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    mgr._compress_tool_result(_tool_msg(tool_name=None))
    messages_arg = mock_model.response.call_args.kwargs.get("messages")
    if messages_arg is None:
      messages_arg = mock_model.response.call_args[0][0]
    user_content = messages_arg[1].content
    assert "Tool: unknown" in user_content

  @patch(GET_MODEL_PATH)
  def test_returns_raw_content_on_model_exception(self, mock_get_model):
    mock_model = _make_mock_model()
    mock_model.response.side_effect = RuntimeError("API failure")
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    result = mgr._compress_tool_result(_tool_msg(content="raw data", tool_name="search"))
    assert result == "Tool: search\nraw data"


# ===========================================================================
# _acompress_tool_result (async)
# ===========================================================================


@pytest.mark.unit
class TestAcompressToolResult:
  """Verify _acompress_tool_result returns compressed content via async model."""

  @pytest.mark.asyncio
  async def test_returns_none_for_falsy_input(self):
    mgr = CompressionManager()
    assert await mgr._acompress_tool_result(None) is None  # type: ignore[arg-type]

  @pytest.mark.asyncio
  @patch(GET_MODEL_PATH, return_value=None)
  async def test_returns_none_when_no_model(self, _mock_get_model):
    mgr = CompressionManager(model=None)
    result = await mgr._acompress_tool_result(_tool_msg())
    assert result is None

  @pytest.mark.asyncio
  async def test_returns_model_response_content(self):
    mock_model = _make_async_mock_model(response_content="async summary")
    with patch(GET_MODEL_PATH, return_value=mock_model):
      mgr = CompressionManager(model=mock_model)
      result = await mgr._acompress_tool_result(_tool_msg(content="long output"))
    assert result == "async summary"

  @pytest.mark.asyncio
  async def test_returns_raw_content_on_model_exception(self):
    mock_model = _make_async_mock_model()
    mock_model.aresponse = AsyncMock(side_effect=RuntimeError("Async API failure"))
    with patch(GET_MODEL_PATH, return_value=mock_model):
      mgr = CompressionManager(model=mock_model)
      result = await mgr._acompress_tool_result(_tool_msg(content="raw data", tool_name="calc"))
    assert result == "Tool: calc\nraw data"


# ===========================================================================
# compress (sync)
# ===========================================================================


@pytest.mark.unit
class TestCompress:
  """Verify compress() mutates messages in-place and tracks stats."""

  def test_noop_when_disabled(self):
    mock_model = _make_mock_model()
    mgr = CompressionManager(model=mock_model, compress_tool_results=False)
    msg = _tool_msg()
    mgr.compress([msg])
    assert msg.compressed_content is None
    mock_model.response.assert_not_called()

  def test_noop_when_no_uncompressed_tools(self):
    mock_model = _make_mock_model()
    mgr = CompressionManager(model=mock_model)
    msgs = [_user_msg(), _assistant_msg()]
    mgr.compress(msgs)
    mock_model.response.assert_not_called()

  def test_noop_when_all_already_compressed(self):
    mock_model = _make_mock_model()
    mgr = CompressionManager(model=mock_model)
    msgs = [_tool_msg(compressed="done"), _tool_msg(compressed="done")]
    mgr.compress(msgs)
    mock_model.response.assert_not_called()

  @patch(GET_MODEL_PATH)
  def test_sets_compressed_content_on_tool_messages(self, mock_get_model):
    mock_model = _make_mock_model(response_content="short")
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    msg = _tool_msg(content="verbose tool output")
    mgr.compress([msg])
    assert msg.compressed_content == "short"

  @patch(GET_MODEL_PATH)
  def test_does_not_touch_non_tool_messages(self, mock_get_model):
    mock_model = _make_mock_model(response_content="short")
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    user = _user_msg()
    assistant = _assistant_msg()
    tool = _tool_msg()
    mgr.compress([user, assistant, tool])
    assert user.compressed_content is None
    assert assistant.compressed_content is None
    assert tool.compressed_content == "short"

  @patch(GET_MODEL_PATH)
  def test_skips_already_compressed_messages(self, mock_get_model):
    mock_model = _make_mock_model(response_content="new")
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    already = _tool_msg(compressed="old")
    fresh = _tool_msg()
    mgr.compress([already, fresh])
    assert already.compressed_content == "old"
    assert fresh.compressed_content == "new"
    assert mock_model.response.call_count == 1

  @patch(GET_MODEL_PATH)
  def test_stats_tool_results_compressed_incremented(self, mock_get_model):
    mock_model = _make_mock_model(response_content="s")
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    mgr.compress([_tool_msg(), _tool_msg()])
    assert mgr.stats["tool_results_compressed"] == 2

  @patch(GET_MODEL_PATH)
  def test_stats_original_and_compressed_size_tracked(self, mock_get_model):
    mock_model = _make_mock_model(response_content="s")
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    mgr.compress([_tool_msg(content="x" * 100)])
    assert mgr.stats["original_size"] == 100
    assert mgr.stats["compressed_size"] == 1

  @patch(GET_MODEL_PATH)
  def test_stats_accumulate_across_calls(self, mock_get_model):
    mock_model = _make_mock_model(response_content="s")
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    mgr.compress([_tool_msg(content="aa")])
    mgr.compress([_tool_msg(content="bbb")])
    assert mgr.stats["tool_results_compressed"] == 2
    assert mgr.stats["original_size"] == 5  # 2 + 3
    assert mgr.stats["compressed_size"] == 2  # 1 + 1

  @patch(GET_MODEL_PATH)
  def test_stats_count_tool_calls_when_present(self, mock_get_model):
    mock_model = _make_mock_model(response_content="s")
    mock_get_model.return_value = mock_model
    mgr = CompressionManager(model=mock_model)
    tool_calls = [{"id": "1", "function": {"name": "a"}}, {"id": "2", "function": {"name": "b"}}]
    msg = _tool_msg(tool_calls=tool_calls)
    mgr.compress([msg])
    assert mgr.stats["tool_results_compressed"] == 2

  @patch(GET_MODEL_PATH, return_value=None)
  def test_compression_failure_does_not_set_compressed_content(self, _mock_get_model):
    mgr = CompressionManager(model=None)
    msg = _tool_msg()
    mgr.compress([msg])
    assert msg.compressed_content is None

  def test_empty_message_list(self):
    mgr = CompressionManager(model=_make_mock_model())
    mgr.compress([])
    assert mgr.stats == {}


# ===========================================================================
# acompress (async)
# ===========================================================================


@pytest.mark.unit
class TestAcompress:
  """Verify acompress() uses asyncio.gather for parallel compression."""

  @pytest.mark.asyncio
  async def test_noop_when_disabled(self):
    mock_model = _make_async_mock_model()
    mgr = CompressionManager(model=mock_model, compress_tool_results=False)
    msg = _tool_msg()
    await mgr.acompress([msg])
    assert msg.compressed_content is None

  @pytest.mark.asyncio
  async def test_noop_when_no_uncompressed_tools(self):
    mock_model = _make_async_mock_model()
    mgr = CompressionManager(model=mock_model)
    await mgr.acompress([_user_msg()])
    mock_model.aresponse.assert_not_awaited()

  @pytest.mark.asyncio
  async def test_sets_compressed_content(self):
    mock_model = _make_async_mock_model(response_content="async-short")
    with patch(GET_MODEL_PATH, return_value=mock_model):
      mgr = CompressionManager(model=mock_model)
      msg = _tool_msg(content="verbose async output")
      await mgr.acompress([msg])
    assert msg.compressed_content == "async-short"

  @pytest.mark.asyncio
  async def test_compresses_multiple_in_parallel(self):
    mock_model = _make_async_mock_model(response_content="p")
    with patch(GET_MODEL_PATH, return_value=mock_model):
      mgr = CompressionManager(model=mock_model)
      msgs = [_tool_msg(), _tool_msg(), _tool_msg()]
      await mgr.acompress(msgs)
    assert all(m.compressed_content == "p" for m in msgs)
    assert mock_model.aresponse.await_count == 3

  @pytest.mark.asyncio
  async def test_skips_already_compressed(self):
    mock_model = _make_async_mock_model(response_content="new")
    with patch(GET_MODEL_PATH, return_value=mock_model):
      mgr = CompressionManager(model=mock_model)
      already = _tool_msg(compressed="old")
      fresh = _tool_msg()
      await mgr.acompress([already, fresh])
    assert already.compressed_content == "old"
    assert fresh.compressed_content == "new"
    assert mock_model.aresponse.await_count == 1

  @pytest.mark.asyncio
  async def test_stats_tracked(self):
    mock_model = _make_async_mock_model(response_content="s")
    with patch(GET_MODEL_PATH, return_value=mock_model):
      mgr = CompressionManager(model=mock_model)
      await mgr.acompress([_tool_msg(content="abc"), _tool_msg(content="defgh")])
    assert mgr.stats["tool_results_compressed"] == 2
    assert mgr.stats["original_size"] == 8  # 3 + 5
    assert mgr.stats["compressed_size"] == 2  # 1 + 1

  @pytest.mark.asyncio
  async def test_stats_count_tool_calls_when_present(self):
    mock_model = _make_async_mock_model(response_content="s")
    with patch(GET_MODEL_PATH, return_value=mock_model):
      mgr = CompressionManager(model=mock_model)
      tool_calls = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
      msg = _tool_msg(tool_calls=tool_calls)
      await mgr.acompress([msg])
    assert mgr.stats["tool_results_compressed"] == 3

  @pytest.mark.asyncio
  @patch(GET_MODEL_PATH, return_value=None)
  async def test_compression_failure_does_not_set_compressed(self, _mock_get_model):
    mgr = CompressionManager(model=None)
    msg = _tool_msg()
    await mgr.acompress([msg])
    assert msg.compressed_content is None

  @pytest.mark.asyncio
  async def test_empty_message_list(self):
    mgr = CompressionManager(model=_make_async_mock_model())
    await mgr.acompress([])
    assert mgr.stats == {}


# ===========================================================================
# DEFAULT_COMPRESSION_PROMPT
# ===========================================================================


@pytest.mark.unit
class TestDefaultCompressionPrompt:
  """Verify the default prompt is well-formed."""

  def test_is_non_empty_string(self):
    assert isinstance(DEFAULT_COMPRESSION_PROMPT, str)
    assert len(DEFAULT_COMPRESSION_PROMPT) > 100

  def test_contains_key_sections(self):
    assert "ALWAYS PRESERVE" in DEFAULT_COMPRESSION_PROMPT
    assert "COMPRESS TO ESSENTIALS" in DEFAULT_COMPRESSION_PROMPT
    assert "REMOVE ENTIRELY" in DEFAULT_COMPRESSION_PROMPT

  def test_does_not_have_leading_whitespace(self):
    assert not DEFAULT_COMPRESSION_PROMPT.startswith(" ")
    assert not DEFAULT_COMPRESSION_PROMPT.startswith("\t")


# ===========================================================================
# __init__.py re-export
# ===========================================================================


@pytest.mark.unit
class TestCompressionInit:
  """Verify the compression package re-exports Compression and CompressionManager."""

  def test_import_compression_manager_from_package(self):
    from definable.agent.compression import CompressionManager as CM

    assert CM is CompressionManager

  def test_import_compression_from_package(self):
    from definable.agent.compression import Compression as C

    assert C is Compression

  def test_import_compression_from_agent(self):
    from definable.agent import Compression as C

    assert C is Compression

  def test_import_compression_from_top_level(self):
    from definable import Compression as C

    assert C is Compression


# ===========================================================================
# single_result_size threshold (CompressionManager)
# ===========================================================================


@pytest.mark.unit
class TestSingleResultSize:
  """Verify compress_single_result_size triggers compression for large individual results."""

  def test_should_compress_when_single_result_exceeds_size(self):
    mgr = CompressionManager(compress_single_result_size=1000, compress_tool_results_limit=None)
    msgs = [_tool_msg(content="x" * 2000)]
    assert mgr.should_compress(msgs) is True

  def test_should_not_compress_when_single_result_below_size(self):
    mgr = CompressionManager(compress_single_result_size=1000, compress_tool_results_limit=None)
    msgs = [_tool_msg(content="x" * 500)]
    assert mgr.should_compress(msgs) is False

  def test_skips_already_compressed(self):
    mgr = CompressionManager(compress_single_result_size=1000, compress_tool_results_limit=None)
    msgs = [_tool_msg(content="x" * 2000, compressed="already done")]
    assert mgr.should_compress(msgs) is False

  def test_none_ignored(self):
    """When compress_single_result_size is None (default), large messages don't trigger."""
    mgr = CompressionManager(compress_single_result_size=None, compress_tool_results_limit=None)
    msgs = [_tool_msg(content="x" * 100_000)]
    assert mgr.should_compress(msgs) is False

  def test_exact_boundary_triggers(self):
    mgr = CompressionManager(compress_single_result_size=100, compress_tool_results_limit=None)
    msgs = [_tool_msg(content="x" * 100)]
    assert mgr.should_compress(msgs) is True

  def test_non_tool_messages_ignored(self):
    mgr = CompressionManager(compress_single_result_size=10, compress_tool_results_limit=None)
    msgs = [_user_msg(content="x" * 1000), _assistant_msg(content="x" * 1000)]
    assert mgr.should_compress(msgs) is False

  @pytest.mark.asyncio
  async def test_async_should_compress_when_single_result_exceeds_size(self):
    mgr = CompressionManager(compress_single_result_size=1000, compress_tool_results_limit=None)
    msgs = [_tool_msg(content="x" * 2000)]
    assert await mgr.ashould_compress(msgs) is True

  @pytest.mark.asyncio
  async def test_async_should_not_compress_when_below_size(self):
    mgr = CompressionManager(compress_single_result_size=1000, compress_tool_results_limit=None)
    msgs = [_tool_msg(content="x" * 500)]
    assert await mgr.ashould_compress(msgs) is False

  @pytest.mark.asyncio
  async def test_async_skips_already_compressed(self):
    mgr = CompressionManager(compress_single_result_size=1000, compress_tool_results_limit=None)
    msgs = [_tool_msg(content="x" * 2000, compressed="done")]
    assert await mgr.ashould_compress(msgs) is False


# ===========================================================================
# Compression config: single_result_size field
# ===========================================================================


@pytest.mark.unit
class TestCompressionSingleResultSizeField:
  """Verify single_result_size on the user-facing Compression config."""

  def test_defaults_none(self):
    cfg = Compression()
    assert cfg.single_result_size is None

  def test_custom_value(self):
    cfg = Compression(single_result_size=5000)
    assert cfg.single_result_size == 5000


# ===========================================================================
# AgentLoop: compress_tool_results flag wiring
# ===========================================================================


@pytest.mark.unit
class TestLoopCompressFlag:
  """Verify AgentLoop passes compress_tool_results to model calls."""

  @pytest.mark.asyncio
  async def test_loop_passes_compress_flag_when_manager_exists(self):
    """When compression_manager is provided and compress_tool_results=True, the flag is passed."""
    from definable.agent.loop import AgentLoop
    from definable.agent.events import RunContext

    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "done"
    mock_response.tool_calls = None
    mock_response.response_usage = None
    mock_response.parsed = None
    mock_model.ainvoke = AsyncMock(return_value=mock_response)
    mock_model.id = "test-model"
    mock_model.provider = "test"

    mgr = CompressionManager(compress_tool_results=True)

    context = RunContext(run_id="r1", session_id="s1")
    loop = AgentLoop(
      model=mock_model,
      tools={},
      messages=[Message(role="user", content="hi")],
      context=context,
      config=MagicMock(max_tool_rounds=5, max_retries=0, retry_transient_errors=False),
      streaming=False,
      compression_manager=mgr,
      emit_fn=lambda e: None,
      agent_id="a1",
      agent_name="test",
    )

    events = []
    async for event in loop.run():
      events.append(event)

    mock_model.ainvoke.assert_awaited_once()
    call_kwargs = mock_model.ainvoke.call_args.kwargs
    assert call_kwargs["compress_tool_results"] is True

  @pytest.mark.asyncio
  async def test_loop_no_compress_flag_when_no_manager(self):
    """Without compression_manager, compress_tool_results should be False."""
    from definable.agent.loop import AgentLoop
    from definable.agent.events import RunContext

    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "done"
    mock_response.tool_calls = None
    mock_response.response_usage = None
    mock_response.parsed = None
    mock_model.ainvoke = AsyncMock(return_value=mock_response)
    mock_model.id = "test-model"
    mock_model.provider = "test"

    context = RunContext(run_id="r1", session_id="s1")
    loop = AgentLoop(
      model=mock_model,
      tools={},
      messages=[Message(role="user", content="hi")],
      context=context,
      config=MagicMock(max_tool_rounds=5, max_retries=0, retry_transient_errors=False),
      streaming=False,
      compression_manager=None,
      emit_fn=lambda e: None,
      agent_id="a1",
      agent_name="test",
    )

    events = []
    async for event in loop.run():
      events.append(event)

    mock_model.ainvoke.assert_awaited_once()
    call_kwargs = mock_model.ainvoke.call_args.kwargs
    assert call_kwargs["compress_tool_results"] is False

  @pytest.mark.asyncio
  async def test_loop_passes_compress_flag_streaming(self):
    """Streaming mode also passes the compress flag."""
    from definable.agent.loop import AgentLoop
    from definable.agent.events import RunContext

    mock_model = MagicMock()
    # Make ainvoke_stream an async generator
    chunk = MagicMock()
    chunk.content = "streamed"
    chunk.tool_calls = None
    chunk.response_usage = None
    chunk.parsed = None

    async def fake_stream(**kwargs):
      yield chunk

    mock_model.ainvoke_stream = MagicMock(side_effect=fake_stream)
    mock_model.id = "test-model"
    mock_model.provider = "test"

    mgr = CompressionManager(compress_tool_results=True)

    context = RunContext(run_id="r1", session_id="s1")
    loop = AgentLoop(
      model=mock_model,
      tools={},
      messages=[Message(role="user", content="hi")],
      context=context,
      config=MagicMock(max_tool_rounds=5, max_retries=0, retry_transient_errors=False),
      streaming=True,
      compression_manager=mgr,
      emit_fn=lambda e: None,
      agent_id="a1",
      agent_name="test",
    )

    events = []
    async for event in loop.run():
      events.append(event)

    mock_model.ainvoke_stream.assert_called_once()
    call_kwargs = mock_model.ainvoke_stream.call_args.kwargs
    assert call_kwargs["compress_tool_results"] is True
