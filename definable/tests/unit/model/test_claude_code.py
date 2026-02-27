"""
Unit tests for ClaudeCode model — the claude-agent-sdk wrapper.

Tests cover:
  - Initialization defaults and custom config
  - Options builder (tools, thinking, streaming, permissions)
  - MCP bridge (tool dict → SDK MCP server conversion)
  - Message serialization (system extraction, prompt flattening)
  - Response parsing (text, thinking, tool calls, metrics, structured output)
  - Tool call parsing (MCP prefix stripping, multiple tools)
  - Stream event parsing (text_delta, thinking_delta, assistant messages)
  - Provider registration (resolve_model_string)
  - Error handling (import errors, SDK errors → ModelProviderError)
  - Thinking fallback (streaming → buffered when thinking enabled)
"""

import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

sdk = pytest.importorskip("claude_agent_sdk", reason="claude-agent-sdk not installed")

from claude_agent_sdk import (  # noqa: E402
  AssistantMessage as SdkAssistantMessage,
  ResultMessage as SdkResultMessage,
  TextBlock as SdkTextBlock,
  ThinkingBlock as SdkThinkingBlock,
  ToolUseBlock as SdkToolUseBlock,
)
from claude_agent_sdk.types import StreamEvent as SdkStreamEvent  # noqa: E402

from definable.exceptions import ModelProviderError  # noqa: E402
from definable.model.message import Message  # noqa: E402
from definable.model.response import ModelResponse  # noqa: E402


# ---------------------------------------------------------------------------
# Helper factories using real SDK types
# ---------------------------------------------------------------------------


def _text(text: str) -> SdkTextBlock:
  return SdkTextBlock(text=text)


def _thinking(thinking: str, signature: str = "sig") -> SdkThinkingBlock:
  return SdkThinkingBlock(thinking=thinking, signature=signature)


def _tool_use(tool_id: str, name: str, tool_input: Dict[str, Any]) -> SdkToolUseBlock:
  return SdkToolUseBlock(id=tool_id, name=name, input=tool_input)  # noqa: A003


def _assistant(content: list, model: str = "sonnet") -> SdkAssistantMessage:
  return SdkAssistantMessage(content=content, model=model)


def _result(
  duration_ms: int = 1500,
  is_error: bool = False,
  total_cost_usd: Optional[float] = 0.005,
  usage: Optional[Dict[str, Any]] = None,
  result: Optional[str] = None,
  structured_output: Any = None,
) -> SdkResultMessage:
  return SdkResultMessage(
    subtype="result",
    duration_ms=duration_ms,
    duration_api_ms=1200,
    is_error=is_error,
    num_turns=1,
    session_id="test-session",
    total_cost_usd=total_cost_usd,
    usage=usage if usage is not None else {"input_tokens": 100, "output_tokens": 50},
    result=result,
    structured_output=structured_output,
  )


def _stream_event(event: Dict[str, Any]) -> SdkStreamEvent:
  return SdkStreamEvent(uuid="test-uuid", session_id="test-session", event=event)


class MockOutputModel(BaseModel):
  name: str
  age: int


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeInit:
  """Verify default and custom initialization."""

  def test_default_attributes(self):
    """ClaudeCode has correct defaults for id, name, provider."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    assert model.id == "sonnet"
    assert model.name == "ClaudeCode"
    assert model.provider == "ClaudeCode"
    assert model.permission_mode == "bypassPermissions"
    assert model.thinking is None
    assert model.cli_path is None
    assert model.cwd is None

  def test_custom_id(self):
    """Custom model id is preserved."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(id="opus")
    assert model.id == "opus"

  def test_custom_thinking(self):
    """Thinking config is stored correctly."""
    from definable.model.claude_code import ClaudeCode

    thinking = {"type": "enabled", "budget_tokens": 10000}
    model = ClaudeCode(thinking=thinking)
    assert model.thinking == thinking

  def test_custom_cli_path(self):
    """Custom CLI path is stored."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(cli_path="/usr/local/bin/claude")
    assert model.cli_path == "/usr/local/bin/claude"

  def test_custom_cwd(self):
    """Custom working directory is stored."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(cwd="/tmp/work")
    assert model.cwd == "/tmp/work"


# ---------------------------------------------------------------------------
# Test: Options builder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeBuildOptions:
  """Verify _build_options produces correct ClaudeAgentOptions."""

  def test_basic_options_no_tools(self):
    """Options without tools should disable all tools."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    opts = model._build_options(system_prompt_text="You are helpful.", stream=False)
    assert opts.max_turns == 1
    assert opts.permission_mode == "bypassPermissions"
    assert opts.system_prompt == "You are helpful."
    assert opts.allowed_tools == []
    assert opts.disallowed_tools == ["*"]

  def test_options_with_mcp_server(self):
    """Options with MCP server should allow definable tools."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    mock_server = MagicMock()
    opts = model._build_options(
      system_prompt_text="",
      stream=False,
      mcp_server=mock_server,
    )
    assert isinstance(opts.mcp_servers, dict)
    assert "definable" in opts.mcp_servers
    assert opts.allowed_tools == ["mcp__definable__*"]
    assert opts.disallowed_tools == ["*"]

  def test_options_streaming_no_thinking(self):
    """Streaming enabled when thinking is off."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    opts = model._build_options(system_prompt_text="", stream=True)
    assert opts.include_partial_messages is True

  def test_options_streaming_with_thinking(self):
    """Streaming disabled when thinking is on (SDK limitation)."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(thinking={"type": "enabled", "budget_tokens": 5000})
    opts = model._build_options(system_prompt_text="", stream=True)
    assert opts.include_partial_messages is False

  def test_options_include_thinking_config(self):
    """Thinking config is passed through to options."""
    from definable.model.claude_code import ClaudeCode

    thinking = {"type": "adaptive"}
    model = ClaudeCode(thinking=thinking)
    opts = model._build_options(system_prompt_text="", stream=False)
    assert opts.thinking == thinking

  def test_options_include_cli_path(self):
    """CLI path is passed through to options."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(cli_path="/custom/claude")
    opts = model._build_options(system_prompt_text="", stream=False)
    assert opts.cli_path == "/custom/claude"

  def test_options_include_cwd(self):
    """Working directory is passed through to options."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(cwd="/workspace")
    opts = model._build_options(system_prompt_text="", stream=False)
    assert opts.cwd == "/workspace"

  def test_options_include_model_id(self):
    """Model ID is passed to options."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(id="haiku")
    opts = model._build_options(system_prompt_text="", stream=False)
    assert opts.model == "haiku"

  def test_options_structured_output(self):
    """Structured output format is built from Pydantic model."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    opts = model._build_options(
      system_prompt_text="",
      stream=False,
      response_format=MockOutputModel,
    )
    assert opts.output_format is not None
    assert opts.output_format["type"] == "json_schema"
    assert "properties" in opts.output_format["schema"]


# ---------------------------------------------------------------------------
# Test: MCP bridge
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeMCPBridge:
  """Verify tool dict → SDK MCP server conversion."""

  def test_single_tool_conversion(self):
    """Single tool dict converts to an MCP server config."""
    from definable.model.claude_code.chat import _build_mcp_server

    tools = [
      {
        "type": "function",
        "function": {
          "name": "search",
          "description": "Search for things",
          "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
          },
        },
      }
    ]
    server = _build_mcp_server(tools)
    # Should return an McpSdkServerConfig (TypedDict)
    assert server is not None
    assert server["name"] == "definable"

  def test_multiple_tools_conversion(self):
    """Multiple tool dicts all convert correctly."""
    from definable.model.claude_code.chat import _build_mcp_server

    tools = [
      {"type": "function", "function": {"name": "search", "description": "Search", "parameters": {}}},
      {"type": "function", "function": {"name": "calculate", "description": "Calculate", "parameters": {}}},
    ]
    server = _build_mcp_server(tools)
    assert server is not None

  def test_empty_tools_list(self):
    """Empty tools list produces a valid server with no tools."""
    from definable.model.claude_code.chat import _build_mcp_server

    server = _build_mcp_server([])
    assert server is not None

  def test_tool_with_empty_params(self):
    """Tool with empty parameters gets default schema."""
    from definable.model.claude_code.chat import _build_mcp_server

    tools = [{"type": "function", "function": {"name": "noop", "description": "Does nothing", "parameters": {}}}]
    server = _build_mcp_server(tools)
    assert server is not None


# ---------------------------------------------------------------------------
# Test: Message conversion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeMessageConversion:
  """Verify message serialization for the SDK."""

  def test_extract_system_prompt(self):
    """System messages are extracted into a single string."""
    from definable.model.claude_code.chat import _extract_system_prompt

    messages = [
      Message(role="system", content="You are a helpful assistant."),
      Message(role="user", content="Hello"),
    ]
    result = _extract_system_prompt(messages)
    assert "helpful assistant" in result

  def test_extract_system_prompt_multiple(self):
    """Multiple system messages are joined."""
    from definable.model.claude_code.chat import _extract_system_prompt

    messages = [
      Message(role="system", content="Instruction 1."),
      Message(role="system", content="Instruction 2."),
      Message(role="user", content="Hello"),
    ]
    result = _extract_system_prompt(messages)
    assert "Instruction 1" in result
    assert "Instruction 2" in result

  def test_extract_system_prompt_none(self):
    """No system messages returns empty string."""
    from definable.model.claude_code.chat import _extract_system_prompt

    messages = [Message(role="user", content="Hello")]
    result = _extract_system_prompt(messages)
    assert result == ""

  def test_messages_to_prompt_user(self):
    """User messages are serialized as plain text."""
    from definable.model.claude_code.chat import _messages_to_prompt

    messages = [Message(role="user", content="What is 2+2?")]
    result = _messages_to_prompt(messages)
    assert "What is 2+2?" in result

  def test_messages_to_prompt_skips_system(self):
    """System messages are skipped in prompt (handled separately)."""
    from definable.model.claude_code.chat import _messages_to_prompt

    messages = [
      Message(role="system", content="Be helpful"),
      Message(role="user", content="Hello"),
    ]
    result = _messages_to_prompt(messages)
    assert "Be helpful" not in result
    assert "Hello" in result

  def test_messages_to_prompt_assistant(self):
    """Assistant messages are formatted with prefix."""
    from definable.model.claude_code.chat import _messages_to_prompt

    messages = [
      Message(role="user", content="Hi"),
      Message(role="assistant", content="Hello!"),
    ]
    result = _messages_to_prompt(messages)
    assert "[Assistant]: Hello!" in result

  def test_messages_to_prompt_tool_result(self):
    """Tool result messages include tool_call_id."""
    from definable.model.claude_code.chat import _messages_to_prompt

    messages = [
      Message(role="tool", content="42", tool_call_id="call_123"),
    ]
    result = _messages_to_prompt(messages)
    assert "[Tool Result (call_123)]: 42" in result

  def test_messages_to_prompt_assistant_with_tool_calls(self):
    """Assistant messages with tool calls include call info."""
    from definable.model.claude_code.chat import _messages_to_prompt

    messages = [
      Message(
        role="assistant",
        content="Let me search.",
        tool_calls=[
          {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "test"}'},
          }
        ],
      ),
    ]
    result = _messages_to_prompt(messages)
    assert "[Assistant Tool Call]: search" in result
    assert "[Assistant]: Let me search." in result

  def test_messages_to_prompt_list_content(self):
    """Messages with list content extract text parts."""
    from definable.model.claude_code.chat import _messages_to_prompt

    messages = [
      Message(role="user", content=[{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]),
    ]
    result = _messages_to_prompt(messages)
    assert "Hello" in result
    assert "World" in result

  def test_extract_system_prompt_list_content(self):
    """System messages with list content extract text."""
    from definable.model.claude_code.chat import _extract_system_prompt

    messages = [
      Message(role="system", content=[{"type": "text", "text": "Rule 1"}, {"type": "text", "text": "Rule 2"}]),
    ]
    result = _extract_system_prompt(messages)
    assert "Rule 1" in result
    assert "Rule 2" in result


# ---------------------------------------------------------------------------
# Test: Response parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeResponseParsing:
  """Verify _parse_provider_response produces correct ModelResponse."""

  def test_text_content(self):
    """Text blocks become response content."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    assistant = _assistant(content=[_text(text="Hello world")])
    result_msg = _result()
    response = model._parse_provider_response((
      [assistant],
      result_msg,
    ))
    assert response.content == "Hello world"
    assert response.role == "assistant"

  def test_multiple_text_blocks(self):
    """Multiple text blocks are concatenated."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    assistant = _assistant(
      content=[
        _text(text="Hello "),
        _text(text="world"),
      ]
    )
    response = model._parse_provider_response(([assistant], _result()))
    assert response.content == "Hello world"

  def test_thinking_blocks(self):
    """Thinking blocks populate reasoning_content."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    assistant = _assistant(
      content=[
        _thinking(thinking="Let me think...", signature="sig123"),
        _text(text="The answer is 4."),
      ]
    )
    response = model._parse_provider_response(([assistant], _result()))
    assert response.reasoning_content == "Let me think..."
    assert response.content == "The answer is 4."
    assert response.provider_data is not None
    assert response.provider_data["signature"] == "sig123"

  def test_metrics_extraction(self):
    """Result message metrics are extracted correctly."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    assistant = _assistant(content=[_text(text="Hi")])
    result_msg = _result(
      duration_ms=2000,
      total_cost_usd=0.01,
      usage={"input_tokens": 200, "output_tokens": 100, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 10},
    )
    response = model._parse_provider_response(([assistant], result_msg))
    assert response.response_usage is not None
    assert response.response_usage.duration == 2.0
    assert response.response_usage.cost == 0.01
    assert response.response_usage.input_tokens == 200
    assert response.response_usage.output_tokens == 100
    assert response.response_usage.total_tokens == 300
    assert response.response_usage.cache_read_tokens == 50
    assert response.response_usage.cache_write_tokens == 10

  def test_no_result_message(self):
    """Missing result message is handled gracefully."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    assistant = _assistant(content=[_text(text="Hello")])
    response = model._parse_provider_response(([assistant], None))
    assert response.content == "Hello"
    assert response.response_usage is None

  def test_result_text_as_fallback(self):
    """Result message text used when no assistant message content."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    result_msg = _result(result="Fallback content")
    response = model._parse_provider_response(([], result_msg))
    assert response.content == "Fallback content"

  def test_error_result(self):
    """Error result is logged but still returned."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    result_msg = _result(is_error=True, result="Something went wrong")
    response = model._parse_provider_response(([], result_msg))
    assert response.content == "Something went wrong"

  def test_empty_response(self):
    """Empty response produces empty ModelResponse."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    response = model._parse_provider_response(([], None))
    assert response.content is None
    assert response.tool_calls == []


# ---------------------------------------------------------------------------
# Test: Tool call parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeToolCallParsing:
  """Verify ToolUseBlock → ToolCallDict conversion."""

  def test_single_tool_call(self):
    """Single ToolUseBlock converts to ToolCallDict."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    blocks = [_tool_use(tool_id="tc_1", name="mcp__definable__search", tool_input={"query": "test"})]
    result = model._parse_tool_use_blocks(blocks)
    assert len(result) == 1
    assert result[0]["id"] == "tc_1"
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "search"
    assert json.loads(result[0]["function"]["arguments"]) == {"query": "test"}

  def test_mcp_prefix_stripping(self):
    """MCP prefix 'mcp__definable__' is stripped from tool names."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    blocks = [_tool_use(tool_id="tc_1", name="mcp__definable__greet", tool_input={"name": "Alice"})]
    result = model._parse_tool_use_blocks(blocks)
    assert result[0]["function"]["name"] == "greet"

  def test_no_prefix_preserved(self):
    """Tool names without MCP prefix are preserved."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    blocks = [_tool_use(tool_id="tc_1", name="custom_tool", tool_input={})]
    result = model._parse_tool_use_blocks(blocks)
    assert result[0]["function"]["name"] == "custom_tool"

  def test_multiple_tool_calls(self):
    """Multiple ToolUseBlocks all convert."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    blocks = [
      _tool_use(tool_id="tc_1", name="mcp__definable__search", tool_input={"q": "a"}),
      _tool_use(tool_id="tc_2", name="mcp__definable__calculate", tool_input={"x": 1}),
    ]
    result = model._parse_tool_use_blocks(blocks)
    assert len(result) == 2
    assert result[0]["function"]["name"] == "search"
    assert result[1]["function"]["name"] == "calculate"

  def test_mixed_blocks(self):
    """Non-ToolUseBlock items are skipped."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    blocks = [
      _text(text="thinking..."),
      _tool_use(tool_id="tc_1", name="mcp__definable__search", tool_input={"q": "test"}),
    ]
    result = model._parse_tool_use_blocks(blocks)
    assert len(result) == 1

  def test_empty_input(self):
    """Tool with None/empty input produces empty JSON."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    blocks = [_tool_use(tool_id="tc_1", name="mcp__definable__noop", tool_input={})]
    result = model._parse_tool_use_blocks(blocks)
    assert result[0]["function"]["arguments"] == "{}"

  def test_tool_calls_in_response(self):
    """ToolUseBlocks in assistant message populate response.tool_calls."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    assistant = _assistant(
      content=[
        _text(text="Let me search."),
        _tool_use(tool_id="tc_1", name="mcp__definable__search", tool_input={"q": "test"}),
      ]
    )
    response = model._parse_provider_response(([assistant], _result()))
    assert response.content == "Let me search."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["function"]["name"] == "search"


# ---------------------------------------------------------------------------
# Test: Stream event parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeStreamParsing:
  """Verify _parse_provider_response_delta handles stream events."""

  def test_text_delta_event(self):
    """text_delta stream event populates content."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    event = _stream_event(
      event={
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "Hello"},
      }
    )
    response = model._parse_provider_response_delta(event)
    assert response.content == "Hello"

  def test_thinking_delta_event(self):
    """thinking_delta stream event populates reasoning_content."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    event = _stream_event(
      event={
        "type": "content_block_delta",
        "delta": {"type": "thinking_delta", "thinking": "Hmm..."},
      }
    )
    response = model._parse_provider_response_delta(event)
    assert response.reasoning_content == "Hmm..."

  def test_unknown_event(self):
    """Unknown event types produce empty response."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    event = _stream_event(event={"type": "message_start"})
    response = model._parse_provider_response_delta(event)
    assert response.content is None
    assert response.reasoning_content is None

  def test_assistant_message_in_stream(self):
    """Full AssistantMessage in stream is parsed completely."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    amsg = _assistant(
      content=[
        _text(text="Complete response"),
        _tool_use(tool_id="tc_1", name="mcp__definable__search", tool_input={"q": "x"}),
      ]
    )
    response = model._parse_provider_response_delta(amsg)
    assert response.content == "Complete response"
    assert len(response.tool_calls) == 1


# ---------------------------------------------------------------------------
# Test: Provider registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeProviderRegistration:
  """Verify string resolution 'claude-code/...' works."""

  def test_resolve_model_string(self):
    """'claude-code/sonnet' resolves to ClaudeCode(id='sonnet')."""
    from definable.model.claude_code import ClaudeCode
    from definable.model.utils import resolve_model_string

    model = resolve_model_string("claude-code/sonnet")
    assert isinstance(model, ClaudeCode)
    assert model.id == "sonnet"
    assert model.provider == "ClaudeCode"

  def test_resolve_opus(self):
    """'claude-code/opus' resolves correctly."""
    from definable.model.claude_code import ClaudeCode
    from definable.model.utils import resolve_model_string

    model = resolve_model_string("claude-code/opus")
    assert isinstance(model, ClaudeCode)
    assert model.id == "opus"

  def test_resolve_haiku(self):
    """'claude-code/haiku' resolves correctly."""
    from definable.model.claude_code import ClaudeCode
    from definable.model.utils import resolve_model_string

    model = resolve_model_string("claude-code/haiku")
    assert isinstance(model, ClaudeCode)
    assert model.id == "haiku"

  def test_lazy_import_from_model(self):
    """ClaudeCode is importable from definable.model."""
    from definable.model import ClaudeCode

    assert ClaudeCode is not None
    assert ClaudeCode.__name__ == "ClaudeCode"


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeErrorHandling:
  """Verify error propagation and import guards."""

  def test_import_error_hint(self):
    """Missing SDK raises ImportError with install hint."""
    with patch.dict("sys.modules", {"claude_agent_sdk": None}):
      # Force re-evaluation of import guard
      from definable.model.claude_code.chat import _ensure_sdk

      # Simulate SDK not available
      import definable.model.claude_code.chat as chat_module

      original = chat_module._SDK_AVAILABLE
      chat_module._SDK_AVAILABLE = False
      try:
        with pytest.raises(ImportError, match="claude-agent-sdk"):
          _ensure_sdk()
      finally:
        chat_module._SDK_AVAILABLE = original

  @pytest.mark.asyncio
  async def test_sdk_error_becomes_model_provider_error(self):
    """SDK exceptions are wrapped in ModelProviderError."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()

    async def _mock_query(**kwargs):
      raise RuntimeError("CLI not found")
      yield  # type: ignore[unreachable]  # make it an async generator

    with patch("definable.model.claude_code.chat.sdk_query", _mock_query):
      with pytest.raises(ModelProviderError, match="CLI not found"):
        await model.ainvoke(
          messages=[Message(role="user", content="Hello")],
          assistant_message=Message(role="assistant", content=""),
        )

  def test_install_hint_in_provider_map(self):
    """claude-code provider has install hint."""
    from definable.model.utils import _INSTALL_HINTS

    assert "claude-code" in _INSTALL_HINTS
    assert "claude-code" in _INSTALL_HINTS["claude-code"]

  def test_import_error_via_resolve(self):
    """Import error from resolve_model_string includes install hint."""
    from unittest.mock import patch as _patch

    from definable.model.utils import resolve_model_string

    with _patch("definable.model.utils.import_module", side_effect=ImportError("No module")):
      with pytest.raises(ImportError, match=r'pip install "definable\[claude-code\]"'):
        resolve_model_string("claude-code/sonnet")


# ---------------------------------------------------------------------------
# Test: Thinking fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeThinkingFallback:
  """Verify streaming falls back to buffered when thinking is enabled."""

  @pytest.mark.asyncio
  async def test_streaming_with_thinking_falls_back(self):
    """ainvoke_stream yields a single buffered result when thinking is on."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(thinking={"type": "enabled", "budget_tokens": 5000})

    # Mock ainvoke to return a simple response
    mock_response = ModelResponse(content="buffered result", role="assistant")

    with patch.object(model, "ainvoke", new_callable=AsyncMock, return_value=mock_response):
      results = []
      async for chunk in model.ainvoke_stream(
        messages=[Message(role="user", content="Hello")],
        assistant_message=Message(role="assistant", content=""),
      ):
        results.append(chunk)

      assert len(results) == 1
      assert results[0].content == "buffered result"


# ---------------------------------------------------------------------------
# Test: Tools streaming fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeToolsStreamingFallback:
  """Verify streaming falls back to buffered when tools are present."""

  @pytest.mark.asyncio
  async def test_streaming_with_tools_falls_back(self):
    """ainvoke_stream yields a single buffered result when tools are provided."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()

    tool_call_response = ModelResponse(
      content="I'll search for that.",
      role="assistant",
      tool_calls=[
        {
          "type": "function",
          "function": {"name": "search", "arguments": json.dumps({"q": "test"})},
        }
      ],
    )

    with patch.object(model, "ainvoke", new_callable=AsyncMock, return_value=tool_call_response) as mock_ainvoke:
      results = []
      async for chunk in model.ainvoke_stream(
        messages=[Message(role="user", content="Search for test")],
        assistant_message=Message(role="assistant", content=""),
        tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
      ):
        results.append(chunk)

      assert len(results) == 1
      assert results[0].content == "I'll search for that."
      assert results[0].tool_calls is not None
      assert len(results[0].tool_calls) == 1
      mock_ainvoke.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_streaming_without_tools_does_not_fallback(self):
    """ainvoke_stream streams via SDK when no tools are provided (no fallback)."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()

    assistant_msg = _assistant([_text("hello world")])

    async def _fake_stream(*a, **kw):
      yield assistant_msg

    with patch.object(model, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
      with patch("definable.model.claude_code.chat.sdk_query", side_effect=_fake_stream):
        results = []
        async for chunk in model.ainvoke_stream(
          messages=[Message(role="user", content="Hi")],
          assistant_message=Message(role="assistant", content=""),
        ):
          results.append(chunk)

        # Should have streamed, NOT fallen back to ainvoke
        mock_ainvoke.assert_not_awaited()
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Test: Structured output
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeStructuredOutput:
  """Verify structured output handling."""

  def test_build_output_format_pydantic(self):
    """Pydantic model generates json_schema output format."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    result = model._build_output_format(MockOutputModel)
    assert result is not None
    assert result["type"] == "json_schema"
    assert "properties" in result["schema"]
    assert "name" in result["schema"]["properties"]
    assert "age" in result["schema"]["properties"]

  def test_build_output_format_dict(self):
    """Dict format is passed through."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    fmt = {"type": "json_schema", "schema": {"type": "object"}}
    result = model._build_output_format(fmt)
    assert result == fmt

  def test_build_output_format_json_object_rejected(self):
    """json_object type returns None (not supported by SDK)."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    result = model._build_output_format({"type": "json_object"})
    assert result is None

  def test_build_output_format_none(self):
    """None input returns None."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    result = model._build_output_format(None)
    assert result is None

  def test_structured_output_in_response(self):
    """Structured output from ResultMessage is parsed into response.parsed."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    result_msg = _result(
      structured_output={"name": "Alice", "age": 30},
    )
    assistant = _assistant(content=[_text(text='{"name": "Alice", "age": 30}')])
    response = model._parse_provider_response(
      ([assistant], result_msg),
      response_format=MockOutputModel,
    )
    assert response.parsed is not None
    assert isinstance(response.parsed, MockOutputModel)
    assert response.parsed.name == "Alice"
    assert response.parsed.age == 30


# ---------------------------------------------------------------------------
# Test: Serialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeSerialization:
  """Verify to_dict output."""

  def test_to_dict_defaults(self):
    """Default model serializes with core fields."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    d = model.to_dict()
    assert d["name"] == "ClaudeCode"
    assert d["id"] == "sonnet"
    assert d["provider"] == "ClaudeCode"
    assert d["permission_mode"] == "bypassPermissions"
    assert "thinking" not in d
    assert "cli_path" not in d

  def test_to_dict_with_thinking(self):
    """Thinking config appears in serialized dict."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(thinking={"type": "adaptive"})
    d = model.to_dict()
    assert d["thinking"] == {"type": "adaptive"}

  def test_to_dict_with_cli_path(self):
    """CLI path appears in serialized dict."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(cli_path="/usr/bin/claude")
    d = model.to_dict()
    assert d["cli_path"] == "/usr/bin/claude"


# ---------------------------------------------------------------------------
# Test: ainvoke integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeAInvoke:
  """Verify ainvoke end-to-end with mocked SDK."""

  @pytest.mark.asyncio
  async def test_ainvoke_text_response(self):
    """ainvoke returns text content from SDK response."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()

    async def mock_query(*, prompt, options=None):
      yield _assistant(content=[_text(text="Hello!")])
      yield _result(duration_ms=1000, usage={"input_tokens": 50, "output_tokens": 20})

    with patch("definable.model.claude_code.chat.sdk_query", mock_query):
      response = await model.ainvoke(
        messages=[Message(role="user", content="Hi")],
        assistant_message=Message(role="assistant", content=""),
      )
      assert response.content == "Hello!"
      assert response.response_usage is not None
      assert response.response_usage.input_tokens == 50

  @pytest.mark.asyncio
  async def test_ainvoke_with_tool_calls(self):
    """ainvoke returns tool calls from SDK response."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()

    async def mock_query(*, prompt, options=None):
      yield _assistant(
        content=[
          _text(text="Let me search."),
          _tool_use(tool_id="tc_1", name="mcp__definable__search", tool_input={"query": "test"}),
        ]
      )
      yield _result()

    with patch("definable.model.claude_code.chat.sdk_query", mock_query):
      response = await model.ainvoke(
        messages=[Message(role="user", content="Search for test")],
        assistant_message=Message(role="assistant", content=""),
        tools=[{"type": "function", "function": {"name": "search", "description": "Search", "parameters": {}}}],
      )
      assert len(response.tool_calls) == 1
      assert response.tool_calls[0]["function"]["name"] == "search"

  @pytest.mark.asyncio
  async def test_ainvoke_with_system_prompt(self):
    """System messages are extracted and passed as system_prompt."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    captured_options = {}

    async def mock_query(*, prompt, options=None):
      captured_options["opts"] = options
      yield _assistant(content=[_text(text="OK")])
      yield _result()

    with patch("definable.model.claude_code.chat.sdk_query", mock_query):
      await model.ainvoke(
        messages=[
          Message(role="system", content="Be concise."),
          Message(role="user", content="Hello"),
        ],
        assistant_message=Message(role="assistant", content=""),
      )
      assert captured_options["opts"].system_prompt == "Be concise."

  @pytest.mark.asyncio
  async def test_ainvoke_multi_turn(self):
    """Multi-turn conversation is serialized into prompt."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    captured_prompts = {}

    async def mock_query(*, prompt, options=None):
      captured_prompts["prompt"] = prompt
      yield _assistant(content=[_text(text="4")])
      yield _result()

    with patch("definable.model.claude_code.chat.sdk_query", mock_query):
      await model.ainvoke(
        messages=[
          Message(role="user", content="What is 2+2?"),
          Message(role="assistant", content="Let me calculate."),
          Message(role="user", content="Please answer."),
        ],
        assistant_message=Message(role="assistant", content=""),
      )
      prompt = captured_prompts["prompt"]
      assert "What is 2+2?" in prompt
      assert "[Assistant]: Let me calculate." in prompt
      assert "Please answer." in prompt


# ---------------------------------------------------------------------------
# Test: Session isolation (setting_sources)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeCodeSettingSources:
  """Verify setting_sources controls CLI session isolation."""

  def test_default_setting_sources_empty(self):
    """Default setting_sources is [] (amnesiac — no CLI settings loaded)."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    assert model.setting_sources == []

  def test_setting_sources_none_not_in_options(self):
    """setting_sources=None omits the key from options (CLI uses defaults)."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(setting_sources=None)
    opts = model._build_options(system_prompt_text="", stream=False)
    assert not hasattr(opts, "setting_sources") or opts.setting_sources is None

  def test_setting_sources_empty_in_options(self):
    """Default [] is forwarded to options (blocks all settings)."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    opts = model._build_options(system_prompt_text="", stream=False)
    assert opts.setting_sources == []

  def test_setting_sources_custom_list(self):
    """Custom list is forwarded to options."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(setting_sources=["project", "user"])
    opts = model._build_options(system_prompt_text="", stream=False)
    assert opts.setting_sources == ["project", "user"]

  def test_setting_sources_project_only(self):
    """["project"] loads only project CLAUDE.md."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(setting_sources=["project"])
    opts = model._build_options(system_prompt_text="", stream=False)
    assert opts.setting_sources == ["project"]

  def test_setting_sources_in_to_dict(self):
    """setting_sources appears in serialized dict when set."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(setting_sources=["project"])
    d = model.to_dict()
    assert d["setting_sources"] == ["project"]

  def test_setting_sources_empty_in_to_dict(self):
    """Default [] appears in serialized dict."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    d = model.to_dict()
    assert d["setting_sources"] == []

  def test_setting_sources_none_not_in_to_dict(self):
    """setting_sources=None is omitted from serialized dict."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode(setting_sources=None)
    d = model.to_dict()
    assert "setting_sources" not in d

  @pytest.mark.asyncio
  async def test_setting_sources_in_ainvoke(self):
    """setting_sources is forwarded through ainvoke → options."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()  # default []
    captured_options: Dict[str, Any] = {}

    async def mock_query(*, prompt, options=None):
      captured_options["opts"] = options
      yield _assistant(content=[_text(text="OK")])
      yield _result()

    with patch("definable.model.claude_code.chat.sdk_query", mock_query):
      await model.ainvoke(
        messages=[Message(role="user", content="Hello")],
        assistant_message=Message(role="assistant", content=""),
      )
      assert captured_options["opts"].setting_sources == []

  def test_isolation_default_amnesiac(self):
    """Default ClaudeCode produces options with empty setting_sources (amnesiac)."""
    from definable.model.claude_code import ClaudeCode

    model = ClaudeCode()
    opts = model._build_options(system_prompt_text="You are helpful.", stream=False)
    assert opts.setting_sources == []
    assert opts.max_turns == 1
    assert opts.system_prompt == "You are helpful."
