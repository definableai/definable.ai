"""ClaudeCode model — wraps the claude-agent-sdk (Claude Code CLI).

Unlike the `Claude` model (direct Anthropic Messages API), this model runs
Claude Code CLI under the hood via the Agent SDK.  It provides session-aware,
higher-level capabilities while integrating with Definable's agent loop
through an MCP tool bridge.

Phase 1: stateless `query()` per invoke, MCP tool bridge, structured output.
"""

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Type, Union

if sys.version_info < (3, 11):
  from exceptiongroup import ExceptionGroup

from pydantic import BaseModel

from definable.run.agent import RunOutput
from definable.exceptions import ModelProviderError
from definable.model.base import Model
from definable.model.message import Message
from definable.model.metrics import Metrics
from definable.model.response import ModelResponse
from definable.utils.log import log_debug, log_error, log_warning

try:
  from claude_agent_sdk import (
    AssistantMessage as SdkAssistantMessage,
    ClaudeAgentOptions,
    ResultMessage as SdkResultMessage,
    SdkMcpTool,
    TextBlock as SdkTextBlock,
    ThinkingBlock as SdkThinkingBlock,
    ToolUseBlock as SdkToolUseBlock,
    create_sdk_mcp_server,
    query as sdk_query,
  )
  from claude_agent_sdk.types import StreamEvent as SdkStreamEvent

  _SDK_AVAILABLE = True
except ImportError:
  _SDK_AVAILABLE = False


def _ensure_sdk() -> None:
  """Raise a helpful ImportError if the SDK is not installed."""
  if not _SDK_AVAILABLE:
    raise ImportError('`claude-agent-sdk` not installed. Install it with: pip install "definable[claude-code]"')


# ---------------------------------------------------------------------------
# MCP bridge — convert Definable tool dicts to SDK MCP tools
# ---------------------------------------------------------------------------


def _build_mcp_server(tools: List[Dict[str, Any]]) -> Any:
  """Convert Definable tool dicts to an in-process SDK MCP server.

  Each tool becomes a no-op handler because we set ``max_turns=1`` so
  the SDK never actually executes them — it returns ``ToolUseBlock``
  requests that Definable's agent loop executes.
  """
  _ensure_sdk()
  sdk_tools: list[SdkMcpTool[Any]] = []

  for tool_def in tools:
    func = tool_def.get("function", {})
    name = func.get("name", "")
    desc = func.get("description", "")
    params = func.get("parameters", {})

    # Build a no-op handler for this tool.
    async def _noop(args: Any) -> dict:  # noqa: ARG001
      return {"content": [{"type": "text", "text": ""}]}

    sdk_tools.append(
      SdkMcpTool(
        name=name,
        description=desc,
        input_schema=params or {"type": "object", "properties": {}},
        handler=_noop,
      )
    )

  return create_sdk_mcp_server(name="definable", version="1.0.0", tools=sdk_tools)


# ---------------------------------------------------------------------------
# Message serialization helpers
# ---------------------------------------------------------------------------

_MCP_PREFIX = "mcp__definable__"


def _extract_system_prompt(messages: List[Message]) -> str:
  """Pull all system messages into a single string."""
  parts: list[str] = []
  for m in messages:
    if m.role == "system" and m.content:
      if isinstance(m.content, str):
        parts.append(m.content)
      elif isinstance(m.content, list):
        for block in m.content:
          if isinstance(block, str):
            parts.append(block)
          elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
  return "\n\n".join(parts)


def _messages_to_prompt(messages: List[Message]) -> str:
  """Serialize non-system messages into a prompt string for the SDK.

  The SDK's ``query()`` accepts a single prompt string.  We flatten the
  multi-turn conversation into a readable transcript that preserves role
  structure.
  """
  lines: list[str] = []
  for m in messages:
    if m.role == "system":
      continue
    content = ""
    if isinstance(m.content, str):
      content = m.content
    elif isinstance(m.content, list):
      text_parts: list[str] = []
      for block in m.content:
        if isinstance(block, str):
          text_parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
          text_parts.append(block.get("text", ""))
      content = "\n".join(text_parts)

    if m.role == "tool" and m.tool_call_id:
      lines.append(f"[Tool Result ({m.tool_call_id})]: {content}")
    elif m.role == "assistant":
      # Include tool calls if present
      if m.tool_calls:
        for tc in m.tool_calls:
          fn = tc.get("function", {})
          lines.append(f"[Assistant Tool Call]: {fn.get('name', '')}({fn.get('arguments', '{}')})")
      if content:
        lines.append(f"[Assistant]: {content}")
    elif m.role == "user":
      lines.append(content)

  return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# ClaudeCode model
# ---------------------------------------------------------------------------


@dataclass
class ClaudeCode(Model):
  """Claude Code model — runs Claude Code CLI via the Agent SDK.

  This model wraps ``claude-agent-sdk``'s ``query()`` function with
  ``max_turns=1`` so it returns tool-call requests without executing them.
  Definable's agent loop handles actual tool execution, guardrails, etc.

  Attributes:
    id: SDK model name (``"sonnet"``, ``"opus"``, ``"haiku"``).
    thinking: SDK thinking config dict, e.g.
      ``{"type": "enabled", "budget_tokens": 10000}`` or
      ``{"type": "adaptive"}`` or ``{"type": "disabled"}``.
    cli_path: Custom path to the Claude Code CLI binary.
    cwd: Working directory for the CLI process.
    permission_mode: SDK permission mode (default: ``"bypassPermissions"``).
    setting_sources: Controls which CLI settings are loaded. Default ``[]``
      (amnesiac — no user/project settings loaded). Set to ``None`` to use
      CLI defaults (loads everything). Set to ``["project"]`` to load only
      the project CLAUDE.md.
  """

  id: str = "sonnet"
  name: str = "ClaudeCode"
  provider: str = "ClaudeCode"

  # SDK options
  permission_mode: str = "bypassPermissions"
  thinking: Optional[Dict[str, Any]] = None
  cli_path: Optional[str] = None
  cwd: Optional[str] = None
  setting_sources: Optional[List[str]] = field(default_factory=list)

  def __post_init__(self) -> None:
    _ensure_sdk()
    super().__post_init__()

  # ---------------------------------------------------------------------------
  # Options builder
  # ---------------------------------------------------------------------------

  def _build_options(
    self,
    system_prompt_text: str,
    stream: bool,
    mcp_server: Any = None,
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
  ) -> "ClaudeAgentOptions":
    """Build SDK ``ClaudeAgentOptions`` for a single invoke."""
    kwargs: Dict[str, Any] = {
      "max_turns": 1,
      "permission_mode": self.permission_mode,
    }

    if system_prompt_text:
      kwargs["system_prompt"] = system_prompt_text

    if self.thinking:
      kwargs["thinking"] = self.thinking

    # Streaming: only when thinking is NOT enabled (SDK limitation)
    if stream and not self.thinking:
      kwargs["include_partial_messages"] = True

    if self.cli_path:
      kwargs["cli_path"] = self.cli_path

    if self.cwd:
      kwargs["cwd"] = self.cwd

    if self.setting_sources is not None:
      kwargs["setting_sources"] = self.setting_sources

    if self.id:
      kwargs["model"] = self.id

    # MCP tool bridge
    if mcp_server is not None:
      kwargs["mcp_servers"] = {"definable": mcp_server}
      kwargs["allowed_tools"] = ["mcp__definable__*"]
      kwargs["disallowed_tools"] = ["*"]
    else:
      kwargs["allowed_tools"] = []
      kwargs["disallowed_tools"] = ["*"]

    # Structured output
    output_format = self._build_output_format(response_format)
    if output_format:
      kwargs["output_format"] = output_format

    return ClaudeAgentOptions(**kwargs)

  # ---------------------------------------------------------------------------
  # Structured output
  # ---------------------------------------------------------------------------

  def _build_output_format(
    self,
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
  ) -> Optional[Dict[str, Any]]:
    """Convert ``response_format`` (Pydantic model or dict) to SDK output_format."""
    if response_format is None:
      return None

    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
      schema = response_format.model_json_schema()
      # Ensure additionalProperties is False
      if isinstance(schema, dict) and "additionalProperties" not in schema:
        schema["additionalProperties"] = False
      return {"type": "json_schema", "schema": schema}

    if isinstance(response_format, dict):
      if response_format.get("type") == "json_object":
        return None
      return response_format

    return None  # type: ignore[unreachable]

  # ---------------------------------------------------------------------------
  # Response parsing
  # ---------------------------------------------------------------------------

  def _parse_tool_use_blocks(self, blocks: list) -> list:
    """Convert SDK ``ToolUseBlock`` items to Definable ``ToolCallDict``s."""
    tool_calls: list[Dict[str, Any]] = []
    for block in blocks:
      if isinstance(block, SdkToolUseBlock):
        name = block.name
        if name.startswith(_MCP_PREFIX):
          name = name[len(_MCP_PREFIX) :]
        tool_calls.append({
          "id": block.id,
          "type": "function",
          "function": {
            "name": name,
            "arguments": json.dumps(block.input) if block.input else "{}",
          },
        })
    return tool_calls

  def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
    """Parse collected SDK messages into a ``ModelResponse``.

    ``response`` here is a tuple of ``(assistant_messages, result_message)``
    collected from the ``query()`` async iterator.
    """
    assistant_msgs: list = response[0]
    result_msg: Optional[Any] = response[1]
    response_format = kwargs.get("response_format")

    model_response = ModelResponse()
    model_response.role = "assistant"

    # Process assistant messages
    for amsg in assistant_msgs:
      if not isinstance(amsg, SdkAssistantMessage):
        continue
      for block in amsg.content:
        if isinstance(block, SdkTextBlock):
          if model_response.content is None:
            model_response.content = block.text
          else:
            model_response.content += block.text
        elif isinstance(block, SdkThinkingBlock):
          if model_response.reasoning_content is None:
            model_response.reasoning_content = block.thinking
          else:
            model_response.reasoning_content += block.thinking
          model_response.provider_data = model_response.provider_data or {}
          model_response.provider_data["signature"] = block.signature

      # Collect tool calls
      tc = self._parse_tool_use_blocks(amsg.content)
      if tc:
        model_response.tool_calls.extend(tc)

    # Process result message for metrics + structured output
    if result_msg is not None and isinstance(result_msg, SdkResultMessage):
      metrics = Metrics()
      metrics.duration = result_msg.duration_ms / 1000.0
      if result_msg.usage:
        metrics.input_tokens = result_msg.usage.get("input_tokens", 0)
        metrics.output_tokens = result_msg.usage.get("output_tokens", 0)
        metrics.total_tokens = metrics.input_tokens + metrics.output_tokens
        metrics.cache_read_tokens = result_msg.usage.get("cache_read_input_tokens", 0)
        metrics.cache_write_tokens = result_msg.usage.get("cache_creation_input_tokens", 0)
      if result_msg.total_cost_usd is not None:
        metrics.cost = result_msg.total_cost_usd
      model_response.response_usage = metrics

      # Structured output
      if result_msg.structured_output is not None:
        if response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel):
          try:
            if isinstance(result_msg.structured_output, dict):
              model_response.parsed = response_format.model_validate(result_msg.structured_output)
            elif isinstance(result_msg.structured_output, str):
              model_response.parsed = response_format.model_validate_json(result_msg.structured_output)
            else:
              model_response.parsed = result_msg.structured_output
          except Exception as e:
            log_warning(f"Failed to validate structured output: {e}")
            model_response.parsed = result_msg.structured_output
        else:
          model_response.parsed = result_msg.structured_output

      # If result text is present and no content from assistant messages
      if result_msg.result and model_response.content is None:
        model_response.content = result_msg.result

      # Check for errors
      if result_msg.is_error:
        log_warning(f"ClaudeCode query returned an error: {result_msg.result}")

    return model_response

  def _parse_provider_response_delta(self, response: Any, **kwargs: Any) -> ModelResponse:
    """Parse a single SDK ``StreamEvent`` into a delta ``ModelResponse``."""
    model_response = ModelResponse()

    if isinstance(response, SdkStreamEvent):
      event_data = response.event
      event_type = event_data.get("type", "")

      if event_type == "content_block_delta":
        delta = event_data.get("delta", {})
        delta_type = delta.get("type", "")
        if delta_type == "text_delta":
          model_response.content = delta.get("text", "")
        elif delta_type == "thinking_delta":
          model_response.reasoning_content = delta.get("thinking", "")
    elif isinstance(response, SdkAssistantMessage):
      # Full assistant message in stream — parse as complete
      for block in response.content:
        if isinstance(block, SdkTextBlock):
          if model_response.content is None:
            model_response.content = block.text
          else:
            model_response.content += block.text
        elif isinstance(block, SdkThinkingBlock):
          if model_response.reasoning_content is None:
            model_response.reasoning_content = block.thinking
          else:
            model_response.reasoning_content += block.thinking
      tc = self._parse_tool_use_blocks(response.content)
      if tc:
        model_response.tool_calls.extend(tc)

    return model_response

  # ---------------------------------------------------------------------------
  # Core invoke methods
  # ---------------------------------------------------------------------------

  async def ainvoke(
    self,
    messages: List[Message],
    assistant_message: Message,
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    run_response: Optional[RunOutput] = None,
    compress_tool_results: bool = False,
  ) -> ModelResponse:
    """Invoke Claude Code via the SDK ``query()`` function."""
    _ensure_sdk()

    try:
      if run_response and run_response.metrics:
        run_response.metrics.set_time_to_first_token()

      system_prompt_text = _extract_system_prompt(messages)
      prompt = _messages_to_prompt(messages)

      mcp_server = None
      if tools:
        mcp_server = _build_mcp_server(tools)

      options = self._build_options(
        system_prompt_text=system_prompt_text,
        stream=False,
        mcp_server=mcp_server,
        response_format=response_format,
      )

      assistant_message.metrics.start_timer()

      # Collect all messages from the async iterator.
      # The SDK has a known race condition: when max_turns=1, the CLI subprocess
      # exits after producing output, but a background TaskGroup task may still
      # try to write to the dead transport.  We catch that ExceptionGroup and
      # use whatever results were already collected.
      assistant_msgs: list = []
      result_msg = None

      try:
        async for msg in sdk_query(prompt=prompt, options=options):
          if isinstance(msg, SdkAssistantMessage):
            assistant_msgs.append(msg)
          elif isinstance(msg, SdkResultMessage):
            result_msg = msg
      except ExceptionGroup as eg:
        # Filter out benign transport-cleanup errors from the SDK.
        real_errors = [e for e in eg.exceptions if not (hasattr(e, "__class__") and "CLIConnectionError" in type(e).__name__)]
        if real_errors:
          raise ExceptionGroup(eg.message, real_errors) from eg
        # All sub-exceptions were transport cleanup noise — continue with
        # whatever messages we already collected.
        log_warning(f"ClaudeCode: suppressed SDK transport cleanup error ({len(eg.exceptions)} sub-exceptions)")

      assistant_message.metrics.stop_timer()

      if not assistant_msgs and result_msg is None:
        raise ModelProviderError(
          message="ClaudeCode SDK returned no messages",
          model_name=self.name,
          model_id=self.id,
        )

      model_response = self._parse_provider_response(
        (assistant_msgs, result_msg),
        response_format=response_format,
      )
      return model_response

    except ImportError:
      raise
    except ModelProviderError:
      raise
    except Exception as e:
      log_error(f"ClaudeCode SDK error: {e}")
      raise ModelProviderError(
        message=str(e),
        model_name=self.name,
        model_id=self.id,
      ) from e

  def invoke(
    self,
    messages: List[Message],
    assistant_message: Message,
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    run_response: Optional[RunOutput] = None,
    compress_tool_results: bool = False,
  ) -> ModelResponse:
    """Synchronous invoke — delegates to ``ainvoke`` via ``asyncio.run``."""
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      import concurrent.futures

      with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(
          asyncio.run,
          self.ainvoke(
            messages=messages,
            assistant_message=assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
            compress_tool_results=compress_tool_results,
          ),
        )
        return future.result()
    else:
      return asyncio.run(
        self.ainvoke(
          messages=messages,
          assistant_message=assistant_message,
          response_format=response_format,
          tools=tools,
          tool_choice=tool_choice,
          run_response=run_response,
          compress_tool_results=compress_tool_results,
        )
      )

  async def ainvoke_stream(
    self,
    messages: List[Message],
    assistant_message: Message,
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    run_response: Optional[RunOutput] = None,
    compress_tool_results: bool = False,
  ) -> AsyncIterator[ModelResponse]:
    """Async streaming invoke.

    When thinking is enabled, falls back to buffered mode (SDK limitation).
    """
    _ensure_sdk()

    # Thinking + streaming are incompatible — fall back to buffered
    if self.thinking:
      log_debug("ClaudeCode: thinking enabled — falling back to buffered invoke for streaming")
      result = await self.ainvoke(
        messages=messages,
        assistant_message=assistant_message,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        run_response=run_response,
        compress_tool_results=compress_tool_results,
      )
      yield result
      return

    # Tools + streaming: SDK tool_use deltas are not fully parsed — fall back to buffered
    if tools:
      log_debug("ClaudeCode: tools present — falling back to buffered invoke for streaming")
      result = await self.ainvoke(
        messages=messages,
        assistant_message=assistant_message,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        run_response=run_response,
        compress_tool_results=compress_tool_results,
      )
      yield result
      return

    try:
      if run_response and run_response.metrics:
        run_response.metrics.set_time_to_first_token()

      system_prompt_text = _extract_system_prompt(messages)
      prompt = _messages_to_prompt(messages)

      mcp_server = None
      if tools:
        mcp_server = _build_mcp_server(tools)

      options = self._build_options(
        system_prompt_text=system_prompt_text,
        stream=True,
        mcp_server=mcp_server,
        response_format=response_format,
      )

      assistant_message.metrics.start_timer()

      try:
        async for msg in sdk_query(prompt=prompt, options=options):
          if isinstance(msg, SdkStreamEvent):
            yield self._parse_provider_response_delta(msg)
          elif isinstance(msg, SdkAssistantMessage):
            yield self._parse_provider_response_delta(msg)
          elif isinstance(msg, SdkResultMessage):
            # Final message — emit metrics
            final = ModelResponse()
            metrics = Metrics()
            metrics.duration = msg.duration_ms / 1000.0
            if msg.usage:
              metrics.input_tokens = msg.usage.get("input_tokens", 0)
              metrics.output_tokens = msg.usage.get("output_tokens", 0)
              metrics.total_tokens = metrics.input_tokens + metrics.output_tokens
            if msg.total_cost_usd is not None:
              metrics.cost = msg.total_cost_usd
            final.response_usage = metrics
            if msg.structured_output is not None:
              final.parsed = msg.structured_output
            yield final
      except ExceptionGroup as eg:
        real_errors = [e for e in eg.exceptions if not (hasattr(e, "__class__") and "CLIConnectionError" in type(e).__name__)]
        if real_errors:
          raise ExceptionGroup(eg.message, real_errors) from eg
        log_warning(f"ClaudeCode: suppressed SDK transport cleanup error ({len(eg.exceptions)} sub-exceptions)")

      assistant_message.metrics.stop_timer()

    except ImportError:
      raise
    except ModelProviderError:
      raise
    except Exception as e:
      log_error(f"ClaudeCode SDK streaming error: {e}")
      raise ModelProviderError(
        message=str(e),
        model_name=self.name,
        model_id=self.id,
      ) from e

  def invoke_stream(
    self,
    messages: List[Message],
    assistant_message: Message,
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    run_response: Optional[RunOutput] = None,
    compress_tool_results: bool = False,
  ) -> Iterator[ModelResponse]:
    """Synchronous streaming invoke — wraps async generator."""

    async def _collect() -> list:
      results: list[ModelResponse] = []
      async for chunk in self.ainvoke_stream(
        messages=messages,
        assistant_message=assistant_message,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        run_response=run_response,
        compress_tool_results=compress_tool_results,
      ):
        results.append(chunk)
      return results

    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      import concurrent.futures

      with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, _collect())
        results = future.result()
    else:
      results = asyncio.run(_collect())

    yield from results

  # ---------------------------------------------------------------------------
  # Serialization
  # ---------------------------------------------------------------------------

  def to_dict(self) -> Dict[str, Any]:
    model_dict = super().to_dict()
    if self.thinking:
      model_dict["thinking"] = self.thinking
    if self.cli_path:
      model_dict["cli_path"] = self.cli_path
    if self.cwd:
      model_dict["cwd"] = self.cwd
    if self.setting_sources is not None:
      model_dict["setting_sources"] = self.setting_sources
    model_dict["permission_mode"] = self.permission_mode
    return {k: v for k, v in model_dict.items() if v is not None}
