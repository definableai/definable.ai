"""Conversation widget — scrollable container of message blocks."""

from __future__ import annotations

from typing import Optional

from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widget import Widget

from definable.agent.interface.cli.tui.widgets.agent_response import AgentResponse
from definable.agent.interface.cli.tui.widgets.system_message import SystemMessage
from definable.agent.interface.cli.tui.widgets.thinking import ThinkingBlock
from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock
from definable.agent.interface.cli.tui.widgets.user_message import UserMessage


class Conversation(VerticalScroll):
  """Scrollable conversation area containing message blocks.

  Supports:
  - Adding user messages
  - Adding agent responses (with streaming via MarkdownStream)
  - Adding thinking blocks (with RichLog streaming)
  - Adding tool call blocks (with auto-expand config)
  - Auto-scrolling to bottom on new content
  - Block navigation with Alt+Up/Down
  """

  BINDINGS = [
    Binding("alt+up", "previous_block", "Previous block", show=False),
    Binding("alt+down", "next_block", "Next block", show=False),
  ]

  DEFAULT_CSS = """
  Conversation {
    height: 1fr;
    padding: 1 0;
    scrollbar-size: 1 1;
  }
  """

  def __init__(self, tools_expand: str = "success") -> None:
    super().__init__()
    self._current_response: Optional[AgentResponse] = None
    self._current_thinking: Optional[ThinkingBlock] = None
    self._tool_calls: dict[str, ToolCallBlock] = {}
    self._auto_scroll = True
    self._focused_block_index: int = -1
    self._tools_expand = tools_expand

  # --- Message management ---

  async def add_user_message(self, text: str) -> UserMessage:
    """Add a user message block."""
    msg = UserMessage(text)
    await self.mount(msg)
    self._scroll_to_bottom()
    return msg

  async def start_response(self, run_id: str = "") -> AgentResponse:
    """Start a new agent response block for streaming."""
    response = AgentResponse(run_id=run_id)
    self._current_response = response
    await self.mount(response)
    self._scroll_to_bottom()
    return response

  async def append_to_response(self, text: str) -> None:
    """Append text to the current streaming response."""
    if self._current_response is not None:
      await self._current_response.append_chunk(text)
      self._scroll_to_bottom()

  async def finish_response(self) -> None:
    """Mark the current response as complete."""
    if self._current_response is not None:
      await self._current_response.finish()
      self._current_response = None

  async def start_thinking(self) -> ThinkingBlock:
    """Add a new thinking block."""
    thinking = ThinkingBlock()
    self._current_thinking = thinking
    await self.mount(thinking)
    self._scroll_to_bottom()
    return thinking

  async def append_to_thinking(self, text: str) -> None:
    """Append text to the current thinking block."""
    if self._current_thinking is not None:
      await self._current_thinking.append_chunk(text)

  async def finish_thinking(self) -> None:
    """Mark thinking as complete."""
    if self._current_thinking is not None:
      await self._current_thinking.finish()
      self._current_thinking = None

  async def add_tool_call(
    self,
    tool_name: str,
    arguments: str = "",
    call_id: str = "",
  ) -> ToolCallBlock:
    """Add a tool call block with auto-expand config."""
    block = ToolCallBlock(
      tool_name=tool_name,
      arguments=arguments,
      call_id=call_id,
      tools_expand=self._tools_expand,
    )
    self._tool_calls[call_id] = block
    await self.mount(block)
    self._scroll_to_bottom()
    return block

  def complete_tool_call(
    self,
    call_id: str,
    result: str = "",
    error: Optional[str] = None,
    duration_ms: Optional[float] = None,
  ) -> None:
    """Mark a tool call as completed."""
    block = self._tool_calls.get(call_id)
    if block is not None:
      block.complete(result=result, error=error, duration_ms=duration_ms)
      self._scroll_to_bottom()

  async def add_system_message(self, content: str, label: str = "Sys") -> SystemMessage:
    """Add a system message block (command output, notifications)."""
    msg = SystemMessage(content=content, label=label)
    await self.mount(msg)
    self._scroll_to_bottom()
    return msg

  async def rebuild_from_messages(self, messages: list) -> None:
    """Reconstruct the conversation from a list of Message objects.

    Used when switching sessions or loading history. Renders
    user and assistant messages as blocks. Tool calls and system
    messages are shown as system messages.
    """
    await self.clear_conversation()
    for msg in messages:
      role = getattr(msg, "role", None) or ""
      content = str(getattr(msg, "content", None) or "")
      if not content:
        continue
      if role == "user":
        await self.add_user_message(content)
      elif role == "assistant":
        response = await self.start_response()
        await response.append_chunk(content)
        await response.finish()
        self._current_response = None
      elif role == "tool":
        tool_name = getattr(msg, "tool_name", None) or "tool"
        preview = content[:500] + "\u2026" if len(content) > 500 else content
        await self.add_system_message(f"[{tool_name}] {preview}")
      elif role == "system":
        preview = content[:500] + "\u2026" if len(content) > 500 else content
        await self.add_system_message(preview, label="Sys")

  async def clear_conversation(self) -> None:
    """Remove all blocks from the conversation."""
    await self.remove_children()
    self._current_response = None
    self._current_thinking = None
    self._tool_calls.clear()
    self._focused_block_index = -1

  # --- Block navigation ---

  def _get_blocks(self) -> list[Widget]:
    """Get all navigable blocks."""
    return list(self.children)

  def action_previous_block(self) -> None:
    """Navigate to the previous block."""
    blocks = self._get_blocks()
    if not blocks:
      return
    self._focused_block_index = max(0, self._focused_block_index - 1)
    blocks[self._focused_block_index].scroll_visible()

  def action_next_block(self) -> None:
    """Navigate to the next block."""
    blocks = self._get_blocks()
    if not blocks:
      return
    self._focused_block_index = min(len(blocks) - 1, self._focused_block_index + 1)
    blocks[self._focused_block_index].scroll_visible()

  # --- Scrolling ---

  def _scroll_to_bottom(self) -> None:
    """Scroll to the bottom if auto-scroll is active."""
    if self._auto_scroll:
      self.scroll_end(animate=False)

  def on_scroll_up(self) -> None:
    """User scrolled up — disable auto-scroll."""
    self._auto_scroll = False

  def watch_scroll_y(self, old_value: float, new_value: float) -> None:
    """Re-enable auto-scroll when user scrolls to bottom."""
    if self.max_scroll_y > 0 and new_value >= self.max_scroll_y - 2:
      self._auto_scroll = True
