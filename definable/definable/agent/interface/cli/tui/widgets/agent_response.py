"""Agent response block — streaming markdown rendering."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Markdown, Static
from textual.widgets.markdown import MarkdownStream


class AgentResponse(Widget):
  """A streaming agent response in the conversation.

  Uses Textual's MarkdownStream for true progressive rendering —
  fragments are queued and batched automatically, so the UI stays
  responsive even at high token throughput.
  """

  DEFAULT_CSS = """
  AgentResponse {
    margin: 0 0 1 0;
    padding: 0 1;
    height: auto;
  }

  AgentResponse .agent-label {
    width: 4;
    color: $success;
    text-style: bold;
  }

  AgentResponse .agent-body {
    width: 1fr;
    height: auto;
  }

  AgentResponse Markdown {
    margin: 0;
    padding: 0;
  }
  """

  def __init__(self, run_id: str = "") -> None:
    super().__init__()
    self.run_id = run_id
    self._content = ""
    self._markdown: Markdown | None = None
    self._stream: MarkdownStream | None = None
    self._finished = False

  def compose(self) -> ComposeResult:
    with Horizontal():
      yield Static("AI", classes="agent-label")
      with Vertical(classes="agent-body"):
        self._markdown = Markdown("", classes="agent-markdown")
        yield self._markdown

  def on_mount(self) -> None:
    """Initialize the markdown stream on mount."""
    if self._markdown is not None:
      self._stream = Markdown.get_stream(self._markdown)
      self._stream.start()

  async def append_chunk(self, text: str) -> None:
    """Append a streaming chunk of markdown content."""
    self._content += text
    if self._stream is not None:
      await self._stream.write(text)

  async def finish(self) -> None:
    """Mark the response as complete."""
    self._finished = True
    if self._stream is not None:
      await self._stream.stop()
      self._stream = None

  @property
  def content(self) -> str:
    """The full accumulated content."""
    return self._content

  @property
  def finished(self) -> bool:
    return self._finished
