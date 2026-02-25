"""Agent response block — streaming markdown rendering."""

from __future__ import annotations

from textual.widgets import Markdown
from textual.widgets.markdown import MarkdownStream


class AgentResponse(Markdown):
  """A streaming agent response in the conversation.

  Extends Markdown directly for proper layout: stream rendering.
  Uses Textual's MarkdownStream for progressive rendering —
  fragments are queued and batched automatically, so the UI stays
  responsive even at high token throughput.
  """

  DEFAULT_CSS = """
  AgentResponse {
    min-height: 1;
    padding: 0 1 0 0;
    overflow-x: auto;
    layout: stream;

    MarkdownBlock:last-child {
      margin-bottom: 0;
    }
  }
  """

  def __init__(self, run_id: str = "", markdown: str | None = None) -> None:
    super().__init__(markdown)
    self.run_id = run_id
    self._content = ""
    self._stream: MarkdownStream | None = None
    self._finished = False

  @property
  def stream(self) -> MarkdownStream:
    """Lazy-initialize the markdown stream."""
    if self._stream is None:
      self._stream = self.get_stream(self)
    return self._stream

  async def append_chunk(self, text: str) -> None:
    """Append a streaming chunk of markdown content."""
    self._content += text
    await self.stream.write(text)

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
