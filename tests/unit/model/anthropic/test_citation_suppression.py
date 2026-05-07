"""format_messages cite_documents flag suppresses citations.enabled when output_format active."""

from definable.media import File
from definable.model.message import Message
from definable.utils.claude import format_messages


def test_url_document_includes_citations_by_default():
  file = File(url="https://example.com/doc.pdf")
  msgs = [Message(role="user", content="summarize", files=[file])]
  formatted, _ = format_messages(msgs)
  doc_block = next(b for b in formatted[0]["content"] if isinstance(b, dict) and b.get("type") == "document")
  assert doc_block["citations"] == {"enabled": True}


def test_url_document_omits_citations_when_cite_false():
  file = File(url="https://example.com/doc.pdf")
  msgs = [Message(role="user", content="summarize", files=[file])]
  formatted, _ = format_messages(msgs, cite_documents=False)
  doc_block = next(b for b in formatted[0]["content"] if isinstance(b, dict) and b.get("type") == "document")
  assert "citations" not in doc_block


def test_inline_text_document_omits_citations_when_cite_false(tmp_path):
  txt = tmp_path / "note.txt"
  txt.write_text("hello world")
  file = File(filepath=str(txt))
  msgs = [Message(role="user", content="summarize", files=[file])]
  formatted, _ = format_messages(msgs, cite_documents=False)
  doc_block = next(b for b in formatted[0]["content"] if isinstance(b, dict) and b.get("type") == "document")
  assert "citations" not in doc_block
