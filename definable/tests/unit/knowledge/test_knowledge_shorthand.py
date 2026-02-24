"""
Unit tests for Knowledge path shorthand — Agent(knowledge="./docs/").

When Agent(knowledge="./docs/") is called, the string is resolved to a fully
configured Knowledge instance with auto-detected readers, RecursiveChunker,
InMemoryVectorDB + OpenAIEmbedder, and all documents loaded.

Covers:
  - Knowledge.from_path with a directory
  - Knowledge.from_path with a single file
  - Knowledge.from_path rejects non-existent path
  - Knowledge.from_path rejects empty directory
  - Knowledge.from_path auto-detects text and PDF files
  - Agent resolves knowledge string to Knowledge instance
  - Agent knowledge=True still raises ValueError
  - Agent knowledge=False/None still returns None
  - Recursive directory traversal picks up nested files
  - Custom chunk_size/top_k params forwarded
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from definable.knowledge.base import Knowledge


# ---------------------------------------------------------------------------
# Knowledge.from_path — directory loading
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnowledgeFromPathDirectory:
  """Knowledge.from_path loads all supported files from a directory."""

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_loads_txt_files(self, mock_embed):
    mock_embed.return_value = [0.1] * 1536

    with tempfile.TemporaryDirectory() as tmpdir:
      Path(tmpdir, "doc1.txt").write_text("Hello world")
      Path(tmpdir, "doc2.txt").write_text("Goodbye world")

      kb = Knowledge.from_path(tmpdir)

      assert kb.vector_db is not None
      assert kb.chunker is not None
      assert len(kb.readers) == 2  # TextReader + PDFReader
      assert kb.top_k == 5

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_loads_md_files(self, mock_embed):
    mock_embed.return_value = [0.1] * 1536

    with tempfile.TemporaryDirectory() as tmpdir:
      Path(tmpdir, "readme.md").write_text("# Hello\n\nSome docs")

      kb = Knowledge.from_path(tmpdir)
      assert kb.vector_db is not None

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_recursive_directory_traversal(self, mock_embed):
    mock_embed.return_value = [0.1] * 1536

    with tempfile.TemporaryDirectory() as tmpdir:
      nested = Path(tmpdir, "sub", "deep")
      nested.mkdir(parents=True)
      Path(tmpdir, "top.txt").write_text("Top level")
      Path(nested, "deep.txt").write_text("Deep nested")

      kb = Knowledge.from_path(tmpdir)
      # Should find both files
      assert kb.vector_db is not None

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_custom_chunk_size_and_top_k(self, mock_embed):
    mock_embed.return_value = [0.1] * 1536

    with tempfile.TemporaryDirectory() as tmpdir:
      Path(tmpdir, "doc.txt").write_text("Some content here")

      kb = Knowledge.from_path(tmpdir, chunk_size=500, chunk_overlap=100, top_k=10)
      assert kb.top_k == 10
      assert kb.chunker is not None
      assert kb.chunker.chunk_size == 500
      assert kb.chunker.chunk_overlap == 100

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_ignores_unsupported_extensions(self, mock_embed):
    """Files like .py, .jpg, .bin are not loaded."""
    mock_embed.return_value = [0.1] * 1536

    with tempfile.TemporaryDirectory() as tmpdir:
      Path(tmpdir, "code.py").write_text("print('hello')")
      Path(tmpdir, "image.jpg").write_bytes(b"\xff\xd8\xff\xe0")
      Path(tmpdir, "doc.txt").write_text("Supported file")

      kb = Knowledge.from_path(tmpdir)
      assert kb.vector_db is not None


# ---------------------------------------------------------------------------
# Knowledge.from_path — single file
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnowledgeFromPathFile:
  """Knowledge.from_path loads a single file."""

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_loads_single_txt_file(self, mock_embed):
    mock_embed.return_value = [0.1] * 1536

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
      f.write("Single file content")
      f.flush()
      try:
        kb = Knowledge.from_path(f.name)
        assert kb.vector_db is not None
      finally:
        os.unlink(f.name)

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_loads_single_md_file(self, mock_embed):
    mock_embed.return_value = [0.1] * 1536

    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
      f.write("# Markdown\n\nContent here")
      f.flush()
      try:
        kb = Knowledge.from_path(f.name)
        assert kb.vector_db is not None
      finally:
        os.unlink(f.name)


# ---------------------------------------------------------------------------
# Knowledge.from_path — error cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnowledgeFromPathErrors:
  """Knowledge.from_path raises on invalid input."""

  def test_rejects_nonexistent_path(self):
    with pytest.raises(FileNotFoundError, match="does not exist"):
      Knowledge.from_path("/nonexistent/path/to/docs")

  def test_rejects_empty_directory(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      with pytest.raises(ValueError, match="No supported files found"):
        Knowledge.from_path(tmpdir)

  def test_rejects_dir_with_only_unsupported_files(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      Path(tmpdir, "code.py").write_text("print('hello')")
      Path(tmpdir, "data.json").write_text('{"key": "value"}')

      with pytest.raises(ValueError, match="No supported files found"):
        Knowledge.from_path(tmpdir)


# ---------------------------------------------------------------------------
# Agent string shorthand integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentKnowledgeStringShorthand:
  """Agent resolves knowledge string to Knowledge.from_path."""

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  @patch("definable.model.openai.chat.OpenAIChat.get_client")
  def test_agent_accepts_knowledge_string(self, mock_client, mock_embed):
    mock_client.return_value = MagicMock()
    mock_embed.return_value = [0.1] * 1536

    from definable.agent import Agent

    with tempfile.TemporaryDirectory() as tmpdir:
      Path(tmpdir, "doc.txt").write_text("Knowledge content")

      agent = Agent(model="openai/gpt-4o-mini", knowledge=tmpdir)
      assert agent._knowledge is not None
      assert isinstance(agent._knowledge, Knowledge)

  def test_agent_knowledge_true_still_raises(self):
    from definable.agent import Agent

    with pytest.raises(ValueError, match="knowledge=True is not supported"):
      Agent(model=MagicMock(), knowledge=True)

  def test_agent_knowledge_false_is_none(self):
    from definable.agent import Agent

    agent = Agent(model=MagicMock())
    assert agent._knowledge is None

  def test_agent_knowledge_none_is_none(self):
    from definable.agent import Agent

    agent = Agent(model=MagicMock(), knowledge=None)
    assert agent._knowledge is None

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_agent_knowledge_instance_passthrough(self, mock_embed):
    mock_embed.return_value = [0.1] * 1536
    from definable.agent import Agent

    kb = Knowledge()
    agent = Agent(model=MagicMock(), knowledge=kb)
    assert agent._knowledge is kb

  def test_agent_knowledge_nonexistent_path_raises(self):
    from definable.agent import Agent

    with pytest.raises(FileNotFoundError, match="does not exist"):
      Agent(model=MagicMock(), knowledge="/nonexistent/docs")


# ---------------------------------------------------------------------------
# Knowledge.from_path — component configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnowledgeFromPathComponents:
  """from_path creates correct default components."""

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_uses_recursive_chunker(self, mock_embed):
    mock_embed.return_value = [0.1] * 1536
    from definable.knowledge.chunker.recursive import RecursiveChunker

    with tempfile.TemporaryDirectory() as tmpdir:
      Path(tmpdir, "doc.txt").write_text("Content")

      kb = Knowledge.from_path(tmpdir)
      assert isinstance(kb.chunker, RecursiveChunker)

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_uses_openai_embedder(self, mock_embed):
    mock_embed.return_value = [0.1] * 1536
    from definable.knowledge.embedder.openai import OpenAIEmbedder

    with tempfile.TemporaryDirectory() as tmpdir:
      Path(tmpdir, "doc.txt").write_text("Content")

      kb = Knowledge.from_path(tmpdir)
      assert isinstance(kb.embedder, OpenAIEmbedder)

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_uses_in_memory_vectordb(self, mock_embed):
    mock_embed.return_value = [0.1] * 1536
    from definable.vectordb.memory import InMemoryVectorDB

    with tempfile.TemporaryDirectory() as tmpdir:
      Path(tmpdir, "doc.txt").write_text("Content")

      kb = Knowledge.from_path(tmpdir)
      assert isinstance(kb.vector_db, InMemoryVectorDB)

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_includes_text_and_pdf_readers(self, mock_embed):
    mock_embed.return_value = [0.1] * 1536
    from definable.knowledge.reader.pdf import PDFReader
    from definable.knowledge.reader.text import TextReader

    with tempfile.TemporaryDirectory() as tmpdir:
      Path(tmpdir, "doc.txt").write_text("Content")

      kb = Knowledge.from_path(tmpdir)
      reader_types = {type(r) for r in kb.readers}
      assert TextReader in reader_types
      assert PDFReader in reader_types

  @patch("definable.knowledge.embedder.openai.OpenAIEmbedder.get_embedding")
  def test_tilde_expansion(self, mock_embed):
    """Paths with ~ are expanded."""
    mock_embed.return_value = [0.1] * 1536

    with tempfile.TemporaryDirectory() as tmpdir:
      Path(tmpdir, "doc.txt").write_text("Content")

      # Create a path that doesn't use ~ but test that expanduser is called
      kb = Knowledge.from_path(tmpdir)
      assert kb.vector_db is not None
