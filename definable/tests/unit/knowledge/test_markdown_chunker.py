"""
Unit tests for MarkdownChunker.

Tests pure logic: does the chunker split markdown correctly?
No API calls. No external dependencies.

Covers:
  - Splits on headings at configurable depth
  - Preserves code blocks across heading splits
  - Falls back to paragraph splitting for large sections
  - Empty/tiny documents handled gracefully
  - Chunk metadata (index, total, parent_id, source, chunker) preserved
  - chunk_many() processes multiple documents
"""

import pytest

from definable.knowledge.chunker.markdown import MarkdownChunker
from definable.knowledge.document import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_doc(content: str, name: str = "test_doc", source: str = "test.md") -> Document:
  return Document(content=content, name=name, source=source, source_type="markdown")


# ---------------------------------------------------------------------------
# Basic splitting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarkdownChunkerBasic:
  """MarkdownChunker splits markdown on headings."""

  def test_empty_document_returns_empty_list(self):
    chunker = MarkdownChunker(chunk_size=100)
    assert chunker.chunk(make_doc("")) == []

  def test_small_document_returns_single_chunk(self):
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc("Hello world")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world"

  def test_splits_on_h1_headings(self):
    md = "# Introduction\n\nSome intro text.\n\n# Chapter 1\n\nChapter text.\n\n# Chapter 2\n\nMore text."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 3
    assert "Introduction" in chunks[0].content
    assert "Chapter 1" in chunks[1].content
    assert "Chapter 2" in chunks[2].content

  def test_splits_on_h2_headings(self):
    md = "## Section A\n\nText A.\n\n## Section B\n\nText B."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2
    assert "Section A" in chunks[0].content
    assert "Section B" in chunks[1].content

  def test_max_heading_depth_limits_splitting(self):
    md = "# H1\n\nText.\n\n## H2\n\nMore.\n\n### H3\n\nDeep."
    # max_heading_depth=1 → only split on H1
    chunker = MarkdownChunker(chunk_size=5000, max_heading_depth=1)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1  # Only one H1, everything under it stays together

  def test_max_heading_depth_2(self):
    md = "# H1\n\nText.\n\n## H2\n\nMore.\n\n### H3\n\nDeep."
    chunker = MarkdownChunker(chunk_size=5000, max_heading_depth=2)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2  # H1 and H2 each start a section

  def test_no_headings_returns_single_chunk(self):
    md = "Just some text without any headings.\n\nAnother paragraph."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].content.strip() == md.strip()

  def test_content_before_first_heading(self):
    md = "Preamble text.\n\n# Title\n\nBody text."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2
    assert "Preamble" in chunks[0].content
    assert "Title" in chunks[1].content


# ---------------------------------------------------------------------------
# Code block preservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarkdownChunkerCodeBlocks:
  """Code blocks must not be split or garbled."""

  def test_code_block_preserved_in_chunk(self):
    md = "# Setup\n\n```python\ndef hello():\n  print('hi')\n```\n\n# Usage\n\nUse it."
    chunker = MarkdownChunker(chunk_size=5000, preserve_code_blocks=True)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2
    assert "```python" in chunks[0].content
    assert "def hello():" in chunks[0].content
    assert "```" in chunks[0].content

  def test_code_block_with_heading_like_content(self):
    """Code blocks containing # should NOT be treated as headings."""
    md = "# Before\n\n```bash\n# This is a comment\necho hello\n```\n\n# After\n\nText."
    chunker = MarkdownChunker(chunk_size=5000, preserve_code_blocks=True)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2
    # The bash comment should be inside the first chunk's code block
    assert "# This is a comment" in chunks[0].content

  def test_multiple_code_blocks_preserved(self):
    md = "# A\n\n```js\nconst x = 1;\n```\n\nText.\n\n```py\ny = 2\n```\n\n# B\n\nMore."
    chunker = MarkdownChunker(chunk_size=5000, preserve_code_blocks=True)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2
    assert "const x = 1;" in chunks[0].content
    assert "y = 2" in chunks[0].content

  def test_preserve_code_blocks_disabled(self):
    md = "# Setup\n\n```python\n# A heading inside code\ndef foo(): pass\n```\n\n# Other\n\nText."
    chunker = MarkdownChunker(chunk_size=5000, preserve_code_blocks=False)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    # Without code block protection, the # inside code might cause extra splits
    # but should still produce valid chunks
    assert len(chunks) >= 2
    combined = " ".join(c.content for c in chunks)
    assert "def foo(): pass" in combined


# ---------------------------------------------------------------------------
# Large section fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarkdownChunkerLargeSections:
  """Sections exceeding chunk_size are further split."""

  def test_large_section_gets_split(self):
    heading = "# Big Section\n\n"
    body = "Word " * 200  # ~1000 chars
    chunker = MarkdownChunker(chunk_size=100)
    doc = make_doc(heading + body)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1

  def test_large_section_splits_on_paragraphs(self):
    heading = "# Section\n\n"
    paragraphs = "\n\n".join([f"Paragraph {i}. " + "Text " * 20 for i in range(10)])
    chunker = MarkdownChunker(chunk_size=200)
    doc = make_doc(heading + paragraphs)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    # All paragraphs should be preserved
    combined = " ".join(c.content for c in chunks)
    for i in range(10):
      assert f"Paragraph {i}" in combined

  def test_multiple_sections_some_large(self):
    sections = [
      "# Small\n\nShort.",
      "# Large\n\n" + "Word " * 200,
      "# Also Small\n\nBrief.",
    ]
    chunker = MarkdownChunker(chunk_size=100)
    doc = make_doc("\n\n".join(sections))
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 3
    combined = " ".join(c.content for c in chunks)
    assert "Short." in combined
    assert "Brief." in combined


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarkdownChunkerMetadata:
  """Chunk metadata must be correct."""

  def test_chunk_indices_sequential(self):
    md = "# A\n\nText.\n\n# B\n\nText.\n\n# C\n\nText."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    for i, chunk in enumerate(chunks):
      assert chunk.chunk_index == i

  def test_chunk_total_consistent(self):
    md = "# A\n\nText.\n\n# B\n\nText."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    totals = {c.chunk_total for c in chunks}
    assert len(totals) == 1
    assert totals.pop() == len(chunks)

  def test_source_preserved(self):
    md = "# A\n\nText.\n\n# B\n\nText."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md, source="doc.md")
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.source == "doc.md"

  def test_source_type_preserved(self):
    md = "# A\n\nText.\n\n# B\n\nText."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.source_type == "markdown"

  def test_parent_id_set(self):
    md = "# A\n\nText.\n\n# B\n\nText."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    parent_ids = {c.parent_id for c in chunks}
    assert len(parent_ids) == 1
    assert parent_ids.pop() is not None

  def test_meta_data_has_chunker_field(self):
    md = "# A\n\nText.\n\n# B\n\nText."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.meta_data.get("chunker") == "markdown"

  def test_meta_data_inherited_from_parent(self):
    md = "# A\n\nText.\n\n# B\n\nText."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = Document(content=md, meta_data={"category": "docs"})
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.meta_data.get("category") == "docs"

  def test_name_includes_chunk_index(self):
    md = "# A\n\nText.\n\n# B\n\nText."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md, name="readme")
    chunks = chunker.chunk(doc)
    for i, chunk in enumerate(chunks):
      assert f"chunk_{i}" in chunk.name  # type: ignore[operator]

  def test_chunk_many_works(self):
    chunker = MarkdownChunker(chunk_size=5000)
    docs = [
      make_doc("# A\n\nText.\n\n# B\n\nMore."),
      make_doc("# C\n\nOther."),
    ]
    chunks = chunker.chunk_many(docs)
    assert len(chunks) >= 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarkdownChunkerEdgeCases:
  """Edge cases and corner cases."""

  def test_heading_only_no_body(self):
    md = "# Title\n\n# Another Title"
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1

  def test_deeply_nested_headings(self):
    md = "# H1\n\n## H2\n\n### H3\n\n#### H4\n\n##### H5\n\n###### H6\n\nDeep."
    chunker = MarkdownChunker(chunk_size=5000, max_heading_depth=6)
    doc = make_doc(md)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1
    combined = " ".join(c.content for c in chunks)
    assert "Deep." in combined

  def test_unclosed_code_block(self):
    md = "# Title\n\n```python\ndef broken():\n  pass\n\n# Should not split here"
    chunker = MarkdownChunker(chunk_size=5000, preserve_code_blocks=True)
    doc = make_doc(md)
    # Should not crash
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1

  def test_whitespace_only_returns_empty(self):
    chunker = MarkdownChunker(chunk_size=100)
    doc = make_doc("   \n\n   ")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 0

  def test_returns_list_of_documents(self):
    md = "# A\n\nText."
    chunker = MarkdownChunker(chunk_size=5000)
    doc = make_doc(md)
    result = chunker.chunk(doc)
    assert isinstance(result, list)
    assert all(isinstance(c, Document) for c in result)
