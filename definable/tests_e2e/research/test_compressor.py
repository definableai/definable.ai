"""Tests for the CKU compressor."""

import pytest

from definable.research.compressor import compress, compress_batch
from definable.research.models import PageContent


@pytest.mark.asyncio
class TestCompress:
  """Test single-page CKU extraction."""

  async def test_compress_valid_page(self, mock_model, sample_pages):
    cku = await compress(
      sample_pages[0],
      sub_question="What are quantum hardware advances?",
      original_query="Quantum computing overview",
      model=mock_model,
    )
    assert cku is not None
    assert len(cku.facts) > 0
    assert cku.source_url == "https://example.com/page1"
    assert cku.relevance_score > 0

  async def test_compress_empty_page(self, mock_model):
    page = PageContent(url="https://empty.com", content="")
    cku = await compress(page, "q", "q", mock_model)
    assert cku is None

  async def test_compress_error_page(self, mock_model):
    page = PageContent(url="https://error.com", error="connection timeout")
    cku = await compress(page, "q", "q", mock_model)
    assert cku is None

  async def test_compress_preserves_source_url(self, mock_model, sample_pages):
    cku = await compress(sample_pages[0], "q", "q", mock_model)
    assert cku is not None
    assert cku.source_url == sample_pages[0].url


@pytest.mark.asyncio
class TestCompressBatch:
  """Test batch CKU extraction."""

  async def test_compress_batch_multiple_pages(self, mock_model, sample_pages):
    ckus = await compress_batch(
      sample_pages,
      sub_question="quantum computing",
      original_query="quantum computing overview",
      model=mock_model,
      max_concurrent=2,
    )
    assert len(ckus) == 2

  async def test_compress_batch_filters_nones(self, mock_model):
    pages = [
      PageContent(url="https://ok.com", title="OK", content="Some content here"),
      PageContent(url="https://err.com", error="fail"),
      PageContent(url="https://empty.com", content=""),
    ]
    ckus = await compress_batch(pages, "q", "q", mock_model, max_concurrent=2)
    # Only the first page should produce a CKU
    assert len(ckus) == 1

  async def test_compress_batch_empty_list(self, mock_model):
    ckus = await compress_batch([], "q", "q", mock_model)
    assert ckus == []
