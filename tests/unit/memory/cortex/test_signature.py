"""Tests for Cortex SignatureBuilder and SignatureIndex."""

import pytest
import numpy as np
from definable.memory.cortex.index.signature import SignatureBuilder, SignatureIndex


class TestSignatureBuilder:
  def test_build_produces_bytes(self):
    builder = SignatureBuilder(dims=1024)
    sig = builder.build("hello world")
    assert isinstance(sig, bytes)
    assert len(sig) == 1024 // 8  # 128 bytes

  def test_empty_text(self):
    builder = SignatureBuilder(dims=1024)
    sig = builder.build("")
    assert sig == b"\x00" * 128

  def test_similar_texts_close(self):
    builder = SignatureBuilder(dims=1024, seed=42)
    s1 = builder.build("python programming language")
    s2 = builder.build("python programming tutorial")
    s3 = builder.build("quantum physics experiment")
    # s1 and s2 should be closer than s1 and s3
    d12 = _hamming(s1, s2)
    d13 = _hamming(s1, s3)
    assert d12 < d13

  def test_deterministic(self):
    b1 = SignatureBuilder(dims=512, seed=99)
    b2 = SignatureBuilder(dims=512, seed=99)
    s1 = b1.build("test text")
    s2 = b2.build("test text")
    assert s1 == s2

  def test_roundtrip(self):
    builder = SignatureBuilder(dims=512)
    builder.build("some text")
    d = builder.to_dict()
    restored = SignatureBuilder.from_dict(d)
    assert restored.dims == 512
    assert restored.build("some text") == builder.build("some text")


@pytest.fixture
async def sig_index(tmp_path):
  import aiosqlite

  db = await aiosqlite.connect(str(tmp_path / "sig.db"))
  idx = SignatureIndex()
  await idx.initialize(db)
  yield idx
  await db.close()


@pytest.mark.asyncio
class TestSignatureIndex:
  async def test_add_and_search(self, sig_index):
    builder = SignatureBuilder(dims=1024, seed=42)
    s1 = builder.build("python programming")
    s2 = builder.build("java programming")
    s3 = builder.build("cooking recipes")
    await sig_index.add("r1", s1)
    await sig_index.add("r2", s2)
    await sig_index.add("r3", s3)

    query = builder.build("python code")
    results = await sig_index.search(query, max_distance=500)
    assert len(results) > 0
    # r1 (python) should be closer than r3 (cooking)
    ids = [r[0] for r in results]
    if "r1" in ids and "r3" in ids:
      idx1 = ids.index("r1")
      idx3 = ids.index("r3")
      assert idx1 < idx3

  async def test_remove(self, sig_index):
    builder = SignatureBuilder(dims=1024)
    await sig_index.add("r1", builder.build("test"))
    await sig_index.remove("r1")
    results = await sig_index.search(builder.build("test"), max_distance=1000)
    assert all(r[0] != "r1" for r in results)

  async def test_max_distance_filter(self, sig_index):
    builder = SignatureBuilder(dims=1024, seed=42)
    await sig_index.add("r1", builder.build("hello world"))
    results = await sig_index.search(builder.build("hello world"), max_distance=0)
    assert len(results) == 1
    assert results[0][1] == 0


def _hamming(a: bytes, b: bytes) -> int:
  """Hamming distance between two byte strings."""
  ba = np.unpackbits(np.frombuffer(a, dtype=np.uint8))
  bb = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
  return int(np.sum(ba != bb))
