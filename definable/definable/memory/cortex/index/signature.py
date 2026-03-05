"""Binary signature index using Random Indexing.

Provides ultra-fast pre-filtering via Hamming distance on binary signatures.
Each text is mapped to a fixed-size binary vector via random indexing,
enabling O(1) per-comparison filtering before expensive similarity search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


class SignatureBuilder:
  """Builds binary signatures from text using Random Indexing.

  Random Indexing: each unique token is assigned a sparse random vector (RI vector).
  A document's signature is the sum of its tokens' RI vectors, binarized.

  Args:
    dims: Dimensionality of the binary signature.
    nnz: Number of non-zero elements per random index vector.
    seed: Random seed for reproducibility.
  """

  def __init__(self, dims: int = 1024, nnz: int = 8, seed: int = 42):
    self.dims = dims
    self.nnz = nnz
    self._rng = np.random.RandomState(seed)
    self._token_vectors: Dict[str, np.ndarray] = {}

  def _get_token_vector(self, token: str) -> np.ndarray:
    """Get or create the RI vector for a token."""
    if token not in self._token_vectors:
      vec = np.zeros(self.dims, dtype=np.float32)
      indices = self._rng.choice(self.dims, size=self.nnz, replace=False)
      signs = self._rng.choice([-1.0, 1.0], size=self.nnz)
      vec[indices] = signs
      self._token_vectors[token] = vec
    return self._token_vectors[token]

  def build(self, text: str) -> bytes:
    """Build a binary signature from text.

    Args:
      text: Input text to create signature for.

    Returns:
      Binary signature as bytes (dims/8 bytes long).
    """
    tokens = text.lower().split()
    if not tokens:
      return b"\x00" * (self.dims // 8)

    acc = np.zeros(self.dims, dtype=np.float32)
    for token in tokens:
      acc += self._get_token_vector(token)

    # Binarize: positive → 1, non-positive → 0
    bits = (acc > 0).astype(np.uint8)
    return np.packbits(bits).tobytes()

  def to_dict(self) -> Dict[str, Any]:
    """Serialize builder state for persistence."""
    token_vecs = {}
    for token, vec in self._token_vectors.items():
      token_vecs[token] = vec.tolist()
    return {
      "dims": self.dims,
      "nnz": self.nnz,
      "token_vectors": token_vecs,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "SignatureBuilder":
    builder = cls(dims=data["dims"], nnz=data["nnz"])
    for token, vec_list in data.get("token_vectors", {}).items():
      builder._token_vectors[token] = np.array(vec_list, dtype=np.float32)
    return builder


class SignatureIndex:
  """Index for fast Hamming-distance search over binary signatures.

  SQLite-backed. Stores signatures as BLOBs and performs Hamming distance
  computation in Python after bulk retrieval.
  """

  def __init__(self, db: Any = None):
    self._db = db
    self._initialized = False

  async def initialize(self, db: Any) -> None:
    """Initialize with a shared aiosqlite connection."""
    self._db = db
    assert self._db is not None
    await self._db.executescript("""
      CREATE TABLE IF NOT EXISTS cortex_signatures (
        record_id TEXT PRIMARY KEY,
        signature BLOB NOT NULL
      );
    """)
    await self._db.commit()
    self._initialized = True

  async def add(self, record_id: str, signature: bytes) -> None:
    """Add a signature for a record."""
    assert self._db is not None
    await self._db.execute(
      "INSERT OR REPLACE INTO cortex_signatures (record_id, signature) VALUES (?, ?)",
      (record_id, signature),
    )
    await self._db.commit()

  async def remove(self, record_id: str) -> None:
    """Remove a signature."""
    assert self._db is not None
    await self._db.execute("DELETE FROM cortex_signatures WHERE record_id = ?", (record_id,))
    await self._db.commit()

  async def search(self, query_signature: bytes, max_distance: int = 128, limit: int = 100) -> List[Tuple[str, int]]:
    """Find records with Hamming distance <= max_distance from query.

    Args:
      query_signature: The query binary signature.
      max_distance: Maximum Hamming distance threshold.
      limit: Max results to return.

    Returns:
      List of (record_id, hamming_distance) sorted by distance ascending.
    """
    assert self._db is not None
    cursor = await self._db.execute("SELECT record_id, signature FROM cortex_signatures")
    rows = await cursor.fetchall()

    query_bits = np.unpackbits(np.frombuffer(query_signature, dtype=np.uint8))
    results: List[Tuple[str, int]] = []

    for record_id, sig_blob in rows:
      sig_bits = np.unpackbits(np.frombuffer(sig_blob, dtype=np.uint8))
      # Hamming distance = number of differing bits
      distance = int(np.sum(query_bits != sig_bits))
      if distance <= max_distance:
        results.append((record_id, distance))

    results.sort(key=lambda x: x[1])
    return results[:limit]

  async def close(self) -> None:
    """No-op — db lifecycle managed by CortexStore."""
    pass
