"""JSON document reader."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from definable.knowledge.document import Document
from definable.knowledge.reader.base import Reader
from definable.utils.log import log_error


@dataclass
class JSONReader(Reader):
  """Read JSON files and convert them into Documents.

  Supports both single JSON objects and JSON arrays. Each top-level
  object becomes a separate Document. Nested objects are serialized
  as their content.

  Args:
      content_key: If set, extract this key's value as document content.
                   If None, the entire JSON object is stringified.
      metadata_keys: Keys to extract as meta_data (rest goes to content).
      flatten: If True, flatten nested objects to "key.subkey" format.
      encoding: File encoding. Default "utf-8".

  Example::

      reader = JSONReader(content_key="text", metadata_keys=["source", "date"])
      docs = reader.read("data.json")
  """

  content_key: Optional[str] = None
  metadata_keys: Optional[List[str]] = None
  flatten: bool = False
  encoding: str = "utf-8"

  def read(self, source: Union[str, Path]) -> List[Document]:
    """Read a JSON file and return Documents."""
    path = Path(source)
    try:
      raw = path.read_text(encoding=self.encoding)
      data = json.loads(raw)
    except FileNotFoundError:
      log_error(f"JSONReader: file not found: {path}")
      return []
    except json.JSONDecodeError as e:
      log_error(f"JSONReader: invalid JSON in {path}: {e}")
      return []

    return self._parse(data, source=str(path))

  async def aread(self, source: Union[str, Path]) -> List[Document]:
    """Async read via thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.read, source)

  def can_read(self, source: Union[str, Path]) -> bool:
    """Check if the source is a JSON file."""
    return str(source).lower().endswith(".json")

  def read_string(self, content: str, source: str = "string") -> List[Document]:
    """Parse a JSON string directly."""
    try:
      data = json.loads(content)
    except json.JSONDecodeError as e:
      log_error(f"JSONReader: invalid JSON string: {e}")
      return []
    return self._parse(data, source=source)

  def _parse(self, data: Any, source: str = "") -> List[Document]:
    """Convert parsed JSON data into Documents."""
    items: List[Dict[str, Any]] = []
    if isinstance(data, list):
      items = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
      items = [data]
    else:
      # Primitive value — single document
      return [
        Document(
          content=str(data),
          source=source,
          source_type="json",
          meta_data={"source": source},
        )
      ]

    documents: List[Document] = []
    for i, item in enumerate(items):
      if self.flatten:
        item = self._flatten_dict(item)

      content = self._extract_content(item)
      meta = self._extract_metadata(item)
      meta["source"] = source
      meta["index"] = i

      documents.append(
        Document(
          content=content,
          name=f"{Path(source).stem}_{i}" if source else f"json_{i}",
          source=source,
          source_type="json",
          meta_data=meta,
        )
      )

    return documents

  def _extract_content(self, item: Dict[str, Any]) -> str:
    """Extract content string from a JSON object."""
    if self.content_key and self.content_key in item:
      value = item[self.content_key]
      return str(value) if not isinstance(value, str) else value

    # Exclude metadata keys from content
    if self.metadata_keys:
      content_item = {k: v for k, v in item.items() if k not in self.metadata_keys}
    else:
      content_item = item

    return json.dumps(content_item, ensure_ascii=False, indent=2)

  def _extract_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from a JSON object."""
    if not self.metadata_keys:
      return {}
    return {k: item[k] for k in self.metadata_keys if k in item}

  @staticmethod
  def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict to dot-separated keys."""
    flat: Dict[str, Any] = {}
    for k, v in d.items():
      key = f"{prefix}.{k}" if prefix else k
      if isinstance(v, dict):
        flat.update(JSONReader._flatten_dict(v, key))
      else:
        flat[key] = v
    return flat
