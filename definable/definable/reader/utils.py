"""Shared I/O helpers for the readers module.

Extracts the I/O logic from the old FileReader base class into
standalone functions usable by both BaseReader and provider readers.
"""

import os
from typing import Optional
from urllib.parse import urlparse

from definable.media import File


def get_file_bytes_sync(file: File, *, encoding: str = "utf-8", timeout: float | None = 30.0) -> bytes:
  """Resolve File content/filepath/url to raw bytes (sync)."""
  if file.content is not None:
    if isinstance(file.content, bytes):
      return file.content
    if isinstance(file.content, str):
      return file.content.encode(encoding)
    return bytes(file.content)

  if file.filepath:
    with open(str(file.filepath), "rb") as f:
      return f.read()

  if file.url:
    import httpx

    response = httpx.get(file.url, timeout=timeout)
    response.raise_for_status()
    return response.content

  raise ValueError("File has no content, filepath, or url")


async def get_file_bytes_async(file: File, *, encoding: str = "utf-8", timeout: float | None = 30.0) -> bytes:
  """Resolve File content/filepath/url to raw bytes (async)."""
  if file.content is not None:
    if isinstance(file.content, bytes):
      return file.content
    if isinstance(file.content, str):
      return file.content.encode(encoding)
    return bytes(file.content)

  if file.filepath:
    import asyncio

    path = str(file.filepath)

    def _read() -> bytes:
      with open(path, "rb") as f:
        return f.read()

    return await asyncio.get_running_loop().run_in_executor(None, _read)

  if file.url:
    import httpx

    async with httpx.AsyncClient() as client:
      response = await client.get(file.url, timeout=timeout)
      response.raise_for_status()
      return response.content

  raise ValueError("File has no content, filepath, or url")


def get_filename(file: File) -> str:
  """Get a display filename for the file."""
  return file.filename or file.name or "unknown"


def extract_extension(file: File) -> Optional[str]:
  """Extract the lowercase file extension from a File object."""
  name = file.filename or file.name
  if name:
    _, ext = os.path.splitext(name)
    return ext.lower() if ext else None

  if file.filepath:
    _, ext = os.path.splitext(str(file.filepath))
    return ext.lower() if ext else None

  if file.url:
    path = urlparse(file.url).path
    _, ext = os.path.splitext(path)
    return ext.lower() if ext else None

  return None
