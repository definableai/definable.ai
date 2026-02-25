"""WebSocket client Protocol for type-safe optional dependency typing.

The ``websockets`` library is an optional dependency (``pip install 'definable[call]'``)
imported lazily at runtime. This Protocol gives mypy the structural type
information it needs without coupling to the websockets package at import time.
"""

from typing import AsyncIterator, Protocol, Union


class WebSocketClient(Protocol):
  """Minimal async WebSocket client interface.

  Describes the subset of ``websockets.asyncio.client.ClientConnection``
  used by Definable's call interface providers (DeepgramSTT, CartesiaTTS,
  OpenAIRealtimeProvider).
  """

  async def send(self, message: Union[str, bytes]) -> None: ...
  async def recv(self) -> Union[str, bytes]: ...
  async def close(self) -> None: ...
  def __aiter__(self) -> AsyncIterator[Union[str, bytes]]: ...
