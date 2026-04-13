from __future__ import annotations

from typing import AsyncIterator

from definable.agent.interface.call import (
  CallInterface,
  RealtimeEvent,
  RealtimeProvider,
  STTProvider,
  TTSProvider,
  Transcript,
)


class DemoSTT(STTProvider):
  async def connect(self, *, sample_rate: int = 8000, encoding: str = "mulaw", channels: int = 1) -> None:
    return None

  async def send_audio(self, audio_bytes: bytes) -> None:
    return None

  async def close(self) -> None:
    return None

  async def _iter(self) -> AsyncIterator[Transcript]:
    if False:
      yield Transcript(text="")

  def receive_transcripts(self) -> AsyncIterator[Transcript]:
    return self._iter()


class DemoTTS(TTSProvider):
  async def _iter(self) -> AsyncIterator[bytes]:
    if False:
      yield b""

  def synthesize_stream(
    self,
    text: str,
    *,
    voice: str = "default",
    encoding: str = "mulaw",
    sample_rate: int = 8000,
  ) -> AsyncIterator[bytes]:
    return self._iter()

  async def close(self) -> None:
    return None


class DemoRealtime(RealtimeProvider):
  async def connect(
    self,
    *,
    instructions: str = "",
    tools=None,
    voice: str = "alloy",
    input_encoding: str = "mulaw",
    input_sample_rate: int = 8000,
    output_encoding: str = "mulaw",
    output_sample_rate: int = 8000,
  ) -> None:
    return None

  async def send_audio(self, audio_bytes: bytes) -> None:
    return None

  async def send_tool_result(self, call_id: str, result: str) -> None:
    return None

  async def interrupt(self) -> None:
    return None

  async def close(self) -> None:
    return None

  async def _iter(self) -> AsyncIterator[RealtimeEvent]:
    if False:
      yield RealtimeEvent(type="turn_complete")

  def receive_events(self) -> AsyncIterator[RealtimeEvent]:
    return self._iter()


def main() -> dict[str, object]:
  managed = CallInterface(
    provider="twilio",
    account_sid="AC123",
    auth_token="token",
    phone_number="+14155550123",
    pipeline="managed",
    welcome_message="Hello from the managed pipeline.",
  )
  cascading = CallInterface(
    provider="twilio",
    account_sid="AC123",
    auth_token="token",
    phone_number="+14155550123",
    pipeline="cascading",
    stt=DemoSTT(),
    tts=DemoTTS(),
  )
  realtime = CallInterface(
    provider="twilio",
    account_sid="AC123",
    auth_token="token",
    phone_number="+14155550123",
    pipeline="realtime",
    realtime=DemoRealtime(),
  )

  router_paths = sorted(route.path for route in managed.create_router().routes)
  summary = {
    "managed_mode": managed.config.pipeline_mode,
    "cascading_mode": cascading.config.pipeline_mode,
    "realtime_mode": realtime.config.pipeline_mode,
    "router_paths": router_paths,
  }

  assert summary["managed_mode"] == "managed"
  assert summary["cascading_mode"] == "cascading"
  assert summary["realtime_mode"] == "realtime"
  assert "/call/incoming" in router_paths
  assert "/call/stream/{call_id}" in router_paths

  return summary


if __name__ == "__main__":
  print(main())
