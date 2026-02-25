# Deepgram Real-Time Streaming STT — SDK & Protocol Research

> **Date**: 2026-02-25
> **Purpose**: Production STT provider for Definable voice pipeline
> **SDK Version**: deepgram-sdk 6.0.1 (released 2026-02-24)
> **Python**: 3.8+

---

## 1. SDK Package Overview

- **Package**: `deepgram-sdk` on PyPI
- **Latest**: 6.0.1 (Feb 24, 2026)
- **Size**: 490.8 KB wheel
- **Python**: >=3.8, <4.0
- **Key deps**: `httpx`, `websockets`
- **Install**: `pip install deepgram-sdk`

### Import Paths (v5+ / v6)
```python
from deepgram import DeepgramClient, AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    ListenV1SocketClientResponse,
    ListenV1MediaMessage,
    ListenV2SocketClientResponse,
    ListenV2MediaMessage,
    ListenV2ControlMessage,
)
```

### Legacy Import Paths (v3-v4, still work in some versions)
```python
from deepgram import LiveOptions, LiveTranscriptionEvents
```

---

## 2. Two API Versions

### Listen V1 (Nova models — `wss://api.deepgram.com/v1/listen`)
- Uses `client.listen.v1.connect(...)` or legacy `client.listen.websocket.v("1")`
- Models: `nova-3`, `nova-2`, `enhanced`, `base`, `phonecall`, `meeting`
- EventType-based callbacks

### Listen V2 (Flux models — `wss://api.deepgram.com/v2/listen`)
- Uses `client.listen.v2.connect(...)`
- Models: `flux-general-en`, `flux-general-*`
- Same EventType system
- Expanded audio format support

### Key Decision: Use V1 (Nova-3) for production STT
- Nova-3 is battle-tested, widely documented
- Flux is newer, may have edge cases
- V1 has more community examples

---

## 3. Connection Patterns

### Pattern A: Context Manager (v5+ recommended)
```python
from deepgram import DeepgramClient
from deepgram.core.events import EventType

client = DeepgramClient(api_key="YOUR_KEY")

with client.listen.v1.connect(
    model="nova-3",
    language="en-US",
    encoding="mulaw",
    sample_rate="8000",
    interim_results=True,
    endpointing=300,
    smart_format=True,
    punctuate=True,
    vad_events=True,
    utterance_end_ms="1000",
) as connection:
    connection.on(EventType.OPEN, on_open)
    connection.on(EventType.MESSAGE, on_message)
    connection.on(EventType.CLOSE, on_close)
    connection.on(EventType.ERROR, on_error)
    connection.start_listening()
    # send audio in loop...
    connection.send_media(audio_bytes)
    connection.finish()
```

### Pattern B: Legacy LiveOptions (v3-v4 style, still works)
```python
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

client = DeepgramClient()
connection = client.listen.websocket.v("1")

@connection.on(LiveTranscriptionEvents.Transcript)
def on_message(self, result, **kwargs):
    transcript = result.channel.alternatives[0].transcript
    if transcript:
        print(f"Transcript: {transcript}")

options = LiveOptions(
    model="nova-3",
    language="en-US",
    encoding="mulaw",
    sample_rate=8000,
    interim_results=True,
    endpointing=300,
)
connection.start(options)
connection.send(audio_bytes)
connection.finish()
```

### Pattern C: Async Context Manager
```python
async with client.listen.v1.connect(
    model="nova-3",
    encoding="linear16",
    sample_rate="16000",
) as connection:
    connection.on(EventType.MESSAGE, on_message)
    connection.start_listening()
    # async audio send loop
```

---

## 4. Audio Format Configuration

### Supported Raw Encodings
| Encoding | Description | Common Use |
|----------|-------------|------------|
| `linear16` | 16-bit signed little-endian PCM | Default, microphone |
| `linear32` | 32-bit signed little-endian float PCM | High-quality |
| `mulaw` | Mu-law encoding | **Telephony (US)** |
| `alaw` | A-law encoding | Telephony (EU) |
| `opus` | Opus codec | WebRTC, VoIP |
| `ogg-opus` | Opus in Ogg container | Browser recording |
| `amr-nb` | AMR narrowband | Mobile (8kHz only) |
| `amr-wb` | AMR wideband | Mobile (16kHz only) |
| `speex` | Speex codec | Legacy VoIP |
| `g729` | G.729 codec | Telephony |
| `flac` | FLAC lossless | Pre-recorded |

### Supported Sample Rates
- `8000` Hz (telephony)
- `16000` Hz (**recommended**)
- `24000` Hz
- `44100` Hz (CD quality)
- `48000` Hz (studio)

### Telephony Config (mulaw 8kHz)
```python
with client.listen.v1.connect(
    model="nova-3",
    encoding="mulaw",
    sample_rate="8000",
    channels=1,
) as connection:
    ...
```

### Containerized Audio (WAV, Ogg, WebM)
- **Omit** `encoding` and `sample_rate` — auto-detected from container headers

### Chunk Size
- **80ms strongly recommended** for optimal latency
- For 8kHz mulaw: 80ms = 640 bytes per chunk
- For 16kHz linear16: 80ms = 2560 bytes per chunk

---

## 5. Event System

### EventType Enum (v5+)
```python
from deepgram.core.events import EventType

EventType.OPEN      # Connection established
EventType.MESSAGE   # Any message received (transcripts, metadata, etc.)
EventType.CLOSE     # Connection closed
EventType.ERROR     # Error occurred
```

### Legacy LiveTranscriptionEvents (v3-v4)
```python
from deepgram import LiveTranscriptionEvents

LiveTranscriptionEvents.Open
LiveTranscriptionEvents.Transcript
LiveTranscriptionEvents.Metadata
LiveTranscriptionEvents.SpeechStarted
LiveTranscriptionEvents.UtteranceEnd
LiveTranscriptionEvents.Close
LiveTranscriptionEvents.Error
LiveTranscriptionEvents.Warning
```

### Message Handler — Parsing Transcripts
```python
def on_message(message):
    msg_type = getattr(message, "type", "Unknown")

    if msg_type == "Results":
        is_final = message.is_final        # True = segment complete
        speech_final = message.speech_final  # True = utterance complete (endpointing)
        channel = message.channel
        transcript = channel.alternatives[0].transcript
        confidence = channel.alternatives[0].confidence
        words = channel.alternatives[0].words  # list of {word, start, end, confidence}

    elif msg_type == "Metadata":
        # Initial connection metadata
        request_id = message.request_id

    elif msg_type == "SpeechStarted":
        # VAD detected speech start
        timestamp = message.timestamp

    elif msg_type == "UtteranceEnd":
        # Utterance boundary detected
        last_word_end = message.last_word_end
```

---

## 6. Interim vs Final Transcripts

### Three Flags in Response
| Flag | Meaning |
|------|---------|
| `is_final: false` | Interim result — may change as more audio arrives |
| `is_final: true` | Segment finalized — this text won't change |
| `speech_final: true` | Endpointing triggered — speaker paused/stopped |

### Reconstruction Logic
1. `is_final: false` + `speech_final: false` = still speaking, interim update
2. `is_final: true` + `speech_final: false` = segment finalized, but speaker continues
3. `is_final: true` + `speech_final: true` = utterance complete, speaker paused

**Rule**: Concatenate all `is_final: true` segments until `speech_final: true` to get full utterance.

### UtteranceEnd Event
- Separate from `speech_final`
- Fires when `utterance_end_ms` timer expires after last word
- Use for "user done speaking" detection in voice agents
- Requires `interim_results=True`

---

## 7. VAD & Endpointing

### Endpointing (Pause Detection)
```python
endpointing=300  # ms of silence to trigger speech_final
```
- **10ms (default)**: Ultra-fast, for chatbots
- **300-500ms**: Natural conversation
- **`false`**: Disable entirely

### VAD Events (Speech Start Detection)
```python
vad_events=True  # Enable SpeechStarted messages
```
- Receives `SpeechStarted` events with timestamp
- Uses tonal analysis to differentiate speech from silence
- **Caveat**: Can trigger on background noise

### Recommended Voice Agent Config
```python
with client.listen.v1.connect(
    model="nova-3",
    encoding="mulaw",
    sample_rate="8000",
    interim_results=True,
    endpointing=300,
    vad_events=True,
    utterance_end_ms="1000",
    smart_format=True,
    punctuate=True,
) as connection:
    ...
```

---

## 8. Connection Lifecycle & Control Messages

### Opening
- SDK handles WebSocket handshake automatically
- Auth via `Authorization: Token <api_key>` header
- Query params for model/encoding/options

### KeepAlive (Prevent Timeout)
```json
{"type": "KeepAlive"}
```
- **MUST** send as text frame (not binary)
- Send every 3-5 seconds during silence
- 10-second timeout without audio/keepalive = `NET-0001` error + disconnect
- SDK handles this automatically with `keepalive: true` config
- Server does NOT respond to KeepAlive

### Finalize (Flush Buffer)
```json
{"type": "Finalize"}
```
- Forces processing of buffered audio
- Treats interim results as final
- Connection stays open after Finalize
- Response includes `from_finalize: true` field
- Use when you need immediate result (e.g., user pressed "send")

### CloseStream (Graceful Close)
```json
{"type": "CloseStream"}
```
- Terminates the connection
- SDK method: `connection.finish()`

### Lifecycle Summary
```
Open -> [Send Audio / KeepAlive] -> [Finalize] -> CloseStream -> Close
```

---

## 9. Raw WebSocket Approach (No SDK)

### Direct `websockets` Library
```python
import asyncio
import json
import websockets

DEEPGRAM_API_KEY = "your-key"
DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-3"
    "&encoding=mulaw"
    "&sample_rate=8000"
    "&interim_results=true"
    "&endpointing=300"
    "&vad_events=true"
    "&utterance_end_ms=1000"
    "&smart_format=true"
    "&punctuate=true"
)

async def transcribe_stream(audio_source):
    extra_headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}"
    }

    async with websockets.connect(
        DEEPGRAM_URL,
        extra_headers=extra_headers,
    ) as ws:
        # Receive task
        async def receive():
            async for msg in ws:
                data = json.loads(msg)
                msg_type = data.get("type", "")

                if msg_type == "Results":
                    transcript = (
                        data["channel"]["alternatives"][0]["transcript"]
                    )
                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)

                    if transcript:
                        print(f"[{'FINAL' if is_final else 'interim'}] {transcript}")

                    if speech_final:
                        print("--- utterance complete ---")

                elif msg_type == "SpeechStarted":
                    print(f"Speech started at {data.get('timestamp')}")

                elif msg_type == "UtteranceEnd":
                    print(f"Utterance ended at {data.get('last_word_end')}")

        # Send task
        async def send():
            async for chunk in audio_source:
                await ws.send(chunk)  # binary frame
            # Finalize
            await ws.send(json.dumps({"type": "CloseStream"}))

        # KeepAlive task
        async def keepalive():
            while True:
                try:
                    await asyncio.sleep(5)
                    await ws.send(json.dumps({"type": "KeepAlive"}))
                except websockets.exceptions.ConnectionClosed:
                    break

        await asyncio.gather(receive(), send(), keepalive())
```

### Raw WebSocket Response JSON Schema
```json
{
    "type": "Results",
    "channel_index": [0, 1],
    "duration": 1.04,
    "start": 0.0,
    "is_final": true,
    "speech_final": true,
    "channel": {
        "alternatives": [
            {
                "transcript": "hello world",
                "confidence": 0.98,
                "words": [
                    {
                        "word": "hello",
                        "start": 0.08,
                        "end": 0.56,
                        "confidence": 0.99,
                        "punctuated_word": "Hello"
                    },
                    {
                        "word": "world",
                        "start": 0.56,
                        "end": 1.04,
                        "confidence": 0.97,
                        "punctuated_word": "world."
                    }
                ]
            }
        ]
    },
    "metadata": {
        "request_id": "uuid-here",
        "model_info": {
            "name": "nova-3",
            "version": "2024.01.01"
        },
        "model_uuid": "uuid-here"
    },
    "from_finalize": false
}
```

---

## 10. SDK vs Raw WebSocket Tradeoffs

| Aspect | SDK (`deepgram-sdk`) | Raw (`websockets`) |
|--------|---------------------|-------------------|
| Size | 490 KB + deps (httpx, websockets) | ~50 KB (websockets only) |
| KeepAlive | Automatic | Manual implementation |
| Reconnection | Built-in (configurable) | Manual implementation |
| Type safety | Response objects with attributes | Raw dict parsing |
| Async | Both sync and async | Async only (websockets) |
| Maintenance | Deepgram maintains | We maintain |
| Overhead | Higher (full SDK surface) | Minimal |

### Recommendation for Definable
**Use raw `websockets`** because:
1. We only need streaming STT (not REST, TTS, agents, etc.)
2. Minimal dependency footprint
3. Full control over reconnection/error handling
4. The WebSocket protocol is simple: send binary audio, receive JSON text
5. ~400 lines of code vs pulling in a 490KB SDK with transitive deps
6. The `websockets` library is already a common Python dependency

---

## 11. Error Handling

### Common Error Codes
- `NET-0001`: Connection timeout (no audio/keepalive for 10s)
- `DATA-0000`: Invalid audio data
- Standard WebSocket close codes (1000=normal, 1001=going away, etc.)

### Reconnection Pattern
```python
async def connect_with_retry(max_retries=5):
    for attempt in range(max_retries):
        try:
            async with websockets.connect(url, extra_headers=headers) as ws:
                await handle_connection(ws)
                return  # Clean exit
        except websockets.exceptions.ConnectionClosed as e:
            if e.code == 1000:  # Normal close
                return
            backoff = min(2 ** attempt, 30)
            await asyncio.sleep(backoff)
        except Exception:
            backoff = min(2 ** attempt, 30)
            await asyncio.sleep(backoff)
    raise ConnectionError("Max retries exceeded")
```
