# Twilio Voice API Research (2026-02-25)

> Research for building a real-time voice calling interface for Definable agents.

## Two Approaches: Media Streams vs ConversationRelay

### 1. Media Streams (Low-Level, Full Control)
- **What**: Raw audio streaming over WebSocket from phone calls
- **Audio format**: `audio/x-mulaw`, 8000 Hz, mono, base64-encoded
- **Bidirectional**: Use `<Connect><Stream url="wss://...">` TwiML
- **Unidirectional**: Use `<Start><Stream url="wss://...">` TwiML (fork-only)
- **You handle**: STT, LLM, TTS, interruption detection, buffering, silence detection
- **Use when**: Need full audio control, custom STT/TTS, Speech-to-Speech models (OpenAI Realtime API)

### 2. ConversationRelay (Managed, Higher-Level) -- GA May 2025
- **What**: Managed voice AI orchestration -- Twilio handles STT + TTS, you handle LLM
- **Protocol**: WebSocket but sends/receives TEXT (not audio) -- Twilio does STT/TTS
- **STT providers**: Deepgram (nova-2/nova-3), Google (telephony model)
- **TTS providers**: ElevenLabs, Google, Amazon
- **Built-in**: Interruption/barge-in detection, silence detection, DTMF, multi-language
- **Latency**: <500ms median, <725ms p95
- **Use when**: Want fast development, text-based LLM interaction, managed STT/TTS

## Architecture: Media Streams (Raw Audio)

```
Phone Call -> Twilio PSTN -> TwiML webhook -> <Connect><Stream>
  -> WebSocket to your server (mulaw/8000 base64)
  -> Your server decodes audio
  -> STT (Deepgram/Whisper/etc)
  -> LLM (OpenAI/etc)
  -> TTS (ElevenLabs/etc)
  -> Encode to mulaw/8000 base64
  -> Send back via WebSocket
  -> Twilio plays to caller
```

### WebSocket Message Format (From Twilio)
```json
{"event": "connected", "protocol": "Call", "version": "1.0.0"}
{"event": "start", "start": {"streamSid": "MZ...", "callSid": "CA...", "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}
{"event": "media", "media": {"track": "inbound", "chunk": "1", "timestamp": "5", "payload": "<base64>"}}
{"event": "dtmf", "dtmf": {"track": "inbound_track", "digit": "1"}}
{"event": "mark", "mark": {"name": "custom_label"}}
{"event": "stop"}
```

### WebSocket Message Format (To Twilio -- bidirectional only)
```json
{"event": "media", "streamSid": "MZ...", "media": {"payload": "<base64 mulaw/8000>"}}
{"event": "mark", "streamSid": "MZ...", "mark": {"name": "identifier"}}
{"event": "clear", "streamSid": "MZ..."}  // flush audio buffer (for interruptions)
```

### Mark Protocol
- Send mark after media -> Twilio sends matching mark back when playback completes
- Useful for tracking what has been played to the caller
- Clear event flushes pending audio + triggers mark events for outstanding marks

## Architecture: ConversationRelay (Text-Based)

```
Phone Call -> Twilio PSTN -> TwiML webhook -> <Connect><ConversationRelay>
  -> Twilio STT (Deepgram/Google)
  -> WebSocket TEXT to your server (transcribed speech)
  -> Your server processes with LLM
  -> Send TEXT back via WebSocket
  -> Twilio TTS (ElevenLabs/Google/Amazon)
  -> Plays to caller
```

### ConversationRelay TwiML
```xml
<Connect>
  <ConversationRelay
    url="wss://your-server.com/ws"
    welcomeGreeting="Hello, how can I help?"
    ttsProvider="ElevenLabs"
    transcriptionProvider="Deepgram"
    voice="Rachel"
    language="en-US"
    interruptible="any"
    interruptSensitivity="high"
    dtmfDetection="true"
  >
    <Parameter name="agentId" value="my-agent" />
  </ConversationRelay>
</Connect>
```

### ConversationRelay WebSocket Messages
```json
// From ConversationRelay (caller speech transcribed)
{"type": "prompt", "voicePrompt": "What's the weather today?"}
{"type": "setup", "callSid": "CA...", "customParameters": {...}}
{"type": "interrupt", "utteranceUntilInterrupt": "The weather in"}
{"type": "dtmf", "digit": "1"}

// To ConversationRelay (your LLM response)
{"type": "text", "token": "The weather", "last": false}  // streaming
{"type": "text", "token": " is sunny.", "last": true}     // final token
// OR non-streaming:
{"type": "text", "token": "The weather is sunny today.", "last": true}
```

## Architecture: OpenAI Realtime API + Media Streams

```
Phone Call -> Twilio -> <Connect><Stream> -> WebSocket
  -> Your server (proxy)
  -> OpenAI Realtime API WebSocket (Speech-to-Speech, no STT/TTS needed)
  -> Audio flows bidirectionally through proxy
```

- OpenAI Realtime API does native S2S -- no separate STT/TTS step
- Audio format: both Twilio and OpenAI support mulaw/8000 (pcmu)
- Server is a thin audio proxy between two WebSockets
- OpenAI handles VAD (Voice Activity Detection), turn-taking, interruptions
- Lowest latency option for OpenAI-powered agents

## Python SDK (`twilio` package)

```bash
pip install twilio
```

Key classes:
- `twilio.twiml.voice_response.VoiceResponse` -- generate TwiML
- `twilio.twiml.voice_response.Connect` -- for Stream/ConversationRelay
- `twilio.rest.Client` -- REST API (make calls, manage numbers, etc.)
- `client.calls.create(to=, from_=, url=)` -- outbound calls

For WebSocket handling, use `fastapi` + `websockets` (not part of twilio SDK):
```python
from fastapi import FastAPI, WebSocket
from twilio.twiml.voice_response import VoiceResponse, Connect

app = FastAPI()

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request):
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{request.url.hostname}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    # Handle bidirectional audio streaming
    stream_sid = None
    async for message in websocket.iter_text():
        data = json.loads(message)
        if data["event"] == "start":
            stream_sid = data["start"]["streamSid"]
        elif data["event"] == "media":
            # data["media"]["payload"] is base64 mulaw audio
            # Process and send back
            pass
```

## Latency Breakdown

Target: 1,115ms mouth-to-ear (total turn gap)

| Component | Target | Upper Limit |
|-----------|--------|-------------|
| Network to media edge | 40ms | - |
| Buffering/decoding | 25ms | - |
| STT | 350ms | 500ms |
| LLM TTFT | 375ms | 750ms |
| TTS TTFB | 100ms | 250ms |
| Inter-service hops | ~30ms | - |
| **Total** | **~920ms** | **~1,500ms** |

ConversationRelay: <500ms median, <725ms p95 (includes STT+TTS)
Human expectation: 300-500ms feels natural

### Optimization Keys
- **Streaming is mandatory** for LLM (TTFT matters, not total generation time)
- **Colocation**: deploy near Twilio media edge
- **Persistent connections**: HTTP keep-alive, reuse WebSockets
- **Audio codec match**: avoid transcoding (mulaw/8000 throughout)
- **Smart endpointing**: context-aware silence detection vs fixed timeout

## Barge-in / Interruption Handling

### Media Streams (manual)
1. Detect caller speech during playback (your VAD or STT streaming)
2. Send `{"event": "clear", "streamSid": "MZ..."}` to flush audio buffer
3. Process the interrupting speech
4. Send new response audio

### ConversationRelay (built-in)
1. `interruptible="any"` enables barge-in
2. `interruptSensitivity="low|medium|high"` controls sensitivity
3. On interruption, CR sends: `{"type": "interrupt", "utteranceUntilInterrupt": "..."}`
4. Your server truncates conversation history at interruption point
5. Process new user input normally

## STT/TTS -- Twilio Does NOT Have Its Own

Twilio does NOT have built-in STT/TTS. They partner with:
- **STT**: Deepgram (nova-2, nova-3), Google (telephony)
- **TTS**: ElevenLabs, Google, Amazon
- These are available through ConversationRelay (managed)
- For Media Streams, you bring your own STT/TTS

Exception: `<Say>` verb uses basic TTS (Amazon Polly under the hood)
Exception: `<Gather>` verb uses basic STT for DTMF/speech input

## Decision Matrix for Definable

| Factor | Media Streams | ConversationRelay | OpenAI Realtime + Media Streams |
|--------|--------------|-------------------|---------------------------------|
| Latency | You control | <500ms managed | Lowest (S2S, no STT/TTS hop) |
| Complexity | High | Low | Medium |
| STT/TTS | BYO | Managed (Deepgram, ElevenLabs, etc) | None needed (S2S) |
| LLM flexibility | Any | Any (text-based) | OpenAI only |
| Audio control | Full | None (text only) | Full |
| Interruptions | Manual | Built-in | OpenAI VAD |
| Cost | Twilio + BYO STT/TTS | Twilio (includes STT/TTS) | Twilio + OpenAI Realtime |
| HIPAA | Manual | Eligible | ? |

### Recommendation for Definable VoiceInterface
- **ConversationRelay** for fast MVP -- text-based interaction fits Agent.arun() perfectly
- **Media Streams** for advanced use cases (custom STT/TTS, audio analysis)
- **OpenAI Realtime proxy** as optional high-performance mode
- All three could be modes of the same VoiceInterface
