# Plivo Voice API Research (2026-02-25)

> Research for building a real-time voice calling interface for Definable agents.
> Companion to: twilio-voice-research.md, voice-ai-research.md

---

## 1. Plivo Voice API Fundamentals

### Call Flow Model
- **Incoming calls**: Caller dials Plivo number -> Plivo sends HTTP callback (POST/GET) to your Answer URL -> you return Plivo XML -> Plivo executes instructions
- **Outgoing calls**: You call `client.calls.create(answer_url=...)` -> Plivo dials destination -> on answer, Plivo fetches your Answer URL -> you return XML
- **XML-driven**: All call control is via Plivo XML (analogous to Twilio's TwiML)

### Webhook Callbacks
| Callback | When | Response Expected |
|----------|------|-------------------|
| `answer_url` (mandatory) | Call answered | Plivo XML |
| `ring_url` | Call starts ringing | None |
| `fallback_url` | answer_url unreachable | Plivo XML |
| `hangup_url` | Call disconnects | None |
| `action` (XML element) | After XML element executes (e.g., GetInput) | Plivo XML |
| `callbackUrl` (XML element) | Event notification | None |
| `statusCallbackUrl` (Stream) | Stream lifecycle events | None |

### Security
- All requests include: `X-Plivo-Signature-V3`, `X-Plivo-Signature-Ma-V3`, `X-Plivo-Signature-V3-Nonce`
- Use `CallUUID` as idempotency key
- Plivo retries webhooks automatically if no HTTP 200 returned

### Plivo XML Elements
| Element | Purpose |
|---------|---------|
| `<Speak>` | Text-to-speech |
| `<Play>` | Play audio file |
| `<GetDigits>` | Collect DTMF input |
| `<GetInput>` | Collect speech OR DTMF input (has built-in ASR) |
| `<Dial>` | Connect to another number/SIP |
| `<Conference>` | Conference room |
| `<MultiPartyCall>` | Advanced conferencing |
| `<Record>` | Record call/message |
| `<Stream>` | Real-time audio streaming via WebSocket |
| `<Redirect>` | Transfer call flow to another URL |
| `<Hangup>` | End call |
| `<Wait>` | Pause execution |
| `<PreAnswer>` | Play media before answering |
| `<DTMF>` | Send DTMF tones |

Max XML response size: 100 KB. Elements execute sequentially. Empty `<Response>` terminates call.

---

## 2. Audio Streaming (Real-Time WebSocket)

### Overview
Plivo's `<Stream>` element gives you raw audio from a live call over WebSocket. Supports both unidirectional (receive only) and bidirectional (send+receive).

### Stream XML Configuration
```xml
<Response>
  <Stream
    streamTimeout="3600"
    keepCallAlive="true"
    bidirectional="true"
    contentType="audio/x-mulaw;rate=8000"
    statusCallbackUrl="https://your-domain.com/stream-status">
    wss://your-domain.com/ws
  </Stream>
</Response>
```

### Key Parameters
- `bidirectional`: `true` for two-way audio (required for voice AI)
- `keepCallAlive`: `true` to prevent call from ending after XML execution (required for agents)
- `contentType`: Audio codec/format
- `streamTimeout`: Max stream duration (default 3600s)
- `statusCallbackUrl`: Webhook for stream lifecycle events

### Supported Audio Formats
| Codec | Sample Rate | Description | Use Case |
|-------|------------|-------------|----------|
| mu-law (PCMU) | 8 kHz | Native telephony, lowest latency | **Recommended** -- no transcoding |
| Linear PCM | 8 kHz | Uncompressed 16-bit | Standard quality |
| Linear PCM | 16 kHz | Wideband audio | High-fidelity STT models |

**Key insight**: mu-law 8kHz eliminates transcoding overhead, reducing latency ~50% vs Linear PCM.

### WebSocket Events (Plivo -> Your Server)

**Start Event**:
```json
{
  "event": "start",
  "start": {
    "streamId": "unique-id",
    "accountId": "your-account",
    "callId": "call-identifier"
  }
}
```

**Media Event** (incoming audio from caller):
```json
{
  "event": "media",
  "sequenceNumber": "3",
  "media": {
    "track": "inbound",
    "chunk": "1",
    "timestamp": "1711216222735",
    "payload": "<base64-encoded audio>"
  },
  "streamId": "...",
  "extra_headers": "{X-PH-key1: value1}"
}
```

**Stop Event**: `{"event": "stop"}` -- no more audio
**DTMF Event**: Key presses during call

### WebSocket Events (Your Server -> Plivo)

**playAudio** (send audio back to caller):
```json
{
  "event": "playAudio",
  "media": {
    "contentType": "audio/x-mulaw",
    "sampleRate": 8000,
    "payload": "<base64-encoded audio>"
  }
}
```

**clearAudio** (interrupt/flush buffered audio):
```json
{
  "event": "clearAudio",
  "streamId": "..."
}
```

**checkPoint** (confirm audio delivery):
- Send checkpoint -> Plivo responds with `played` event confirming all audio up to that point was delivered

### Status Callback Events
- **StartStream**: Audio streaming began
- **StopStream**: Streaming terminated
- **DroppedStream**: WebSocket failure during init or streaming, or slow connection
- **DegradedStream**: Buffer reaching capacity (triggered at 30%, 60%, 90%)

### Connection Handling
- Plivo retries WebSocket connection **twice** before disconnecting
- Plivo buffers audio packets up to **40 seconds**
- Auto-disconnects WebSocket when call terminates (no manual cleanup needed)
- DegradedStream events at 30/60/90% buffer capacity

---

## 3. STT/TTS Capabilities

### Built-in ASR via `<GetInput>`
Plivo HAS built-in speech recognition (unlike Twilio which has none natively):

**Speech Models**:
- `default`: General long-form audio
- `command_and_search`: Short commands, voice search
- `phone_call`: Optimized for phone call audio quality

**Key Attributes**:
- `inputType`: `speech`, `dtmf`, or `dtmf speech`
- `language`: 27+ languages (default `en-US`)
- `speechModel`: Model selection
- `hints`: Comma-separated phrases for accuracy boost (max 500 phrases, 10K chars)
- `speechEndTimeout`: Silence detection (2-10s or `auto`)
- `executionTimeout`: Max wait (5-60s)
- `profanityFilter`: Filter profane words
- `interimSpeechResultsCallback`: URL for real-time partial transcriptions

**Transcription Results**:
- `Speech`: Transcribed text
- `SpeechConfidenceScore`: 0.0-1.0
- `InputType`: speech or dtmf
- Interim: `StableSpeech`, `UnstableSpeech`, `Stability`, `SequenceNumber`

**Pricing**: $0.02 per 15 seconds of speech recognition via GetInput

### Agentic STT (Purpose-Built for Voice AI)
Plivo's advanced STT model combines 3 functions competitors need separate services for:
1. **Real-time noise cancellation** -- filters background audio
2. **Interruption handling** -- manages cross-talk naturally
3. **Turn detection** -- knows when caller finished speaking

Available as standalone service or part of full voice AI platform.

### Built-in TTS via `<Speak>`
Basic TTS for playing synthesized speech. Supports multiple languages and voices.

### Full Voice AI Stack (Managed)
Plivo offers modular options:
- **Full platform**: STT + LLM + TTS + telephony pre-configured
- **Agentic STT only**: Just noise cancellation + interruption + turn detection
- **Audio streaming**: Raw WebSocket (BYO STT/TTS/LLM)
- **SIP trunking**: BYO everything

### Third-Party Integration
Common pattern: Plivo audio streaming + Deepgram (STT) + OpenAI (LLM) + ElevenLabs (TTS)

---

## 4. Architecture: Real-Time Voice AI with Plivo

### Pattern A: Audio Streaming + BYO STT/TTS/LLM
```
Phone Call -> Plivo PSTN -> Answer URL webhook -> <Stream bidirectional="true">
  -> WebSocket to your server (mulaw/8000 base64)
  -> Decode audio
  -> STT (Deepgram/AssemblyAI/etc)
  -> LLM (OpenAI/DeepSeek/etc) with streaming
  -> TTS (Cartesia/ElevenLabs/Deepgram Aura/etc)
  -> Encode to mulaw/8000 base64
  -> Send back via WebSocket (playAudio event)
  -> Plivo plays to caller
```

### Pattern B: Plivo Managed Voice AI
- Use Plivo's full platform with pre-configured STT + LLM + TTS
- Sub-700ms latency
- Less control but faster deployment

### Pattern C: GetInput for Simple IVR
```
Phone Call -> Plivo -> <Speak>How can I help?</Speak>
  -> <GetInput inputType="speech" action="/process">
  -> Plivo ASR transcribes speech
  -> POST to your /process endpoint with transcript
  -> Your server calls LLM
  -> Return <Speak>response</Speak> XML
```
Good for simple command-response, NOT for natural conversation (high latency per turn).

### Pattern D: Pipecat Framework Integration
```python
from pipecat.transports.websocket import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.plivo import PlivoFrameSerializer

serializer = PlivoFrameSerializer(
    stream_id=stream_id,
    call_id=call_id,
    auth_id=os.getenv("PLIVO_AUTH_ID"),
    auth_token=os.getenv("PLIVO_AUTH_TOKEN"),
)

transport = FastAPIWebsocketTransport(
    websocket=websocket,
    params=FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        add_wav_header=False,
        serializer=serializer,
    ),
)

# Pipeline: audio_in -> STT -> LLM -> TTS -> audio_out
task = PipelineTask(
    pipeline,
    params=PipelineParams(
        audio_in_sample_rate=8000,
        audio_out_sample_rate=8000,
    ),
)
```

Auto-terminates calls when pipeline ends (if credentials provided to serializer).

---

## 5. Python SDK (`plivo` package)

### Installation
```bash
pip install plivo
```

### Authentication
```python
import plivo
client = plivo.RestClient(auth_id='AUTH_ID', auth_token='AUTH_TOKEN')
# Or via env: PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN
```

### Make Outbound Call
```python
call = client.calls.create(
    from_='the_from_number',
    to_='the_to_number',
    answer_url='https://your-domain.com/answer'
)
```

### Generate XML
```python
from plivo import plivoxml

response = plivoxml.ResponseElement()
response.add_speak('Hello, world!')

# Stream element
stream = response.add_stream(
    'wss://your-domain.com/ws',
    bidirectional='true',
    keep_call_alive='true',
    content_type='audio/x-mulaw;rate=8000',
    status_callback_url='https://your-domain.com/status'
)

print(response.to_string())
```

### SDK Scope
- Voice: calls CRUD, XML generation, number management
- SMS: send/receive
- WhatsApp: templates, interactive
- PHLO: trigger workflows
- Number lookup
- **Does NOT include**: WebSocket handling (use `websockets` or `fastapi` for that)

---

## 6. Latency Considerations

### Plivo Infrastructure
- 7 global PoPs: California, Virginia, Frankfurt, Mumbai, Singapore, Sydney, Sao Paulo
- 99.99% uptime since 2011
- Sub-2-second failover on backbone issues
- Routes calls through edge closest to caller

### Latency Budget (Audio Streaming + BYO pipeline)
| Component | Target | Notes |
|-----------|--------|-------|
| Codec Processing (mu-law) | ~0ms | Native format, no transcoding |
| Network (WebSocket) | <100ms | Deploy near caller regions |
| STT | <200ms | Streaming providers (Deepgram) |
| LLM Processing | <500ms | Stream responses, use fast models |
| TTS | <200ms | Streaming providers (Cartesia, Deepgram) |
| **Total End-to-End** | **<1 second** | Target for natural conversation |

### Plivo Managed Platform
- Sub-700ms latency (full STT + LLM + TTS)
- Competitive with Twilio ConversationRelay (<500ms median)

### Optimization Keys
1. Use mu-law 8kHz (zero transcoding overhead)
2. Co-locate WebSocket server near Plivo PoPs
3. Stream everything (STT partial results, LLM tokens, TTS chunks)
4. Use `clearAudio` for instant barge-in
5. Optimize model size (GPT-4o-mini vs GPT-4o)

---

## 7. Interruption & Silence Handling

### Barge-in with Audio Streaming
1. Detect caller speech during AI playback (your VAD or STT streaming)
2. Send `{"event": "clearAudio", "streamId": "..."}` to flush Plivo's audio buffer
3. Process the interrupting speech
4. Send new response audio via `playAudio`

### Checkpoint Protocol
- Send `checkPoint` event -> Plivo responds with `played` event
- Confirms all audio up to checkpoint was played to caller
- Use to track what the caller actually heard (important for interruption context)

### Silence Detection
- **GetInput**: `speechEndTimeout` (2-10s or `auto`) -- built-in
- **Audio Streaming**: BYO VAD (Silero, Cobra, or STT provider's VAD)
- **Plivo Agentic STT**: Built-in turn detection

### Buffer Management
- Plivo buffers up to 40 seconds of audio
- DegradedStream events at 30%, 60%, 90% capacity
- On buffer full: stream may be dropped

---

## 8. Plivo vs Twilio Comparison

### Feature Parity
| Feature | Plivo | Twilio |
|---------|-------|--------|
| Raw audio WebSocket streaming | Yes (`<Stream>`) | Yes (`<Connect><Stream>`) |
| Bidirectional audio | Yes | Yes |
| Audio format | mu-law 8kHz (+ Linear PCM 8/16kHz) | mu-law 8kHz only |
| DTMF during stream | Yes | Yes (inbound only) |
| Audio buffer flush | `clearAudio` | `clear` event |
| Playback confirmation | `checkPoint` -> `played` | `mark` -> `mark` |
| Built-in ASR | Yes (`<GetInput>` with 3 models) | Basic (`<Gather>`) |
| Built-in TTS | Yes (`<Speak>`) | Yes (`<Say>`, Polly) |
| Managed voice AI | Yes (full platform + Agentic STT) | Yes (ConversationRelay) |
| Pipecat integration | Yes (PlivoFrameSerializer) | Yes (TwilioFrameSerializer) |
| Visual workflow builder | PHLO (simpler) | Studio (more powerful) |

### What Plivo Has That Twilio Doesn't
- **Agentic STT**: Purpose-built for voice AI (noise cancellation + interruption + turn detection in one model)
- **More audio formats**: Linear PCM 8kHz + 16kHz in addition to mu-law
- **Built-in ASR models**: 3 speech models in GetInput (Twilio's Gather is basic)
- **Full managed voice AI platform**: STT + LLM + TTS pre-integrated
- **Automated QA testing**: Built-in agent testing and scoring
- **30-35% lower pricing** on like-for-like traffic

### What Twilio Has That Plivo Doesn't
- **ConversationRelay**: Higher-level text-based abstraction (you get transcribed text, send text back; Twilio handles STT/TTS)
- **Multi-track streaming**: Up to 4 unidirectional tracks per call (Plivo: single stream)
- **Voice Intelligence**: Deep analytics and insights platform
- **Larger ecosystem**: More third-party integrations, larger community
- **More documentation**: More code examples, tutorials, quickstarts
- **<Connect> nesting**: Stream inside Connect for more complex call flows

### Pricing Comparison
| Item | Plivo | Twilio |
|------|-------|--------|
| Audio streaming | $0.003-0.004/min/stream | Included with voice |
| Voice (inbound) | ~30-35% less | Higher |
| Built-in ASR | $0.02/15 seconds | Basic (via Gather) |
| Managed voice AI | $0.05/min (all-inclusive) | ConversationRelay pricing varies |

### Decision Factors
- **Choose Plivo if**: Cost-sensitive, want built-in ASR, need 16kHz audio, want managed voice AI platform, simpler use case
- **Choose Twilio if**: Need ConversationRelay (text-based abstraction), need Voice Intelligence analytics, need multi-track streaming, larger team with existing Twilio infra
- **Both work equally well for**: Audio streaming + BYO pipeline (Deepgram + OpenAI + Cartesia/ElevenLabs)

---

## 9. Implementation Skeleton for Definable

### FastAPI + Plivo Audio Streaming
```python
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
import json, base64

app = FastAPI()

@app.post("/answer")
async def answer_call(request: Request):
    """Plivo calls this when a call is answered."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <Response>
      <Stream bidirectional="true" keepCallAlive="true"
             contentType="audio/x-mulaw;rate=8000"
             statusCallbackUrl="https://your-domain.com/stream-status">
        wss://your-domain.com/ws
      </Stream>
    </Response>"""
    return Response(content=xml, media_type="application/xml")

@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    stream_id = None

    async for message in websocket.iter_text():
        data = json.loads(message)

        if data["event"] == "start":
            stream_id = data["start"]["streamId"]
            # Initialize STT, LLM, TTS services

        elif data["event"] == "media":
            audio_bytes = base64.b64decode(data["media"]["payload"])
            # Feed to STT -> LLM -> TTS pipeline

            # When TTS generates audio, send back:
            # await websocket.send_text(json.dumps({
            #     "event": "playAudio",
            #     "media": {
            #         "contentType": "audio/x-mulaw",
            #         "sampleRate": 8000,
            #         "payload": base64.b64encode(tts_audio).decode()
            #     }
            # }))

        elif data["event"] == "stop":
            break

@app.post("/stream-status")
async def stream_status(request: Request):
    """Handle stream lifecycle events."""
    form = await request.form()
    event = form.get("Event")  # StartStream, StopStream, DroppedStream, DegradedStream
    # Log/handle accordingly
    return Response(status_code=200)
```

---

## 10. Key Takeaways for Definable VoiceInterface

1. **Plivo is a viable alternative to Twilio** for voice AI with feature parity on audio streaming
2. **Audio streaming protocol is nearly identical** to Twilio Media Streams -- same concepts (WebSocket, mulaw/8000, base64, bidirectional, buffer clear)
3. **Plivo has MORE built-in STT** than Twilio (GetInput with 3 speech models vs Twilio's basic Gather)
4. **Plivo's Agentic STT** is a differentiator -- combined noise cancellation + interruption + turn detection
5. **30-35% cheaper** than Twilio on voice minutes
6. **Pipecat supports both** Plivo and Twilio with minimal code changes (just swap serializer)
7. **Missing**: No equivalent to Twilio's ConversationRelay (managed text-based abstraction)
8. **For Definable**: A `VoiceInterface` could support both Plivo and Twilio via a common protocol, since the WebSocket audio streaming is nearly identical between them
