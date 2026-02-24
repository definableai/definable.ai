# Voice AI Research: STT, TTS, and Real-time Voice Providers

> **Date**: 2026-02-25
> **Purpose**: Production-grade voice AI interface for Definable agents
> **Status**: Research complete. Ready for architecture decisions.

---

## Architecture Patterns

### 1. Cascading Pipeline (STT -> LLM -> TTS)
- Sequential processing: user speaks -> STT completes -> LLM processes -> TTS generates -> response plays
- **Typical latency: 2-4 seconds** (cumulative across all components)
- Component breakdown: STT (100-500ms) + LLM (200-2000ms) + TTS (200-800ms) + network (50-200ms)
- **Advantage**: Full control over each component. Text transcript available. Can use any LLM.
- **Disadvantage**: Latency stacks. Feels unnatural for conversation.

### 2. Speech-to-Speech (S2S) - Single Model
- One model processes audio in -> audio out directly (e.g., OpenAI Realtime API)
- **Typical latency: 200-300ms** (85% reduction vs cascading)
- **Advantage**: Ultra-low latency. More natural prosody. Handles interruptions natively.
- **Disadvantage**: Locked to one provider. Less control over intermediate text. Debugging harder.

### 3. Optimized Cascading (Streaming Pipeline)
- Each component streams to the next (no waiting for full completion)
- Frameworks like Pipecat orchestrate this: STT streams partial transcripts -> LLM streams tokens -> TTS streams audio chunks
- **Typical latency: 500-800ms** with Pipecat
- **Best middle ground**: Provider flexibility + reasonable latency

### Natural Conversation Threshold
- Human conversation expects 300-500ms response window
- Delays beyond 500ms feel unnatural
- Sub-300ms TTFB needed for truly conversational experience

---

## STT Providers (Real-time Streaming)

### Deepgram Nova-3 -- RECOMMENDED for streaming STT
- **Latency**: 200-300ms (partial results)
- **API**: WebSocket streaming (stateful, no per-packet handshake)
- **WER**: 6.84% median (54.2% better than next-best at 14.92%)
- **Audio**: 100ms chunks optimal (200ms for degraded networks)
- **Price**: Starting at $0.0043/min
- **VAD**: Server-side included
- **Languages**: 36+
- **Key strength**: Purpose-built for real-time. Best accuracy-to-latency ratio.

### AssemblyAI Universal-Streaming
- **Latency**: ~307ms P50 word emission (41% faster than Deepgram Nova-3 at 516ms in their benchmark)
- **P99 Latency**: 1,012ms vs Deepgram's 1,907ms
- **API**: WebSocket
- **Features**: Intelligent endpointing (semantic + acoustic), word-level timestamps, confidence scores
- **Languages**: EN, ES, FR, DE, IT, PT (expanding)
- **Price**: ~$0.0065/min (real-time tier)
- **Key strength**: Immutable transcripts, smart endpointing, unlimited concurrent streams

### Google Cloud Speech-to-Text v2 (Chirp 3)
- **Latency**: Sub-second (not precise public numbers), estimated 300-500ms
- **API**: gRPC streaming (StreamingRecognize)
- **Features**: 100ms frame size recommended
- **Price**: $0.003/min (dynamic batch), $0.016/min (standard)
- **Languages**: 125+
- **Key strength**: Language coverage, enterprise compliance

### Azure Speech Services
- **Latency**: ~1-2 seconds typical; 3-5 seconds cold start
- **API**: WebSocket via Speech SDK
- **Features**: 140+ languages, custom speech models
- **Price**: $0.016/min (standard)
- **Key weakness**: Cold start latency. Needs warm endpoint strategy.

### OpenAI Whisper API
- **NOT real-time**. Designed for offline/batch transcription.
- Custom streaming hacks get 500ms+ latency with boundary errors
- **Price**: Batch at $0.36/hour (~$0.006/min)
- **Use for**: File transcription only. Not for voice agents.
- **Alternative**: OpenAI Realtime API for real-time voice

---

## TTS Providers (Low-latency Streaming)

### Cartesia Sonic-3 -- RECOMMENDED for lowest latency TTS
- **TTFB**: 40-90ms (fastest in industry)
- **API**: WebSocket streaming
- **Features**: Laughter, breathing, emotional inflections, voice cloning
- **Languages**: 15+
- **Price**: ~$0.04/1K chars
- **Key strength**: Purpose-built for real-time conversation. State-space model architecture.

### ElevenLabs Flash v2.5
- **TTFB**: ~75ms (Flash model)
- **TTFB**: 250-300ms (Turbo v2.5, higher quality)
- **API**: WebSocket + REST streaming
- **Features**: Best voice quality. Voice cloning. chunk_length_schedule tuning.
- **Regions**: USA, Netherlands, Singapore
- **Price**: $180/1M chars (expensive). Subscription tiers: $5-$330/month.
- **Key strength**: Best quality voices. Good for premium experiences.

### Deepgram Aura-2
- **TTFB**: 90ms optimized, sub-200ms baseline
- **API**: WebSocket + REST streaming
- **Features**: Entity-aware text normalization (addresses, phone numbers, account numbers without SSML)
- **Price**: $0.030/1K chars ($0.01/min Falcon model)
- **Key strength**: Cost-effective. Great for enterprise/telephony.

### OpenAI TTS
- **TTFB**: Higher than competitors (not competitive for real-time)
- **API**: REST streaming (chunked response)
- **Models**: tts-1 (faster, lower quality), tts-1-hd (slower, higher quality)
- **Price**: $15-30/1M chars (cheapest for quality)
- **Key weakness**: Not designed for real-time conversational agents
- **Best for**: Offline generation, batch content

### Google Cloud TTS (Neural2 voices)
- **TTFB**: 200-250ms (Neural2), up to 3.5 seconds (Chirp3-HD)
- **API**: REST + streaming (text streaming for Chirp3)
- **Price**: $0.016/1M chars (Neural2), $0.016/1M chars (Studio)
- **Languages**: 50+, 380+ voices
- **Key strength**: Language variety. Neural2 is fast enough for real-time.

### Azure TTS
- **TTFB**: Variable. Cold starts of 3-5 seconds.
- **API**: Speech SDK (WebSocket under the hood)
- **Features**: 400+ voices, 140+ languages, Custom Neural Voice
- **Price**: $0.016/1M chars
- **Key weakness**: Cold start latency. Need warm endpoints.

---

## OpenAI Realtime API -- Deep Dive

### Overview
- Voice-to-voice natively. Single model handles audio in -> audio out.
- Latest model: `gpt-realtime-2025-08-28` (GA)
- **Approach**: Speech-to-speech (S2S), NOT cascading pipeline

### Transport
- **WebRTC**: For client-side (browser). Peer-to-peer. Lowest latency.
- **WebSocket**: For server-side (phone bots, backend agents). Bidirectional events.

### Audio Formats
- PCM16 at 24kHz (high-fidelity)
- G.711 at 8kHz (telephony/VoIP)
- Stream in 20-100ms chunks via `input_audio_buffer.append` events

### Function Calling / Tool Use
- Full function calling support during voice sessions
- ComplexFuncBench score: 66.5% (up from 49.7%)
- **Async function calling**: Long-running calls don't block conversation flow
- **Sideband control channel**: Server monitors session, updates instructions, handles tool calls
- Architecture: Client (WebRTC) + Server (WebSocket) to same session

### Interruption Handling (Barge-in)
- Built-in server-side VAD
- Model detects user speech and stops generating automatically
- No client-side VAD needed (but can add for better UX)

### Production Architecture (3 layers)
1. **Edge (Client)**: Mic input, audio playback, WebRTC negotiation
2. **Control Plane (Backend)**: Auth, ephemeral tokens, context rehydration, tool execution
3. **Model Session**: Persistent WebSocket, bidirectional audio + events

### Price
- $1.00/hour (~$0.0167/min) -- 75% more expensive than cascading pipeline alternatives

### Best For
- Premium conversational experiences requiring ultra-low latency
- Use cases where natural interruption handling matters
- When you want to minimize infrastructure complexity

---

## Voice Activity Detection (VAD)

### Server-side VAD (included with providers)
- OpenAI Realtime: Built-in server VAD
- Deepgram: Built-in server VAD
- AssemblyAI: Intelligent endpointing (semantic + acoustic)

### Client-side VAD Options (for custom pipelines)
| VAD Engine | Accuracy | Latency | Notes |
|-----------|----------|---------|-------|
| Cobra VAD (Picovoice) | 99% | Low | Best for production. Cross-platform SDKs. |
| Silero VAD | Good | 85-100ms | Open source. Deep learning. ~10 speech cut-offs per session. |
| WebRTC VAD (Google) | Moderate | Low | Open source. ~62 speech cut-offs per session. |

### Barge-in Detection Requirements
- VAD latency: 85-100ms
- Barge-in stop latency: <200ms for natural feel
- Accuracy: 95%+ to avoid false triggers on background noise

---

## Interruption Handling (Barge-in)

### How It Works
1. User starts speaking while AI is outputting audio
2. VAD detects user speech
3. TTS output stops immediately
4. STT captures user input
5. New LLM turn begins

### Provider Support
- **OpenAI Realtime**: Native barge-in. Model handles it internally.
- **Deepgram Voice Agent API**: Built-in turn management
- **Pipecat framework**: Handles barge-in across any provider combination
- **Custom pipeline**: Need client-side VAD -> cancel TTS -> buffer STT -> restart LLM

---

## Audio Format Guide

| Format | Sample Rate | Bitrate | Use Case |
|--------|------------|---------|----------|
| PCM16 (LINEAR16) | 16-24kHz | ~256-384 kbps | Best quality for STT. Raw, uncompressed. |
| G.711 u-law | 8kHz | 64 kbps | Telephony/PSTN. Required for phone bots. |
| Opus | 8-48kHz | 6-510 kbps | WebRTC default. Best quality-to-bandwidth ratio. |
| WAV | Any | Variable | Container format. PCM16 inside usually. |
| MP3 | Any | 128-320 kbps | Lossy. Good for TTS output delivery. Not ideal for STT. |

### Key Rules
- Use mono audio, not stereo
- 16kHz minimum for quality STT (8kHz for telephony degrades accuracy significantly)
- 20ms frame size for streaming (balance latency vs overhead)
- Opus for browser-to-server (WebRTC handles encoding)
- PCM16 for server-to-STT (most ASR models expect this)

---

## Cost Comparison (per minute of voice conversation)

### STT Only
| Provider | Price/min | Notes |
|---------|-----------|-------|
| Deepgram Nova-3 | $0.0043 | Cheapest |
| Google Cloud STT | $0.003-0.016 | Batch vs standard |
| AssemblyAI | ~$0.0065 | Real-time tier |
| Azure Speech | $0.016 | Standard |
| OpenAI Whisper | ~$0.006 | Batch only |

### TTS Only (per ~150 words/min spoken)
| Provider | Price/1K chars | ~Price/min spoken |
|---------|---------------|-------------------|
| OpenAI TTS | $0.015-0.030 | ~$0.012-0.024 |
| Deepgram Aura-2 | $0.030 | ~$0.024 |
| Cartesia Sonic | ~$0.040 | ~$0.032 |
| Google Cloud TTS | $0.016 | ~$0.013 |
| ElevenLabs | $0.180 | ~$0.144 |

### Full Voice Agent (STT + LLM + TTS combined)
| Solution | ~Price/hour | Notes |
|---------|-------------|-------|
| Deepgram Voice Agent API | $4.50 | All-in-one |
| Cascading (Deepgram + GPT-4o-mini + Cartesia) | ~$3-5 | Estimated |
| OpenAI Realtime API | ~$1.00 (audio only) | + token costs |
| ElevenLabs Conversational AI | ~$5.90 | All-in-one |

---

## Framework: Pipecat (Open Source Orchestrator)

- **What**: Python framework for real-time voice + multimodal AI
- **GitHub**: github.com/pipecat-ai/pipecat
- **Latency**: 500-800ms round-trip typical
- **Supports**: Pluggable STT, LLM, TTS providers
- **Transport**: Daily (WebRTC), Twilio, custom WebSocket
- **Key value**: Provider-agnostic. Swap STT/TTS/LLM without rewriting.
- **Relevance to Definable**: Could be used as inspiration or integration layer for voice interface

---

## Recommendations for Definable Voice Interface

### Tier 1: Best Latency (Speech-to-Speech)
- **OpenAI Realtime API** via WebSocket
- Native voice-to-voice, function calling, barge-in
- Best for premium conversational agents
- Trade-off: Vendor lock-in, higher cost, less control

### Tier 2: Best Flexibility (Optimized Cascading)
- **STT**: Deepgram Nova-3 (WebSocket streaming)
- **LLM**: Any (OpenAI, DeepSeek, etc.) with streaming
- **TTS**: Cartesia Sonic-3 (40-90ms TTFB) or Deepgram Aura-2 (90ms)
- **VAD**: Silero VAD (open source) or Cobra VAD (production)
- **Orchestration**: Custom pipeline or Pipecat-inspired
- Expected total latency: 800-1200ms
- Trade-off: More infrastructure, but full provider choice

### Tier 3: Best Cost (Budget Pipeline)
- **STT**: Deepgram Nova-3 ($0.0043/min)
- **LLM**: GPT-4o-mini or DeepSeek
- **TTS**: Deepgram Aura-2 ($0.030/1K chars)
- All Deepgram = simpler billing, single vendor for audio

### Architecture Decision for Definable
The framework should support BOTH approaches:
1. `SpeechToSpeechInterface` -- wraps OpenAI Realtime API (or future S2S providers)
2. `VoicePipelineInterface` -- cascading with pluggable STT/TTS providers
Both implement a common `VoiceInterface` protocol for the agent layer.
