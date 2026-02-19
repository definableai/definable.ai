"""Multimodal agent — Files, images, and audio with readers.

Shows how to combine document processing (via readers) with native
multimodal inputs (images, audio) in a single agent call.

- Documents (CSV, JSON, etc.) → readers parse text → injected into context
- Images → passed directly to the model via images= for vision analysis
- Audio → passed directly to the model via audio= for transcription

Usage:
    # MockModel demo (no API key needed):
    python definable/examples/readers/06_multimodal_agent.py

    # Live model demo (requires API key):
    export OPENAI_API_KEY=your-key
    python definable/examples/readers/06_multimodal_agent.py --live
"""

import struct
import sys
import zlib

sys.path.append("/Users/hash/work/definable.ai/definable")

from definable.agent import Agent, MockModel
from definable.agent.tracing import Tracing
from definable.media import Audio, File, Image
from definable.reader import BaseReader

# ---------------------------------------------------------------------------
# Synthetic test data (no external files needed)
# ---------------------------------------------------------------------------

CSV_DATA = b"product,units,revenue\nWidget A,150,$4500\nWidget B,90,$3600\nWidget C,210,$8400"

JSON_DATA = b"""{
  "app": "multimodal-demo",
  "version": "1.0.0",
  "features": ["readers", "vision", "audio"]
}"""


def make_minimal_png() -> bytes:
  """Create a minimal valid 1x1 red PNG image."""
  # PNG signature
  sig = b"\x89PNG\r\n\x1a\n"

  def chunk(chunk_type: bytes, data: bytes) -> bytes:
    c = chunk_type + data
    return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

  # IHDR: 1x1 px, 8-bit RGB
  ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
  # IDAT: single red pixel, filter byte 0
  raw_row = b"\x00\xff\x00\x00"
  idat = zlib.compress(raw_row)

  return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def make_minimal_wav() -> bytes:
  """Create a minimal valid WAV file (~0.1s of silence, 16-bit mono 16kHz)."""
  sample_rate = 16000
  num_samples = 1600  # 0.1 seconds
  bits_per_sample = 16
  num_channels = 1
  byte_rate = sample_rate * num_channels * bits_per_sample // 8
  block_align = num_channels * bits_per_sample // 8
  data_size = num_samples * block_align

  header = struct.pack(
    "<4sI4s4sIHHIIHH4sI",
    b"RIFF",
    36 + data_size,
    b"WAVE",
    b"fmt ",
    16,  # chunk size
    1,  # PCM
    num_channels,
    sample_rate,
    byte_rate,
    block_align,
    bits_per_sample,
    b"data",
    data_size,
  )
  return header + b"\x00\x00" * num_samples


PNG_BYTES = make_minimal_png()
WAV_BYTES = make_minimal_wav()

# ---------------------------------------------------------------------------
# Part 1: Standalone reader — inspect ContentBlock types
# ---------------------------------------------------------------------------


def demo_standalone_reader():
  """Use BaseReader directly to see how different file types produce different ContentBlock types."""
  print("=" * 60)
  print("Part 1: Standalone Reader — ContentBlock types")
  print("=" * 60)

  reader = BaseReader()

  files = {
    "CSV document": File(content=CSV_DATA, filename="sales.csv", mime_type="text/csv"),
    "JSON document": File(content=JSON_DATA, filename="config.json", mime_type="application/json"),
    "PNG image": File(content=PNG_BYTES, filename="photo.png", mime_type="image/png"),
  }

  for label, file in files.items():
    result = reader.read(file)
    print(f"\n--- {label}: {result.filename} ---")
    print(f"  MIME type:  {result.mime_type}")
    print(f"  Blocks:     {len(result.blocks)}")

    for i, block in enumerate(result.blocks):
      print(f"  Block {i}:")
      print(f"    content_type: {block.content_type}")
      print(f"    as_text():    {block.as_text()[:80]}...")

      msg = block.as_message_content()
      print(f"    msg type:     {msg['type']}")

    # Show how as_text() behaves for the full output
    print(f"  result.content: {result.content[:80]}...")

  print()


# ---------------------------------------------------------------------------
# Part 2: MockModel agent — verify file content injection
# ---------------------------------------------------------------------------


def demo_mock_agent():
  """Run agent with MockModel to see how readers inject file content."""
  print("=" * 60)
  print("Part 2: MockModel Agent — File content injection")
  print("=" * 60)

  model = MockModel(responses=["I analyzed the CSV, image, and audio you provided."])

  agent = Agent(
    model=model,  # type: ignore[arg-type]
    instructions="You are a multimodal assistant. Analyze all provided inputs.",
    readers=True,
    tracing=Tracing(enabled=False),
  )

  # Documents go through readers (text extraction into context)
  csv_file = File(content=CSV_DATA, filename="sales.csv", mime_type="text/csv")
  json_file = File(content=JSON_DATA, filename="config.json", mime_type="application/json")

  # Images and audio go through native model params (binary → model)
  image = Image(content=PNG_BYTES, format="png", mime_type="image/png")
  audio_clip = Audio(content=WAV_BYTES, format="wav", mime_type="audio/wav")

  output = agent.run(
    "Analyze the sales CSV, the config JSON, describe the image, and transcribe the audio.",
    files=[csv_file, json_file],
    images=[image],
    audio=[audio_clip],
  )

  print(f"\nAgent response: {output.content}")
  print(f"Model called:   {model.call_count} time(s)")

  # Inspect what was sent to the model
  # Readers inject <file_contents> into the last user message, not the system message
  call = model.call_history[0]
  messages = call.get("messages", [])

  for msg in messages:
    content = getattr(msg, "content", "") or ""
    if "<file_contents>" in content:
      print("\nReaders injected file contents into user message:")
      start = content.index("<file_contents>")
      end = content.index("</file_contents>") + len("</file_contents>")
      snippet = content[start:end]
      for line in snippet.split("\n")[:10]:
        print(f"  {line}")
      if snippet.count("\n") > 10:
        print("  ...")
      break
  else:
    print("\nNote: <file_contents> not found in messages")

  print()


# ---------------------------------------------------------------------------
# Part 3: Live model agent (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------


def demo_live_agent():
  """Run agent with a real model — requires OPENAI_API_KEY."""
  print("=" * 60)
  print("Part 3: Live Model Agent — gpt-4o-mini")
  print("=" * 60)

  from definable.model.openai import OpenAIChat

  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a multimodal assistant. Summarize all inputs concisely.",
    readers=True,
  )

  csv_file = File(content=CSV_DATA, filename="sales.csv", mime_type="text/csv")
  image = Image(content=PNG_BYTES, format="png", mime_type="image/png")

  output = agent.run(
    "Summarize the sales data from the CSV. Also describe what you see in the image.",
    files=[csv_file],
    images=[image],
  )

  print(f"\nAgent response:\n{output.content}")
  print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  live = "--live" in sys.argv

  demo_standalone_reader()
  demo_mock_agent()

  if live:
    demo_live_agent()
  else:
    print("Skipping live model demo. Pass --live to run with OPENAI_API_KEY.")
