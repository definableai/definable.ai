"""Shell, Camera, and Screen Recording — direct BridgeClient usage.

Demonstrates the new bridge capabilities without an LLM agent:
1. Run shell commands (ls, curl, brew)
2. List cameras and take a photo
3. Record the screen for 5 seconds

Run: .venv/bin/python definable/examples/desktop/03_shell_and_camera.py
Requires: Desktop bridge running (~/.definable/bin/desktop-bridge serve)
"""

import asyncio
from pathlib import Path

from definable.agent.interface.desktop.bridge_client import BridgeClient

OUTPUT_DIR = Path("/tmp/definable-desktop-demo")


async def main() -> None:
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  async with BridgeClient() as client:
    # ── Shell Commands ────────────────────────────────────────────────
    print("=== Shell Commands ===")

    result = await client.run_shell(["uname", "-a"])
    print(f"uname: {result.get('stdout', '').strip()}")

    result = await client.run_shell(["ls", "-la", "/tmp"], timeout=5)
    stdout = result.get("stdout", "")
    lines = stdout.strip().split("\n")
    print(f"ls /tmp: {len(lines)} entries")
    for line in lines[:5]:
      print(f"  {line}")
    if len(lines) > 5:
      print(f"  ... and {len(lines) - 5} more")

    # Curl a URL
    result = await client.run_shell(
      ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "https://httpbin.org/get"],
      timeout=10,
    )
    print(f"curl httpbin: HTTP {result.get('stdout', '?')}")
    print()

    # ── Camera ────────────────────────────────────────────────────────
    print("=== Camera ===")

    cameras = await client.camera_list()
    if cameras:
      for cam in cameras:
        print(f"  [{cam.get('id', '?')}] {cam.get('name', '?')} ({cam.get('position', '?')})")

      # Take a photo
      print("Taking photo...")
      try:
        jpeg_bytes = await client.camera_snap(facing="front", max_width=1280)
        path = OUTPUT_DIR / "camera_snap.jpg"
        path.write_bytes(jpeg_bytes)
        print(f"  Saved: {path} ({len(jpeg_bytes):,} bytes)")
      except Exception as e:
        print(f"  Camera snap failed: {e}")

      # Record a 3-second clip
      print("Recording 3s clip...")
      try:
        clip = await client.camera_clip(facing="front", duration_ms=3000)
        clip_path = clip.get("path", "")
        print(f"  Clip: {clip_path} ({clip.get('size_bytes', 0):,} bytes)")
      except Exception as e:
        print(f"  Camera clip failed: {e}")
    else:
      print("  No cameras found (or camera permission not granted)")
    print()

    # ── Screen Recording ──────────────────────────────────────────────
    print("=== Screen Recording ===")
    print("Recording screen for 5 seconds...")
    try:
      recording = await client.screen_record(
        screen_index=0,
        duration_ms=5000,
        fps=10,
      )
      rec_path = recording.get("path", "")
      print(f"  Recording: {rec_path} ({recording.get('size_bytes', 0):,} bytes)")
    except Exception as e:
      print(f"  Screen recording failed: {e}")

    print()
    print(f"All outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
  asyncio.run(main())
