"""Desktop Bridge Validation — Phase 5

Tests the new desktop bridge by:
1. Checking bridge health and permissions
2. Taking a screenshot and saving it
3. Opening Safari and navigating to Hacker News
4. Taking a screenshot of the page
5. Using OCR to read the page content
6. Extracting the top 10 posts

Run: .venv/bin/python definable/examples/desktop/01_bridge_validation.py
Requires: Desktop bridge running (./desktop-bridge serve or ~/.definable/bin/desktop-bridge serve)
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from definable.agent.interface.desktop.bridge_client import BridgeClient


OUTPUT_DIR = Path("/tmp/definable-bridge-validation")


async def save_screenshot(client: BridgeClient, filename: str, max_width: int = 1024) -> str:
  """Take a screenshot and save it to disk."""
  jpeg_bytes = await client.capture_screen(max_width=max_width)
  path = OUTPUT_DIR / filename
  path.write_bytes(jpeg_bytes)
  print(f"  Screenshot saved: {path} ({len(jpeg_bytes):,} bytes)")
  return str(path)


async def main():
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  print("=" * 60)
  print("  Definable Desktop Bridge — Validation")
  print("=" * 60)
  print()

  async with BridgeClient() as client:
    # ── Step 1: Health Check ──
    print("Step 1: Health Check")
    health = await client.health()
    perms = health.get("permissions", {})
    print(f"  Status: {health.get('status', 'unknown')}")
    print(f"  Accessibility: {'✓' if perms.get('accessibility') else '✗'}")
    print(f"  Screen Recording: {'✓' if perms.get('screen_recording') else '✗'}")
    print(f"  Full Disk Access: {'✓' if perms.get('full_disk_access') else '✗'}")
    print()

    # ── Step 2: System Info ──
    print("Step 2: System Info")
    info = await client.system_info()
    print(f"  Hostname: {info.get('hostname')}")
    print(f"  OS: {info.get('os_version')}")
    print(f"  CPU: {info.get('cpu')}")
    print(f"  Memory: {info.get('memory_gb'):.1f} GB")
    print()

    # ── Step 3: Initial Screenshot ──
    print("Step 3: Taking initial screenshot...")
    await save_screenshot(client, "01_initial.jpg")
    print()

    # ── Step 4: List Running Apps ──
    print("Step 4: Running Apps")
    apps = await client.list_apps()
    for app in apps[:5]:
      print(f"  {app.name} (pid={app.pid})")
    if len(apps) > 5:
      print(f"  ... and {len(apps) - 5} more")
    print()

    # ── Step 5: Open Safari ──
    print("Step 5: Opening Safari...")
    try:
      pid = await client.open_app("Safari")
      print(f"  Safari opened (pid={pid})")
    except Exception as e:
      print(f"  Failed to open Safari: {e}")
      return
    await asyncio.sleep(2)
    print()

    # ── Step 6: Navigate to Hacker News via AppleScript ──
    print("Step 6: Navigating to news.ycombinator.com...")
    script = """
    tell application "Safari"
      activate
      if (count of documents) is 0 then
        make new document with properties {URL:"https://news.ycombinator.com"}
      else
        set URL of document 1 to "https://news.ycombinator.com"
      end if
    end tell
    """
    result = await client.run_applescript(script)
    if result.get("error"):
      print(f"  AppleScript error: {result['error']}")
      # Fallback: use open URL
      print("  Falling back to open_url...")
      await client.open_url("https://news.ycombinator.com")
    else:
      print("  Navigation started")
    await asyncio.sleep(5)  # Wait for page load
    print()

    # ── Step 7: Screenshot of HN ──
    # Safari is on display 1 (second monitor), capture that
    print("Step 7: Taking screenshot of Hacker News...")
    # Try each display to find Safari
    for display_idx in range(3):
      try:
        jpeg_bytes = await client.capture_screen(display=display_idx, max_width=1280)
        path = OUTPUT_DIR / f"02_hackernews_display{display_idx}.jpg"
        path.write_bytes(jpeg_bytes)
        print(f"  Display {display_idx}: saved {path} ({len(jpeg_bytes):,} bytes)")
      except Exception:
        break
    print()

    # Also get page content via AppleScript (more reliable than OCR)
    print("Step 7b: Getting HN content via AppleScript...")
    js_script = """
    tell application "Safari"
      set pageText to do JavaScript "
        var titles = [];
        var rows = document.querySelectorAll('.titleline > a');
        for (var i = 0; i < Math.min(rows.length, 10); i++) {
          titles.push((i+1) + '. ' + rows[i].textContent);
        }
        titles.join('\\n');
      " in document 1
      return pageText
    end tell
    """
    js_result = await client.run_applescript(js_script)
    hn_content = js_result.get("output", "")
    if hn_content:
      print("  Top HN posts via JavaScript:")
      for line in hn_content.strip().split("\n")[:10]:
        print(f"    {line}")
    elif js_result.get("error"):
      print(f"  JS error: {js_result['error']}")
    print()

    # ── Step 8: OCR the second display (where Safari is) ──
    print("Step 8: Reading page with OCR (display 1)...")
    # Use the shell to get HN content as a fallback
    shell_result = await client.run_shell(["curl", "-s", "https://news.ycombinator.com"], timeout=10)
    if shell_result.get("success"):
      import re

      html = shell_result.get("stdout", "")
      titles = re.findall(r'class="titleline"><a[^>]*>([^<]+)</a>', html)
      if titles:
        print(f"  Found {len(titles)} HN posts via curl:")
        for i, t in enumerate(titles[:10], 1):
          print(f"    {i}. {t}")

    # Also do OCR on primary display for reference
    ocr_result = await client.ocr_screen()
    full_text = ocr_result.get("text", "")
    elements = ocr_result.get("elements", [])
    print(f"  OCR detected {len(elements)} text elements")
    print(f"  Total text length: {len(full_text)} chars")
    print()

    # ── Step 9: Extract top posts ──
    print("Step 9: Extracting top Hacker News posts...")
    lines = full_text.split("\n")
    # HN posts have numbered titles like "1. Title here"
    posts = []
    for line in lines:
      stripped = line.strip()
      if not stripped:
        continue
      # Check if line starts with a number followed by a period
      parts = stripped.split(".", 1)
      if len(parts) >= 2 and parts[0].strip().isdigit():
        num = int(parts[0].strip())
        title = parts[1].strip()
        if 1 <= num <= 30 and len(title) > 5:
          posts.append((num, title))

    if posts:
      print(f"  Found {len(posts)} numbered posts:")
      for num, title in posts[:10]:
        print(f"    {num}. {title}")
    else:
      # Fallback: just print interesting lines from the OCR
      print("  Could not parse numbered posts. OCR text excerpt:")
      interesting = [l for l in lines if len(l.strip()) > 20 and not l.strip().startswith(("http", "Hacker", "new |", "past |"))]
      for line in interesting[:10]:
        print(f"    {line.strip()}")
    print()

    # ── Step 10: Window list ──
    print("Step 10: Window List")
    windows = await client.list_windows()
    safari_windows = [w for w in windows if w.app == "Safari"]
    for w in safari_windows:
      print(f"  Safari: '{w.title}' at ({w.bounds.x}, {w.bounds.y}) {w.bounds.width}x{w.bounds.height}")
    print()

    # ── Step 11: Final screenshot ──
    print("Step 11: Final screenshot...")
    await save_screenshot(client, "03_final.jpg", max_width=1280)
    print()

    # Save OCR text for reference
    ocr_path = OUTPUT_DIR / "ocr_output.txt"
    ocr_path.write_text(full_text)
    print(f"  OCR text saved: {ocr_path}")

  print()
  print("=" * 60)
  print("  Validation Complete!")
  print(f"  Screenshots saved in: {OUTPUT_DIR}")
  print("=" * 60)


if __name__ == "__main__":
  asyncio.run(main())
