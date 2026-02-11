"""Live integration test — MistralReader against real asset files.

Reads every file in examples/assets/readers/ through MistralReader and
prints a summary for each: classification, page count, word count, first
300 chars of content, and any error.

Usage:
  MISTRAL_API_KEY=<key> python definable/examples/readers/05_mistral_ocr.py
"""

from pathlib import Path

from definable.media import File
from definable.readers.providers.mistral import MistralReader

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "readers"


def main():
  reader = MistralReader()

  files = sorted(ASSETS_DIR.iterdir())
  if not files:
    print(f"No files found in {ASSETS_DIR}")
    return

  print(f"Found {len(files)} files in {ASSETS_DIR}\n")
  print("=" * 80)

  for path in files:
    if path.is_dir():
      continue

    file = File(filepath=path, filename=path.name)

    # Get classification before reading
    classification = reader._classify(file)

    print(f"\n  File: {path.name}")
    print(f"  Classification: {classification}")

    result = reader.read(file)

    print(f"  Pages: {result.page_count}")
    print(f"  Words: {result.word_count}")

    if result.error:
      print(f"  Error: {result.error}")
    else:
      preview = result.content[:300].replace("\n", " ")
      print(f"  Preview: {preview}...")

    print("-" * 80)

  print("\nDone.")


if __name__ == "__main__":
  main()
