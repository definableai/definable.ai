"""Standalone usage — Use BaseReader without an agent.

The readers module can be used independently for file processing,
without any agent or model involvement. This is useful for:
- Pre-processing files in a pipeline
- Extracting text for indexing
- Building custom file processing workflows
"""

from definable.media import File
from definable.reader import BaseReader

# Create a reader with default parsers
reader = BaseReader()

# Process different file types
files = [
  File(content=b"Hello, world!", filename="greeting.txt", mime_type="text/plain"),
  File(content=b'{"name": "Alice", "age": 30}', filename="user.json", mime_type="application/json"),
  File(
    content=b"name,score\nAlice,95\nBob,87\nCharlie,92",
    filename="scores.csv",
    mime_type="text/csv",
  ),
]

for file in files:
  result = reader.read(file)
  print(f"\n--- {result.filename} ---")
  print(f"Content ({result.word_count} words):")
  print(result.content[:200])
  if result.error:
    print(f"Error: {result.error}")

# Check which parser handles a file
parser = reader.get_parser(files[0])
print(f"\nParser for .txt: {type(parser).__name__}")
