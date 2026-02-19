"""Provider override — Replace the default PDF parser with a custom one.

Demonstrates the priority-based override system. When you register a
custom parser with a higher priority, it takes precedence over
defaults for overlapping formats. This lets you:
- Swap a local parser for an API-based one (e.g., pypdf → Mistral OCR)
- Use different providers for different environments
- Add support for formats not covered by built-ins
"""

from typing import List, Set

from definable.agent import Agent
from definable.agent.testing import MockModel
from definable.media import File
from definable.reader.base import BaseReader
from definable.reader.models import ContentBlock, ReaderConfig
from definable.reader.parsers.base_parser import BaseParser
from definable.reader.registry import ParserRegistry


class CloudPDFParser(BaseParser):
  """Simulated cloud-based PDF parser."""

  def supported_mime_types(self) -> List[str]:
    return ["application/pdf"]

  def supported_extensions(self) -> Set[str]:
    return {".pdf"}

  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    return [
      ContentBlock(
        content_type="text",
        content="[CloudPDF] Extracted text with high-quality OCR",
        mime_type="application/pdf",
      )
    ]


# --- Default reader ---
default_reader = BaseReader()
pdf_file = File(content=b"%PDF", filename="report.pdf", mime_type="application/pdf")

parser = default_reader.get_parser(pdf_file)
print(f"Default PDF parser: {type(parser).__name__}")

# --- Override with cloud provider ---
registry = ParserRegistry()
registry.register(CloudPDFParser(), priority=200)  # Higher priority wins
custom_reader = BaseReader(registry=registry)

parser = custom_reader.get_parser(pdf_file)
print(f"Custom PDF parser: {type(parser).__name__}")

result = custom_reader.read(pdf_file)
print(f"Result: {result.content}")

# --- Use with Agent ---
model = MockModel(responses=["I analyzed the PDF."])
agent = Agent(model=model, readers=custom_reader)  # type: ignore[arg-type]
output = agent.run("Analyze this PDF.", files=[pdf_file])
print(f"\nAgent response: {output.content}")
