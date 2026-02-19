"""Format parsers — stateless bytes-to-ContentBlock converters.

Every parser receives raw bytes and returns a list of ContentBlock.
Parsers never perform I/O; the async boundary lives in BaseReader.
"""

from definable.reader.parsers.base_parser import BaseParser

__all__ = ["BaseParser"]
