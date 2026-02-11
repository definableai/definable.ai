"""ODS parser — extracts text from OpenDocument spreadsheets using odfpy.

Requires optional dependency: ``odfpy>=1.4.0``
"""

import io
from typing import List, Set

from definable.readers.models import ContentBlock, ReaderConfig
from definable.readers.parsers.base_parser import BaseParser


def _import_odf():
  try:
    from odf import opendocument, table, text

    return opendocument, table, text
  except ImportError as e:
    raise ImportError(
      "OdsParser requires the 'odfpy' package. Install it with: pip install 'odfpy>=1.4.0' or: pip install 'definable-ai[readers]'"
    ) from e


class OdsParser(BaseParser):
  """Parses ODS files using odfpy (local, no API calls).

  Extracts sheets as tab-separated text with a guard against
  very large spreadsheets.
  """

  def __init__(self, max_rows: int = 1000) -> None:
    _import_odf()
    self.max_rows = max_rows

  def supported_mime_types(self) -> List[str]:
    return ["application/vnd.oasis.opendocument.spreadsheet"]

  def supported_extensions(self) -> Set[str]:
    return {".ods"}

  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    opendocument, table_mod, text_mod = _import_odf()
    doc = opendocument.load(io.BytesIO(data))
    blocks: List[ContentBlock] = []
    sheet_count = 0

    for sheet in doc.spreadsheet.getElementsByType(table_mod.Table):
      sheet_count += 1
      sheet_name = sheet.getAttribute("name") or f"Sheet{sheet_count}"
      rows: List[str] = []
      row_count = 0

      for row in sheet.getElementsByType(table_mod.TableRow):
        if row_count >= self.max_rows:
          break
        cells: List[str] = []
        for cell in row.getElementsByType(table_mod.TableCell):
          # Extract all text content from the cell
          text_parts: List[str] = []
          for p in cell.getElementsByType(text_mod.P):
            text_content = ""
            for node in p.childNodes:
              if hasattr(node, "data"):
                text_content += node.data
              elif hasattr(node, "__str__"):
                text_content += str(node)
            text_parts.append(text_content)
          cell_text = "\n".join(text_parts)

          # Handle repeated columns
          repeat = cell.getAttribute("numbercolumnsrepeated")
          if repeat:
            try:
              repeat_count = int(repeat)
              # Cap to avoid memory issues with large empty repeats
              repeat_count = min(repeat_count, 100)
              cells.extend([cell_text] * repeat_count)
            except (ValueError, TypeError):
              cells.append(cell_text)
          else:
            cells.append(cell_text)

        # Strip trailing empty cells
        while cells and not cells[-1]:
          cells.pop()

        if cells:
          rows.append("\t".join(cells))
          row_count += 1

      if rows:
        sheet_text = f"[Sheet: {sheet_name}]\n" + "\n".join(rows)
        blocks.append(
          ContentBlock(
            content_type="table",
            content=sheet_text,
            mime_type="application/vnd.oasis.opendocument.spreadsheet",
            metadata={"sheet_name": sheet_name},
          )
        )

    return blocks
