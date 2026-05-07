"""CSV skill — read, analyze, and query CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from definable.agent.skill.base import Skill
from definable.tool.decorator import tool


class CSVTools(Skill):
  """Read, analyze, and query CSV files.

  Optionally integrates with DuckDB for SQL queries on CSV data.

  Args:
      csv_files: List of CSV file paths to register.
      row_limit: Default maximum rows to return. Default 50.
      delimiter: CSV delimiter. Default "," (auto-detected per file).
      enable_query: Enable SQL queries via DuckDB. Default False.
      enable_write: Enable writing CSV files. Default True.

  Example::

      from definable.agent.skill.builtin import CSVTools
      agent = Agent(model=model, skills=[CSVTools(csv_files=["data.csv", "users.csv"])])
  """

  name = "csv_tools"
  instructions = (
    "You have access to CSV file tools. Use list_csv_files to see available files, "
    "read_csv to view data, and get_csv_columns to inspect structure. "
    "If SQL queries are enabled, use query_csv for complex data analysis."
  )

  def __init__(
    self,
    *,
    csv_files: Optional[List[str]] = None,
    row_limit: int = 50,
    delimiter: str = ",",
    enable_query: bool = False,
    enable_write: bool = True,
  ):
    super().__init__()
    self._csv_files: Dict[str, Path] = {}
    for f in csv_files or []:
      p = Path(f)
      self._csv_files[p.stem] = p.resolve()
    self._row_limit = row_limit
    self._delimiter = delimiter
    self._enable_query = enable_query
    self._enable_write = enable_write
    self._duckdb_conn: Any = None

  def _read_csv(self, path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
      reader = csv.DictReader(f, delimiter=self._delimiter)
      rows = []
      max_rows = limit or self._row_limit
      for i, row in enumerate(reader):
        if i >= max_rows:
          break
        rows.append(dict(row))
      return rows

  def _resolve_csv(self, csv_name: str) -> Optional[Path]:
    if csv_name in self._csv_files:
      return self._csv_files[csv_name]
    p = Path(csv_name)
    if p.exists():
      return p.resolve()
    p_csv = Path(f"{csv_name}.csv")
    if p_csv.exists():
      return p_csv.resolve()
    return None

  @property
  def tools(self) -> list:
    skill = self
    result: list = []

    @tool
    def list_csv_files() -> str:
      """List all registered CSV files."""
      try:
        files = {}
        for name, path in skill._csv_files.items():
          files[name] = str(path)
        return json.dumps({"files": files, "count": len(files)})
      except Exception as e:
        return json.dumps({"error": str(e)})

    @tool
    def read_csv(csv_name: str, row_limit: int = 0) -> str:
      """Read rows from a CSV file. Returns data as JSON array."""
      try:
        path = skill._resolve_csv(csv_name)
        if not path or not path.exists():
          return json.dumps({"error": f"CSV file not found: {csv_name}"})
        rows = skill._read_csv(path, limit=row_limit)
        return json.dumps({"file": csv_name, "rows": rows, "count": len(rows)}, indent=2, default=str)
      except Exception as e:
        return json.dumps({"error": str(e)})

    @tool
    def get_csv_columns(csv_name: str) -> str:
      """Get column names and sample values from a CSV file."""
      try:
        path = skill._resolve_csv(csv_name)
        if not path or not path.exists():
          return json.dumps({"error": f"CSV file not found: {csv_name}"})
        rows = skill._read_csv(path, limit=3)
        if not rows:
          return json.dumps({"file": csv_name, "columns": [], "sample": []})
        columns = list(rows[0].keys())
        return json.dumps({"file": csv_name, "columns": columns, "sample_rows": rows}, indent=2, default=str)
      except Exception as e:
        return json.dumps({"error": str(e)})

    result.extend([list_csv_files, read_csv, get_csv_columns])

    if self._enable_query:

      @tool
      def query_csv(csv_name: str, sql_query: str) -> str:
        """Run a SQL query on a CSV file using DuckDB. Reference the file as the table name."""
        try:
          import duckdb
        except ImportError:
          return json.dumps({"error": "`duckdb` not installed. Run: pip install duckdb"})
        try:
          path = skill._resolve_csv(csv_name)
          if not path or not path.exists():
            return json.dumps({"error": f"CSV file not found: {csv_name}"})
          if skill._duckdb_conn is None:
            skill._duckdb_conn = duckdb.connect()
          conn = skill._duckdb_conn
          table = path.stem.replace("-", "_").replace(" ", "_")
          conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto('{path}')")
          result_set = conn.execute(sql_query.replace(csv_name, table))
          columns = [desc[0] for desc in result_set.description]
          rows = [dict(zip(columns, row)) for row in result_set.fetchmany(skill._row_limit)]
          return json.dumps({"rows": rows, "count": len(rows)}, indent=2, default=str)
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(query_csv)

    if self._enable_write:

      @tool
      def write_csv(file_path: str, data: str) -> str:
        """Write data to a CSV file. Data should be a JSON array of objects."""
        try:
          rows = json.loads(data)
          if not isinstance(rows, list) or not rows:
            return json.dumps({"error": "Data must be a non-empty JSON array of objects."})
          path = Path(file_path)
          fieldnames = list(rows[0].keys())
          with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
          # Register the new file
          skill._csv_files[path.stem] = path.resolve()
          return json.dumps({"status": "ok", "path": str(path), "rows_written": len(rows)})
        except json.JSONDecodeError:
          return json.dumps({"error": "Invalid JSON data. Provide a JSON array of objects."})
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(write_csv)

    return result

  def teardown(self) -> None:
    if self._duckdb_conn is not None:
      self._duckdb_conn.close()
      self._duckdb_conn = None
