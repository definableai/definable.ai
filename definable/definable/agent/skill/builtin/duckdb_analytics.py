"""DuckDB analytics skill — embedded SQL analytics on local/S3 files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from definable.agent.skill.base import Skill
from definable.tool.decorator import tool


class DuckDBAnalytics(Skill):
  """Embedded SQL analytics using DuckDB. No server required.

  Supports querying CSV, Parquet, JSON files directly. Can load from
  local paths or S3. Full-text search via FTS extension.

  Requires ``duckdb``: ``pip install duckdb``

  Args:
      db_path: Path to DuckDB file. None for in-memory (default).
      read_only: Open in read-only mode. Default False.
      init_commands: SQL commands to run on initialization.
      max_rows: Maximum rows to return. Default 200.
      enable_load: Enable file loading tools. Default True.
      enable_fts: Enable full-text search tools. Default True.
      enable_export: Enable export tools. Default True.

  Example::

      from definable.agent.skill.builtin import DuckDBAnalytics
      agent = Agent(model=model, skills=[DuckDBAnalytics()])
      # Agent can now: load CSVs, query with SQL, export results
  """

  name = "duckdb_analytics"
  instructions = (
    "You have access to DuckDB, an embedded SQL analytics engine. "
    "You can load CSV/Parquet/JSON files into tables, query them with SQL, "
    "and export results. Use show_tables and describe_table to explore data. "
    "You can also query files directly: SELECT * FROM 'file.csv' LIMIT 10."
  )

  def __init__(
    self,
    *,
    db_path: Optional[str] = None,
    read_only: bool = False,
    init_commands: Optional[List[str]] = None,
    max_rows: int = 200,
    enable_load: bool = True,
    enable_fts: bool = True,
    enable_export: bool = True,
  ):
    super().__init__()
    self._db_path = db_path
    self._read_only = read_only
    self._init_commands = init_commands or []
    self._max_rows = max_rows
    self._enable_load = enable_load
    self._enable_fts = enable_fts
    self._enable_export = enable_export
    self._conn: Any = None

  @property
  def connection(self) -> Any:
    if self._conn is not None:
      return self._conn
    try:
      import duckdb
    except ImportError:
      raise ImportError("`duckdb` not installed. Run: pip install duckdb")
    self._conn = duckdb.connect(database=self._db_path or ":memory:", read_only=self._read_only)
    for cmd in self._init_commands:
      self._conn.execute(cmd)
    return self._conn

  def _query(self, sql: str) -> list:
    result = self.connection.execute(sql)
    columns = [desc[0] for desc in result.description]
    rows = result.fetchmany(self._max_rows)
    return [dict(zip(columns, row)) for row in rows]

  @staticmethod
  def _sanitize_sql(sql: str) -> str:
    """Take only the first SQL statement."""
    return sql.split(";")[0].strip()

  @property
  def tools(self) -> list:
    skill = self
    result: list = []

    @tool
    def show_tables() -> str:
      """List all tables in the DuckDB database."""
      try:
        rows = skill._query("SHOW TABLES")
        return json.dumps({"tables": [r.get("name", r) for r in rows]}, indent=2, default=str)
      except Exception as e:
        return json.dumps({"error": str(e)})

    @tool
    def describe_table(table_name: str) -> str:
      """Describe a table's schema: columns, types, nullable."""
      try:
        rows = skill._query(f"DESCRIBE {table_name}")
        return json.dumps({"table": table_name, "columns": rows}, indent=2, default=str)
      except Exception as e:
        return json.dumps({"error": str(e)})

    @tool
    def run_query(query: str) -> str:
      """Execute a SQL query on DuckDB. Supports direct file queries like: SELECT * FROM 'file.csv'."""
      try:
        sql = skill._sanitize_sql(query)
        rows = skill._query(sql)
        return json.dumps({"rows": rows, "count": len(rows)}, indent=2, default=str)
      except Exception as e:
        return json.dumps({"error": str(e)})

    @tool
    def summarize_table(table_name: str) -> str:
      """Get summary statistics for a table (min, max, avg, count, nulls)."""
      try:
        rows = skill._query(f"SUMMARIZE {table_name}")
        return json.dumps({"summary": rows}, indent=2, default=str)
      except Exception as e:
        return json.dumps({"error": str(e)})

    result.extend([show_tables, describe_table, run_query, summarize_table])

    if self._enable_load:

      @tool
      def load_file(file_path: str, table_name: str = "") -> str:
        """Load a CSV, Parquet, or JSON file into a DuckDB table. Auto-detects format from extension."""
        try:
          p = Path(file_path)
          if not table_name:
            table_name = p.stem.replace("-", "_").replace(" ", "_")
          sql = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM '{file_path}'"
          skill.connection.execute(sql)
          count = skill._query(f"SELECT COUNT(*) as cnt FROM {table_name}")[0]["cnt"]
          return json.dumps({"table": table_name, "rows_loaded": count, "source": file_path})
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(load_file)

    if self._enable_fts:

      @tool
      def create_fts_index(table_name: str, columns: str) -> str:
        """Create a full-text search index on specified columns (comma-separated)."""
        try:
          skill.connection.execute("INSTALL fts; LOAD fts;")
          col_list = ", ".join(c.strip() for c in columns.split(","))
          skill.connection.execute(f"PRAGMA create_fts_index('{table_name}', '*', '{col_list}')")
          return json.dumps({"status": "ok", "table": table_name, "indexed_columns": col_list})
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(create_fts_index)

    if self._enable_export:

      @tool
      def export_to_file(query: str, output_path: str, output_format: str = "csv") -> str:
        """Export query results to a file. output_format: 'csv' or 'parquet'."""
        try:
          sql = skill._sanitize_sql(query)
          fmt = output_format.upper()
          skill.connection.execute(f"COPY ({sql}) TO '{output_path}' (FORMAT {fmt})")
          return json.dumps({"status": "ok", "path": output_path, "format": fmt})
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(export_to_file)

    return result

  def teardown(self) -> None:
    if self._conn is not None:
      self._conn.close()
      self._conn = None
