"""SQL Database skill — query any SQL database via SQLAlchemy."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from definable.agent.skill.base import Skill
from definable.tool.decorator import tool


class SQLDatabase(Skill):
  """Query SQL databases (PostgreSQL, MySQL, SQLite, etc.) via SQLAlchemy.

  Requires ``sqlalchemy``: ``pip install sqlalchemy``
  Plus a driver for your database (e.g., ``psycopg2``, ``pymysql``).

  Args:
      connection_url: SQLAlchemy connection URL. Falls back to DATABASE_URL env var.
          Examples: "postgresql://user:pass@host/db", "sqlite:///data.db", "mysql+pymysql://user:pass@host/db"
      schema: Database schema to use. Default "public" (Postgres) or None.
      read_only: If True, only allow SELECT queries. Default True.
      max_rows: Maximum rows to return from queries. Default 100.
      enable_describe: Enable table description tools. Default True.
      enable_query: Enable query execution. Default True.
      enable_write: Enable INSERT/UPDATE/DELETE. Default False (requires read_only=False).

  Example::

      from definable.agent.skill.builtin import SQLDatabase
      agent = Agent(model=model, skills=[SQLDatabase(connection_url="postgresql://localhost/mydb")])
  """

  name = "sql_database"
  instructions = (
    "You have access to a SQL database. Use show_tables to list available tables, "
    "describe_table to understand schema, and run_query to execute SQL. "
    "Always inspect the schema before writing queries. Prefer SELECT queries."
  )

  def __init__(
    self,
    *,
    connection_url: Optional[str] = None,
    schema: Optional[str] = None,
    read_only: bool = True,
    max_rows: int = 100,
    enable_describe: bool = True,
    enable_query: bool = True,
    enable_write: bool = False,
  ):
    super().__init__()
    self._url = connection_url or os.getenv("DATABASE_URL")
    self._schema = schema
    self._read_only = read_only
    self._max_rows = max_rows
    self._enable_describe = enable_describe
    self._enable_query = enable_query
    self._enable_write = enable_write
    self._engine: Any = None

  @property
  def engine(self) -> Any:
    if self._engine is not None:
      return self._engine
    try:
      from sqlalchemy import create_engine
    except ImportError:
      raise ImportError("`sqlalchemy` not installed. Run: pip install sqlalchemy")
    if not self._url:
      raise ValueError("Database connection URL required. Set connection_url or DATABASE_URL env var.")
    self._engine = create_engine(self._url)
    return self._engine

  def _execute(self, query: str) -> list:
    from sqlalchemy import text

    with self.engine.connect() as conn:
      result = conn.execute(text(query))
      if result.returns_rows:
        columns = list(result.keys())
        rows = [dict(zip(columns, row)) for row in result.fetchmany(self._max_rows)]
        return rows
      conn.commit()
      return [{"affected_rows": result.rowcount}]

  @staticmethod
  def _is_read_only(query: str) -> bool:
    normalized = query.strip().upper()
    return normalized.startswith(("SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "WITH"))

  @property
  def tools(self) -> list:
    skill = self
    result: list = []

    if self._enable_describe:

      @tool
      def show_tables() -> str:
        """List all tables in the database."""
        try:
          from sqlalchemy import inspect

          inspector = inspect(skill.engine)
          tables = inspector.get_table_names(schema=skill._schema)
          return json.dumps({"tables": tables, "count": len(tables)}, indent=2)
        except Exception as e:
          return json.dumps({"error": str(e)})

      @tool
      def describe_table(table_name: str) -> str:
        """Describe a table's columns, types, and constraints."""
        try:
          from sqlalchemy import inspect

          inspector = inspect(skill.engine)
          columns = inspector.get_columns(table_name, schema=skill._schema)
          pk = inspector.get_pk_constraint(table_name, schema=skill._schema)
          col_info = []
          for col in columns:
            col_info.append({
              "name": col["name"],
              "type": str(col["type"]),
              "nullable": col.get("nullable", True),
              "default": str(col.get("default")) if col.get("default") else None,
            })
          return json.dumps({"table": table_name, "columns": col_info, "primary_key": pk.get("constrained_columns", [])}, indent=2)
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.extend([show_tables, describe_table])

    if self._enable_query:

      @tool
      def run_query(query: str) -> str:
        """Execute a SQL query and return results as JSON. Use SELECT for reads."""
        try:
          if skill._read_only and not skill._is_read_only(query):
            return json.dumps({"error": "Read-only mode. Only SELECT/SHOW/DESCRIBE/EXPLAIN queries allowed."})
          rows = skill._execute(query)
          return json.dumps({"rows": rows, "count": len(rows)}, indent=2, default=str)
        except Exception as e:
          return json.dumps({"error": str(e)})

      @tool
      def explain_query(query: str) -> str:
        """Get the execution plan for a SQL query."""
        try:
          rows = skill._execute(f"EXPLAIN {query}")
          return json.dumps({"plan": rows}, indent=2, default=str)
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.extend([run_query, explain_query])

    if self._enable_write and not self._read_only:

      @tool
      def execute_statement(statement: str) -> str:
        """Execute a write SQL statement (INSERT, UPDATE, DELETE). Use with caution."""
        try:
          rows = skill._execute(statement)
          return json.dumps({"result": rows}, default=str)
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(execute_statement)

    return result
