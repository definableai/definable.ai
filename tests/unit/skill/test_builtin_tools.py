"""
Unit tests for Phase 5 built-in skills: GitHub, SQLDatabase, DuckDBAnalytics,
SlackTools, EmailTools, PythonExec, Firecrawl, CSVTools.

Tests pure logic and configuration. Uses mocks for all external APIs.
No real API calls, no real database connections.

Covers:
  - Default configuration and custom params
  - Lazy client initialization
  - ImportError handling for missing SDKs
  - Tool generation (correct tools returned based on flags)
  - Tool function signatures and return types
  - Error handling (returns JSON error, never raises)
  - Feature flag toggling (enable_*/disable_*)
  - Import paths (lazy loading from builtin)
"""

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGitHubSkill:
  """GitHub skill configuration and tool generation."""

  def test_default_config(self):
    from definable.agent.skill.builtin.github import GitHub

    gh = GitHub(access_token="test-token")
    assert gh.name == "github"
    assert gh._token == "test-token"

  def test_env_var_fallback(self):
    from definable.agent.skill.builtin.github import GitHub

    with patch.dict("os.environ", {"GITHUB_ACCESS_TOKEN": "env-token"}):
      gh = GitHub()
      assert gh._token == "env-token"

  def test_tools_generated_with_defaults(self):
    from definable.agent.skill.builtin.github import GitHub

    gh = GitHub(access_token="test")
    tools = gh.tools
    names = {t.name for t in tools}
    assert "search_repos" in names
    assert "get_repo" in names
    assert "list_issues" in names
    assert "get_issue" in names
    assert "list_pull_requests" in names
    assert "get_pull_request" in names
    assert "list_branches" in names
    assert "get_file_content" in names

  def test_write_tools_included_by_default(self):
    from definable.agent.skill.builtin.github import GitHub

    gh = GitHub(access_token="test")
    names = {t.name for t in gh.tools}
    assert "create_issue" in names
    assert "comment_on_issue" in names
    assert "close_issue" in names

  def test_write_tools_disabled(self):
    from definable.agent.skill.builtin.github import GitHub

    gh = GitHub(access_token="test", enable_write=False)
    names = {t.name for t in gh.tools}
    assert "create_issue" not in names
    assert "comment_on_issue" not in names

  def test_disable_repos(self):
    from definable.agent.skill.builtin.github import GitHub

    gh = GitHub(access_token="test", enable_repos=False)
    names = {t.name for t in gh.tools}
    assert "get_repo" not in names

  def test_disable_prs(self):
    from definable.agent.skill.builtin.github import GitHub

    gh = GitHub(access_token="test", enable_prs=False)
    names = {t.name for t in gh.tools}
    assert "list_pull_requests" not in names
    assert "get_pull_request" not in names

  def test_search_repos_with_mock(self):
    from definable.agent.skill.builtin.github import GitHub

    gh = GitHub(access_token="test")
    mock_repo = MagicMock()
    mock_repo.full_name = "owner/repo"
    mock_repo.description = "A test repo"
    mock_repo.stargazers_count = 100
    mock_repo.html_url = "https://github.com/owner/repo"
    mock_client = MagicMock()
    mock_client.search_repositories.return_value = [mock_repo]
    gh._client = mock_client

    search_tool = next(t for t in gh.tools if t.name == "search_repos")
    result = search_tool.entrypoint("python")
    data = json.loads(result)
    assert isinstance(data, list)
    assert data[0]["full_name"] == "owner/repo"

  def test_error_returns_json(self):
    from definable.agent.skill.builtin.github import GitHub

    gh = GitHub(access_token="test")
    mock_client = MagicMock()
    mock_client.search_repositories.side_effect = RuntimeError("API error")
    gh._client = mock_client

    search_tool = next(t for t in gh.tools if t.name == "search_repos")
    result = search_tool.entrypoint("query")
    data = json.loads(result)
    assert "error" in data

  def test_import_error_helpful(self):
    from definable.agent.skill.builtin.github import GitHub

    gh = GitHub(access_token="test")
    with patch.dict("sys.modules", {"github": None}):
      with pytest.raises(ImportError, match="PyGithub"):
        gh.client

  def test_missing_token_raises(self):
    from definable.agent.skill.builtin.github import GitHub

    gh = GitHub()
    gh._token = None
    with pytest.raises((ValueError, ImportError)):
      gh.client


# ---------------------------------------------------------------------------
# SQLDatabase
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSQLDatabaseSkill:
  """SQLDatabase skill configuration and tool generation."""

  def test_default_config(self):
    from definable.agent.skill.builtin.sql_database import SQLDatabase

    db = SQLDatabase(connection_url="sqlite:///test.db")
    assert db.name == "sql_database"
    assert db._read_only is True
    assert db._max_rows == 100

  def test_env_var_fallback(self):
    from definable.agent.skill.builtin.sql_database import SQLDatabase

    with patch.dict("os.environ", {"DATABASE_URL": "sqlite:///env.db"}):
      db = SQLDatabase()
      assert db._url == "sqlite:///env.db"

  def test_tools_generated(self):
    from definable.agent.skill.builtin.sql_database import SQLDatabase

    db = SQLDatabase(connection_url="sqlite:///test.db")
    names = {t.name for t in db.tools}
    assert "show_tables" in names
    assert "describe_table" in names
    assert "run_query" in names
    assert "explain_query" in names

  def test_write_disabled_by_default(self):
    from definable.agent.skill.builtin.sql_database import SQLDatabase

    db = SQLDatabase(connection_url="sqlite:///test.db")
    names = {t.name for t in db.tools}
    assert "execute_statement" not in names

  def test_write_enabled(self):
    from definable.agent.skill.builtin.sql_database import SQLDatabase

    db = SQLDatabase(connection_url="sqlite:///test.db", read_only=False, enable_write=True)
    names = {t.name for t in db.tools}
    assert "execute_statement" in names

  def test_read_only_check(self):
    from definable.agent.skill.builtin.sql_database import SQLDatabase

    assert SQLDatabase._is_read_only("SELECT * FROM users") is True
    assert SQLDatabase._is_read_only("SHOW TABLES") is True
    assert SQLDatabase._is_read_only("EXPLAIN SELECT 1") is True
    assert SQLDatabase._is_read_only("WITH cte AS (SELECT 1) SELECT * FROM cte") is True
    assert SQLDatabase._is_read_only("INSERT INTO users VALUES (1)") is False
    assert SQLDatabase._is_read_only("DELETE FROM users") is False
    assert SQLDatabase._is_read_only("DROP TABLE users") is False

  def test_run_query_blocks_writes_in_read_only(self):
    from definable.agent.skill.builtin.sql_database import SQLDatabase

    db = SQLDatabase(connection_url="sqlite:///test.db", read_only=True)
    db._engine = MagicMock()
    query_tool = next(t for t in db.tools if t.name == "run_query")
    result = query_tool.entrypoint("DELETE FROM users")
    data = json.loads(result)
    assert "error" in data
    assert "Read-only" in data["error"]

  def test_import_error(self):
    from definable.agent.skill.builtin.sql_database import SQLDatabase

    db = SQLDatabase(connection_url="sqlite:///test.db")
    with patch.dict("sys.modules", {"sqlalchemy": None}):
      with pytest.raises(ImportError, match="sqlalchemy"):
        db.engine


# ---------------------------------------------------------------------------
# DuckDBAnalytics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDuckDBAnalyticsSkill:
  """DuckDBAnalytics skill configuration and tool generation."""

  def test_default_config(self):
    from definable.agent.skill.builtin.duckdb_analytics import DuckDBAnalytics

    db = DuckDBAnalytics()
    assert db.name == "duckdb_analytics"
    assert db._db_path is None
    assert db._max_rows == 200

  def test_tools_generated(self):
    from definable.agent.skill.builtin.duckdb_analytics import DuckDBAnalytics

    db = DuckDBAnalytics()
    names = {t.name for t in db.tools}
    assert "show_tables" in names
    assert "describe_table" in names
    assert "run_query" in names
    assert "summarize_table" in names
    assert "load_file" in names
    assert "create_fts_index" in names
    assert "export_to_file" in names

  def test_disable_features(self):
    from definable.agent.skill.builtin.duckdb_analytics import DuckDBAnalytics

    db = DuckDBAnalytics(enable_load=False, enable_fts=False, enable_export=False)
    names = {t.name for t in db.tools}
    assert "load_file" not in names
    assert "create_fts_index" not in names
    assert "export_to_file" not in names

  def test_sanitize_sql(self):
    from definable.agent.skill.builtin.duckdb_analytics import DuckDBAnalytics

    assert DuckDBAnalytics._sanitize_sql("SELECT 1; DROP TABLE x") == "SELECT 1"
    assert DuckDBAnalytics._sanitize_sql("SELECT 1") == "SELECT 1"

  def test_teardown_closes_connection(self):
    from definable.agent.skill.builtin.duckdb_analytics import DuckDBAnalytics

    db = DuckDBAnalytics()
    mock_conn = MagicMock()
    db._conn = mock_conn
    db.teardown()
    mock_conn.close.assert_called_once()
    assert db._conn is None

  def test_import_error(self):
    from definable.agent.skill.builtin.duckdb_analytics import DuckDBAnalytics

    db = DuckDBAnalytics()
    with patch.dict("sys.modules", {"duckdb": None}):
      with pytest.raises(ImportError, match="duckdb"):
        db.connection


# ---------------------------------------------------------------------------
# SlackTools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlackToolsSkill:
  """SlackTools skill configuration and tool generation."""

  def test_default_config(self):
    from definable.agent.skill.builtin.slack_tools import SlackTools

    sl = SlackTools(token="xoxb-test")
    assert sl.name == "slack_tools"
    assert sl._token == "xoxb-test"

  def test_env_var_fallback(self):
    from definable.agent.skill.builtin.slack_tools import SlackTools

    with patch.dict("os.environ", {"SLACK_TOKEN": "xoxb-env"}):
      sl = SlackTools()
      assert sl._token == "xoxb-env"

  def test_default_tools(self):
    from definable.agent.skill.builtin.slack_tools import SlackTools

    sl = SlackTools(token="xoxb-test")
    names = {t.name for t in sl.tools}
    assert "list_channels" in names
    assert "get_channel_history" in names
    assert "send_message" in names
    assert "reply_in_thread" in names

  def test_search_disabled_by_default(self):
    from definable.agent.skill.builtin.slack_tools import SlackTools

    sl = SlackTools(token="xoxb-test")
    names = {t.name for t in sl.tools}
    assert "search_messages" not in names

  def test_search_enabled(self):
    from definable.agent.skill.builtin.slack_tools import SlackTools

    sl = SlackTools(token="xoxb-test", enable_search=True)
    names = {t.name for t in sl.tools}
    assert "search_messages" in names
    assert "get_thread" in names

  def test_users_enabled(self):
    from definable.agent.skill.builtin.slack_tools import SlackTools

    sl = SlackTools(token="xoxb-test", enable_users=True)
    names = {t.name for t in sl.tools}
    assert "list_users" in names

  def test_send_message_with_mock(self):
    from definable.agent.skill.builtin.slack_tools import SlackTools

    sl = SlackTools(token="xoxb-test")
    mock_client = MagicMock()
    mock_client.chat_postMessage.return_value = {"ok": True, "channel": "C123", "ts": "1234567.890"}
    sl._client = mock_client

    send_tool = next(t for t in sl.tools if t.name == "send_message")
    result = send_tool.entrypoint("C123", "hello")
    data = json.loads(result)
    assert data["ok"] is True

  def test_error_returns_json(self):
    from definable.agent.skill.builtin.slack_tools import SlackTools

    sl = SlackTools(token="xoxb-test")
    mock_client = MagicMock()
    mock_client.conversations_list.side_effect = RuntimeError("rate limited")
    sl._client = mock_client

    list_tool = next(t for t in sl.tools if t.name == "list_channels")
    result = list_tool.entrypoint()
    data = json.loads(result)
    assert "error" in data

  def test_import_error(self):
    from definable.agent.skill.builtin.slack_tools import SlackTools

    sl = SlackTools(token="xoxb-test")
    with patch.dict("sys.modules", {"slack_sdk": None}):
      with pytest.raises(ImportError, match="slack-sdk"):
        sl.client

  def test_missing_token_raises(self):
    from definable.agent.skill.builtin.slack_tools import SlackTools

    sl = SlackTools()
    sl._token = None
    with pytest.raises((ValueError, ImportError)):
      sl.client


# ---------------------------------------------------------------------------
# EmailTools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmailToolsSkill:
  """EmailTools skill configuration and tool generation."""

  def test_default_config(self):
    from definable.agent.skill.builtin.email_tools import EmailTools

    em = EmailTools(sender_email="test@example.com", sender_password="pass")
    assert em.name == "email_tools"
    assert em._smtp_host == "smtp.gmail.com"
    assert em._smtp_port == 587

  def test_tools_generated(self):
    from definable.agent.skill.builtin.email_tools import EmailTools

    em = EmailTools(sender_email="test@example.com", sender_password="pass")
    names = {t.name for t in em.tools}
    assert "send_email" in names
    assert "send_html_email" in names

  def test_missing_credentials_returns_error(self):
    from definable.agent.skill.builtin.email_tools import EmailTools

    em = EmailTools()
    em._sender_email = None
    em._sender_password = None
    result = em._send("to@example.com", "subject", "body")
    assert "error" in result

  def test_send_with_mock_smtp(self):
    from definable.agent.skill.builtin.email_tools import EmailTools

    em = EmailTools(sender_email="test@example.com", sender_password="pass")
    with patch("definable.skill.builtin.email_tools.smtplib.SMTP") as mock_smtp:
      mock_server = MagicMock()
      mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
      mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
      send_tool = next(t for t in em.tools if t.name == "send_email")
      result = send_tool.entrypoint("to@example.com", "Test Subject", "Test Body")
      data = json.loads(result)
      assert data["ok"] is True


# ---------------------------------------------------------------------------
# PythonExec
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPythonExecSkill:
  """PythonExec skill configuration and tool generation."""

  def test_default_config(self):
    from definable.agent.skill.builtin.python_exec import PythonExec

    pe = PythonExec()
    assert pe.name == "python_exec"
    assert pe._timeout == 30

  def test_tools_generated(self):
    from definable.agent.skill.builtin.python_exec import PythonExec

    pe = PythonExec()
    names = {t.name for t in pe.tools}
    assert "run_python" in names

  def test_file_ops_enabled(self):
    from definable.agent.skill.builtin.python_exec import PythonExec

    pe = PythonExec(enable_file_ops=True)
    names = {t.name for t in pe.tools}
    assert "save_and_run" in names

  def test_pip_disabled_by_default(self):
    from definable.agent.skill.builtin.python_exec import PythonExec

    pe = PythonExec()
    names = {t.name for t in pe.tools}
    assert "pip_install" not in names

  def test_pip_enabled(self):
    from definable.agent.skill.builtin.python_exec import PythonExec

    pe = PythonExec(enable_pip=True)
    names = {t.name for t in pe.tools}
    assert "pip_install" in names

  def test_run_python_basic(self):
    from definable.agent.skill.builtin.python_exec import PythonExec

    pe = PythonExec()
    run_tool = next(t for t in pe.tools if t.name == "run_python")
    result = run_tool.entrypoint("x = 2 + 2", "x")
    data = json.loads(result)
    assert data["result"] == "4"

  def test_run_python_stdout(self):
    from definable.agent.skill.builtin.python_exec import PythonExec

    pe = PythonExec()
    run_tool = next(t for t in pe.tools if t.name == "run_python")
    result = run_tool.entrypoint("print('hello world')")
    data = json.loads(result)
    assert "hello world" in data["stdout"]

  def test_run_python_error(self):
    from definable.agent.skill.builtin.python_exec import PythonExec

    pe = PythonExec()
    run_tool = next(t for t in pe.tools if t.name == "run_python")
    result = run_tool.entrypoint("raise ValueError('test')")
    data = json.loads(result)
    assert "error" in data
    assert "ValueError" in data["error"]

  def test_path_restriction(self):
    from definable.agent.skill.builtin.python_exec import PythonExec

    pe = PythonExec(base_dir="/tmp/test_sandbox", restrict_to_base_dir=True)
    with pytest.raises(PermissionError, match="escapes base directory"):
      pe._check_path("../../etc/passwd")


# ---------------------------------------------------------------------------
# Firecrawl
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFirecrawlSkill:
  """Firecrawl skill configuration and tool generation."""

  def test_default_config(self):
    from definable.agent.skill.builtin.firecrawl import Firecrawl

    fc = Firecrawl(api_key="fc-test")
    assert fc.name == "firecrawl"
    assert fc._formats == ["markdown"]

  def test_default_tools(self):
    from definable.agent.skill.builtin.firecrawl import Firecrawl

    fc = Firecrawl(api_key="fc-test")
    names = {t.name for t in fc.tools}
    assert "scrape_page" in names
    assert "crawl_site" not in names
    assert "map_site" not in names

  def test_all_tools_enabled(self):
    from definable.agent.skill.builtin.firecrawl import Firecrawl

    fc = Firecrawl(api_key="fc-test", enable_crawl=True, enable_map=True, enable_search=True)
    names = {t.name for t in fc.tools}
    assert "scrape_page" in names
    assert "crawl_site" in names
    assert "map_site" in names
    assert "search_web" in names

  def test_scrape_with_mock(self):
    from definable.agent.skill.builtin.firecrawl import Firecrawl

    fc = Firecrawl(api_key="fc-test")
    mock_app = MagicMock()
    mock_app.scrape_url.return_value = {"markdown": "# Hello World", "metadata": {"title": "Test"}}
    fc._app = mock_app

    scrape_tool = next(t for t in fc.tools if t.name == "scrape_page")
    result = scrape_tool.entrypoint("https://example.com")
    data = json.loads(result)
    assert data["content"] == "# Hello World"

  def test_import_error(self):
    from definable.agent.skill.builtin.firecrawl import Firecrawl

    fc = Firecrawl(api_key="fc-test")
    with patch.dict("sys.modules", {"firecrawl": None}):
      with pytest.raises(ImportError, match="firecrawl-py"):
        fc.app

  def test_missing_key_raises(self):
    from definable.agent.skill.builtin.firecrawl import Firecrawl

    fc = Firecrawl()
    fc._api_key = None
    with pytest.raises((ValueError, ImportError)):
      fc.app


# ---------------------------------------------------------------------------
# CSVTools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCSVToolsSkill:
  """CSVTools skill configuration and tool generation."""

  def _make_csv(self, rows: list) -> str:
    """Write a temporary CSV and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    f.close()
    return f.name

  def test_default_config(self):
    from definable.agent.skill.builtin.csv_tools import CSVTools

    ct = CSVTools()
    assert ct.name == "csv_tools"
    assert ct._row_limit == 50

  def test_tools_generated(self):
    from definable.agent.skill.builtin.csv_tools import CSVTools

    ct = CSVTools()
    names = {t.name for t in ct.tools}
    assert "list_csv_files" in names
    assert "read_csv" in names
    assert "get_csv_columns" in names

  def test_query_disabled_by_default(self):
    from definable.agent.skill.builtin.csv_tools import CSVTools

    ct = CSVTools()
    names = {t.name for t in ct.tools}
    assert "query_csv" not in names

  def test_query_enabled(self):
    from definable.agent.skill.builtin.csv_tools import CSVTools

    ct = CSVTools(enable_query=True)
    names = {t.name for t in ct.tools}
    assert "query_csv" in names

  def test_write_enabled_by_default(self):
    from definable.agent.skill.builtin.csv_tools import CSVTools

    ct = CSVTools()
    names = {t.name for t in ct.tools}
    assert "write_csv" in names

  def test_read_csv_file(self):
    from definable.agent.skill.builtin.csv_tools import CSVTools

    path = self._make_csv([{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}])
    ct = CSVTools(csv_files=[path])
    read_tool = next(t for t in ct.tools if t.name == "read_csv")
    result = read_tool.entrypoint(Path(path).stem)
    data = json.loads(result)
    assert data["count"] == 2
    assert data["rows"][0]["name"] == "Alice"

  def test_get_csv_columns(self):
    from definable.agent.skill.builtin.csv_tools import CSVTools

    path = self._make_csv([{"x": "1", "y": "2"}])
    ct = CSVTools(csv_files=[path])
    col_tool = next(t for t in ct.tools if t.name == "get_csv_columns")
    result = col_tool.entrypoint(Path(path).stem)
    data = json.loads(result)
    assert "x" in data["columns"]
    assert "y" in data["columns"]

  def test_csv_not_found(self):
    from definable.agent.skill.builtin.csv_tools import CSVTools

    ct = CSVTools()
    read_tool = next(t for t in ct.tools if t.name == "read_csv")
    result = read_tool.entrypoint("nonexistent")
    data = json.loads(result)
    assert "error" in data

  def test_write_csv_file(self):
    from definable.agent.skill.builtin.csv_tools import CSVTools

    ct = CSVTools()
    write_tool = next(t for t in ct.tools if t.name == "write_csv")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
      path = f.name
    data_str = json.dumps([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    result = write_tool.entrypoint(path, data_str)
    data = json.loads(result)
    assert data["rows_written"] == 2

  def test_teardown(self):
    from definable.agent.skill.builtin.csv_tools import CSVTools

    ct = CSVTools()
    mock_conn = MagicMock()
    ct._duckdb_conn = mock_conn
    ct.teardown()
    mock_conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# Lazy import from builtin package
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuiltinImports:
  """All new skills are importable from definable.agent.skill.builtin."""

  def test_github_import(self):
    from definable.agent.skill.builtin import GitHub

    assert GitHub.__name__ == "GitHub"

  def test_sql_database_import(self):
    from definable.agent.skill.builtin import SQLDatabase

    assert SQLDatabase.__name__ == "SQLDatabase"

  def test_duckdb_import(self):
    from definable.agent.skill.builtin import DuckDBAnalytics

    assert DuckDBAnalytics.__name__ == "DuckDBAnalytics"

  def test_slack_import(self):
    from definable.agent.skill.builtin import SlackTools

    assert SlackTools.__name__ == "SlackTools"

  def test_email_import(self):
    from definable.agent.skill.builtin import EmailTools

    assert EmailTools.__name__ == "EmailTools"

  def test_python_exec_import(self):
    from definable.agent.skill.builtin import PythonExec

    assert PythonExec.__name__ == "PythonExec"

  def test_firecrawl_import(self):
    from definable.agent.skill.builtin import Firecrawl

    assert Firecrawl.__name__ == "Firecrawl"

  def test_csv_tools_import(self):
    from definable.agent.skill.builtin import CSVTools

    assert CSVTools.__name__ == "CSVTools"

  def test_existing_skills_still_work(self):
    from definable.agent.skill.builtin import Calculator, DateTime, FileOperations, Shell, WebSearch

    assert Calculator.__name__ == "Calculator"
    assert WebSearch.__name__ == "WebSearch"
    assert Shell.__name__ == "Shell"
    assert FileOperations.__name__ == "FileOperations"
    assert DateTime.__name__ == "DateTime"
