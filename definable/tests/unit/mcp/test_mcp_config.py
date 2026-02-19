"""Unit tests for MCPConfig and MCPServerConfig.

Tests cover field defaults, validation, serialization,
tool filtering, and error paths. No network calls.
"""

import pytest

from definable.mcp.config import MCPConfig, MCPServerConfig


@pytest.mark.unit
class TestMCPServerConfigCreation:
  """Tests for MCPServerConfig instantiation and validation."""

  def test_stdio_server_valid(self):
    """A stdio server with a command is valid."""
    cfg = MCPServerConfig(name="fs", command="npx", args=["-y", "server"])
    assert cfg.name == "fs"
    assert cfg.transport == "stdio"
    assert cfg.command == "npx"
    assert cfg.args == ["-y", "server"]

  def test_sse_server_valid(self):
    """An SSE server with a url is valid."""
    cfg = MCPServerConfig(name="web", transport="sse", url="http://localhost:3000/sse")
    assert cfg.name == "web"
    assert cfg.transport == "sse"
    assert cfg.url == "http://localhost:3000/sse"

  def test_http_server_valid(self):
    """An HTTP server with a url is valid."""
    cfg = MCPServerConfig(name="api", transport="http", url="http://localhost:8080")
    assert cfg.transport == "http"

  def test_stdio_requires_command(self):
    """stdio transport without command raises ValueError."""
    with pytest.raises(ValueError, match="stdio transport requires 'command'"):
      MCPServerConfig(name="bad", transport="stdio")

  def test_sse_requires_url(self):
    """sse transport without url raises ValueError."""
    with pytest.raises(ValueError, match="sse transport requires 'url'"):
      MCPServerConfig(name="bad", transport="sse")

  def test_http_requires_url(self):
    """http transport without url raises ValueError."""
    with pytest.raises(ValueError, match="http transport requires 'url'"):
      MCPServerConfig(name="bad", transport="http")

  def test_default_timeouts(self):
    """Default connect_timeout=30 and request_timeout=60."""
    cfg = MCPServerConfig(name="t", command="cmd")
    assert cfg.connect_timeout == 30.0
    assert cfg.request_timeout == 60.0

  def test_default_reconnect_settings(self):
    """Default reconnection settings are sensible."""
    cfg = MCPServerConfig(name="r", command="cmd")
    assert cfg.reconnect_on_failure is True
    assert cfg.max_reconnect_attempts == 3
    assert cfg.reconnect_delay == 1.0


@pytest.mark.unit
class TestMCPServerConfigToolFiltering:
  """Tests for MCPServerConfig.is_tool_allowed()."""

  def test_no_filter_allows_all(self):
    """With no allowed/blocked lists every tool is allowed."""
    cfg = MCPServerConfig(name="t", command="cmd")
    assert cfg.is_tool_allowed("any_tool") is True

  def test_allowlist_permits_listed_tool(self):
    """A tool in the allowlist is permitted."""
    cfg = MCPServerConfig(name="t", command="cmd", allowed_tools=["read", "write"])
    assert cfg.is_tool_allowed("read") is True

  def test_allowlist_blocks_unlisted_tool(self):
    """A tool not in the allowlist is blocked."""
    cfg = MCPServerConfig(name="t", command="cmd", allowed_tools=["read"])
    assert cfg.is_tool_allowed("delete") is False

  def test_blocklist_blocks_listed_tool(self):
    """A tool in the blocklist is blocked."""
    cfg = MCPServerConfig(name="t", command="cmd", blocked_tools=["dangerous"])
    assert cfg.is_tool_allowed("dangerous") is False

  def test_blocklist_allows_unlisted_tool(self):
    """A tool not in the blocklist is allowed."""
    cfg = MCPServerConfig(name="t", command="cmd", blocked_tools=["dangerous"])
    assert cfg.is_tool_allowed("safe") is True


@pytest.mark.unit
class TestMCPServerConfigSerialization:
  """Tests for to_dict / from_dict."""

  def test_to_dict_stdio(self):
    """to_dict includes transport-specific fields for stdio."""
    cfg = MCPServerConfig(name="s", command="npx", args=["--flag"])
    d = cfg.to_dict()
    assert d["name"] == "s"
    assert d["transport"] == "stdio"
    assert d["command"] == "npx"
    assert d["args"] == ["--flag"]

  def test_to_dict_sse(self):
    """to_dict includes url for sse."""
    cfg = MCPServerConfig(name="s", transport="sse", url="http://x")
    d = cfg.to_dict()
    assert d["url"] == "http://x"

  def test_from_dict_roundtrip(self):
    """from_dict(to_dict(cfg)) reproduces the original config."""
    original = MCPServerConfig(name="rt", command="node", args=["server.js"])
    rebuilt = MCPServerConfig.from_dict(original.to_dict())
    assert rebuilt.name == original.name
    assert rebuilt.command == original.command
    assert rebuilt.args == original.args
    assert rebuilt.transport == original.transport


@pytest.mark.unit
class TestMCPConfigCreation:
  """Tests for MCPConfig instantiation and validation."""

  def test_empty_config(self):
    """MCPConfig with no servers is valid."""
    cfg = MCPConfig()
    assert cfg.servers == []
    assert cfg.auto_connect is True

  def test_config_with_servers(self):
    """MCPConfig stores multiple servers."""
    s1 = MCPServerConfig(name="a", command="cmd_a")
    s2 = MCPServerConfig(name="b", transport="sse", url="http://b")
    cfg = MCPConfig(servers=[s1, s2])
    assert len(cfg.servers) == 2

  def test_duplicate_server_names_raises(self):
    """Duplicate server names in MCPConfig raise ValueError."""
    s1 = MCPServerConfig(name="dup", command="cmd1")
    s2 = MCPServerConfig(name="dup", command="cmd2")
    with pytest.raises(ValueError, match="Duplicate server names"):
      MCPConfig(servers=[s1, s2])

  def test_auto_connect_default_true(self):
    """auto_connect defaults to True."""
    cfg = MCPConfig()
    assert cfg.auto_connect is True


@pytest.mark.unit
class TestMCPConfigServerOps:
  """Tests for get_server, add_server, remove_server."""

  def test_get_server_found(self):
    """get_server returns the matching server config."""
    s = MCPServerConfig(name="target", command="cmd")
    cfg = MCPConfig(servers=[s])
    assert cfg.get_server("target") is s

  def test_get_server_not_found(self):
    """get_server returns None for unknown name."""
    cfg = MCPConfig()
    assert cfg.get_server("missing") is None

  def test_add_server(self):
    """add_server appends a new server."""
    cfg = MCPConfig()
    cfg.add_server(MCPServerConfig(name="new", command="cmd"))
    assert len(cfg.servers) == 1

  def test_add_duplicate_server_raises(self):
    """add_server with an existing name raises ValueError."""
    cfg = MCPConfig(servers=[MCPServerConfig(name="x", command="cmd")])
    with pytest.raises(ValueError, match="already exists"):
      cfg.add_server(MCPServerConfig(name="x", command="cmd2"))

  def test_remove_server_found(self):
    """remove_server returns True and removes the server."""
    cfg = MCPConfig(servers=[MCPServerConfig(name="rm", command="cmd")])
    assert cfg.remove_server("rm") is True
    assert len(cfg.servers) == 0

  def test_remove_server_not_found(self):
    """remove_server returns False for unknown name."""
    cfg = MCPConfig()
    assert cfg.remove_server("ghost") is False


@pytest.mark.unit
class TestMCPConfigSerialization:
  """Tests for MCPConfig to_dict / from_dict / file I/O."""

  def test_to_dict(self):
    """to_dict produces a serializable dictionary."""
    s = MCPServerConfig(name="s", command="cmd")
    cfg = MCPConfig(servers=[s], auto_connect=False)
    d = cfg.to_dict()
    assert d["auto_connect"] is False
    assert len(d["servers"]) == 1

  def test_from_dict_roundtrip(self):
    """from_dict(to_dict(cfg)) reproduces the original config."""
    s = MCPServerConfig(name="rt", command="node")
    original = MCPConfig(servers=[s], auto_connect=False, reconnect_on_failure=False)
    rebuilt = MCPConfig.from_dict(original.to_dict())
    assert rebuilt.auto_connect is False
    assert rebuilt.reconnect_on_failure is False
    assert len(rebuilt.servers) == 1
    assert rebuilt.servers[0].name == "rt"

  def test_from_file_and_to_file(self, tmp_path):
    """to_file writes valid JSON; from_file reconstructs the config."""
    s = MCPServerConfig(name="file_test", command="echo")
    cfg = MCPConfig(servers=[s])
    path = tmp_path / "mcp.json"
    cfg.to_file(path)

    loaded = MCPConfig.from_file(path)
    assert len(loaded.servers) == 1
    assert loaded.servers[0].name == "file_test"

  def test_from_file_missing_raises(self, tmp_path):
    """from_file with a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
      MCPConfig.from_file(tmp_path / "no_such_file.json")
