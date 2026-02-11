# mcp

Model Context Protocol (MCP) client — connect to MCP servers to access tools, resources, and prompts.

## Quick Start

```python
from definable.agents import Agent
from definable.mcp import MCPToolkit, MCPConfig, MCPServerConfig

config = MCPConfig(servers=[
  MCPServerConfig(
    name="filesystem",
    transport="stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
  ),
])

async with MCPToolkit(config=config) as toolkit:
  await toolkit.initialize()

  agent = Agent(
    model=model,
    toolkits=[toolkit],
  )
  response = await agent.arun("List files in /tmp")
```

## Module Structure

```
mcp/
├── __init__.py      # Public API exports
├── config.py        # MCPConfig, MCPServerConfig
├── client.py        # MCPClient, MCPServerConnection
├── toolkit.py       # MCPToolkit — exposes MCP tools as Function objects
├── resources.py     # MCPResourceProvider
├── prompts.py       # MCPPromptProvider
├── protocol.py      # JSON-RPC 2.0 utilities
├── types.py         # Pydantic models for protocol types
├── errors.py        # MCP error hierarchy
└── transports/
    ├── base.py      # BaseTransport ABC
    ├── stdio.py     # StdioTransport (subprocess stdin/stdout)
    ├── sse.py       # SSETransport (HTTP + Server-Sent Events)
    └── http.py      # HTTPTransport (Streamable HTTP)
```

## API Reference

### Configuration

```python
from definable.mcp import MCPConfig, MCPServerConfig
```

**MCPServerConfig** — Configuration for a single MCP server:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Unique server identifier |
| `transport` | `str` | `"stdio"` | Transport type: `"stdio"`, `"sse"`, or `"http"` |
| `command` | `Optional[str]` | `None` | Command for stdio transport |
| `args` | `Optional[List[str]]` | `None` | Command arguments |
| `env` | `Optional[Dict]` | `None` | Environment variables |
| `url` | `Optional[str]` | `None` | Server URL for SSE/HTTP transports |
| `headers` | `Optional[Dict]` | `None` | HTTP headers |
| `allowed_tools` | `Optional[List[str]]` | `None` | Tool whitelist (None = allow all) |
| `blocked_tools` | `Optional[List[str]]` | `None` | Tool blacklist |
| `reconnect_on_failure` | `bool` | `True` | Auto-reconnect |

**MCPConfig** — Top-level configuration:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `servers` | `List[MCPServerConfig]` | required | Server configurations |
| `auto_connect` | `bool` | `True` | Connect on client creation |

`MCPConfig` can be loaded from/saved to JSON files via `from_file(path)` / `to_file(path)`.

### MCPClient

```python
from definable.mcp import MCPClient
```

Manages multiple MCP server connections.

| Method | Description |
|--------|-------------|
| `connect()` / `disconnect()` | Connect/disconnect all servers |
| `connect_server(name)` | Connect a specific server |
| `list_all_tools()` | Aggregate tools from all servers |
| `call_tool(server, name, args)` | Execute a tool on a specific server |
| `list_all_resources()` | Aggregate resources |
| `read_resource(server, uri)` | Read a resource |
| `list_all_prompts()` | Aggregate prompts |
| `get_prompt(server, name, args)` | Get a prompt |

### MCPToolkit

```python
from definable.mcp import MCPToolkit
```

Exposes MCP server tools as `Function` objects for agent integration.

```python
toolkit = MCPToolkit(
  config=config,
  tool_name_prefix="",           # Optional prefix for tool names
  include_server_prefix=True,    # Prefix tool names with server name
  require_confirmation=False,    # HITL confirmation for tool calls
)
```

| Method | Description |
|--------|-------------|
| `initialize()` | Connect to servers and discover tools |
| `shutdown()` | Disconnect from servers |
| `refresh_tools()` | Refresh tool list |
| `get_tool_server(name)` | Get which server owns a tool |

### Providers

```python
from definable.mcp import MCPResourceProvider, MCPPromptProvider
```

- `MCPResourceProvider(client)` — Access resources: `list_resources()`, `read_resource()`, `read_text()`, `find_resource()`.
- `MCPPromptProvider(client)` — Access prompts: `list_prompts()`, `get_prompt()`, `get_messages()`, `get_text()`, `find_prompt()`.

### Transports

| Transport | Protocol | Use Case |
|-----------|----------|----------|
| `StdioTransport` | Newline-delimited JSON-RPC over stdin/stdout | Local subprocess servers |
| `SSETransport` | HTTP POST + Server-Sent Events | Remote servers (legacy) |
| `HTTPTransport` | Streamable HTTP (POST with SSE responses) | Remote servers (current) |

### Errors

```python
from definable.mcp import (
  MCPError,                # Base (500)
  MCPConnectionError,      # 503 — connection failures
  MCPTimeoutError,         # 504 — operation timeouts
  MCPProtocolError,        # 502 — protocol violations
  MCPToolNotFoundError,    # 404 — tool not found
  MCPServerNotFoundError,  # 404 — server not found
  MCPResourceNotFoundError,# 404 — resource not found
  MCPPromptNotFoundError,  # 404 — prompt not found
)
```

### Types

```python
from definable.mcp import (
  MCPToolDefinition,    # Tool metadata (name, description, inputSchema)
  MCPToolCallResult,    # Tool execution result
  MCPResource,          # Resource metadata (uri, name, mimeType)
  MCPResourceContent,   # Resource content (text or blob)
  MCPPromptDefinition,  # Prompt metadata (name, description, arguments)
  MCPPromptMessage,     # Message in prompt result
)
```

## See Also

- `agents/` — Agent class accepts `toolkits=[MCPToolkit(...)]`
- `tools/` — `Function` class that MCPToolkit wraps MCP tools into
