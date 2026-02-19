#!/usr/bin/env python3
"""Review test: MCP through Agent composition.

Tests MCPConfig, MCPServerConfig, MCPToolkit construction.
Actual MCP server connection requires npx — skips if not available.
"""

import sys, os, shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

passed, failed, skipped = 0, 0, 0


def check(name, condition, error=""):
  global passed, failed
  if condition:
    print(f"✅ PASS: {name}")
    passed += 1
  else:
    print(f"❌ FAIL: {name} — {error}")
    failed += 1


def skip(name, reason):
  global skipped
  print(f"⚠️  SKIP: {name} — {reason}")
  skipped += 1


try:
  from definable.mcp import MCPConfig, MCPServerConfig, MCPToolkit
  from definable.mcp import MCPError, MCPConnectionError, MCPTimeoutError
  from definable.agent import Agent, MockModel

  check("Import mcp + agent", True)
except Exception as e:
  check("Import mcp + agent", False, str(e))
  print(f"\n{'=' * 60}\nRESULT: {passed} passed | {failed} failed | {skipped} skipped")
  sys.exit(1)


# ── MCPServerConfig ─────────────────────────────────────────────
try:
  server = MCPServerConfig(name="test-fs", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
  check("MCPServerConfig constructs", True)
  check("MCPServerConfig.name", server.name == "test-fs", f"got: {server.name}")
except Exception as e:
  check("MCPServerConfig", False, str(e))


# ── MCPConfig ───────────────────────────────────────────────────
try:
  config = MCPConfig(
    servers=[
      MCPServerConfig(name="fs", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
    ]
  )
  check("MCPConfig(servers=[...]) constructs", True)
except Exception as e:
  check("MCPConfig", False, str(e))

try:
  config = MCPConfig(servers=[])
  check("MCPConfig(servers=[]) empty", True)
except Exception as e:
  check("MCPConfig empty", False, str(e))


# ── MCPToolkit ──────────────────────────────────────────────────
try:
  config = MCPConfig(
    servers=[
      MCPServerConfig(name="fs", command="echo", args=["test"]),
    ]
  )
  toolkit = MCPToolkit(config=config)
  check("MCPToolkit(config=...) constructs", True)
except Exception as e:
  check("MCPToolkit construction", False, str(e))


# ── MCP Error types importable ──────────────────────────────────
check("MCPError importable", MCPError is not None, "None")
check("MCPConnectionError importable", MCPConnectionError is not None, "None")
check("MCPTimeoutError importable", MCPTimeoutError is not None, "None")


# ── Agent + MCPToolkit (construction only, no server) ───────────
try:
  config = MCPConfig(
    servers=[
      MCPServerConfig(name="dummy", command="echo", args=["dummy"]),
    ]
  )
  toolkit = MCPToolkit(config=config)
  agent = Agent(model=MockModel(), toolkits=[toolkit])
  check("Agent(toolkits=[MCPToolkit(...)]) constructs", True)
except Exception as e:
  check("Agent + MCPToolkit construction", False, str(e))


# ── Live MCP test (npx required) ───────────────────────────────
if not shutil.which("npx"):
  skip("MCP live connection", "npx not found in PATH")
else:
  import asyncio

  try:
    config = MCPConfig(
      servers=[
        MCPServerConfig(name="fs", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
      ]
    )
    toolkit = MCPToolkit(config=config)

    async def test_mcp_lifecycle():
      async with toolkit:
        tools = toolkit.tools
        return len(tools)

    tool_count = asyncio.run(test_mcp_lifecycle())
    check("MCPToolkit async lifecycle: init + tools", tool_count > 0, f"tool_count={tool_count}")
  except Exception as e:
    check("MCPToolkit async lifecycle", False, f"{type(e).__name__}: {e}")


print(f"\n{'=' * 60}")
print(f"RESULT: {passed} passed | {failed} failed | {skipped} skipped")
sys.exit(1 if failed else 0)
