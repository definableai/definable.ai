import asyncio
import sys

from definable.mcp import MCPClient, MCPConfig, MCPServerConfig, MCPToolkit

from support import mock_mcp_server_path


config = MCPConfig(
  servers=[
    MCPServerConfig(
      name="mock",
      command=sys.executable,
      args=[str(mock_mcp_server_path())],
    )
  ]
)


async def main() -> None:
  async with MCPClient(config) as client:
    tools = await client.list_tools("mock")
    result = await client.call_tool("mock", "echo", {"text": "hello"})
    resources = await client.list_all_resources()
    prompt = await client.get_prompt("mock", "summarize", {"topic": "testing"})

    assert [tool.name for tool in tools] == ["echo"]
    assert result.content[0].text == "Echo: hello"
    assert resources["mock"][0].uri == "docs://handbook"
    assert prompt.messages[0].content.text == "Summarize testing in three bullets."

  async with MCPToolkit(config) as toolkit:
    assert [tool.name for tool in toolkit.tools] == ["mock_echo"]


asyncio.run(main())
