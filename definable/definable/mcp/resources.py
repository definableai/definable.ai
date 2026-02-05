"""MCPResourceProvider - Access MCP resources from connected servers."""

from typing import Dict, List, Optional

from definable.mcp.client import MCPClient
from definable.mcp.errors import MCPServerNotFoundError
from definable.mcp.types import MCPResource, MCPResourceContent
from definable.utils.log import log_debug, log_warning


class MCPResourceProvider:
  """Provider for accessing MCP resources from connected servers.

  Resources in MCP represent data sources like files, databases,
  or API endpoints that can be read by agents.

  Example:
      client = MCPClient(config)
      async with client:
          resources = MCPResourceProvider(client)

          # List all resources
          all_resources = await resources.list_resources()
          for server, res_list in all_resources.items():
              print(f"{server}: {[r.name for r in res_list]}")

          # Read a specific resource
          content = await resources.read_resource("fs", "file:///tmp/config.json")
          print(content[0].text)
  """

  def __init__(self, client: MCPClient) -> None:
    """Initialize the resource provider.

    Args:
        client: Connected MCP client.
    """
    self._client = client

  async def list_resources(
    self,
    server_name: Optional[str] = None,
  ) -> Dict[str, List[MCPResource]]:
    """List available resources from servers.

    Args:
        server_name: Specific server to query (None = all servers).

    Returns:
        Dictionary mapping server name to list of resources.

    Raises:
        MCPServerNotFoundError: If specified server not found.
    """
    if server_name:
      connection = self._client.get_connection(server_name)
      if not connection:
        raise MCPServerNotFoundError(
          f"Server '{server_name}' not found",
          server_name=server_name,
          available_servers=self._client.servers,
        )
      resources = await connection.list_resources()
      return {server_name: resources}

    return await self._client.list_all_resources()

  async def read_resource(
    self,
    server_name: str,
    uri: str,
  ) -> List[MCPResourceContent]:
    """Read content from a resource.

    Args:
        server_name: Server that hosts the resource.
        uri: Resource URI.

    Returns:
        List of resource content items.

    Raises:
        MCPServerNotFoundError: If server not found.
        MCPResourceNotFoundError: If resource not found.
        MCPProtocolError: If read fails.
    """
    log_debug(f"MCPResourceProvider: Reading {uri} from {server_name}")
    return await self._client.read_resource(server_name, uri)

  async def read_text(
    self,
    server_name: str,
    uri: str,
  ) -> str:
    """Read text content from a resource.

    Convenience method that extracts text from the first content item.

    Args:
        server_name: Server that hosts the resource.
        uri: Resource URI.

    Returns:
        Text content.

    Raises:
        ValueError: If resource has no text content.
    """
    contents = await self.read_resource(server_name, uri)
    for content in contents:
      if hasattr(content, "text"):
        return content.text
    raise ValueError(f"Resource '{uri}' has no text content")

  async def find_resource(self, uri: str) -> Optional[str]:
    """Find which server has a resource with the given URI.

    Args:
        uri: Resource URI to find.

    Returns:
        Server name if found, None otherwise.
    """
    all_resources = await self._client.list_all_resources()
    for server_name, resources in all_resources.items():
      for resource in resources:
        if resource.uri == uri:
          return server_name
    return None

  async def get_resource_info(
    self,
    server_name: str,
    uri: str,
  ) -> Optional[MCPResource]:
    """Get metadata for a specific resource.

    Args:
        server_name: Server to query.
        uri: Resource URI.

    Returns:
        Resource metadata if found, None otherwise.
    """
    connection = self._client.get_connection(server_name)
    if not connection:
      return None

    try:
      resources = await connection.list_resources()
      for resource in resources:
        if resource.uri == uri:
          return resource
    except Exception as e:
      log_warning(f"MCPResourceProvider: Failed to get resource info: {e}")

    return None
