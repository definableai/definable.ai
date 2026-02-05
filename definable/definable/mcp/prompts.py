"""MCPPromptProvider - Access MCP prompts from connected servers."""

from typing import Dict, List, Optional

from definable.mcp.client import MCPClient
from definable.mcp.errors import MCPPromptNotFoundError, MCPServerNotFoundError
from definable.mcp.types import MCPPromptDefinition, MCPPromptGetResult, MCPPromptMessage
from definable.utils.log import log_debug, log_warning


class MCPPromptProvider:
  """Provider for accessing MCP prompts from connected servers.

  Prompts in MCP are templated messages that can be customized with
  arguments and used to generate agent instructions or user messages.

  Example:
      client = MCPClient(config)
      async with client:
          prompts = MCPPromptProvider(client)

          # List all prompts
          all_prompts = await prompts.list_prompts()
          for server, prompt_list in all_prompts.items():
              print(f"{server}: {[p.name for p in prompt_list]}")

          # Get a rendered prompt
          result = await prompts.get_prompt(
              "assistant",
              "code_review",
              {"language": "python", "style": "detailed"}
          )
          for msg in result.messages:
              print(f"{msg.role}: {msg.content.text}")
  """

  def __init__(self, client: MCPClient) -> None:
    """Initialize the prompt provider.

    Args:
        client: Connected MCP client.
    """
    self._client = client

  async def list_prompts(
    self,
    server_name: Optional[str] = None,
  ) -> Dict[str, List[MCPPromptDefinition]]:
    """List available prompts from servers.

    Args:
        server_name: Specific server to query (None = all servers).

    Returns:
        Dictionary mapping server name to list of prompts.

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
      prompts = await connection.list_prompts()
      return {server_name: prompts}

    return await self._client.list_all_prompts()

  async def get_prompt(
    self,
    server_name: str,
    prompt_name: str,
    arguments: Optional[Dict[str, str]] = None,
  ) -> MCPPromptGetResult:
    """Get a rendered prompt from a server.

    Args:
        server_name: Server that provides the prompt.
        prompt_name: Name of the prompt.
        arguments: Prompt arguments for template substitution.

    Returns:
        Rendered prompt result with messages.

    Raises:
        MCPServerNotFoundError: If server not found.
        MCPPromptNotFoundError: If prompt not found.
        MCPProtocolError: If request fails.
    """
    log_debug(f"MCPPromptProvider: Getting prompt '{prompt_name}' from {server_name}")
    return await self._client.get_prompt(server_name, prompt_name, arguments)

  async def get_messages(
    self,
    server_name: str,
    prompt_name: str,
    arguments: Optional[Dict[str, str]] = None,
  ) -> List[MCPPromptMessage]:
    """Get prompt messages as a list.

    Convenience method that returns just the messages.

    Args:
        server_name: Server that provides the prompt.
        prompt_name: Name of the prompt.
        arguments: Prompt arguments.

    Returns:
        List of prompt messages.
    """
    result = await self.get_prompt(server_name, prompt_name, arguments)
    return result.messages

  async def get_text(
    self,
    server_name: str,
    prompt_name: str,
    arguments: Optional[Dict[str, str]] = None,
  ) -> str:
    """Get prompt as concatenated text.

    Convenience method that concatenates all message content.

    Args:
        server_name: Server that provides the prompt.
        prompt_name: Name of the prompt.
        arguments: Prompt arguments.

    Returns:
        Combined text from all messages.
    """
    result = await self.get_prompt(server_name, prompt_name, arguments)
    parts = []
    for msg in result.messages:
      parts.append(f"{msg.role}: {msg.content.text}")
    return "\n\n".join(parts)

  async def find_prompt(self, prompt_name: str) -> Optional[str]:
    """Find which server has a prompt with the given name.

    Args:
        prompt_name: Prompt name to find.

    Returns:
        Server name if found, None otherwise.
    """
    all_prompts = await self._client.list_all_prompts()
    for server_name, prompts in all_prompts.items():
      for prompt in prompts:
        if prompt.name == prompt_name:
          return server_name
    return None

  async def get_prompt_info(
    self,
    server_name: str,
    prompt_name: str,
  ) -> Optional[MCPPromptDefinition]:
    """Get metadata for a specific prompt.

    Args:
        server_name: Server to query.
        prompt_name: Prompt name.

    Returns:
        Prompt definition if found, None otherwise.
    """
    connection = self._client.get_connection(server_name)
    if not connection:
      return None

    try:
      prompts = await connection.list_prompts()
      for prompt in prompts:
        if prompt.name == prompt_name:
          return prompt
    except Exception as e:
      log_warning(f"MCPPromptProvider: Failed to get prompt info: {e}")

    return None

  async def get_prompt_arguments(
    self,
    server_name: str,
    prompt_name: str,
  ) -> List[str]:
    """Get the list of argument names for a prompt.

    Args:
        server_name: Server to query.
        prompt_name: Prompt name.

    Returns:
        List of argument names (required and optional).

    Raises:
        MCPPromptNotFoundError: If prompt not found.
    """
    info = await self.get_prompt_info(server_name, prompt_name)
    if not info:
      raise MCPPromptNotFoundError(
        f"Prompt '{prompt_name}' not found on server '{server_name}'",
        prompt_name=prompt_name,
        server_name=server_name,
      )

    if not info.arguments:
      return []

    return [arg.name for arg in info.arguments]
