"""JSON-RPC 2.0 protocol utilities for MCP.

Helpers for encoding/decoding MCP protocol messages.
"""

import json
from typing import Any, Dict, Optional, Tuple, Union

from definable.mcp.errors import MCPProtocolError
from definable.mcp.types import (
  JSONRPCErrorCode,
  JSONRPCNotification,
  JSONRPCRequest,
  JSONRPCResponse,
)


def encode_request(
  method: str,
  params: Optional[Dict[str, Any]] = None,
  request_id: Optional[Union[str, int]] = None,
) -> str:
  """Encode a JSON-RPC request to string.

  Args:
      method: RPC method name.
      params: Method parameters (optional).
      request_id: Request ID (optional, omit for notification).

  Returns:
      JSON string representation.
  """
  if request_id is None:
    # Notification (no response expected)
    msg = JSONRPCNotification(method=method, params=params)
  else:
    # Request (expects response)
    msg = JSONRPCRequest(method=method, params=params, id=request_id)

  return msg.model_dump_json(exclude_none=True)


def decode_response(data: Union[str, bytes, Dict[str, Any]]) -> JSONRPCResponse:
  """Decode a JSON-RPC response from string, bytes, or dict.

  Args:
      data: Response data to decode.

  Returns:
      Parsed JSONRPCResponse.

  Raises:
      MCPProtocolError: If data is invalid JSON-RPC.
  """
  if isinstance(data, bytes):
    data = data.decode("utf-8")

  if isinstance(data, str):
    try:
      data = json.loads(data)
    except json.JSONDecodeError as e:
      raise MCPProtocolError(f"Invalid JSON: {e}")

  try:
    return JSONRPCResponse.model_validate(data)
  except Exception as e:
    raise MCPProtocolError(f"Invalid JSON-RPC response: {e}")


def decode_message(
  data: Union[str, bytes, Dict[str, Any]],
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Union[str, int]]]:
  """Decode any JSON-RPC message and extract components.

  Args:
      data: Message data to decode.

  Returns:
      Tuple of (method, params, id). ID is None for notifications.

  Raises:
      MCPProtocolError: If data is invalid JSON-RPC.
  """
  if isinstance(data, bytes):
    data = data.decode("utf-8")

  if isinstance(data, str):
    try:
      data = json.loads(data)
    except json.JSONDecodeError as e:
      raise MCPProtocolError(f"Invalid JSON: {e}")

  if not isinstance(data, dict):
    raise MCPProtocolError("JSON-RPC message must be an object")

  if data.get("jsonrpc") != "2.0":
    raise MCPProtocolError("Missing or invalid jsonrpc version")

  method = data.get("method")
  if not method:
    raise MCPProtocolError("Missing method field")

  return method, data.get("params"), data.get("id")


def create_error_response(
  error_code: int,
  error_message: str,
  request_id: Optional[Union[str, int]] = None,
  error_data: Optional[Any] = None,
) -> JSONRPCResponse:
  """Create a JSON-RPC error response.

  Args:
      error_code: JSON-RPC error code.
      error_message: Human-readable error message.
      request_id: Original request ID.
      error_data: Additional error data (optional).

  Returns:
      JSONRPCResponse with error.
  """
  from definable.mcp.types import JSONRPCErrorData

  return JSONRPCResponse(
    id=request_id,
    error=JSONRPCErrorData(
      code=error_code,
      message=error_message,
      data=error_data,
    ),
  )


def is_error_response(response: JSONRPCResponse) -> bool:
  """Check if response is an error response.

  Args:
      response: JSON-RPC response to check.

  Returns:
      True if response contains an error.
  """
  return response.error is not None


def get_error_message(response: JSONRPCResponse) -> Optional[str]:
  """Extract error message from response.

  Args:
      response: JSON-RPC response.

  Returns:
      Error message if response is an error, None otherwise.
  """
  if response.error:
    return response.error.message
  return None


def validate_response(response: JSONRPCResponse, server_name: Optional[str] = None) -> Any:
  """Validate response and extract result, raising on error.

  Args:
      response: JSON-RPC response to validate.
      server_name: Server name for error messages.

  Returns:
      Response result if successful.

  Raises:
      MCPProtocolError: If response contains an error.
  """
  if response.error:
    error = response.error
    raise MCPProtocolError(
      f"RPC error: {error.message}",
      server_name=server_name,
      error_code=error.code,
      error_data=error.data if hasattr(error, "data") else None,
    )
  return response.result


# Standard JSON-RPC error constructors
def parse_error(request_id: Optional[Union[str, int]] = None) -> JSONRPCResponse:
  """Create a Parse Error response (-32700)."""
  return create_error_response(
    JSONRPCErrorCode.PARSE_ERROR,
    "Parse error",
    request_id,
  )


def invalid_request(request_id: Optional[Union[str, int]] = None) -> JSONRPCResponse:
  """Create an Invalid Request response (-32600)."""
  return create_error_response(
    JSONRPCErrorCode.INVALID_REQUEST,
    "Invalid request",
    request_id,
  )


def method_not_found(
  method: str,
  request_id: Optional[Union[str, int]] = None,
) -> JSONRPCResponse:
  """Create a Method Not Found response (-32601)."""
  return create_error_response(
    JSONRPCErrorCode.METHOD_NOT_FOUND,
    f"Method not found: {method}",
    request_id,
  )


def invalid_params(
  message: str,
  request_id: Optional[Union[str, int]] = None,
) -> JSONRPCResponse:
  """Create an Invalid Params response (-32602)."""
  return create_error_response(
    JSONRPCErrorCode.INVALID_PARAMS,
    message,
    request_id,
  )


def internal_error(
  message: str,
  request_id: Optional[Union[str, int]] = None,
) -> JSONRPCResponse:
  """Create an Internal Error response (-32603)."""
  return create_error_response(
    JSONRPCErrorCode.INTERNAL_ERROR,
    message,
    request_id,
  )
