"""Unit tests for MCP error classes and protocol utilities.

Covers error hierarchy, attributes, status codes, and JSON-RPC protocol
encoding/decoding/validation functions.
"""

import json

import pytest

from definable.agent.mcp.errors import (
  MCPConnectionError,
  MCPError,
  MCPPromptNotFoundError,
  MCPProtocolError,
  MCPResourceNotFoundError,
  MCPServerNotFoundError,
  MCPTimeoutError,
  MCPToolNotFoundError,
)
from definable.agent.mcp.protocol import (
  create_error_response,
  decode_message,
  decode_response,
  encode_request,
  get_error_message,
  internal_error,
  invalid_params,
  invalid_request,
  is_error_response,
  method_not_found,
  parse_error,
  validate_response,
)
from definable.agent.mcp.types import (
  JSONRPCErrorCode,
  JSONRPCErrorData,
  JSONRPCNotification,
  JSONRPCRequest,
  JSONRPCResponse,
  MCPCapabilities,
  MCPClientInfo,
  MCPImplementation,
  MCPPromptArgument,
  MCPPromptDefinition,
  MCPPromptGetResult,
  MCPPromptListResult,
  MCPPromptMessage,
  MCPPromptMessageRole,
  MCPResource,
  MCPResourceListResult,
  MCPResourceTemplate,
  MCPResourceTemplateListResult,
  MCPServerInfo,
  MCPTextContent,
  MCPTextResourceContent,
  MCPToolCallResult,
  MCPToolDefinition,
  MCPToolInputSchema,
  MCPToolListResult,
)


# ===========================================================================
# Error classes
# ===========================================================================


@pytest.mark.unit
class TestMCPErrorHierarchy:
  """Tests for the MCP error class hierarchy."""

  def test_mcp_error_is_exception(self):
    e = MCPError("test")
    assert isinstance(e, Exception)

  def test_mcp_error_default_status(self):
    e = MCPError("test")
    assert e.status_code == 500

  def test_mcp_error_custom_status(self):
    e = MCPError("test", status_code=418)
    assert e.status_code == 418

  def test_mcp_error_server_name(self):
    e = MCPError("test", server_name="my-server")
    assert e.server_name == "my-server"

  def test_mcp_error_type_field(self):
    e = MCPError("test")
    assert e.type == "mcp_error"

  def test_mcp_error_str(self):
    e = MCPError("something went wrong")
    assert "something went wrong" in str(e)


@pytest.mark.unit
class TestMCPConnectionError:
  """Tests for MCPConnectionError."""

  def test_status_code_503(self):
    e = MCPConnectionError("cannot connect")
    assert e.status_code == 503

  def test_original_error_stored(self):
    orig = ValueError("port refused")
    e = MCPConnectionError("fail", original_error=orig)
    assert e.original_error is orig

  def test_original_error_default_none(self):
    e = MCPConnectionError("fail")
    assert e.original_error is None

  def test_type_field(self):
    e = MCPConnectionError("fail")
    assert e.type == "mcp_connection_error"

  def test_is_mcp_error_subclass(self):
    assert issubclass(MCPConnectionError, MCPError)


@pytest.mark.unit
class TestMCPTimeoutError:
  """Tests for MCPTimeoutError."""

  def test_status_code_504(self):
    e = MCPTimeoutError("timed out")
    assert e.status_code == 504

  def test_timeout_seconds_stored(self):
    e = MCPTimeoutError("timed out", timeout_seconds=30.0)
    assert e.timeout_seconds == 30.0

  def test_timeout_seconds_default_none(self):
    e = MCPTimeoutError("timed out")
    assert e.timeout_seconds is None

  def test_type_field(self):
    e = MCPTimeoutError("timed out")
    assert e.type == "mcp_timeout_error"


@pytest.mark.unit
class TestMCPProtocolError:
  """Tests for MCPProtocolError."""

  def test_status_code_502(self):
    e = MCPProtocolError("bad message")
    assert e.status_code == 502

  def test_error_code_stored(self):
    e = MCPProtocolError("bad", error_code=-32700)
    assert e.error_code == -32700

  def test_error_data_stored(self):
    e = MCPProtocolError("bad", error_data={"detail": "parse error"})
    assert e.error_data == {"detail": "parse error"}

  def test_type_field(self):
    e = MCPProtocolError("bad")
    assert e.type == "mcp_protocol_error"


@pytest.mark.unit
class TestMCPToolNotFoundError:
  """Tests for MCPToolNotFoundError."""

  def test_status_code_404(self):
    e = MCPToolNotFoundError("not found", tool_name="search")
    assert e.status_code == 404

  def test_tool_name_stored(self):
    e = MCPToolNotFoundError("not found", tool_name="search")
    assert e.tool_name == "search"

  def test_available_tools_stored(self):
    e = MCPToolNotFoundError("not found", tool_name="x", available_tools=["a", "b"])
    assert e.available_tools == ["a", "b"]

  def test_server_name_passed(self):
    e = MCPToolNotFoundError("not found", tool_name="x", server_name="srv")
    assert e.server_name == "srv"

  def test_type_field(self):
    e = MCPToolNotFoundError("not found", tool_name="x")
    assert e.type == "mcp_tool_not_found_error"


@pytest.mark.unit
class TestMCPServerNotFoundError:
  """Tests for MCPServerNotFoundError."""

  def test_server_name_stored(self):
    e = MCPServerNotFoundError("not found", server_name="srv")
    assert e.server_name == "srv"

  def test_available_servers_stored(self):
    e = MCPServerNotFoundError("not found", server_name="x", available_servers=["a"])
    assert e.available_servers == ["a"]


@pytest.mark.unit
class TestMCPResourceNotFoundError:
  """Tests for MCPResourceNotFoundError."""

  def test_resource_uri_stored(self):
    e = MCPResourceNotFoundError("not found", resource_uri="file:///etc/hosts")
    assert e.resource_uri == "file:///etc/hosts"

  def test_status_code_404(self):
    e = MCPResourceNotFoundError("not found", resource_uri="x")
    assert e.status_code == 404


@pytest.mark.unit
class TestMCPPromptNotFoundError:
  """Tests for MCPPromptNotFoundError."""

  def test_prompt_name_stored(self):
    e = MCPPromptNotFoundError("not found", prompt_name="greet")
    assert e.prompt_name == "greet"

  def test_status_code_404(self):
    e = MCPPromptNotFoundError("not found", prompt_name="x")
    assert e.status_code == 404


# ===========================================================================
# JSON-RPC types (Pydantic models)
# ===========================================================================


@pytest.mark.unit
class TestJSONRPCTypes:
  """Tests for JSON-RPC 2.0 Pydantic models."""

  def test_request_defaults(self):
    r = JSONRPCRequest(method="test")
    assert r.jsonrpc == "2.0"
    assert r.method == "test"
    assert r.params is None
    assert r.id is None

  def test_request_with_id(self):
    r = JSONRPCRequest(method="tools/list", id=42)
    assert r.id == 42

  def test_notification_defaults(self):
    n = JSONRPCNotification(method="ping")
    assert n.jsonrpc == "2.0"
    assert n.params is None

  def test_error_data(self):
    e = JSONRPCErrorData(code=-32700, message="Parse error")
    assert e.code == -32700
    assert e.data is None

  def test_response_success(self):
    r = JSONRPCResponse(id=1, result={"tools": []})
    assert r.result == {"tools": []}
    assert r.error is None

  def test_response_error(self):
    r = JSONRPCResponse(id=1, error=JSONRPCErrorData(code=-32600, message="Invalid"))
    assert r.error is not None
    assert r.error.code == -32600

  def test_error_code_enum_values(self):
    assert JSONRPCErrorCode.PARSE_ERROR.value == -32700
    assert JSONRPCErrorCode.INVALID_REQUEST.value == -32600
    assert JSONRPCErrorCode.METHOD_NOT_FOUND.value == -32601
    assert JSONRPCErrorCode.INVALID_PARAMS.value == -32602
    assert JSONRPCErrorCode.INTERNAL_ERROR.value == -32603


@pytest.mark.unit
class TestMCPProtocolTypes:
  """Tests for MCP protocol Pydantic models."""

  def test_implementation(self):
    impl = MCPImplementation(name="test-server", version="1.0.0")
    assert impl.name == "test-server"
    assert impl.version == "1.0.0"

  def test_capabilities_defaults(self):
    c = MCPCapabilities()
    assert c.tools is None
    assert c.resources is None
    assert c.prompts is None

  def test_server_info(self):
    info = MCPServerInfo(
      protocolVersion="2024-11-05",
      capabilities=MCPCapabilities(tools={}),
      serverInfo=MCPImplementation(name="test", version="1.0"),
    )
    assert info.protocolVersion == "2024-11-05"
    assert info.capabilities.tools == {}

  def test_client_info_defaults(self):
    c = MCPClientInfo()
    assert c.protocolVersion == "2024-11-05"
    assert c.clientInfo.name == "definable"

  def test_tool_input_schema_defaults(self):
    s = MCPToolInputSchema()
    assert s.type == "object"
    assert s.properties == {}

  def test_tool_definition(self):
    t = MCPToolDefinition(
      name="search",
      description="Search the web",
      inputSchema=MCPToolInputSchema(
        properties={"query": {"type": "string"}},
        required=["query"],
      ),
    )
    assert t.name == "search"
    assert "query" in t.inputSchema.properties

  def test_tool_call_result(self):
    r = MCPToolCallResult(
      content=[MCPTextContent(text="result data")],
      isError=False,
    )
    assert len(r.content) == 1
    assert isinstance(r.content[0], MCPTextContent)
    assert r.content[0].text == "result data"

  def test_tool_list_result(self):
    r = MCPToolListResult(tools=[], nextCursor=None)
    assert r.tools == []

  def test_resource(self):
    r = MCPResource(uri="file:///test.txt", name="test.txt")
    assert r.uri == "file:///test.txt"

  def test_resource_template(self):
    t = MCPResourceTemplate(uriTemplate="file:///{path}", name="files")
    assert "{path}" in t.uriTemplate

  def test_resource_list_result(self):
    r = MCPResourceListResult(resources=[])
    assert r.nextCursor is None

  def test_resource_template_list_result(self):
    r = MCPResourceTemplateListResult(resourceTemplates=[])
    assert r.nextCursor is None

  def test_text_resource_content(self):
    c = MCPTextResourceContent(uri="file:///a.txt", text="hello")
    assert c.text == "hello"

  def test_prompt_argument(self):
    a = MCPPromptArgument(name="language", description="Target language", required=True)
    assert a.name == "language"
    assert a.required is True

  def test_prompt_definition(self):
    p = MCPPromptDefinition(name="translate", arguments=[MCPPromptArgument(name="lang")])
    assert p.name == "translate"
    assert p.arguments is not None
    assert len(p.arguments) == 1

  def test_prompt_list_result(self):
    r = MCPPromptListResult(prompts=[])
    assert r.nextCursor is None

  def test_prompt_message_role_enum(self):
    assert MCPPromptMessageRole.USER.value == "user"
    assert MCPPromptMessageRole.ASSISTANT.value == "assistant"

  def test_prompt_message(self):
    m = MCPPromptMessage(
      role=MCPPromptMessageRole.USER,
      content=MCPTextContent(text="Hello"),
    )
    assert m.role == MCPPromptMessageRole.USER

  def test_prompt_get_result(self):
    r = MCPPromptGetResult(
      messages=[
        MCPPromptMessage(
          role=MCPPromptMessageRole.ASSISTANT,
          content=MCPTextContent(text="Hi"),
        )
      ]
    )
    assert len(r.messages) == 1


# ===========================================================================
# Protocol utilities
# ===========================================================================


@pytest.mark.unit
class TestEncodeRequest:
  """Tests for encode_request()."""

  def test_request_with_id(self):
    raw = encode_request("tools/list", params={"cursor": None}, request_id=1)
    data = json.loads(raw)
    assert data["jsonrpc"] == "2.0"
    assert data["method"] == "tools/list"
    assert data["id"] == 1

  def test_notification_without_id(self):
    raw = encode_request("notifications/initialized")
    data = json.loads(raw)
    assert data["method"] == "notifications/initialized"
    assert "id" not in data

  def test_params_omitted_when_none(self):
    raw = encode_request("ping", request_id=1)
    data = json.loads(raw)
    assert "params" not in data

  def test_string_request_id(self):
    raw = encode_request("test", request_id="abc-123")
    data = json.loads(raw)
    assert data["id"] == "abc-123"


@pytest.mark.unit
class TestDecodeResponse:
  """Tests for decode_response()."""

  def test_from_dict(self):
    r = decode_response({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
    assert r.id == 1
    assert r.result == {"tools": []}

  def test_from_string(self):
    raw = '{"jsonrpc": "2.0", "id": 1, "result": "ok"}'
    r = decode_response(raw)
    assert r.result == "ok"

  def test_from_bytes(self):
    raw = b'{"jsonrpc": "2.0", "id": 1, "result": null}'
    r = decode_response(raw)
    assert r.result is None

  def test_invalid_json_raises(self):
    with pytest.raises(MCPProtocolError, match="Invalid JSON"):
      decode_response("not json {{{")

  def test_accepts_minimal_response(self):
    """JSONRPCResponse fields are all optional, so minimal dicts are valid."""
    r = decode_response({"jsonrpc": "2.0"})
    assert r.result is None
    assert r.error is None


@pytest.mark.unit
class TestDecodeMessage:
  """Tests for decode_message()."""

  def test_request_decoded(self):
    method, params, req_id = decode_message({"jsonrpc": "2.0", "method": "tools/list", "params": {"cursor": None}, "id": 1})
    assert method == "tools/list"
    assert params == {"cursor": None}
    assert req_id == 1

  def test_notification_decoded(self):
    method, params, req_id = decode_message({"jsonrpc": "2.0", "method": "ping"})
    assert method == "ping"
    assert req_id is None

  def test_from_string(self):
    raw = '{"jsonrpc": "2.0", "method": "test", "id": 5}'
    method, _, req_id = decode_message(raw)
    assert method == "test"
    assert req_id == 5

  def test_from_bytes(self):
    raw = b'{"jsonrpc": "2.0", "method": "test"}'
    method, _, _ = decode_message(raw)
    assert method == "test"

  def test_invalid_json_raises(self):
    with pytest.raises(MCPProtocolError, match="Invalid JSON"):
      decode_message("broken{")

  def test_non_object_raises(self):
    with pytest.raises(MCPProtocolError, match="must be an object"):
      decode_message('"just a string"')

  def test_missing_jsonrpc_raises(self):
    with pytest.raises(MCPProtocolError, match="jsonrpc version"):
      decode_message({"method": "test"})

  def test_wrong_jsonrpc_version_raises(self):
    with pytest.raises(MCPProtocolError, match="jsonrpc version"):
      decode_message({"jsonrpc": "1.0", "method": "test"})

  def test_missing_method_raises(self):
    with pytest.raises(MCPProtocolError, match="Missing method"):
      decode_message({"jsonrpc": "2.0"})


@pytest.mark.unit
class TestErrorResponseHelpers:
  """Tests for error response creation and inspection."""

  def test_create_error_response(self):
    r = create_error_response(-32700, "Parse error", request_id=1)
    assert r.error is not None
    assert r.error.code == -32700
    assert r.error.message == "Parse error"
    assert r.id == 1

  def test_create_error_response_with_data(self):
    r = create_error_response(-32600, "Bad", error_data={"detail": "x"})
    assert r.error is not None
    assert r.error.data == {"detail": "x"}

  def test_is_error_response_true(self):
    r = JSONRPCResponse(error=JSONRPCErrorData(code=-1, message="err"))
    assert is_error_response(r) is True

  def test_is_error_response_false(self):
    r = JSONRPCResponse(id=1, result="ok")
    assert is_error_response(r) is False

  def test_get_error_message_present(self):
    r = JSONRPCResponse(error=JSONRPCErrorData(code=-1, message="oops"))
    assert get_error_message(r) == "oops"

  def test_get_error_message_absent(self):
    r = JSONRPCResponse(id=1, result="ok")
    assert get_error_message(r) is None

  def test_validate_response_success(self):
    r = JSONRPCResponse(id=1, result={"tools": []})
    result = validate_response(r)
    assert result == {"tools": []}

  def test_validate_response_error_raises(self):
    r = JSONRPCResponse(error=JSONRPCErrorData(code=-32700, message="Parse error"))
    with pytest.raises(MCPProtocolError, match="Parse error"):
      validate_response(r)

  def test_validate_response_error_with_server_name(self):
    r = JSONRPCResponse(error=JSONRPCErrorData(code=-1, message="fail"))
    with pytest.raises(MCPProtocolError) as exc_info:
      validate_response(r, server_name="my-server")
    assert exc_info.value.server_name == "my-server"


@pytest.mark.unit
class TestStandardErrorConstructors:
  """Tests for standard JSON-RPC error response constructors."""

  def test_parse_error(self):
    r = parse_error(request_id=1)
    assert r.error is not None
    assert r.error.code == JSONRPCErrorCode.PARSE_ERROR
    assert r.error.message == "Parse error"

  def test_invalid_request(self):
    r = invalid_request(request_id=2)
    assert r.error is not None
    assert r.error.code == JSONRPCErrorCode.INVALID_REQUEST

  def test_method_not_found(self):
    r = method_not_found("nonexistent", request_id=3)
    assert r.error is not None
    assert r.error.code == JSONRPCErrorCode.METHOD_NOT_FOUND
    assert "nonexistent" in r.error.message

  def test_invalid_params(self):
    r = invalid_params("missing query", request_id=4)
    assert r.error is not None
    assert r.error.code == JSONRPCErrorCode.INVALID_PARAMS
    assert "missing query" in r.error.message

  def test_internal_error(self):
    r = internal_error("server crashed", request_id=5)
    assert r.error is not None
    assert r.error.code == JSONRPCErrorCode.INTERNAL_ERROR

  def test_constructors_without_request_id(self):
    r = parse_error()
    assert r.id is None
