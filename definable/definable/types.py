"""Shared type aliases — semantic precision without runtime cost."""

from typing import Any, Dict

# Tool call dicts (OpenAI format):
# {"id": str, "type": "function", "function": {"name": str, "arguments": str}}
ToolCallDict = Dict[str, Any]

# JSON Schema objects — recursive structure, cannot constrain further
JsonSchema = Dict[str, Any]

# Open key-value bags (metadata, session_state, provider_data, etc.)
Metadata = Dict[str, Any]

# Generic JSON object — for API payloads, serialization, etc.
JsonDict = Dict[str, Any]
