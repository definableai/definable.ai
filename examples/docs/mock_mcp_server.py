from __future__ import annotations

import json
import sys
from typing import Any


def send(message: dict[str, Any]) -> None:
  sys.stdout.write(json.dumps(message) + "\n")
  sys.stdout.flush()


def text_content(text: str) -> dict[str, Any]:
  return {"type": "text", "text": text}


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
  request_id = request.get("id")
  method = request.get("method")
  params = request.get("params") or {}

  if method == "notifications/initialized":
    return None

  if method == "initialize":
    return {
      "jsonrpc": "2.0",
      "id": request_id,
      "result": {
        "protocolVersion": "2024-11-05",
        "capabilities": {
          "tools": {},
          "resources": {},
          "prompts": {},
        },
        "serverInfo": {
          "name": "docs-mock-server",
          "version": "1.0.0",
        },
      },
    }

  if method == "tools/list":
    return {
      "jsonrpc": "2.0",
      "id": request_id,
      "result": {
        "tools": [
          {
            "name": "echo",
            "description": "Echo back the provided text.",
            "inputSchema": {
              "type": "object",
              "properties": {
                "text": {"type": "string"},
              },
              "required": ["text"],
              "additionalProperties": False,
            },
          }
        ]
      },
    }

  if method == "tools/call":
    tool_name = params.get("name")
    arguments = params.get("arguments") or {}
    if tool_name == "echo":
      text = arguments.get("text", "")
      return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
          "content": [text_content(f"Echo: {text}")],
          "isError": False,
        },
      }
    return {
      "jsonrpc": "2.0",
      "id": request_id,
      "error": {
        "code": -32601,
        "message": f"Unknown tool: {tool_name}",
      },
    }

  if method == "resources/list":
    return {
      "jsonrpc": "2.0",
      "id": request_id,
      "result": {
        "resources": [
          {
            "uri": "docs://handbook",
            "name": "handbook",
            "description": "Internal handbook excerpt",
            "mimeType": "text/plain",
          }
        ]
      },
    }

  if method == "resources/read":
    return {
      "jsonrpc": "2.0",
      "id": request_id,
      "result": {
        "contents": [
          {
            "uri": params.get("uri", "docs://handbook"),
            "mimeType": "text/plain",
            "text": "Definable documentation favors small, verified examples.",
          }
        ]
      },
    }

  if method == "prompts/list":
    return {
      "jsonrpc": "2.0",
      "id": request_id,
      "result": {
        "prompts": [
          {
            "name": "summarize",
            "description": "Return a short summary prompt.",
            "arguments": [{"name": "topic", "required": True}],
          }
        ]
      },
    }

  if method == "prompts/get":
    topic = (params.get("arguments") or {}).get("topic", "the topic")
    return {
      "jsonrpc": "2.0",
      "id": request_id,
      "result": {
        "description": "Summary prompt",
        "messages": [
          {
            "role": "user",
            "content": text_content(f"Summarize {topic} in three bullets."),
          }
        ],
      },
    }

  return {
    "jsonrpc": "2.0",
    "id": request_id,
    "error": {
      "code": -32601,
      "message": f"Unknown method: {method}",
    },
  }


def main() -> None:
  for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
      continue
    try:
      request = json.loads(line)
    except json.JSONDecodeError:
      continue
    response = handle(request)
    if response is not None:
      send(response)


if __name__ == "__main__":
  main()
