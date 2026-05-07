"""DEPRECATED — re-export shim for `definable.agent.toolkit.function`."""

from definable.agent.toolkit.function import (
  Function,
  FunctionCall,
  FunctionExecutionResult,
  ToolResult,
  UserInputField,
  get_entrypoint_docstring,
)

__all__ = [
  "Function",
  "FunctionCall",
  "FunctionExecutionResult",
  "ToolResult",
  "UserInputField",
  "get_entrypoint_docstring",
]
