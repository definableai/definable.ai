"""ToolRetry exception — PydanticAI-style tool retry with model feedback."""


class ToolRetry(Exception):
  """Raise inside a @tool function to ask the model to retry with better args.

  The message is sent back to the model as the tool result, prefixed with
  ``[RETRY]``. The model sees the feedback, adjusts its arguments, and
  retries the tool call. The loop tracks retry count per tool_call_id and
  respects ``max_retries``.

  Example::

      from definable.agent.pipeline import ToolRetry
      from definable.tool.decorator import tool

      @tool
      def search(query: str) -> str:
          if len(query) < 3:
              raise ToolRetry("Query too short. Provide at least 3 characters.")
          return do_search(query)

  Args:
    message: Feedback for the model explaining what went wrong.
    max_retries: Maximum retry attempts for this tool call (default 3).
  """

  def __init__(self, message: str, max_retries: int = 3) -> None:
    self.message = message
    self.max_retries = max_retries
    super().__init__(message)
