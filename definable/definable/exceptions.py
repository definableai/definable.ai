"""Definable exception hierarchy.

Two separate trees:

1. **Control flow** — intentional redirection, not errors:
   - ``RetryAgentRun`` — retry tool call with feedback to the model
   - ``StopAgentRun``  — terminate execution gracefully

2. **Errors** — something broke:
   - ``DefinableError`` (base)
     - ``ModelAuthenticationError`` (401)
     - ``ModelProviderError`` (502)
       - ``ModelRateLimitError`` (429)
     - ``RemoteServerUnavailableError`` (503)
   - ``InputCheckError`` / ``OutputCheckError`` — guardrail violations

3. **Developer mistakes**:
   - ``UserError`` — the developer configured something wrong (always actionable)
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from definable.model.message import Message


# ---------------------------------------------------------------------------
# Control flow exceptions — intentional redirection, not errors
# ---------------------------------------------------------------------------


class ControlFlowException(Exception):
  """Base for exceptions that redirect execution flow, not signal errors.

  Catch these when you want to handle *all* intentional redirections
  (retries, stops) without catching actual errors.
  """

  def __init__(
    self,
    exc,
    user_message: Optional[Union[str, Message]] = None,
    agent_message: Optional[Union[str, Message]] = None,
    messages: Optional[List[Union[dict, Message]]] = None,
    stop_execution: bool = False,
  ):
    super().__init__(exc)
    self.user_message = user_message
    self.agent_message = agent_message
    self.messages = messages
    self.stop_execution = stop_execution
    self.type = "control_flow"
    self.error_id = "control_flow"


class RetryAgentRun(ControlFlowException):
  """Signal that a tool call should be retried with feedback to the model."""

  def __init__(
    self,
    exc,
    user_message: Optional[Union[str, Message]] = None,
    agent_message: Optional[Union[str, Message]] = None,
    messages: Optional[List[Union[dict, Message]]] = None,
  ):
    super().__init__(exc, user_message=user_message, agent_message=agent_message, messages=messages, stop_execution=False)
    self.error_id = "retry_agent_run"


class StopAgentRun(ControlFlowException):
  """Signal that the agent should stop executing entirely."""

  def __init__(
    self,
    exc,
    user_message: Optional[Union[str, Message]] = None,
    agent_message: Optional[Union[str, Message]] = None,
    messages: Optional[List[Union[dict, Message]]] = None,
  ):
    super().__init__(exc, user_message=user_message, agent_message=agent_message, messages=messages, stop_execution=True)
    self.error_id = "stop_agent_run"


# ---------------------------------------------------------------------------
# Backward compatibility — AgentRunException is now an alias
# ---------------------------------------------------------------------------

# AgentRunException was the original base for both RetryAgentRun and StopAgentRun.
# Code that catches AgentRunException will still work because ControlFlowException
# is now the actual base class. We keep AgentRunException as an alias so that:
#   - ``except AgentRunException`` still catches RetryAgentRun/StopAgentRun
#   - ``isinstance(e, AgentRunException)`` still returns True
AgentRunException = ControlFlowException


# ---------------------------------------------------------------------------
# Error exceptions — something actually broke
# ---------------------------------------------------------------------------


class DefinableError(Exception):
  """Base for internal framework errors."""

  def __init__(self, message: str, status_code: int = 500):
    super().__init__(message)
    self.message = message
    self.status_code = status_code
    self.type = "definable_error"
    self.error_id = "definable_error"

  def __str__(self) -> str:
    return str(self.message)


class UserError(DefinableError):
  """Developer mistake — always actionable.

  Raise when the developer configured something incorrectly:
  bad argument combinations, missing required settings, etc.
  The message should tell them exactly what to fix.
  """

  def __init__(self, message: str):
    super().__init__(message, status_code=400)
    self.error_id = "user_error"


class ModelAuthenticationError(DefinableError):
  """Raised when model authentication fails."""

  def __init__(self, message: str, status_code: int = 401, model_name: Optional[str] = None):
    super().__init__(message, status_code)
    self.model_name = model_name
    self.type = "model_authentication_error"
    self.error_id = "model_authentication_error"


class ModelProviderError(DefinableError):
  """Exception raised when a model provider returns an error."""

  def __init__(self, message: str, status_code: int = 502, model_name: Optional[str] = None, model_id: Optional[str] = None):
    super().__init__(message, status_code)
    self.model_name = model_name
    self.model_id = model_id
    self.type = "model_provider_error"
    self.error_id = "model_provider_error"


class ModelRateLimitError(ModelProviderError):
  """Exception raised when a model provider returns a rate limit error."""

  def __init__(self, message: str, status_code: int = 429, model_name: Optional[str] = None, model_id: Optional[str] = None):
    super().__init__(message, status_code, model_name, model_id)
    self.error_id = "model_rate_limit_error"


class ContextWindowExceededError(ModelProviderError):
  """Raised when a request exceeds the model's context window.

  Distinct from a generic 400 so callers can react specifically — e.g.
  trim history, summarize, or pick a model with a larger window —
  instead of treating it as an opaque provider error.
  """

  def __init__(self, message: str, status_code: int = 400, model_name: Optional[str] = None, model_id: Optional[str] = None):
    super().__init__(message, status_code, model_name, model_id)
    self.error_id = "context_window_exceeded"


# Substrings that indicate a context-window / token-limit problem.
# Centralized so providers don't ship divergent pattern lists.
CONTEXT_WINDOW_PATTERNS: frozenset[str] = frozenset({
  "context_length_exceeded",
  "context window",
  "maximum context length",
  "token limit",
  "max_tokens",
  "too many tokens",
  "payload too large",
  "content_too_large",
  "request too large",
  "input too long",
  "prompt is too long",  # Anthropic context-window message
  "exceeds the model",
})

# HTTP status codes that providers return for non-retryable client errors.
NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404, 413, 422})


def classify_model_error(
  message: str,
  status_code: int = 502,
  model_name: Optional[str] = None,
  model_id: Optional[str] = None,
) -> ModelProviderError:
  """Map a raw provider error into the most specific ``ModelProviderError`` subclass.

  Returns a ``ContextWindowExceededError`` when the message matches any
  ``CONTEXT_WINDOW_PATTERNS`` substring, a ``ModelRateLimitError`` for 429,
  or a plain ``ModelProviderError`` otherwise. Providers should funnel raised
  errors through this helper so retry/failover logic sees stable subclasses.
  """
  lowered = message.lower()
  if any(pattern in lowered for pattern in CONTEXT_WINDOW_PATTERNS):
    return ContextWindowExceededError(message, status_code=status_code or 400, model_name=model_name, model_id=model_id)
  # 429 is the standard rate limit; Anthropic uses 529 / "overloaded" as a retry-friendly load-shed signal.
  if status_code in (429, 529) or "overloaded" in lowered:
    return ModelRateLimitError(message, status_code=status_code, model_name=model_name, model_id=model_id)
  return ModelProviderError(message, status_code=status_code, model_name=model_name, model_id=model_id)


def is_retryable_model_error(error: ModelProviderError) -> bool:
  """Return ``True`` when ``error`` is a transient provider error worth retrying.

  Centralized classification — fast-path on subclass identity
  (``ContextWindowExceededError`` is non-retryable by construction), then
  fall back to status code and pattern matching for raw provider errors.
  """
  if isinstance(error, ContextWindowExceededError):
    return False
  if error.status_code in NON_RETRYABLE_STATUS_CODES:
    return False
  lowered = str(error.message).lower()
  if any(pattern in lowered for pattern in CONTEXT_WINDOW_PATTERNS):
    return False
  return True


# ---------------------------------------------------------------------------
# Guardrail exceptions
# ---------------------------------------------------------------------------


class CheckTrigger(Enum):
  """Enum for guardrail triggers."""

  OFF_TOPIC = "off_topic"
  INPUT_NOT_ALLOWED = "input_not_allowed"
  OUTPUT_NOT_ALLOWED = "output_not_allowed"
  VALIDATION_FAILED = "validation_failed"

  PROMPT_INJECTION = "prompt_injection"
  PII_DETECTED = "pii_detected"
  GUARDRAIL_BLOCKED = "guardrail_blocked"


class InputCheckError(Exception):
  """Exception raised when an input check fails."""

  def __init__(
    self,
    message: str,
    check_trigger: CheckTrigger = CheckTrigger.INPUT_NOT_ALLOWED,
    additional_data: Optional[Dict[str, Any]] = None,
  ):
    super().__init__(message)
    self.type = "input_check_error"
    self.error_id = check_trigger.value

    self.message = message
    self.check_trigger = check_trigger
    self.additional_data = additional_data


class OutputCheckError(Exception):
  """Exception raised when an output check fails."""

  def __init__(
    self,
    message: str,
    check_trigger: CheckTrigger = CheckTrigger.OUTPUT_NOT_ALLOWED,
    additional_data: Optional[Dict[str, Any]] = None,
  ):
    super().__init__(message)
    self.type = "output_check_error"
    self.error_id = check_trigger.value

    self.message = message
    self.check_trigger = check_trigger
    self.additional_data = additional_data


# ---------------------------------------------------------------------------
# Retryable model errors — guidance-based retry
# ---------------------------------------------------------------------------


class RetryableModelProviderError(Exception):
  """Raised when a model invocation can be retried with guidance.

  The retry_guidance_message is appended to the conversation to help the model
  avoid the same error on the next attempt (e.g., malformed function call).
  """

  def __init__(
    self,
    original_error: Optional[str] = None,
    retry_guidance_message: Optional[str] = None,
  ):
    super().__init__(original_error or "Retryable model provider error")
    self.original_error = original_error
    self.retry_guidance_message = retry_guidance_message


# ---------------------------------------------------------------------------
# Remote server errors
# ---------------------------------------------------------------------------


class RemoteServerUnavailableError(DefinableError):
  """Exception raised when a remote server is unavailable.

  This can happen due to:
  - Connection refused (server not running)
  - Connection timeout
  - Network errors
  - DNS resolution failures
  """

  def __init__(
    self,
    message: str,
    base_url: Optional[str] = None,
    original_error: Optional[Exception] = None,
  ):
    super().__init__(message, status_code=503)
    self.base_url = base_url
    self.original_error = original_error
    self.type = "remote_server_unavailable_error"
    self.error_id = "remote_server_unavailable_error"
