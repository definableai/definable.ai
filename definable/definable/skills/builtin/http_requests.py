"""HTTP requests skill — make HTTP calls to external APIs.

Gives the agent the ability to interact with REST APIs and web services.
Supports GET, POST, PUT, PATCH, and DELETE methods with configurable
headers, timeouts, and allowed domains for safety.

Example:
    from definable.skills.builtin import HTTPRequests

    agent = Agent(
        model=model,
        skills=[HTTPRequests(allowed_domains=["api.github.com"])],
    )
    output = agent.run("Get the top 5 trending repos on GitHub.")
"""

import json as _json
import contextlib
from typing import Dict, Optional, Set

from definable.skills.base import Skill
from definable.tools.decorator import tool


class HTTPRequests(Skill):
  """Skill for making HTTP requests to external APIs.

  Provides tools for GET, POST, PUT, PATCH, and DELETE requests.
  Includes safety controls via domain allowlists and request timeouts.

  Args:
    allowed_domains: Set of allowed domains. If empty, all domains are allowed.
    default_headers: Headers to include in every request.
    timeout: Request timeout in seconds (default: 30).
    verify_ssl: Whether to verify SSL certificates (default: True).

  Example:
    # Unrestricted
    agent = Agent(model=model, skills=[HTTPRequests()])

    # Restricted to specific APIs
    agent = Agent(
        model=model,
        skills=[HTTPRequests(allowed_domains={"api.github.com", "jsonplaceholder.typicode.com"})],
    )
  """

  name = "http_requests"

  def __init__(
    self,
    *,
    allowed_domains: Optional[Set[str]] = None,
    default_headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    verify_ssl: bool = True,
  ):
    super().__init__()
    self._allowed_domains = allowed_domains or set()
    self._default_headers = default_headers or {}
    self._timeout = timeout
    self._verify_ssl = verify_ssl

  @property
  def instructions(self) -> str:
    domain_note = f"You can only make requests to: {', '.join(sorted(self._allowed_domains))}. " if self._allowed_domains else ""
    return (
      "You have access to HTTP request tools for interacting with web APIs. "
      f"{domain_note}"
      "Use http_get for fetching data, http_post for creating resources, "
      "http_put/http_patch for updates, and http_delete for removal. "
      "Always handle errors gracefully and parse JSON responses when applicable."
    )

  def _check_domain(self, url: str) -> Optional[str]:
    """Check if a URL's domain is allowed. Returns error message or None."""
    if not self._allowed_domains:
      return None
    from urllib.parse import urlparse

    parsed = urlparse(url)
    domain = parsed.hostname or ""
    if domain not in self._allowed_domains:
      return f"Error: Domain '{domain}' is not in the allowed list: {sorted(self._allowed_domains)}"
    return None

  def _make_request(self, method: str, url: str, headers: Optional[str] = None, body: Optional[str] = None) -> str:
    """Make an HTTP request and return the response."""
    domain_error = self._check_domain(url)
    if domain_error:
      return domain_error

    try:
      from urllib.request import Request, urlopen
      from urllib.error import HTTPError, URLError
    except ImportError:
      return "Error: urllib not available."

    # Merge headers
    merged_headers = dict(self._default_headers)
    if headers:
      try:
        parsed_headers = _json.loads(headers)
        merged_headers.update(parsed_headers)
      except _json.JSONDecodeError:
        return 'Error: headers must be valid JSON (e.g. \'{"Content-Type": "application/json"}\')'

    # Prepare body
    data = None
    if body:
      data = body.encode("utf-8")
      if "Content-Type" not in merged_headers:
        merged_headers["Content-Type"] = "application/json"

    try:
      req = Request(url, data=data, headers=merged_headers, method=method)
      with urlopen(req, timeout=self._timeout) as resp:
        response_body = resp.read().decode("utf-8", errors="replace")
        status = resp.status

        # Truncate very large responses
        if len(response_body) > 10000:
          response_body = response_body[:10000] + "\n... [truncated]"

        return f"Status: {status}\n\n{response_body}"

    except HTTPError as e:
      body_text = ""
      with contextlib.suppress(Exception):
        body_text = e.read().decode("utf-8", errors="replace")[:2000]
      return f"HTTP Error {e.code}: {e.reason}\n{body_text}"
    except URLError as e:
      return f"URL Error: {e.reason}"
    except TimeoutError:
      return f"Error: Request timed out after {self._timeout}s"
    except Exception as e:
      return f"Error: {e}"

  @property
  def tools(self) -> list:
    skill = self

    @tool
    def http_get(url: str, headers: Optional[str] = None) -> str:
      """Make an HTTP GET request to fetch data from a URL.

      Args:
        url: The full URL to request (e.g. "https://api.github.com/repos/python/cpython").
        headers: Optional JSON string of HTTP headers (e.g. '{"Authorization": "Bearer token"}').

      Returns:
        The HTTP status code and response body.
      """
      return skill._make_request("GET", url, headers=headers)

    @tool
    def http_post(url: str, body: str, headers: Optional[str] = None) -> str:
      """Make an HTTP POST request to create a resource or send data.

      Args:
        url: The full URL to send the request to.
        body: The request body as a JSON string.
        headers: Optional JSON string of HTTP headers.

      Returns:
        The HTTP status code and response body.
      """
      return skill._make_request("POST", url, headers=headers, body=body)

    @tool
    def http_put(url: str, body: str, headers: Optional[str] = None) -> str:
      """Make an HTTP PUT request to replace a resource.

      Args:
        url: The full URL to send the request to.
        body: The request body as a JSON string.
        headers: Optional JSON string of HTTP headers.

      Returns:
        The HTTP status code and response body.
      """
      return skill._make_request("PUT", url, headers=headers, body=body)

    @tool
    def http_patch(url: str, body: str, headers: Optional[str] = None) -> str:
      """Make an HTTP PATCH request to partially update a resource.

      Args:
        url: The full URL to send the request to.
        body: The request body as a JSON string with the fields to update.
        headers: Optional JSON string of HTTP headers.

      Returns:
        The HTTP status code and response body.
      """
      return skill._make_request("PATCH", url, headers=headers, body=body)

    @tool
    def http_delete(url: str, headers: Optional[str] = None) -> str:
      """Make an HTTP DELETE request to remove a resource.

      Args:
        url: The full URL to send the request to.
        headers: Optional JSON string of HTTP headers.

      Returns:
        The HTTP status code and response body.
      """
      return skill._make_request("DELETE", url, headers=headers)

    return [http_get, http_post, http_put, http_patch, http_delete]
