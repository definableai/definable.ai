"""GitHub skill — repository, issue, PR, and branch management."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from definable.agent.skill.base import Skill
from definable.agent.toolkit.decorator import tool


class GitHub(Skill):
  """Interact with GitHub repositories, issues, PRs, and branches.

  Requires ``PyGithub``: ``pip install PyGithub``

  Args:
      access_token: GitHub personal access token. Falls back to GITHUB_ACCESS_TOKEN env var.
      base_url: Custom GitHub Enterprise URL. None for github.com.
      enable_repos: Enable repository tools. Default True.
      enable_issues: Enable issue tools. Default True.
      enable_prs: Enable pull request tools. Default True.
      enable_branches: Enable branch tools. Default True.
      enable_search: Enable search tools. Default True.
      enable_files: Enable file content tools. Default True.
      enable_write: Enable write operations (create, close, comment). Default True.

  Example::

      from definable.agent.skill.builtin import GitHub
      agent = Agent(model=model, skills=[GitHub(access_token="ghp_...")])
  """

  name = "github"
  instructions = (
    "You have access to GitHub tools for managing repositories, issues, pull requests, and branches. "
    "Use search_repos to find repositories. Use get_issue and list_issues for issue tracking. "
    "Use get_pull_request for PR details. Always specify repos as 'owner/repo' format."
  )

  def __init__(
    self,
    *,
    access_token: Optional[str] = None,
    base_url: Optional[str] = None,
    enable_repos: bool = True,
    enable_issues: bool = True,
    enable_prs: bool = True,
    enable_branches: bool = True,
    enable_search: bool = True,
    enable_files: bool = True,
    enable_write: bool = True,
  ):
    super().__init__()
    self._token = access_token or os.getenv("GITHUB_ACCESS_TOKEN")
    self._base_url = base_url
    self._enable_repos = enable_repos
    self._enable_issues = enable_issues
    self._enable_prs = enable_prs
    self._enable_branches = enable_branches
    self._enable_search = enable_search
    self._enable_files = enable_files
    self._enable_write = enable_write
    self._client: Any = None

  @property
  def client(self) -> Any:
    if self._client is not None:
      return self._client
    try:
      from github import Auth, Github
    except ImportError:
      raise ImportError("`PyGithub` not installed. Run: pip install PyGithub")
    if not self._token:
      raise ValueError("GitHub access token required. Set access_token or GITHUB_ACCESS_TOKEN env var.")
    kwargs: Dict[str, Any] = {"auth": Auth.Token(self._token)}
    if self._base_url:
      kwargs["base_url"] = self._base_url
    self._client = Github(**kwargs)
    return self._client

  def _repo(self, repo: str) -> Any:
    return self.client.get_repo(repo)

  @staticmethod
  def _error(e: Exception) -> str:
    return json.dumps({"error": str(e)})

  @property
  def tools(self) -> list:
    skill = self
    result: list = []

    if self._enable_search:

      @tool
      def search_repos(query: str, max_results: int = 10) -> str:
        """Search GitHub repositories by query. Returns repo names, descriptions, and stars."""
        try:
          repos = skill.client.search_repositories(query=query)
          items = []
          for r in repos[:max_results]:
            items.append({"full_name": r.full_name, "description": r.description or "", "stars": r.stargazers_count, "url": r.html_url})
          return json.dumps(items, indent=2)
        except Exception as e:
          return skill._error(e)

      result.append(search_repos)

    if self._enable_repos:

      @tool
      def get_repo(repo: str) -> str:
        """Get repository details. Repo format: 'owner/repo'."""
        try:
          r = skill._repo(repo)
          return json.dumps(
            {
              "full_name": r.full_name,
              "description": r.description or "",
              "stars": r.stargazers_count,
              "forks": r.forks_count,
              "language": r.language,
              "default_branch": r.default_branch,
              "open_issues": r.open_issues_count,
              "url": r.html_url,
            },
            indent=2,
          )
        except Exception as e:
          return skill._error(e)

      result.append(get_repo)

    if self._enable_issues:

      @tool
      def list_issues(repo: str, state: str = "open", max_results: int = 20) -> str:
        """List issues in a repository. State: 'open', 'closed', or 'all'."""
        try:
          issues = skill._repo(repo).get_issues(state=state)
          items = []
          for iss in issues[:max_results]:
            if iss.pull_request is None:
              items.append({
                "number": iss.number,
                "title": iss.title,
                "state": iss.state,
                "author": iss.user.login if iss.user else None,
                "labels": [l.name for l in iss.labels],
                "created_at": str(iss.created_at),
              })
          return json.dumps(items, indent=2)
        except Exception as e:
          return skill._error(e)

      @tool
      def get_issue(repo: str, issue_number: int) -> str:
        """Get a specific issue by number."""
        try:
          iss = skill._repo(repo).get_issue(number=issue_number)
          return json.dumps(
            {
              "number": iss.number,
              "title": iss.title,
              "state": iss.state,
              "body": iss.body or "",
              "author": iss.user.login if iss.user else None,
              "labels": [l.name for l in iss.labels],
              "assignees": [a.login for a in iss.assignees],
              "created_at": str(iss.created_at),
              "comments": iss.comments,
            },
            indent=2,
          )
        except Exception as e:
          return skill._error(e)

      result.extend([list_issues, get_issue])

      if self._enable_write:

        @tool
        def create_issue(repo: str, title: str, body: str = "", labels: str = "") -> str:
          """Create a new issue. Labels as comma-separated string."""
          try:
            kwargs: Dict[str, Any] = {"title": title, "body": body}
            if labels:
              kwargs["labels"] = [l.strip() for l in labels.split(",")]
            iss = skill._repo(repo).create_issue(**kwargs)
            return json.dumps({"number": iss.number, "url": iss.html_url, "title": iss.title})
          except Exception as e:
            return skill._error(e)

        @tool
        def comment_on_issue(repo: str, issue_number: int, comment: str) -> str:
          """Add a comment to an issue or pull request."""
          try:
            iss = skill._repo(repo).get_issue(number=issue_number)
            c = iss.create_comment(body=comment)
            return json.dumps({"id": c.id, "url": c.html_url})
          except Exception as e:
            return skill._error(e)

        @tool
        def close_issue(repo: str, issue_number: int) -> str:
          """Close an issue."""
          try:
            iss = skill._repo(repo).get_issue(number=issue_number)
            iss.edit(state="closed")
            return json.dumps({"number": iss.number, "state": "closed"})
          except Exception as e:
            return skill._error(e)

        result.extend([create_issue, comment_on_issue, close_issue])

    if self._enable_prs:

      @tool
      def list_pull_requests(repo: str, state: str = "open", max_results: int = 20) -> str:
        """List pull requests. State: 'open', 'closed', or 'all'."""
        try:
          prs = skill._repo(repo).get_pulls(state=state)
          items = []
          for pr in prs[:max_results]:
            items.append({
              "number": pr.number,
              "title": pr.title,
              "state": pr.state,
              "author": pr.user.login if pr.user else None,
              "base": pr.base.ref,
              "head": pr.head.ref,
              "mergeable": pr.mergeable,
              "created_at": str(pr.created_at),
            })
          return json.dumps(items, indent=2)
        except Exception as e:
          return skill._error(e)

      @tool
      def get_pull_request(repo: str, pull_number: int) -> str:
        """Get pull request details including diff stats."""
        try:
          pr = skill._repo(repo).get_pull(number=pull_number)
          return json.dumps(
            {
              "number": pr.number,
              "title": pr.title,
              "state": pr.state,
              "body": pr.body or "",
              "author": pr.user.login if pr.user else None,
              "base": pr.base.ref,
              "head": pr.head.ref,
              "mergeable": pr.mergeable,
              "additions": pr.additions,
              "deletions": pr.deletions,
              "changed_files": pr.changed_files,
              "commits": pr.commits,
              "url": pr.html_url,
            },
            indent=2,
          )
        except Exception as e:
          return skill._error(e)

      @tool
      def get_pull_request_files(repo: str, pull_number: int) -> str:
        """Get files changed in a pull request with patches."""
        try:
          pr = skill._repo(repo).get_pull(number=pull_number)
          files = []
          for f in pr.get_files():
            files.append({
              "filename": f.filename,
              "status": f.status,
              "additions": f.additions,
              "deletions": f.deletions,
              "patch": (f.patch or "")[:2000],
            })
          return json.dumps(files, indent=2)
        except Exception as e:
          return skill._error(e)

      result.extend([list_pull_requests, get_pull_request, get_pull_request_files])

    if self._enable_branches:

      @tool
      def list_branches(repo: str) -> str:
        """List all branches in a repository."""
        try:
          branches = skill._repo(repo).get_branches()
          return json.dumps([{"name": b.name, "protected": b.protected} for b in branches], indent=2)
        except Exception as e:
          return skill._error(e)

      result.append(list_branches)

    if self._enable_files:

      @tool
      def get_file_content(repo: str, path: str, ref: str = "") -> str:
        """Get a file's content from a repository. Optionally specify a branch/ref."""
        try:
          kwargs: Dict[str, Any] = {}
          if ref:
            kwargs["ref"] = ref
          content = skill._repo(repo).get_contents(path, **kwargs)
          if isinstance(content, list):
            return json.dumps([{"path": c.path, "type": c.type, "size": c.size} for c in content], indent=2)
          return content.decoded_content.decode("utf-8", errors="replace")[:50000]
        except Exception as e:
          return skill._error(e)

      result.append(get_file_content)

    return result
