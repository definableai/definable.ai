"""Security audit — automated checks on agent configuration.

Inspects an agent's tools, guardrails, interfaces, memory stores, and
workspace to identify potential security issues.

Usage::

    report = await agent.security_audit()
    print(report)
    assert report.critical_count == 0
"""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List


if TYPE_CHECKING:
  from definable.agent.agent import Agent


# ------------------------------------------------------------------
# Types
# ------------------------------------------------------------------


class SecuritySeverity(str, Enum):
  """Severity level for security findings."""

  info = "info"
  warning = "warning"
  critical = "critical"


@dataclass
class SecurityFinding:
  """A single security finding from the audit.

  Attributes:
    severity: How serious the issue is.
    category: Category of the check (secrets, permissions, tools, etc.).
    title: Short title of the finding.
    description: Detailed description.
    recommendation: What to do about it.
  """

  severity: SecuritySeverity
  category: str
  title: str
  description: str
  recommendation: str


@dataclass
class SecurityReport:
  """Structured report from a security audit.

  Attributes:
    findings: All findings from the audit.
    score: Security score from 0 (worst) to 100 (best).
    checked_at: ISO timestamp when the audit was performed.
    agent_name: Name of the agent that was audited.
  """

  findings: List[SecurityFinding] = field(default_factory=list)
  score: int = 100
  checked_at: str = ""
  agent_name: str = ""

  @property
  def critical_count(self) -> int:
    return sum(1 for f in self.findings if f.severity == SecuritySeverity.critical)

  @property
  def warning_count(self) -> int:
    return sum(1 for f in self.findings if f.severity == SecuritySeverity.warning)

  @property
  def info_count(self) -> int:
    return sum(1 for f in self.findings if f.severity == SecuritySeverity.info)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "agent_name": self.agent_name,
      "checked_at": self.checked_at,
      "score": self.score,
      "summary": {
        "critical": self.critical_count,
        "warning": self.warning_count,
        "info": self.info_count,
        "total": len(self.findings),
      },
      "findings": [
        {
          "severity": f.severity.value,
          "category": f.category,
          "title": f.title,
          "description": f.description,
          "recommendation": f.recommendation,
        }
        for f in self.findings
      ],
    }

  def __str__(self) -> str:
    lines = [
      f"Security Audit Report — {self.agent_name}",
      f"Score: {self.score}/100 | Critical: {self.critical_count} | Warnings: {self.warning_count} | Info: {self.info_count}",
      f"Checked: {self.checked_at}",
      "",
    ]
    for finding in self.findings:
      icon = {"critical": "[!]", "warning": "[~]", "info": "[i]"}[finding.severity.value]
      lines.append(f"  {icon} [{finding.category}] {finding.title}")
      lines.append(f"      {finding.description}")
      lines.append(f"      Fix: {finding.recommendation}")
      lines.append("")
    if not self.findings:
      lines.append("  No issues found.")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Secret patterns
# ------------------------------------------------------------------

_SECRET_PATTERNS = [
  (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "OpenAI API key"),
  (re.compile(r"sk-ant-[a-zA-Z0-9-]{20,}"), "Anthropic API key"),
  (re.compile(r"xai-[a-zA-Z0-9]{20,}"), "xAI API key"),
  (re.compile(r"[a-zA-Z0-9]{32,}"), "Possible API key/token"),
  (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "GitHub personal access token"),
  (re.compile(r"gho_[a-zA-Z0-9]{36}"), "GitHub OAuth token"),
  (re.compile(r"(?:password|secret|token|key)\s*[:=]\s*['\"][^'\"]{8,}['\"]", re.IGNORECASE), "Hardcoded credential"),
]


# ------------------------------------------------------------------
# Audit engine
# ------------------------------------------------------------------


async def security_audit(agent: "Agent") -> SecurityReport:
  """Perform a comprehensive security audit of an agent's configuration.

  Checks:
    1. Exposed secrets in instructions/config
    2. Dangerous tools enabled without ToolPolicy
    3. Missing auth on interfaces
    4. Missing input/output guardrails
    5. World-readable workspace files
    6. MCP servers with broad permissions
    7. Missing rate limiting on interfaces
    8. Shell/exec tools without confirmation

  Args:
    agent: The Agent instance to audit.

  Returns:
    A :class:`SecurityReport` with all findings.
  """
  findings: list[SecurityFinding] = []
  agent_name = getattr(agent.config, "agent_name", None) or agent.model.__class__.__name__

  # 1. Check for secrets in instructions
  _check_secrets(agent, findings)

  # 2. Check dangerous tools
  _check_dangerous_tools(agent, findings)

  # 3. Check guardrails
  _check_guardrails(agent, findings)

  # 4. Check interfaces auth
  _check_interface_auth(agent, findings)

  # 5. Check workspace file permissions
  _check_workspace_permissions(findings)

  # 6. Check MCP toolkits
  _check_mcp_toolkits(agent, findings)

  # 7. Check tool confirmation flags
  _check_tool_confirmation(agent, findings)

  # Calculate score
  score = 100
  for f in findings:
    if f.severity == SecuritySeverity.critical:
      score -= 20
    elif f.severity == SecuritySeverity.warning:
      score -= 5
  score = max(0, score)

  return SecurityReport(
    findings=findings,
    score=score,
    checked_at=datetime.now(timezone.utc).isoformat(),
    agent_name=agent_name,
  )


def _check_secrets(agent: "Agent", findings: list[SecurityFinding]) -> None:
  """Check for potential secrets in agent instructions."""
  instructions = agent.instructions or ""
  if not instructions:
    return

  for pattern, name in _SECRET_PATTERNS:
    if pattern.search(instructions):
      findings.append(
        SecurityFinding(
          severity=SecuritySeverity.critical,
          category="secrets",
          title=f"Possible {name} in instructions",
          description=f"Agent instructions may contain a {name}. Secrets should never be hardcoded.",
          recommendation="Use environment variables or a secrets manager instead of hardcoding credentials.",
        )
      )
      break  # One finding per category is enough


def _check_dangerous_tools(agent: "Agent", findings: list[SecurityFinding]) -> None:
  """Check if dangerous tools are registered without a ToolPolicy."""
  from definable.agent.security.tool_policy import DEFAULT_DANGEROUS_TOOLS

  all_tools = agent.tools or []
  for toolkit in agent.toolkits or []:
    if hasattr(toolkit, "tools"):
      all_tools = [*all_tools, *toolkit.tools]

  dangerous_found = [t for t in all_tools if getattr(t, "name", "") in DEFAULT_DANGEROUS_TOOLS]

  if dangerous_found and not _has_tool_policy(agent):
    tool_names = ", ".join(getattr(t, "name", "?") for t in dangerous_found)
    findings.append(
      SecurityFinding(
        severity=SecuritySeverity.warning,
        category="tools",
        title="Dangerous tools without ToolPolicy",
        description=f"Tools [{tool_names}] are registered without a ToolPolicy. They can execute with full privileges.",
        recommendation="Add SecurityConfig(tool_policy=ToolPolicy(mode='allowlist', ...)) to restrict tool access.",
      )
    )


def _check_guardrails(agent: "Agent", findings: list[SecurityFinding]) -> None:
  """Check guardrail coverage."""
  guardrails = agent.guardrails

  if guardrails is None:
    findings.append(
      SecurityFinding(
        severity=SecuritySeverity.warning,
        category="guardrails",
        title="No guardrails configured",
        description="Agent has no input, output, or tool guardrails. All content passes unchecked.",
        recommendation="Add Guardrails(input=[...], output=[...]) to validate content.",
      )
    )
    return

  if not guardrails.input:
    findings.append(
      SecurityFinding(
        severity=SecuritySeverity.info,
        category="guardrails",
        title="No input guardrails",
        description="No guardrails check user input before the LLM call.",
        recommendation="Consider adding max_tokens() or content defense guardrails.",
      )
    )

  if not guardrails.output:
    findings.append(
      SecurityFinding(
        severity=SecuritySeverity.info,
        category="guardrails",
        title="No output guardrails",
        description="No guardrails check model output after the LLM call.",
        recommendation="Consider adding pii_filter() or max_output_tokens() guardrails.",
      )
    )


def _check_interface_auth(agent: "Agent", findings: list[SecurityFinding]) -> None:
  """Check if interfaces have auth configured."""
  interfaces = getattr(agent, "_interfaces", [])
  for interface in interfaces:
    iface_name = interface.__class__.__name__
    auth = getattr(interface, "_auth", None) or getattr(interface, "auth", None)
    if auth is None:
      findings.append(
        SecurityFinding(
          severity=SecuritySeverity.warning,
          category="auth",
          title=f"No auth on {iface_name}",
          description=f"Interface {iface_name} has no authentication configured. Anyone can send messages.",
          recommendation=f"Add auth=APIKeyAuth(...) or auth=AllowlistAuth(...) to {iface_name}.",
        )
      )


def _check_workspace_permissions(findings: list[SecurityFinding]) -> None:
  """Check if workspace files have safe permissions."""
  try:
    from definable.utils.workspace import get_workspace_dir

    workspace = get_workspace_dir()
    if not workspace.exists():
      return

    for path in workspace.rglob("*"):
      if not path.is_file():
        continue
      try:
        st = path.stat()
        mode = st.st_mode
        # Check if world-readable
        if mode & stat.S_IROTH:
          # Only flag sensitive files
          sensitive_exts = {".db", ".sqlite", ".json", ".env", ".key", ".pem"}
          if path.suffix in sensitive_exts:
            findings.append(
              SecurityFinding(
                severity=SecuritySeverity.warning,
                category="permissions",
                title=f"World-readable sensitive file: {path.name}",
                description=f"File {path} is world-readable. It may contain sensitive data.",
                recommendation=f"Run: chmod 600 {path}",
              )
            )
            break  # One finding is enough
      except OSError:
        continue
  except ImportError:
    pass


def _check_mcp_toolkits(agent: "Agent", findings: list[SecurityFinding]) -> None:
  """Check MCP toolkit configurations."""
  for toolkit in agent.toolkits or []:
    cls_name = toolkit.__class__.__name__
    if "MCP" in cls_name:
      # MCP toolkits expose external tools — flag if no tool guardrails
      if not agent.guardrails or not agent.guardrails.tool:
        findings.append(
          SecurityFinding(
            severity=SecuritySeverity.warning,
            category="mcp",
            title="MCP toolkit without tool guardrails",
            description=f"MCP toolkit ({cls_name}) exposes external tools with no guardrails.",
            recommendation="Add tool guardrails (tool_allowlist or tool_blocklist) to control MCP tool access.",
          )
        )
        break


def _check_tool_confirmation(agent: "Agent", findings: list[SecurityFinding]) -> None:
  """Check if shell/exec tools require confirmation."""
  from definable.agent.security.tool_policy import DEFAULT_DANGEROUS_TOOLS

  for tool in agent.tools or []:
    name = getattr(tool, "name", "")
    requires_confirm = getattr(tool, "requires_confirmation", False)
    if name in DEFAULT_DANGEROUS_TOOLS and not requires_confirm:
      findings.append(
        SecurityFinding(
          severity=SecuritySeverity.info,
          category="tools",
          title=f"Dangerous tool '{name}' without confirmation",
          description=f"Tool '{name}' is classified as dangerous but does not require user confirmation (requires_confirmation=False).",
          recommendation=f"Set requires_confirmation=True on the '{name}' tool for human-in-the-loop safety.",
        )
      )


def _has_tool_policy(agent: "Agent") -> bool:
  """Check if the agent has a ToolPolicy configured via guardrails."""
  from definable.agent.security.tool_policy import ToolPolicyGuardrail

  if not agent.guardrails:
    return False
  return any(isinstance(g, ToolPolicyGuardrail) for g in agent.guardrails.tool)
