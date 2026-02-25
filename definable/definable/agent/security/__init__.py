"""Security module — hardened tool execution, rate limiting, content defense, and auditing.

Provides production-grade security features for Definable agents:

- **ToolPolicy**: Declarative tool execution control (deny/allowlist/full).
- **RateLimiter**: Sliding-window throttling for interface messages.
- **ContentDefense**: Prompt injection detection, XML content wrapping.
- **SSRFGuard**: SSRF protection for outbound HTTP requests.
- **EnvSanitizer**: Dangerous environment variable stripping.
- **SecurityAudit**: Automated agent configuration auditing.

Quick Start::

    from definable.agent import Agent
    from definable.agent.security import SecurityConfig, ToolPolicy

    agent = Agent(
        model="gpt-4o-mini",
        security=SecurityConfig(
            tool_policy=ToolPolicy(mode="allowlist", allowed_tools={"search"}),
        ),
    )

    # Run security audit
    report = await agent.security_audit()
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from definable.agent.security.content_defense import (
  ContentDefenseConfig,
  ContentDefenseGuardrail,
  InjectionScanResult,
  PromptInjectionDetector,
  xml_wrap_content,
)
from definable.agent.security.env_sanitizer import (
  DANGEROUS_ENV_VARS,
  EnvSanitizeConfig,
  is_env_safe,
  sanitize_env,
)
from definable.agent.security.rate_limiter import (
  RateLimitConfig,
  RateLimitHook,
  SlidingWindowRateLimiter,
)
from definable.agent.security.ssrf import (
  SSRFBlockedError,
  SSRFGuard,
  SSRFGuardConfig,
  is_private_ip,
  resolve_and_check,
)
from definable.agent.security.tool_policy import (
  DEFAULT_DANGEROUS_TOOLS,
  ToolPolicy,
  ToolPolicyGuardrail,
)
from definable.agent.security.audit import (
  SecurityFinding,
  SecurityReport,
  SecuritySeverity,
  security_audit,
)


# ------------------------------------------------------------------
# Unified SecurityConfig
# ------------------------------------------------------------------


@dataclass
class SecurityConfig:
  """Unified security configuration for an agent.

  Attach to ``Agent(security=SecurityConfig(...))`` to enable security
  features. Each field is independently optional — configure only what
  you need.

  Attributes:
    tool_policy: Declarative tool execution policy.
    rate_limit: Rate limiting for interface messages.
    content_defense: External content defense settings.
    ssrf_guard: SSRF protection for outbound HTTP.
    env_sanitize: Environment sanitization for subprocess tools.
  """

  tool_policy: Optional[ToolPolicy] = None
  rate_limit: Optional[RateLimitConfig] = None
  content_defense: Optional[ContentDefenseConfig] = None
  ssrf_guard: Optional[SSRFGuardConfig] = None
  env_sanitize: Optional[EnvSanitizeConfig] = None


__all__ = [
  # Unified config
  "SecurityConfig",
  # Tool policy
  "ToolPolicy",
  "ToolPolicyGuardrail",
  "DEFAULT_DANGEROUS_TOOLS",
  # Rate limiting
  "RateLimitConfig",
  "RateLimitHook",
  "SlidingWindowRateLimiter",
  # Content defense
  "ContentDefenseConfig",
  "ContentDefenseGuardrail",
  "PromptInjectionDetector",
  "InjectionScanResult",
  "xml_wrap_content",
  # SSRF
  "SSRFGuard",
  "SSRFGuardConfig",
  "SSRFBlockedError",
  "is_private_ip",
  "resolve_and_check",
  # Environment
  "EnvSanitizeConfig",
  "DANGEROUS_ENV_VARS",
  "sanitize_env",
  "is_env_safe",
  # Audit
  "SecurityReport",
  "SecurityFinding",
  "SecuritySeverity",
  "security_audit",
]
