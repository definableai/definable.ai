"""Environment sanitization for subprocess execution.

Strips dangerous environment variables (LD_PRELOAD, DYLD_*, etc.) before
passing environments to subprocess tools.

Usage::

    from definable.agent.security.env_sanitizer import sanitize_env

    safe_env = sanitize_env()  # Clean copy of os.environ
    subprocess.run(["ls"], env=safe_env)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


# ------------------------------------------------------------------
# Dangerous environment variables
# ------------------------------------------------------------------

DANGEROUS_ENV_VARS: frozenset[str] = frozenset({
  # Linux dynamic linker — code injection
  "LD_PRELOAD",
  "LD_LIBRARY_PATH",
  "LD_AUDIT",
  "LD_DEBUG",
  "LD_PROFILE",
  # macOS dynamic linker — code injection
  "DYLD_INSERT_LIBRARIES",
  "DYLD_LIBRARY_PATH",
  "DYLD_FRAMEWORK_PATH",
  "DYLD_FALLBACK_LIBRARY_PATH",
  "DYLD_PRINT_LIBRARIES",
  # Python — arbitrary code at startup
  "PYTHONSTARTUP",
  "PYTHONPATH",
  "PYTHONHOME",
  # Shell — arbitrary code at shell init
  "BASH_ENV",
  "ENV",
  "CDPATH",
  "IFS",
  "PROMPT_COMMAND",
  # Perl / Ruby — code injection
  "PERL5OPT",
  "RUBYOPT",
  "RUBYLIB",
  # Node.js
  "NODE_OPTIONS",
})


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass
class EnvSanitizeConfig:
  """Configuration for environment sanitization.

  Attributes:
    blocked_vars: Additional env vars to strip (merged with defaults).
    allow_path_override: If False, PATH is locked to a safe default.
    safe_path: The PATH value to use when allow_path_override is False.
  """

  blocked_vars: Set[str] = field(default_factory=set)
  allow_path_override: bool = True
  safe_path: str = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


# ------------------------------------------------------------------
# Sanitizer
# ------------------------------------------------------------------


def sanitize_env(
  env: Optional[Dict[str, str]] = None,
  config: Optional[EnvSanitizeConfig] = None,
) -> Dict[str, str]:
  """Return a sanitized copy of the environment dict.

  Strips all :data:`DANGEROUS_ENV_VARS` plus any extras in
  ``config.blocked_vars``. Optionally locks PATH to a safe default.

  Args:
    env: Source environment dict. Defaults to ``os.environ``.
    config: Sanitization configuration.

  Returns:
    A new dict with dangerous variables removed.
  """
  cfg = config or EnvSanitizeConfig()
  source = dict(env) if env is not None else dict(os.environ)

  # Build full block list
  blocked = DANGEROUS_ENV_VARS | cfg.blocked_vars

  # Strip dangerous vars
  result = {k: v for k, v in source.items() if k not in blocked}

  # Lock PATH if configured
  if not cfg.allow_path_override:
    result["PATH"] = cfg.safe_path

  return result


def is_env_safe(env: Optional[Dict[str, str]] = None) -> list[str]:
  """Check an environment dict for dangerous variables.

  Args:
    env: Environment dict to check. Defaults to ``os.environ``.

  Returns:
    List of dangerous variable names found (empty if safe).
  """
  source = env if env is not None else dict(os.environ)
  return [k for k in source if k in DANGEROUS_ENV_VARS]
