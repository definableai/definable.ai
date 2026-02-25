"""Tests for environment sanitization."""

from definable.agent.security.env_sanitizer import (
  DANGEROUS_ENV_VARS,
  EnvSanitizeConfig,
  is_env_safe,
  sanitize_env,
)


class TestSanitizeEnv:
  def test_strips_ld_preload(self):
    env = {"PATH": "/usr/bin", "LD_PRELOAD": "/tmp/evil.so", "HOME": "/home/user"}
    result = sanitize_env(env)
    assert "LD_PRELOAD" not in result
    assert "PATH" in result
    assert "HOME" in result

  def test_strips_dyld_vars(self):
    env = {"DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib", "SHELL": "/bin/zsh"}
    result = sanitize_env(env)
    assert "DYLD_INSERT_LIBRARIES" not in result
    assert "SHELL" in result

  def test_strips_python_startup(self):
    env = {"PYTHONSTARTUP": "/tmp/evil.py", "PYTHONPATH": "/tmp", "USER": "test"}
    result = sanitize_env(env)
    assert "PYTHONSTARTUP" not in result
    assert "PYTHONPATH" not in result
    assert "USER" in result

  def test_strips_shell_vars(self):
    env = {"BASH_ENV": "/tmp/evil.sh", "IFS": " "}
    result = sanitize_env(env)
    assert "BASH_ENV" not in result
    assert "IFS" not in result

  def test_strips_node_options(self):
    env = {"NODE_OPTIONS": "--require /tmp/evil.js"}
    result = sanitize_env(env)
    assert "NODE_OPTIONS" not in result

  def test_preserves_safe_vars(self):
    env = {"PATH": "/usr/bin", "HOME": "/home/user", "LANG": "en_US.UTF-8"}
    result = sanitize_env(env)
    assert result == env

  def test_custom_blocked_vars(self):
    config = EnvSanitizeConfig(blocked_vars={"CUSTOM_SECRET"})
    env = {"CUSTOM_SECRET": "password", "HOME": "/home"}
    result = sanitize_env(env, config)
    assert "CUSTOM_SECRET" not in result
    assert "HOME" in result

  def test_path_override_locked(self):
    config = EnvSanitizeConfig(allow_path_override=False)
    env = {"PATH": "/tmp/evil:/usr/bin"}
    result = sanitize_env(env, config)
    assert result["PATH"] == config.safe_path

  def test_defaults_to_os_environ(self):
    result = sanitize_env()
    # Should work without error
    assert isinstance(result, dict)

  def test_returns_copy(self):
    env = {"HOME": "/home"}
    result = sanitize_env(env)
    result["NEW_VAR"] = "value"
    assert "NEW_VAR" not in env


class TestIsEnvSafe:
  def test_clean_env(self):
    env = {"PATH": "/usr/bin", "HOME": "/home"}
    assert is_env_safe(env) == []

  def test_dangerous_env(self):
    env = {"LD_PRELOAD": "/tmp/evil.so", "DYLD_INSERT_LIBRARIES": "/tmp/evil"}
    dangers = is_env_safe(env)
    assert "LD_PRELOAD" in dangers
    assert "DYLD_INSERT_LIBRARIES" in dangers

  def test_dangerous_vars_set_not_empty(self):
    assert len(DANGEROUS_ENV_VARS) > 15
