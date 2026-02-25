"""Tests for SSRF protection."""

import pytest

from definable.agent.security.ssrf import (
  SSRFBlockedError,
  SSRFGuard,
  SSRFGuardConfig,
  is_private_ip,
  resolve_and_check,
)


class TestIsPrivateIp:
  def test_loopback_v4(self):
    assert is_private_ip("127.0.0.1") is True
    assert is_private_ip("127.0.0.2") is True

  def test_loopback_v6(self):
    assert is_private_ip("::1") is True

  def test_rfc1918_10(self):
    assert is_private_ip("10.0.0.1") is True
    assert is_private_ip("10.255.255.255") is True

  def test_rfc1918_172(self):
    assert is_private_ip("172.16.0.1") is True
    assert is_private_ip("172.31.255.255") is True
    assert is_private_ip("172.32.0.1") is False

  def test_rfc1918_192(self):
    assert is_private_ip("192.168.0.1") is True
    assert is_private_ip("192.168.255.255") is True

  def test_link_local(self):
    assert is_private_ip("169.254.0.1") is True
    assert is_private_ip("169.254.169.254") is True  # Cloud metadata

  def test_public_ip(self):
    assert is_private_ip("8.8.8.8") is False
    assert is_private_ip("1.1.1.1") is False
    assert is_private_ip("93.184.216.34") is False

  def test_invalid_ip(self):
    assert is_private_ip("not_an_ip") is False


class TestResolveAndCheck:
  def test_public_url_passes(self):
    # google.com should resolve to public IP
    result = resolve_and_check("https://google.com")
    assert result == "https://google.com"

  def test_private_url_blocked(self):
    with pytest.raises(SSRFBlockedError):
      resolve_and_check("http://127.0.0.1:8080/admin")

  def test_allowed_private_host(self):
    result = resolve_and_check(
      "http://localhost:7777/health",
      allowed_private={"localhost"},
    )
    assert result == "http://localhost:7777/health"

  def test_no_hostname_raises(self):
    with pytest.raises(ValueError, match="Cannot extract hostname"):
      resolve_and_check("not-a-url")

  def test_metadata_endpoint_blocked(self):
    with pytest.raises(SSRFBlockedError):
      resolve_and_check("http://169.254.169.254/latest/meta-data/")


class TestSSRFGuard:
  def test_check_url_public(self):
    guard = SSRFGuard()
    result = guard.check_url("https://example.com")
    assert result == "https://example.com"

  def test_check_url_private_blocked(self):
    guard = SSRFGuard()
    with pytest.raises(SSRFBlockedError):
      guard.check_url("http://127.0.0.1/secret")

  def test_disabled_guard_passes_all(self):
    guard = SSRFGuard(SSRFGuardConfig(enabled=False))
    result = guard.check_url("http://10.0.0.1/internal")
    assert result == "http://10.0.0.1/internal"

  def test_allowed_hosts_config(self):
    guard = SSRFGuard(SSRFGuardConfig(allowed_private_hosts={"localhost"}))
    result = guard.check_url("http://localhost:7777/health")
    assert "localhost" in result
