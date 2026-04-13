"""Tests for WhatsApp sender policy engine."""

from definable.agent.interface.whatsapp.policy import WhatsAppPolicy


def _check(policy: WhatsAppPolicy, *, from_phone: str, is_group: bool = False, is_from_me: bool = False) -> bool:
  """Helper — calls check_access with sensible defaults."""
  jid_suffix = "@g.us" if is_group else "@s.whatsapp.net"
  from_jid = f"{from_phone.lstrip('+')}{jid_suffix}"
  chat_jid = from_jid if not is_group else f"120363012345{jid_suffix}"
  return policy.check_access(
    from_phone=from_phone,
    chat_jid=chat_jid,
    from_jid=from_jid,
    is_group=is_group,
    is_from_me=is_from_me,
  )


# --------------------------------------------------------------------------- #
# DM policy                                                                    #
# --------------------------------------------------------------------------- #


class TestDmPolicy:
  def test_allowlist_allowed(self):
    policy = WhatsAppPolicy(dm_policy="allowlist", allow_from=["+15551234567"])
    assert _check(policy, from_phone="+15551234567") is True

  def test_allowlist_blocked(self):
    policy = WhatsAppPolicy(dm_policy="allowlist", allow_from=["+15551234567"])
    assert _check(policy, from_phone="+19999999999") is False

  def test_allowlist_wildcard(self):
    policy = WhatsAppPolicy(dm_policy="allowlist", allow_from=["*"])
    assert _check(policy, from_phone="+19999999999") is True

  def test_allowlist_empty_with_self(self):
    """Empty allowlist + self_phone → auto-allow self only."""
    policy = WhatsAppPolicy(dm_policy="allowlist", allow_from=[], self_phone="+15551234567")
    assert _check(policy, from_phone="+15551234567") is True
    assert _check(policy, from_phone="+19999999999") is False

  def test_allowlist_empty_no_self(self):
    """Empty allowlist + no self_phone → block everyone."""
    policy = WhatsAppPolicy(dm_policy="allowlist", allow_from=[])
    assert _check(policy, from_phone="+15551234567") is False

  def test_open(self):
    policy = WhatsAppPolicy(dm_policy="open")
    assert _check(policy, from_phone="+19999999999") is True

  def test_disabled(self):
    policy = WhatsAppPolicy(dm_policy="disabled")
    assert _check(policy, from_phone="+15551234567") is False

  def test_e164_normalization(self):
    """Numbers are normalized before comparison."""
    policy = WhatsAppPolicy(dm_policy="allowlist", allow_from=["+1-555-123-4567"])
    assert _check(policy, from_phone="15551234567") is True

  def test_multiple_allowed(self):
    policy = WhatsAppPolicy(dm_policy="allowlist", allow_from=["+15551234567", "+44770090012"])
    assert _check(policy, from_phone="+15551234567") is True
    assert _check(policy, from_phone="+44770090012") is True
    assert _check(policy, from_phone="+33123456789") is False


# --------------------------------------------------------------------------- #
# Group policy                                                                 #
# --------------------------------------------------------------------------- #


class TestGroupPolicy:
  def test_open(self):
    policy = WhatsAppPolicy(group_policy="open")
    assert _check(policy, from_phone="+19999999999", is_group=True) is True

  def test_disabled(self):
    policy = WhatsAppPolicy(group_policy="disabled")
    assert _check(policy, from_phone="+15551234567", is_group=True) is False

  def test_allowlist_with_group_allow_from(self):
    policy = WhatsAppPolicy(
      group_policy="allowlist",
      allow_from=["+15551234567"],
      group_allow_from=["+19999999999"],
    )
    # Group allowlist is separate from DM allowlist
    assert _check(policy, from_phone="+19999999999", is_group=True) is True
    assert _check(policy, from_phone="+15551234567", is_group=True) is False

  def test_allowlist_falls_back_to_allow_from(self):
    """When group_allow_from is None, falls back to allow_from."""
    policy = WhatsAppPolicy(
      group_policy="allowlist",
      allow_from=["+15551234567"],
      group_allow_from=None,
    )
    assert _check(policy, from_phone="+15551234567", is_group=True) is True
    assert _check(policy, from_phone="+19999999999", is_group=True) is False

  def test_allowlist_wildcard(self):
    policy = WhatsAppPolicy(group_policy="allowlist", group_allow_from=["*"])
    assert _check(policy, from_phone="+19999999999", is_group=True) is True


# --------------------------------------------------------------------------- #
# Self messages                                                                #
# --------------------------------------------------------------------------- #


class TestSelfMessages:
  def test_self_chat_allowed(self):
    """Messages from self in self-chat (from_jid == chat_jid) are allowed."""
    policy = WhatsAppPolicy(self_phone="+15551234567")
    jid = "15551234567@s.whatsapp.net"
    assert (
      policy.check_access(
        from_phone="+15551234567",
        chat_jid=jid,
        from_jid=jid,
        is_group=False,
        is_from_me=True,
      )
      is True
    )

  def test_self_echo_in_other_chat_blocked(self):
    """Self-sent messages in other chats (echoes) are blocked."""
    policy = WhatsAppPolicy(self_phone="+15551234567")
    assert (
      policy.check_access(
        from_phone="+15551234567",
        chat_jid="19999999999@s.whatsapp.net",
        from_jid="15551234567@s.whatsapp.net",
        is_group=False,
        is_from_me=True,
      )
      is False
    )

  def test_self_echo_in_group_blocked(self):
    """Self-sent messages in groups (echoes) are blocked."""
    policy = WhatsAppPolicy(self_phone="+15551234567")
    assert (
      policy.check_access(
        from_phone="+15551234567",
        chat_jid="120363012345@g.us",
        from_jid="15551234567@s.whatsapp.net",
        is_group=True,
        is_from_me=True,
      )
      is False
    )


# --------------------------------------------------------------------------- #
# Combined scenarios                                                           #
# --------------------------------------------------------------------------- #


class TestCombinedScenarios:
  def test_dm_allowlist_group_open(self):
    """DMs restricted, groups open."""
    policy = WhatsAppPolicy(
      dm_policy="allowlist",
      allow_from=["+15551234567"],
      group_policy="open",
    )
    # DM from allowed sender → ok
    assert _check(policy, from_phone="+15551234567", is_group=False) is True
    # DM from unknown → blocked
    assert _check(policy, from_phone="+19999999999", is_group=False) is False
    # Group from anyone → ok
    assert _check(policy, from_phone="+19999999999", is_group=True) is True

  def test_both_disabled(self):
    policy = WhatsAppPolicy(dm_policy="disabled", group_policy="disabled")
    assert _check(policy, from_phone="+15551234567", is_group=False) is False
    assert _check(policy, from_phone="+15551234567", is_group=True) is False

  def test_both_open(self):
    policy = WhatsAppPolicy(dm_policy="open", group_policy="open")
    assert _check(policy, from_phone="+19999999999", is_group=False) is True
    assert _check(policy, from_phone="+19999999999", is_group=True) is True
