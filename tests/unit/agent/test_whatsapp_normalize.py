"""Tests for WhatsApp phone number and JID normalization."""

from definable.agent.interface.whatsapp.normalize import (
  is_group_jid,
  is_user_target,
  normalize_e164,
  normalize_whatsapp_target,
  phone_to_jid,
  redact_phone,
)


# --------------------------------------------------------------------------- #
# normalize_e164                                                               #
# --------------------------------------------------------------------------- #


class TestNormalizeE164:
  def test_plain_digits(self):
    assert normalize_e164("15551234567") == "15551234567"

  def test_with_plus(self):
    assert normalize_e164("+15551234567") == "15551234567"

  def test_with_dashes(self):
    assert normalize_e164("+1-555-123-4567") == "15551234567"

  def test_with_spaces(self):
    assert normalize_e164("+1 555 123 4567") == "15551234567"

  def test_with_parens(self):
    assert normalize_e164("+1 (555) 123-4567") == "15551234567"

  def test_with_dots(self):
    assert normalize_e164("+1.555.123.4567") == "15551234567"

  def test_whatsapp_prefix(self):
    assert normalize_e164("whatsapp:+15551234567") == "15551234567"

  def test_double_whatsapp_prefix(self):
    assert normalize_e164("whatsapp:whatsapp:+15551234567") == "15551234567"

  def test_too_short(self):
    assert normalize_e164("123456") is None

  def test_too_long(self):
    assert normalize_e164("1234567890123456") is None

  def test_empty(self):
    assert normalize_e164("") is None

  def test_non_numeric(self):
    assert normalize_e164("not-a-number") is None

  def test_letters_mixed(self):
    assert normalize_e164("+1555abc4567") is None

  def test_minimum_length(self):
    assert normalize_e164("1234567") == "1234567"

  def test_maximum_length(self):
    assert normalize_e164("123456789012345") == "123456789012345"

  def test_whitespace_only(self):
    assert normalize_e164("   ") is None

  def test_swiss_number(self):
    assert normalize_e164("+41 79 666 68 64") == "41796666864"


# --------------------------------------------------------------------------- #
# is_group_jid                                                                 #
# --------------------------------------------------------------------------- #


class TestIsGroupJid:
  def test_simple_group(self):
    assert is_group_jid("120363012345@g.us") is True

  def test_group_with_dash(self):
    assert is_group_jid("120363012345-1234567890@g.us") is True

  def test_user_jid(self):
    assert is_group_jid("15551234567@s.whatsapp.net") is False

  def test_plain_number(self):
    assert is_group_jid("+15551234567") is False

  def test_whatsapp_prefix(self):
    assert is_group_jid("whatsapp:120363012345@g.us") is True

  def test_case_insensitive(self):
    assert is_group_jid("120363012345@G.US") is True

  def test_empty(self):
    assert is_group_jid("") is False

  def test_at_sign_in_local(self):
    assert is_group_jid("foo@bar@g.us") is False


# --------------------------------------------------------------------------- #
# is_user_target                                                               #
# --------------------------------------------------------------------------- #


class TestIsUserTarget:
  def test_simple_jid(self):
    assert is_user_target("15551234567@s.whatsapp.net") is True

  def test_jid_with_device(self):
    assert is_user_target("41796666864:0@s.whatsapp.net") is True

  def test_lid(self):
    assert is_user_target("123456789@lid") is True

  def test_group_jid(self):
    assert is_user_target("120363012345@g.us") is False

  def test_plain_number(self):
    assert is_user_target("+15551234567") is False

  def test_whatsapp_prefix(self):
    assert is_user_target("whatsapp:15551234567@s.whatsapp.net") is True


# --------------------------------------------------------------------------- #
# normalize_whatsapp_target                                                    #
# --------------------------------------------------------------------------- #


class TestNormalizeWhatsAppTarget:
  def test_plain_e164(self):
    assert normalize_whatsapp_target("+15551234567") == "15551234567"

  def test_user_jid(self):
    assert normalize_whatsapp_target("41796666864:0@s.whatsapp.net") == "41796666864"

  def test_group_jid(self):
    result = normalize_whatsapp_target("120363012345@g.us")
    assert result == "120363012345@g.us"

  def test_group_jid_with_dash(self):
    result = normalize_whatsapp_target("120363012345-9876@g.us")
    assert result == "120363012345-9876@g.us"

  def test_whatsapp_prefix(self):
    assert normalize_whatsapp_target("whatsapp:+15551234567") == "15551234567"

  def test_lid(self):
    assert normalize_whatsapp_target("123456789@lid") == "123456789"

  def test_unknown_jid_format(self):
    assert normalize_whatsapp_target("foo@bar.com") is None

  def test_empty(self):
    assert normalize_whatsapp_target("") is None

  def test_invalid_number(self):
    assert normalize_whatsapp_target("abc") is None

  def test_none_equivalent(self):
    assert normalize_whatsapp_target("   ") is None


# --------------------------------------------------------------------------- #
# phone_to_jid                                                                 #
# --------------------------------------------------------------------------- #


class TestPhoneToJid:
  def test_basic(self):
    assert phone_to_jid("15551234567") == "15551234567@s.whatsapp.net"

  def test_with_plus(self):
    assert phone_to_jid("+15551234567") == "15551234567@s.whatsapp.net"


# --------------------------------------------------------------------------- #
# redact_phone                                                                 #
# --------------------------------------------------------------------------- #


class TestRedactPhone:
  def test_full_number(self):
    assert redact_phone("+15551234567") == "+155******67"

  def test_no_plus(self):
    assert redact_phone("15551234567") == "155******67"

  def test_short_number(self):
    # Too short to redact — return as-is
    assert redact_phone("12345") == "12345"

  def test_empty(self):
    assert redact_phone("") == ""

  def test_six_digits(self):
    assert redact_phone("123456") == "123*56"
