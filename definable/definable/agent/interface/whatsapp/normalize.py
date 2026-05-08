"""WhatsApp phone number and JID normalization."""

from __future__ import annotations

import re
from typing import Optional

# WhatsApp user JID: "15551234567:0@s.whatsapp.net" or "15551234567@s.whatsapp.net"
_USER_JID_RE = re.compile(r"^(\d+)(?::\d+)?@s\.whatsapp\.net$", re.IGNORECASE)
# WhatsApp linked ID: "123456@lid"
_LID_RE = re.compile(r"^(\d+)@lid$", re.IGNORECASE)
# WhatsApp group JID: "120363012345@g.us" or "120363012345-1234567890@g.us"
_GROUP_JID_RE = re.compile(r"^\d+(-\d+)*@g\.us$", re.IGNORECASE)
# Loose E.164: optional +, 7-15 digits
_E164_RE = re.compile(r"^\+?\d{7,15}$")


def _strip_whatsapp_prefix(value: str) -> str:
  """Strip leading ``whatsapp:`` prefixes (case-insensitive, repeated)."""
  candidate = value.strip()
  while True:
    lower = candidate.lower()
    if lower.startswith("whatsapp:"):
      candidate = candidate[len("whatsapp:") :].strip()
    else:
      return candidate


def is_group_jid(value: str) -> bool:
  """Check if *value* looks like a WhatsApp group JID (``…@g.us``)."""
  candidate = _strip_whatsapp_prefix(value)
  return bool(_GROUP_JID_RE.match(candidate))


def is_user_target(value: str) -> bool:
  """Check if *value* looks like a WhatsApp user JID or LID."""
  candidate = _strip_whatsapp_prefix(value)
  return bool(_USER_JID_RE.match(candidate) or _LID_RE.match(candidate))


def normalize_e164(value: str) -> Optional[str]:
  """Normalize a phone number to bare E.164 (digits only, no ``+``).

  Strips whitespace, dashes, parens, dots, and the leading ``+``.
  Returns ``None`` if the result doesn't look like a phone number.

  Examples::

    normalize_e164("+1-555-123-4567")  # → "15551234567"
    normalize_e164("15551234567")      # → "15551234567"
    normalize_e164("whatsapp:+1555")   # → "1555" (too short → None)
    normalize_e164("not-a-number")     # → None
  """
  if not value:
    return None
  candidate = _strip_whatsapp_prefix(value)
  # Remove common formatting chars
  candidate = re.sub(r"[\s\-\(\)\.]", "", candidate)
  # Strip leading +
  if candidate.startswith("+"):
    candidate = candidate[1:]
  # Must be 7-15 digits
  if not candidate or not candidate.isdigit() or len(candidate) < 7 or len(candidate) > 15:
    return None
  return candidate


def normalize_whatsapp_target(value: str) -> Optional[str]:
  """Normalize any WhatsApp target (E.164, user JID, group JID) to a canonical form.

  - Group JIDs (``@g.us``) → returned as-is (lowercased domain).
  - User JIDs (``@s.whatsapp.net``) → extracted phone, normalized via E.164.
  - LIDs (``@lid``) → extracted number, normalized.
  - Plain phone numbers → normalized via E.164.
  - Unknown JID formats (``@`` present but unrecognized) → ``None``.

  Returns ``None`` if the input cannot be normalized.
  """
  if not value:
    return None

  candidate = _strip_whatsapp_prefix(value)
  if not candidate:
    return None

  # Group JID
  if is_group_jid(candidate):
    local_part = candidate[: candidate.lower().index("@g.us")]
    return f"{local_part}@g.us"

  # User JID
  user_match = _USER_JID_RE.match(candidate)
  if user_match:
    phone = user_match.group(1)
    normalized = normalize_e164(phone)
    return normalized

  # LID
  lid_match = _LID_RE.match(candidate)
  if lid_match:
    return normalize_e164(lid_match.group(1))

  # Unknown JID format — fail fast
  if "@" in candidate:
    return None

  # Plain phone number
  return normalize_e164(candidate)


def phone_to_jid(phone: str) -> str:
  """Convert a bare E.164 phone number to a WhatsApp user JID.

  Args:
    phone: Bare digits (e.g. ``"15551234567"``).

  Returns:
    JID string (e.g. ``"15551234567@s.whatsapp.net"``).
  """
  digits = normalize_e164(phone) or phone
  return f"{digits}@s.whatsapp.net"


def redact_phone(phone: str) -> str:
  """Redact a phone number for safe logging.

  Keeps first 3 and last 2 digits, masks the rest.

  Examples::

    redact_phone("+15551234567")  # → "+155*****67"
    redact_phone("5551234567")    # → "555****67"
  """
  if not phone:
    return phone
  has_plus = phone.startswith("+")
  digits = phone.lstrip("+")
  if len(digits) <= 5:
    return phone  # too short to redact meaningfully
  prefix = digits[:3]
  suffix = digits[-2:]
  masked = "*" * (len(digits) - 3 - 2)
  result = f"{prefix}{masked}{suffix}"
  return f"+{result}" if has_plus else result
