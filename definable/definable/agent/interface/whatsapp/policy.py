"""WhatsApp sender policy — access control for inbound messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set

from definable.agent.interface.whatsapp.normalize import normalize_e164
from definable.utils.log import log_debug


@dataclass
class WhatsAppPolicy:
  """Sender access policy for WhatsApp messages.

  Controls which inbound messages are allowed to reach the agent.

  Args:
    dm_policy: Direct message policy.

      - ``"allowlist"`` — only senders in *allow_from* are accepted.
      - ``"open"`` — all DMs accepted.
      - ``"disabled"`` — all DMs blocked.

    allow_from: List of allowed sender phone numbers (E.164) or
      ``"*"`` for wildcard. Used by both DM and group allowlist modes.
    group_policy: Group message policy.

      - ``"open"`` — all group messages accepted.
      - ``"allowlist"`` — only senders in *group_allow_from* (falling
        back to *allow_from*) are accepted.
      - ``"disabled"`` — all group messages blocked.

    group_allow_from: Separate allowlist for group senders. Falls back
      to *allow_from* if ``None``.
    self_phone: Own phone number (E.164). Used for self-chat detection
      and as a fallback when *allow_from* is empty (auto-allow self).

  Example::

    policy = WhatsAppPolicy(
      dm_policy="allowlist",
      allow_from=["+15551234567"],
      group_policy="open",
    )
  """

  dm_policy: Literal["allowlist", "open", "disabled"] = "allowlist"
  allow_from: List[str] = field(default_factory=list)
  group_policy: Literal["open", "allowlist", "disabled"] = "open"
  group_allow_from: Optional[List[str]] = None
  self_phone: Optional[str] = None

  def __post_init__(self) -> None:
    self._allow_set: Optional[Set[str]] = None
    self._group_allow_set: Optional[Set[str]] = None

  def _get_allow_set(self) -> Set[str]:
    if self._allow_set is None:
      self._allow_set = set()
      for entry in self.allow_from:
        if entry == "*":
          self._allow_set.add("*")
        else:
          normalized = normalize_e164(entry)
          if normalized:
            self._allow_set.add(normalized)
    return self._allow_set

  def _get_group_allow_set(self) -> Set[str]:
    if self._group_allow_set is None:
      source = self.group_allow_from if self.group_allow_from is not None else self.allow_from
      self._group_allow_set = set()
      for entry in source:
        if entry == "*":
          self._group_allow_set.add("*")
        else:
          normalized = normalize_e164(entry)
          if normalized:
            self._group_allow_set.add(normalized)
    return self._group_allow_set

  # --- Public API ---

  def check_access(self, *, from_phone: str, chat_jid: str, from_jid: str, is_group: bool, is_from_me: bool) -> bool:
    """Decide whether an inbound message should be processed.

    Args:
      from_phone: Sender E.164 phone number.
      chat_jid: Chat JID (group or 1:1).
      from_jid: Sender JID.
      is_group: Whether the message is from a group.
      is_from_me: Whether the message was sent by self.

    Returns:
      ``True`` if the message is allowed, ``False`` to drop it.
    """
    if is_from_me:
      return self._check_self_message(from_phone, chat_jid, from_jid)

    if is_group:
      return self._check_group(from_phone)

    return self._check_dm(from_phone)

  # --- Internal ---

  def _check_dm(self, from_phone: str) -> bool:
    if self.dm_policy == "disabled":
      log_debug("[whatsapp:policy] Blocked DM (policy=disabled)")
      return False

    if self.dm_policy == "open":
      return True

    # allowlist
    return self._is_sender_in(from_phone, self._get_allow_set())

  def _check_group(self, from_phone: str) -> bool:
    if self.group_policy == "disabled":
      log_debug("[whatsapp:policy] Blocked group message (policy=disabled)")
      return False

    if self.group_policy == "open":
      return True

    # allowlist
    return self._is_sender_in(from_phone, self._get_group_allow_set())

  def _check_self_message(self, from_phone: str, chat_jid: str, from_jid: str) -> bool:
    """Allow self-chat (messaging yourself), block self-echoes in other chats."""
    # Self-chat: chat_jid matches from_jid (you're talking to yourself)
    if chat_jid == from_jid:
      return True
    # In all other cases, this is an echo of our own outbound message — skip it
    log_debug("[whatsapp:policy] Skipping self-echo in non-self chat")
    return False

  def _is_sender_in(self, phone: str, allow_set: Set[str]) -> bool:
    if "*" in allow_set:
      return True
    # If no allowlist and self_phone is set, auto-allow self
    non_wildcard = allow_set - {"*"}
    if not non_wildcard and self.self_phone:
      self_normalized = normalize_e164(self.self_phone)
      sender_normalized = normalize_e164(phone)
      if self_normalized and sender_normalized and self_normalized == sender_normalized:
        return True
      log_debug("[whatsapp:policy] Blocked sender (empty allowlist, not self)")
      return False
    sender_normalized = normalize_e164(phone)
    if sender_normalized and sender_normalized in non_wildcard:
      return True
    log_debug("[whatsapp:policy] Blocked sender (not in allowlist)")
    return False
