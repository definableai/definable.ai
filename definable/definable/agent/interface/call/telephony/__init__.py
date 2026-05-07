"""Telephony providers for the call interface."""

from definable.agent.interface.call.telephony.base import TelephonyEvent, TelephonyProvider

__all__ = [
  "TelephonyProvider",
  "TelephonyEvent",
]


def __getattr__(name: str):
  if name == "TwilioProvider":
    from definable.agent.interface.call.telephony.twilio import TwilioProvider

    return TwilioProvider
  if name == "PlivoProvider":
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    return PlivoProvider
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
