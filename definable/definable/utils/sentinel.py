"""UNSET sentinel — distinguishes 'not provided' from None.

Use when a parameter accepts None as a meaningful value (e.g., "explicitly disabled")
and you need to distinguish it from "the caller didn't pass anything."

Usage:
    from definable.utils.sentinel import UNSET, _Unset

    def run(self, *, model: Model | str | _Unset = UNSET):
        if model is not UNSET:
            resolved = model  # Caller provided a value (could be None)
        else:
            resolved = self._default  # Use default

Type annotations: use ``_Unset`` in union types (e.g., ``int | _Unset``).
Runtime checks: always compare with ``is`` / ``is not``, never ``==``.
"""


class _Unset:
  """Singleton sentinel for 'not provided'.

  Falsy (bool(UNSET) is False) so it works naturally in ``if value:`` guards,
  but always compare with ``is`` / ``is not`` for clarity.
  """

  _instance: "_Unset | None" = None

  def __new__(cls) -> "_Unset":
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance

  def __repr__(self) -> str:
    return "UNSET"

  def __bool__(self) -> bool:
    return False

  def __reduce__(self) -> str:
    return "UNSET"


UNSET = _Unset()
