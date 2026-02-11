"""Multi-interface supervisor — run multiple interfaces concurrently with auto-restart.

This module delegates to :func:`definable.utils.supervisor.supervise_interfaces`
and exists for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from definable.interfaces.base import BaseInterface

if TYPE_CHECKING:
  from definable.interfaces.identity import IdentityResolver


async def serve(
  *interfaces: BaseInterface,
  name: Optional[str] = None,
  identity_resolver: Optional["IdentityResolver"] = None,
) -> None:
  """Run multiple interfaces concurrently with automatic restart on failure.

  Delegates to :func:`definable.utils.supervisor.supervise_interfaces`.
  See that function for full documentation.

  Args:
    *interfaces: One or more BaseInterface instances to run.
    name: Optional prefix for log messages (defaults to "serve").
    identity_resolver: Optional shared resolver for cross-platform user identity.

  Raises:
    ValueError: If no interfaces are provided.
  """
  from definable.utils.supervisor import supervise_interfaces

  await supervise_interfaces(*interfaces, name=name, identity_resolver=identity_resolver)
