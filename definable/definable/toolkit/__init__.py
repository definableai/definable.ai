"""Toolkit base class — a root-level composable block.

Usage:
    from definable.toolkit import Toolkit

    class MyToolkit(Toolkit):
        @property
        def tools(self):
            return [tool1, tool2]
"""

from definable.agent.toolkit import Toolkit

__all__ = ["Toolkit"]
