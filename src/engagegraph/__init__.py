"""EngageGraph: reproducible app-engagement analytics and next-app recommendation.

The public API is intentionally small.  Most users should start with the CLI
(``engagegraph --help``) or the HTTP service (``engagegraph serve``).  The
Python objects below are re-exported for notebooks and downstream services.
"""

from __future__ import annotations

__version__ = "2.2.1"
__all__ = ["__version__"]
