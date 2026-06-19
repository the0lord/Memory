"""solmem — a shared, self-improving solution memory for teams on Claude Code.

Records problem->solution pairs, recalls them when similar problems reappear, and
evolves which memories to trust from real outcomes. Delivered as an MCP server
(``solmem.server``) on top of the selfmod self-modification engine.
"""

from .records import SolutionRecord
from .store import Store

__all__ = ["SolutionRecord", "Store"]
