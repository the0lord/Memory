"""Stakeholder feedback and consensus for major self-edits.

Minor edits flow through the pipeline automatically. Major edits (large moves or
sign flips, per ``weights.is_major_change``) must clear the GATED stage, which
requires *consensus* among stakeholders rather than a single approver.
"""

from __future__ import annotations

from dataclasses import dataclass

from .router import MemoryRouter
from .weights import FEATURES, WeightVector
from .workload import Interaction


@dataclass
class Stakeholder:
    """A party that cares about routing quality, possibly with a feature bias.

    Each stakeholder approves an edit when it does not meaningfully hurt the
    outcome they observe, and does not gut the feature they most rely on.
    """

    name: str
    guarded_feature: str          # the feature this stakeholder won't let collapse
    tolerance: float = 0.01       # how much outcome regression they'll accept

    def vote(self, router: MemoryRouter, base: WeightVector, proposed: WeightVector,
             batch: list[Interaction]) -> bool:
        before = router.evaluate(base, batch)
        after = router.evaluate(proposed, batch)
        if after < before - self.tolerance:
            return False
        # veto edits that slash their guarded feature toward zero
        b, p = base.values[self.guarded_feature], proposed.values[self.guarded_feature]
        if b > 0.05 and p < b * 0.4:
            return False
        return True


class StakeholderPanel:
    def __init__(self, stakeholders: list[Stakeholder], consensus: float = 0.66):
        self.stakeholders = stakeholders
        self.consensus = consensus

    def deliberate(self, router: MemoryRouter, base: WeightVector,
                   proposed: WeightVector, batch: list[Interaction]) -> tuple[bool, str]:
        votes = {s.name: s.vote(router, base, proposed, batch) for s in self.stakeholders}
        yes = sum(votes.values())
        ratio = yes / len(votes) if votes else 0.0
        approved = ratio >= self.consensus
        detail = ", ".join(f"{n}:{'✓' if v else '✗'}" for n, v in votes.items())
        return approved, f"consensus {yes}/{len(votes)} ({ratio:.0%} ≥ {self.consensus:.0%}) — {detail}"
