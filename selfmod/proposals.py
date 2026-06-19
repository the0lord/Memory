"""Proposal engine: evolve decision weights from real-world outcomes.

The system learns by replaying recent *real* interactions and estimating, by
coordinate-wise finite differences, how each weight affects the outcome metric.
It then steps in the improving direction. Two safeguards shape the raw step:

  * the rejected-edit buffer damps directions that resemble past failures;
  * the edit budget hard-bounds the magnitude of the final proposal.

This is a gradient-free, outcome-driven update -- no labels beyond the relevance
feedback already present in the replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .buffers import RejectedEditBuffer
from .router import MemoryRouter
from .weights import EditBudget, FEATURES, WeightVector
from .workload import Interaction


@dataclass
class Proposal:
    base: WeightVector
    proposed: WeightVector
    rationale: str


class ProposalEngine:
    def __init__(self, router: MemoryRouter, budget: EditBudget,
                 lr: float = 0.6, eps: float = 0.04):
        self.router = router
        self.budget = budget
        self.lr = lr
        self.eps = eps

    def gradient(self, base: WeightVector, replay: list[Interaction]) -> list[float]:
        """Per-feature finite-difference sensitivity of the outcome metric.

        Public so an LLM proposer can be handed the same real-world signal the
        numeric engine uses, rather than guessing from raw weights.
        """
        grad = []
        for i, f in enumerate(FEATURES):
            up = base.vec()[:]
            down = base.vec()[:]
            up[i] += self.eps
            down[i] -= self.eps
            u = self.router.evaluate(WeightVector.from_vec(up), replay)
            d = self.router.evaluate(WeightVector.from_vec(down), replay)
            grad.append((u - d) / (2 * self.eps))
        return grad

    def propose(self, base: WeightVector, replay: list[Interaction],
                rejected: RejectedEditBuffer) -> Proposal:
        grad = self.gradient(base, replay)
        step = [g * self.lr for g in grad]

        # shrink the step if it points like a previously rejected (harmful) edit
        damp = rejected.damping(step)
        if damp < 1.0:
            step = [s * damp for s in step]

        # Normalize the *target* first, then bound the delta to it. Bounding and
        # then normalizing would let normalization reintroduce magnitude past the
        # budget; this order keeps the edit-budget guarantee strict.
        raw = WeightVector.from_vec([b + s for b, s in zip(base.vec(), step)])
        bounded = self.budget.apply(base, raw.normalized())

        moved = base.distance(bounded)
        top = max(range(len(FEATURES)), key=lambda i: abs(grad[i]))
        rationale = (
            f"outcome-gradient step (‖Δ‖={moved:.3f}, damp={damp:.2f}); "
            f"strongest signal on '{FEATURES[top]}' (∂={grad[top]:+.3f})"
        )
        return Proposal(base, bounded, rationale)
