"""The decision engine that the system modifies: the memory router.

The router ranks candidate memories for a query using a linear score over the
feature vector, weighted by the current genome. Its quality is measured with
NDCG@k against ground-truth relevance -- this scalar is the "real-world outcome"
the whole self-modification loop optimizes.
"""

from __future__ import annotations

import math

from .weights import WeightVector
from .workload import Interaction, Memory


def _ndcg(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for i, mid in enumerate(ranked_ids[:k]):
        if mid in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


class MemoryRouter:
    """Scores and ranks memories under a given genome."""

    def __init__(self, k: int = 5):
        self.k = k

    def score(self, weights: WeightVector, interaction: Interaction, mem: Memory) -> float:
        feats = interaction.features(mem)
        return sum(weights.values[f] * feats[f] for f in feats)

    def rank(self, weights: WeightVector, interaction: Interaction) -> list[int]:
        scored = sorted(
            interaction.candidates,
            key=lambda m: self.score(weights, interaction, m),
            reverse=True,
        )
        return [m.id for m in scored]

    def evaluate(self, weights: WeightVector, batch: list[Interaction]) -> float:
        """Mean NDCG@k over a batch -- the outcome metric (0..1, higher better)."""
        if not batch:
            return 0.0
        total = 0.0
        for it in batch:
            total += _ndcg(self.rank(weights, it), it.relevant_ids, self.k)
        return total / len(batch)
