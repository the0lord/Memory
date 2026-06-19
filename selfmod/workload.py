"""Simulated memory store and workloads with ground-truth relevance.

This stands in for the real world. It lets the prototype run end-to-end with no
external services: every query carries a known set of relevant memories, so we can
actually *measure* whether a self-edit improved routing.

Two distributions are exposed deliberately:
  * ``synthetic`` - a narrow, clustered distribution used by the SANDBOX stage.
  * ``real``      - the broader live distribution used by SHADOW / MONITORED.

A self-edit can look great on the narrow synthetic set yet regress on the broad
real set. That gap is exactly what the SHADOW stage is built to catch.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

TOPIC_DIM = 6


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


@dataclass
class Memory:
    id: int
    topic: list[float]      # latent topic embedding
    recency: float          # 0..1, higher = more recently used
    frequency: float        # 0..1
    importance: float       # 0..1
    salience: float         # 0..1


@dataclass
class Interaction:
    """One retrieval episode: a query, its candidate pool, and the truth."""

    query_topic: list[float]
    candidates: list[Memory]
    relevant_ids: set[int]

    def features(self, mem: Memory) -> dict[str, float]:
        """The per-(query, memory) feature vector the router sees."""
        sim = (_cosine(self.query_topic, mem.topic) + 1) / 2  # -> 0..1
        return {
            "similarity": sim,
            "recency": mem.recency,
            "frequency": mem.frequency,
            "importance": mem.importance,
            "salience": mem.salience,
        }


class World:
    """Generates a memory pool plus synthetic and real query streams.

    Ground truth is intentionally driven by *similarity* and *importance*:
    a memory is relevant when it is both topically close and reasonably important.
    Recency / frequency / salience are plausible-looking distractors. An ideal
    genome therefore learns to up-weight similarity & importance and suppress the
    rest -- a target we can verify the system actually discovers.
    """

    def __init__(self, n_memories: int = 240, seed: int = 7):
        self.rng = random.Random(seed)
        self.memories = [self._make_memory(i) for i in range(n_memories)]

    def _make_memory(self, mid: int) -> Memory:
        topic = [self.rng.gauss(0, 1) for _ in range(TOPIC_DIM)]
        importance = self.rng.random()
        return Memory(
            id=mid,
            topic=topic,
            recency=self.rng.random(),
            frequency=self.rng.random(),
            # importance is mildly correlated with topical "centrality"
            importance=importance,
            salience=self.rng.random(),
        )

    def _sample_interaction(self, spread: float, pool: int = 18, k_true: int = 5) -> Interaction:
        # A query is a point in topic space; `spread` controls how far queries
        # roam (synthetic = tight cluster, real = broad).
        center = [self.rng.gauss(0, spread) for _ in range(TOPIC_DIM)]
        candidates = self.rng.sample(self.memories, pool)

        def true_score(m: Memory) -> float:
            sim = (_cosine(center, m.topic) + 1) / 2
            return 0.7 * sim + 0.3 * m.importance  # the world's hidden utility

        ranked = sorted(candidates, key=true_score, reverse=True)
        relevant = {m.id for m in ranked[:k_true]}
        return Interaction(center, candidates, relevant)

    def batch(self, n: int, kind: str) -> list[Interaction]:
        spread = 0.45 if kind == "synthetic" else 1.0
        return [self._sample_interaction(spread) for _ in range(n)]
