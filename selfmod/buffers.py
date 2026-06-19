"""Memory of the past: golden traces (what worked) and rejected edits (what didn't).

Failed experiments are not discarded. Their *signal* -- the direction that was
tried and the outcome it produced -- is preserved so the proposal engine can avoid
re-walking known-bad directions and can learn from near-misses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .weights import FEATURES, WeightVector


@dataclass
class RejectedEdit:
    """A self-edit that failed somewhere in the pipeline, kept for its signal."""

    direction: list[float]   # unit-ish delta that was attempted (proposed - base)
    stage: str               # where it died
    delta_metric: float      # measured change in outcome (negative = it hurt)
    note: str = ""
    ts: float = field(default_factory=time.time)


class RejectedEditBuffer:
    """Bounded store of failed edits, with similarity-aware damping."""

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.items: list[RejectedEdit] = []

    def add(self, base: WeightVector, proposed: WeightVector, stage: str,
            delta_metric: float, note: str = "") -> None:
        direction = [p - b for b, p in zip(base.vec(), proposed.vec())]
        self.items.append(RejectedEdit(direction, stage, delta_metric, note))
        if len(self.items) > self.capacity:
            self.items.pop(0)

    def damping(self, candidate_direction: list[float]) -> float:
        """Return a factor in (0, 1]: smaller when the direction resembles a
        previously rejected (harmful) one. Used to shrink risky proposals."""
        if not self.items:
            return 1.0
        cand_norm = _norm(candidate_direction)
        if cand_norm == 0:
            return 1.0
        worst = 0.0
        for it in self.items:
            sim = _cosine(candidate_direction, it.direction)
            if sim > 0 and it.delta_metric < 0:
                # weight by how badly that edit hurt
                worst = max(worst, sim * min(1.0, abs(it.delta_metric) * 20))
        return max(0.25, 1.0 - worst)

    def summary(self) -> str:
        if not self.items:
            return "rejected-edit buffer: empty"
        by_stage: dict[str, int] = {}
        for it in self.items:
            by_stage[it.stage] = by_stage.get(it.stage, 0) + 1
        parts = ", ".join(f"{s}:{n}" for s, n in sorted(by_stage.items()))
        return f"rejected-edit buffer: {len(self.items)} held ({parts})"


@dataclass
class GoldenTrace:
    """A genome that was promoted to production, with the outcome it achieved."""

    weights: WeightVector
    metric: float
    cycle: int
    ts: float = field(default_factory=time.time)


class GoldenTraceLog:
    """History of promoted genomes; the best one anchors drift correction."""

    def __init__(self) -> None:
        self.traces: list[GoldenTrace] = []

    def record(self, weights: WeightVector, metric: float, cycle: int) -> None:
        self.traces.append(GoldenTrace(weights, metric, cycle))

    def best(self) -> GoldenTrace | None:
        return max(self.traces, key=lambda t: t.metric) if self.traces else None


def _norm(v: list[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
