"""Decision weights (the genome) and the edit budget that bounds change.

The memory router ranks candidate memories with a linear score over a small set
of features. Those per-feature weights ARE the thing the system modifies about
itself. Every proposed modification is forced through an ``EditBudget`` so that no
single self-edit can move the genome more than a bounded amount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The feature axes the router scores memories on. Order is stable so a genome can
# be treated as a vector when convenient.
FEATURES: tuple[str, ...] = (
    "similarity",   # semantic closeness of memory to the query
    "recency",      # how recently the memory was touched
    "frequency",    # how often the memory is accessed
    "importance",   # stored importance score
    "salience",     # emotional / attentional salience
)


@dataclass(frozen=True)
class WeightVector:
    """An immutable set of per-feature decision weights."""

    values: dict[str, float]

    def __post_init__(self) -> None:
        missing = set(FEATURES) - set(self.values)
        if missing:
            raise ValueError(f"weight vector missing features: {sorted(missing)}")

    # -- vector helpers -------------------------------------------------
    def vec(self) -> list[float]:
        return [self.values[f] for f in FEATURES]

    @classmethod
    def from_vec(cls, vec: list[float]) -> "WeightVector":
        return cls({f: float(v) for f, v in zip(FEATURES, vec)})

    def distance(self, other: "WeightVector") -> float:
        """L2 distance between two genomes."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.vec(), other.vec())))

    def blend(self, other: "WeightVector", t: float) -> "WeightVector":
        """Linear interpolation toward ``other`` by fraction ``t`` in [0, 1]."""
        return WeightVector.from_vec(
            [a + (b - a) * t for a, b in zip(self.vec(), other.vec())]
        )

    def normalized(self) -> "WeightVector":
        """Scale so weights sum to 1 (keeps scores comparable across genomes)."""
        total = sum(abs(v) for v in self.vec()) or 1.0
        return WeightVector.from_vec([v / total for v in self.vec()])

    def pretty(self) -> str:
        return "  ".join(f"{f}={self.values[f]:+.3f}" for f in FEATURES)


@dataclass(frozen=True)
class EditBudget:
    """Bounds the magnitude of any single self-edit.

    Three independent guards:
      * ``max_l2``           - the whole change cannot exceed this L2 norm.
      * ``max_per_feature``  - no single weight may move more than this.
      * ``max_features``     - at most this many weights may change at once;
                               the smallest-magnitude deltas are zeroed out.
    """

    max_l2: float = 0.15
    max_per_feature: float = 0.10
    max_features: int = 3

    def apply(self, base: WeightVector, proposed: WeightVector) -> WeightVector:
        """Return ``proposed`` clipped so the delta from ``base`` fits the budget."""
        delta = [p - b for b, p in zip(base.vec(), proposed.vec())]

        # 1. keep only the K largest-magnitude coordinates of the change
        if self.max_features < len(delta):
            ranked = sorted(range(len(delta)), key=lambda i: abs(delta[i]), reverse=True)
            keep = set(ranked[: self.max_features])
            delta = [d if i in keep else 0.0 for i, d in enumerate(delta)]

        # 2. clamp each coordinate
        delta = [max(-self.max_per_feature, min(self.max_per_feature, d)) for d in delta]

        # 3. scale the whole vector down if its norm is still too large
        norm = math.sqrt(sum(d * d for d in delta))
        if norm > self.max_l2 and norm > 0:
            scale = self.max_l2 / norm
            delta = [d * scale for d in delta]

        return WeightVector.from_vec([b + d for b, d in zip(base.vec(), delta)])


def is_major_change(base: WeightVector, proposed: WeightVector, threshold: float = 0.06) -> bool:
    """A change is 'major' if it is large OR flips the sign of any weight.

    Major changes are routed to stakeholder consensus at the GATED stage.
    """
    if base.distance(proposed) >= threshold:
        return True
    for f in FEATURES:
        b, p = base.values[f], proposed.values[f]
        if b != 0 and (b > 0) != (p > 0):
            return True
    return False
