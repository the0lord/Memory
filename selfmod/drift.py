"""Value-drift detection and correction toward the charter goal.

The 'charter' is the system's standing goal: keep production outcome at or above a
floor, anchored to the best genome ever promoted (its golden reference). Drift is
when production quality decays over time, or the live genome wanders far from the
golden reference without earning a better outcome. When detected, the orchestrator
issues a corrective edit that pulls the genome back toward the golden reference --
bounded, like every other edit, by the edit budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .weights import WeightVector


@dataclass
class Charter:
    """The standing goal the system must not drift away from.

    Drift is *backsliding*, not *not-yet-arrived*. So the effective floor is
    adaptive: it tracks the best outcome ever promoted, minus a regression band.
    While the system is still climbing toward the goal, the floor stays low and
    healthy learning is never mistaken for drift. Once high performance is
    reached, the floor rises with it and guards against regression.
    """

    metric_floor: float = 0.30     # absolute safety net (real collapse only)
    regression_band: float = 0.05  # how far below the best we tolerate
    max_drift_distance: float = 0.60  # genome may not stray this far from golden
    window: int = 4                # how many recent cycles define 'recent'


@dataclass
class DriftReport:
    drifting: bool
    reason: str
    recent_metric: float


class DriftMonitor:
    def __init__(self, charter: Charter):
        self.charter = charter
        self.history: list[float] = []

    def observe(self, production_metric: float) -> None:
        self.history.append(production_metric)

    def check(self, current: WeightVector, golden: WeightVector | None,
              best_metric: float | None = None) -> DriftReport:
        recent = self.history[-self.charter.window:]
        recent_mean = sum(recent) / len(recent) if recent else 1.0

        # adaptive floor: the higher of the absolute safety net and
        # (best-ever outcome - regression band). We test the *latest* outcome,
        # not the trailing mean -- otherwise rapid improvement (whose mean lags
        # the rising best) is misread as a backslide.
        latest = self.history[-1] if self.history else 1.0
        floor = self.charter.metric_floor
        if best_metric is not None:
            floor = max(floor, best_metric - self.charter.regression_band)

        if self.history and latest < floor:
            reason = (
                f"latest {latest:.3f} backslid below floor {floor:.2f} (best {best_metric:.3f})"
                if best_metric is not None
                else f"latest {latest:.3f} below floor {floor:.2f}"
            )
            return DriftReport(True, reason, recent_mean)

        # downward trend across the window
        if len(recent) >= self.charter.window and recent[-1] < recent[0] - 0.03:
            return DriftReport(
                True,
                f"outcome trending down ({recent[0]:.3f} → {recent[-1]:.3f})",
                recent_mean,
            )

        if golden is not None and current.distance(golden) > self.charter.max_drift_distance:
            return DriftReport(
                True,
                f"genome strayed {current.distance(golden):.3f} from golden reference",
                recent_mean,
            )

        return DriftReport(False, "within charter", recent_mean)

    @staticmethod
    def correct(current: WeightVector, golden: WeightVector, pull: float = 0.5) -> WeightVector:
        """Produce a corrective genome that leans back toward the golden reference."""
        return current.blend(golden, pull)
