"""The 5-stage deployment pipeline that gates every self-edit.

    PROPOSAL → SANDBOX → SHADOW → GATED → MONITORED → PROMOTED
                  │         │        │         │
                  └─────────┴────────┴─────────┴──► ROLLBACK (signal → rejected buffer)

Each stage can only pass a candidate forward or reject it. A rejection at any
stage routes the candidate's signal into the rejected-edit buffer so nothing is
wasted. Promotion records a golden trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .buffers import GoldenTraceLog, RejectedEditBuffer
from .feedback import StakeholderPanel
from .proposals import Proposal
from .router import MemoryRouter
from .weights import WeightVector, is_major_change
from .workload import Interaction, World


class Stage(str, Enum):
    PROPOSAL = "PROPOSAL"
    SANDBOX = "SANDBOX"
    SHADOW = "SHADOW"
    GATED = "GATED"
    MONITORED = "MONITORED"
    PROMOTED = "PROMOTED"
    ROLLBACK = "ROLLBACK"


@dataclass
class StageEvent:
    stage: Stage
    passed: bool
    detail: str
    metric: float = 0.0


@dataclass
class PipelineResult:
    promoted: bool
    final_stage: Stage
    events: list[StageEvent] = field(default_factory=list)
    new_weights: WeightVector | None = None
    new_metric: float = 0.0


class DeploymentPipeline:
    """Runs a proposal through all five gates."""

    def __init__(self, world: World, router: MemoryRouter, panel: StakeholderPanel,
                 rejected: RejectedEditBuffer, golden: GoldenTraceLog,
                 sandbox_threshold: float = 0.0,
                 shadow_regression_tol: float = 0.005,
                 monitor_windows: int = 3,
                 rollback_tol: float = 0.02):
        self.world = world
        self.router = router
        self.panel = panel
        self.rejected = rejected
        self.golden = golden
        self.sandbox_threshold = sandbox_threshold
        self.shadow_regression_tol = shadow_regression_tol
        self.monitor_windows = monitor_windows
        self.rollback_tol = rollback_tol

    def run(self, proposal: Proposal, production_metric: float, cycle: int) -> PipelineResult:
        base, cand = proposal.base, proposal.proposed
        events: list[StageEvent] = [
            StageEvent(Stage.PROPOSAL, True, proposal.rationale)
        ]

        # 1. SANDBOX — narrow synthetic workload; cheap gate on obvious losers.
        sb = self.world.batch(60, "synthetic")
        sb_base = self.router.evaluate(base, sb)
        sb_cand = self.router.evaluate(cand, sb)
        if sb_cand < sb_base + self.sandbox_threshold:
            return self._reject(Stage.SANDBOX, base, cand, sb_cand - sb_base, events,
                                f"no synthetic gain ({sb_base:.3f} → {sb_cand:.3f})")
        events.append(StageEvent(Stage.SANDBOX, True,
                                 f"synthetic {sb_base:.3f} → {sb_cand:.3f}", sb_cand))

        # 2. SHADOW — broad real workload, scored in parallel, no user impact.
        #    Catches edits that overfit the sandbox but regress on live traffic.
        sh = self.world.batch(150, "real")
        sh_base = self.router.evaluate(base, sh)
        sh_cand = self.router.evaluate(cand, sh)
        if sh_cand < sh_base - self.shadow_regression_tol:
            return self._reject(Stage.SHADOW, base, cand, sh_cand - sh_base, events,
                                f"shadow regression ({sh_base:.3f} → {sh_cand:.3f})")
        events.append(StageEvent(Stage.SHADOW, True,
                                 f"shadow {sh_base:.3f} → {sh_cand:.3f}", sh_cand))

        # 3. GATED — major changes require stakeholder consensus.
        if is_major_change(base, cand):
            approved, detail = self.panel.deliberate(self.router, base, cand, sh)
            if not approved:
                return self._reject(Stage.GATED, base, cand, sh_cand - sh_base, events,
                                    f"consensus denied — {detail}")
            events.append(StageEvent(Stage.GATED, True, f"major change approved — {detail}"))
        else:
            events.append(StageEvent(Stage.GATED, True, "minor change — auto-approved"))

        # 4. MONITORED — live for several windows with an auto-rollback trigger.
        baseline = production_metric
        for w in range(self.monitor_windows):
            window = self.world.batch(80, "real")
            live = self.router.evaluate(cand, window)
            if live < baseline - self.rollback_tol:
                return self._reject(Stage.MONITORED, base, cand, live - baseline, events,
                                    f"auto-rollback at window {w+1} ({live:.3f} < {baseline:.3f})")
        events.append(StageEvent(Stage.MONITORED, True,
                                 f"{self.monitor_windows} windows stable ≥ {baseline:.3f}"))

        # 5. PROMOTED — adopt as production, record golden trace.
        final_metric = self.router.evaluate(cand, self.world.batch(150, "real"))
        self.golden.record(cand, final_metric, cycle)
        events.append(StageEvent(Stage.PROMOTED, True,
                                 f"golden trace recorded @ {final_metric:.3f}", final_metric))
        return PipelineResult(True, Stage.PROMOTED, events, cand, final_metric)

    def _reject(self, stage: Stage, base: WeightVector, cand: WeightVector,
                delta_metric: float, events: list[StageEvent], detail: str) -> PipelineResult:
        self.rejected.add(base, cand, stage.value, delta_metric, detail)
        events.append(StageEvent(stage, False, detail, 0.0))
        events.append(StageEvent(Stage.ROLLBACK, False, "signal → rejected-edit buffer"))
        return PipelineResult(False, stage, events)
