"""The self-modifying memory AI: the loop that ties every component together.

Each cycle:
  1. measure current production outcome on live traffic;
  2. check for value drift -- if drifting, the next proposal is a corrective
     pull toward the golden reference instead of a fresh gradient step;
  3. generate a proposal (outcome-gradient, budget-bounded, buffer-damped);
  4. run it through the 5-stage pipeline;
  5. if promoted, adopt the new genome; otherwise keep production and retain the
     failed signal in the rejected-edit buffer.
  6. auto-save state to disk so the next session resumes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .buffers import GoldenTraceLog, RejectedEditBuffer
from .drift import Charter, DriftMonitor
from .feedback import StakeholderPanel
from .llm import select_proposer
from .persistence import STATE_FILE, load, save
from .pipeline import DeploymentPipeline, PipelineResult
from .proposals import Proposal
from .router import MemoryRouter
from .weights import EditBudget, WeightVector
from .workload import World


@dataclass
class CycleLog:
    cycle: int
    production_metric: float
    drift: str
    result: PipelineResult
    weights: WeightVector


class SelfModifyingMemoryAI:
    def __init__(self, initial: WeightVector, seed: int = 7, proposer=None,
                 state_path: Path = STATE_FILE, resume: bool = True):
        self.world = World(seed=seed)
        self.router = MemoryRouter(k=5)
        self.budget = EditBudget()
        self.state_path = state_path
        self.cycle_offset = 0
        self.resumed = False

        # Try to restore saved state; fall back to initial genome if none exists.
        saved = load(state_path) if resume else None
        if saved is not None:
            production, self.golden, self.rejected, self.cycle_offset = saved
            self.resumed = True
        else:
            self.rejected = RejectedEditBuffer()
            self.golden = GoldenTraceLog()
            production = initial.normalized()

        self.engine = proposer or select_proposer(self.router, self.budget)
        self.panel = StakeholderPanel(_default_panel())
        self.pipeline = DeploymentPipeline(
            self.world, self.router, self.panel, self.rejected, self.golden
        )
        self.drift = DriftMonitor(Charter())
        self.production = production
        self.history: list[CycleLog] = []

    def measure(self) -> float:
        """Current production outcome on a fresh sample of live traffic."""
        return self.router.evaluate(self.production, self.world.batch(200, "real"))

    def run_cycle(self, cycle: int) -> CycleLog:
        abs_cycle = self.cycle_offset + cycle
        production_metric = self.measure()
        self.drift.observe(production_metric)

        best = self.golden.best()
        report = self.drift.check(
            self.production,
            best.weights if best else None,
            best.metric if best else None,
        )

        if report.drifting and best is not None:
            corrected = self.drift.correct(self.production, best.weights)
            bounded = self.budget.apply(self.production, corrected.normalized())
            proposal = Proposal(
                self.production, bounded,
                f"DRIFT CORRECTION — {report.reason}; pulling toward golden reference",
            )
        else:
            replay = self.world.batch(120, "real")
            proposal = self.engine.propose(self.production, replay, self.rejected)

        result = self.pipeline.run(proposal, production_metric, abs_cycle)
        if result.promoted and result.new_weights is not None:
            self.production = result.new_weights

        log = CycleLog(abs_cycle, production_metric, report.reason, result, self.production)
        self.history.append(log)

        # persist after every cycle so a crash loses at most one cycle
        save(self.production, self.golden, self.rejected, abs_cycle, self.state_path)

        return log


def _default_panel():
    from .feedback import Stakeholder
    return [
        Stakeholder("relevance-eng", guarded_feature="similarity", tolerance=0.008),
        Stakeholder("product",       guarded_feature="importance", tolerance=0.012),
        Stakeholder("ux-recency",    guarded_feature="recency",    tolerance=0.020),
    ]
