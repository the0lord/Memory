"""Evolve the recall brain from real outcomes, through the full selfmod pipeline.

Every graded recall episode is a labeled ranking example: we surfaced a set of
candidate solutions, and the developer/tests told us which one actually worked.
That is exactly the signal the selfmod engine optimizes -- so here we wrap recall
history as a ``World`` of ``Interaction``s and run the proposal + 5-stage pipeline
verbatim. SANDBOX/SHADOW become a train/held-out split, which catches edits that
overfit the cases we happen to have graded.

The brain's 5 weights are the genome. If a proposed re-weighting ranks the
known-good solutions higher on held-out episodes, it graduates and becomes the new
brain; otherwise it rolls back and its signal is kept in the rejected-edit buffer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from selfmod.feedback import Stakeholder, StakeholderPanel
from selfmod.pipeline import DeploymentPipeline
from selfmod.proposals import ProposalEngine
from selfmod.router import MemoryRouter
from selfmod.weights import EditBudget, WeightVector

from .store import Store


# --- recall history as router-compatible Interactions ---------------------- #
@dataclass
class _Candidate:
    id: str
    feats: dict


@dataclass
class _Episode:
    """One graded recall, duck-typed to look like a selfmod Interaction."""
    candidates: list[_Candidate]
    relevant_ids: set[str]

    def features(self, mem: _Candidate) -> dict:
        return mem.feats


class _ReplayWorld:
    """Serves batches from graded recall history. ``synthetic`` draws from the
    train split (the SANDBOX gate), ``real`` from the held-out split (SHADOW /
    MONITORED) so an overfit re-weighting gets caught just like in selfmod."""

    def __init__(self, train: list[_Episode], heldout: list[_Episode], seed: int = 7):
        self.train = train
        self.heldout = heldout
        self.rng = random.Random(seed)

    def batch(self, n: int, kind: str) -> list[_Episode]:
        pool = self.train if kind == "synthetic" else self.heldout
        if not pool:
            pool = self.train or self.heldout
        if len(pool) >= n:
            return self.rng.sample(pool, n)
        return [self.rng.choice(pool) for _ in range(n)]  # bootstrap when sparse


def build_episodes(store: Store) -> list[_Episode]:
    """Graded recalls with >=2 candidates and >=1 solution that actually worked.
    A failed-only episode gives no ranking target, so it is skipped here (its
    signal still lives in each record's success rate via the salience feature)."""
    episodes: list[_Episode] = []
    for ep in store.all_recalls():
        outcomes = ep.get("outcomes", {})
        relevant = {cid for cid, ok in outcomes.items() if ok}
        cands = [_Candidate(c["id"], c["features"]) for c in ep.get("candidates", [])]
        if len(cands) >= 2 and relevant & {c.id for c in cands}:
            episodes.append(_Episode(cands, relevant & {c.id for c in cands}))
    return episodes


def _default_panel() -> StakeholderPanel:
    return StakeholderPanel([
        Stakeholder("relevance-eng", "similarity", 0.008),
        Stakeholder("product",       "salience",   0.012),
        Stakeholder("ux-recency",    "recency",    0.020),
    ])


@dataclass
class EvolveReport:
    ran: bool
    detail: str
    before: float = 0.0
    after: float = 0.0
    promoted: bool = False


def evolve(store: Store, min_episodes: int = 6, heldout_frac: float = 0.35,
           k: int = 5, seed: int = 7) -> EvolveReport:
    """Attempt one self-modification of the recall brain from graded outcomes."""
    episodes = build_episodes(store)
    if len(episodes) < min_episodes:
        return EvolveReport(
            False,
            f"not enough graded recalls to evolve (have {len(episodes)}, "
            f"need {min_episodes}). Keep using recall + report_outcome.",
        )

    rng = random.Random(seed)
    rng.shuffle(episodes)
    cut = max(1, int(len(episodes) * heldout_frac))
    heldout, train = episodes[:cut], episodes[cut:]

    router = MemoryRouter(k=k)
    budget = EditBudget()
    base = store.load_brain()
    golden = store.load_golden()
    rejected = store.load_rejected()

    before = router.evaluate(base, heldout)
    engine = ProposalEngine(router, budget)
    proposal = engine.propose(base, train, rejected)

    pipeline = DeploymentPipeline(_ReplayWorld(train, heldout, seed),
                                  router, _default_panel(), rejected, golden)
    cycle = len(golden.traces) + 1
    result = pipeline.run(proposal, before, cycle)

    # persist what the pipeline learned, win or lose
    store.save_rejected(rejected)
    if result.promoted and result.new_weights is not None:
        after = router.evaluate(result.new_weights, heldout)
        promotion = {
            "before_ndcg": round(before, 4), "after_ndcg": round(after, 4),
            "episodes": len(episodes),
            "stages_passed": [e.stage.value for e in result.events if e.passed],
        }
        new_version = store.promote_brain(result.new_weights, promotion)
        store.save_golden(golden)
        detail = (f"PROMOTED → {new_version} — brain evolved on {len(episodes)} graded recalls.\n"
                  f"  held-out NDCG@{k}: {before:.3f} → {after:.3f}\n"
                  f"  {proposal.rationale}\n"
                  f"  new brain: {result.new_weights.pretty()}")
        return EvolveReport(True, detail, before, after, True)

    stage = result.final_stage.value
    reason = next((e.detail for e in result.events if not e.passed), stage)
    detail = (f"NO CHANGE — proposal rejected at {stage} (kept signal in rejected "
              f"buffer).\n  held-out NDCG@{k}: {before:.3f} (unchanged)\n"
              f"  reason: {reason}\n  {proposal.rationale}")
    return EvolveReport(True, detail, before, before, False)
