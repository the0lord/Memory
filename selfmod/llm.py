"""Live LLM proposal author (Mistral) — a drop-in for the numeric ProposalEngine.

The LLM is handed the *same real-world signal* the numeric engine uses (per-feature
outcome sensitivities), plus the rejected-edit buffer, and asked to author a weight
delta with a rationale. Its answer is then forced through the exact same edit
budget and five-stage pipeline as any other proposal — the model authors, the
guards still rule.

Design choices:
  * Key comes from the MISTRAL_API_KEY environment variable. Never a file.
  * stdlib-only HTTP (urllib) so the project keeps zero dependencies.
  * Any failure (no key, network, bad JSON, out-of-range) falls back to the
    numeric ProposalEngine so the loop never breaks.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .buffers import RejectedEditBuffer
from .proposals import Proposal, ProposalEngine
from .router import MemoryRouter
from .weights import EditBudget, FEATURES, WeightVector
from .workload import Interaction

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

FEATURE_DOC = {
    "similarity": "semantic closeness of the memory to the query",
    "recency": "how recently the memory was touched",
    "frequency": "how often the memory is accessed",
    "importance": "stored importance score",
    "salience": "emotional / attentional salience",
}

SYSTEM_PROMPT = (
    "You tune the decision weights of a memory-retrieval router. The router ranks "
    "candidate memories by a linear score over five features; higher mean NDCG@5 on "
    "real traffic is better. You are given the current weights, the measured "
    "sensitivity of NDCG@5 to each weight, and directions that were tried before and "
    "failed. Propose a SMALL, bounded weight delta that should raise NDCG@5. "
    "Respect the edit budget. Avoid directions similar to ones that already failed. "
    "Reply with JSON only."
)


class MistralProposer:
    """Authors proposals with Mistral; falls back to the numeric engine on error."""

    def __init__(self, router: MemoryRouter, budget: EditBudget,
                 api_key: str | None = None, model: str | None = None,
                 timeout: float = 30.0, temperature: float = 0.2,
                 fallback: ProposalEngine | None = None,
                 call_cooldown: float | None = None):
        self.router = router
        self.budget = budget
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.model = model or os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
        self.timeout = timeout
        self.temperature = temperature
        self.fallback = fallback or ProposalEngine(router, budget)
        # Free-tier Mistral Large is 2-5 RPM. Default to 20 s between calls
        # (3 RPM) so variable server load can't push us into 429s. Override via
        # MISTRAL_COOLDOWN_SECS or the constructor arg.
        env_cd = os.environ.get("MISTRAL_COOLDOWN_SECS")
        self.call_cooldown: float = (
            call_cooldown if call_cooldown is not None
            else float(env_cd) if env_cd
            else 20.0
        )
        self._last_call_ts: float = 0.0

    # -- public API (matches ProposalEngine.propose) --------------------
    def propose(self, base: WeightVector, replay: list[Interaction],
                rejected: RejectedEditBuffer) -> Proposal:
        try:
            if not self.api_key:
                raise RuntimeError("MISTRAL_API_KEY not set")
            grad = self.fallback.gradient(base, replay)
            metric = self.router.evaluate(base, replay)
            self._rate_limit()
            content = self._call(self._user_prompt(base, grad, metric, rejected))
            self._last_call_ts = time.monotonic()
            delta = self._parse_delta(content)
            rationale = self._parse_rationale(content)

            # normalize the target, then bound the delta (keeps the budget strict)
            raw = WeightVector.from_vec([b + delta[f] for b, f in zip(base.vec(), FEATURES)])
            bounded = self.budget.apply(base, raw.normalized())

            # buffer-damp risky directions, exactly like the numeric path
            step = [p - b for b, p in zip(base.vec(), bounded.vec())]
            damp = rejected.damping(step)
            if damp < 1.0:
                bounded = self.budget.apply(
                    base, WeightVector.from_vec([b + s * damp for b, s in zip(base.vec(), step)])
                )

            moved = base.distance(bounded)
            return Proposal(base, bounded,
                            f"[mistral:{self.model}] {rationale} (‖Δ‖={moved:.3f}, damp={damp:.2f})")
        except Exception as exc:  # noqa: BLE001 — any failure must fall back safely
            prop = self.fallback.propose(base, replay, rejected)
            return Proposal(prop.base, prop.proposed,
                            f"[mistral fallback: {type(exc).__name__}: {exc}] {prop.rationale}")

    # -- rate limiting --------------------------------------------------
    def _rate_limit(self) -> None:
        """Block until the cooldown since the last call has elapsed, then print
        a countdown so the terminal doesn't look frozen during long demos."""
        if self._last_call_ts == 0.0:
            return  # first call — no wait
        elapsed = time.monotonic() - self._last_call_ts
        wait = self.call_cooldown - elapsed
        if wait <= 0:
            return
        # progress bar so the user can see it's alive
        import sys
        total = int(wait)
        for remaining in range(total, 0, -1):
            bar = "█" * (total - remaining) + "░" * remaining
            sys.stdout.write(f"\r  rate-limit [{bar}] {remaining:2d}s  ")
            sys.stdout.flush()
            time.sleep(min(1.0, wait - (total - remaining)))
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

    # -- prompt / parse -------------------------------------------------
    def _user_prompt(self, base: WeightVector, grad: list[float], metric: float,
                     rejected: RejectedEditBuffer) -> str:
        weights = {f: round(base.values[f], 4) for f in FEATURES}
        sensitivity = {f: round(g, 4) for f, g in zip(FEATURES, grad)}
        bad_dirs = [
            {"direction": {f: round(d, 3) for f, d in zip(FEATURES, it.direction)},
             "hurt_by": round(it.delta_metric, 4), "stage": it.stage}
            for it in rejected.items[-6:]
        ]
        spec = {
            "features": FEATURE_DOC,
            "current_weights": weights,
            "ndcg_at_5_now": round(metric, 4),
            "ndcg_sensitivity_per_weight": sensitivity,
            "recently_failed_directions": bad_dirs,
            "edit_budget": {
                "max_abs_change_per_weight": self.budget.max_per_feature,
                "max_total_L2_change": self.budget.max_l2,
                "max_weights_changed": self.budget.max_features,
            },
            "respond_with": {
                "delta": {f: "float change to add to this weight" for f in FEATURES},
                "rationale": "one short sentence",
            },
        }
        return json.dumps(spec, indent=2)

    def _parse_delta(self, content: str) -> dict[str, float]:
        data = json.loads(content)
        raw = data.get("delta", data)
        return {f: float(raw.get(f, 0.0)) for f in FEATURES}

    def _parse_rationale(self, content: str) -> str:
        try:
            return str(json.loads(content).get("rationale", "")).strip()[:160] or "no rationale"
        except Exception:  # noqa: BLE001
            return "no rationale"

    # -- transport ------------------------------------------------------
    def _call(self, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            MISTRAL_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        return body["choices"][0]["message"]["content"]


def select_proposer(router: MemoryRouter, budget: EditBudget):
    """Pick the proposer: Mistral when a key is present, else the numeric engine."""
    if os.environ.get("MISTRAL_API_KEY"):
        return MistralProposer(router, budget)  # cooldown loaded from env/default
    return ProposalEngine(router, budget)
