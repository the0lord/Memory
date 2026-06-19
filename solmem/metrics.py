"""ROI & quality metrics — measure whether recall actually helps before adding more.

Two kinds of signal, both honest:

  * **Ranking quality**, computed from the *real* graded recall history — hit-rate,
    top-1 success, MRR. Answers "is recall surfacing the solution that works?" with
    zero synthetic data. (For robustness under a growing store, see bench_solmem.py.)
  * **Store health** — active / superseded / unverified / stale counts and provenance
    coverage.

True token savings can only be measured inside live sessions, which the server
can't see. So we report the *countable* facts (hits, applied recalls) and gate any
token estimate on an explicit per-hit assumption (SOLMEM_TOKENS_PER_HIT) rather
than inventing a number.
"""

from __future__ import annotations

import os
import time

from .store import Store

_STALE_RECENCY = 0.3        # matches explain_recall's "stale" threshold
_RECENCY_HALFLIFE_DAYS = 30.0
_DAY = 86_400.0


def _recency(updated: float, now: float) -> float:
    age_days = max(0.0, (now - updated) / _DAY)
    return 0.5 ** (age_days / _RECENCY_HALFLIFE_DAYS)


def compute(store: Store, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    records = store.all()
    recalls = store.all_recalls()
    graded = [ep for ep in recalls if ep.get("outcomes")]

    hits = top1_graded = top1_worked = applied = 0
    rr_sum = 0.0
    rr_n = 0
    for ep in graded:
        order = [c["id"] for c in ep.get("candidates", [])]  # candidates are logged ranked
        outcomes = ep.get("outcomes", {})
        applied += len(outcomes)
        worked = {rid for rid, ok in outcomes.items() if ok}
        if worked:
            hits += 1
            for i, rid in enumerate(order):  # MRR: first worked in the shown order
                if rid in worked:
                    rr_sum += 1.0 / (i + 1)
                    rr_n += 1
                    break
        if order and order[0] in outcomes:   # top-1 only counts when #1 was graded
            top1_graded += 1
            top1_worked += int(bool(outcomes[order[0]]))

    n_graded = len(graded)
    active = [r for r in records if r.active]
    non_neg_active = [r for r in active if r.kind != "negative"]
    authored = [r for r in records if r.author and r.author != "unknown"]

    return {
        "records": len(records),
        "active": len(active),
        "superseded": sum(1 for r in records if not r.active),
        "negatives": sum(1 for r in records if r.kind == "negative"),
        "recalls": len(recalls),
        "graded_recalls": n_graded,
        "applied_outcomes": applied,
        "hits": hits,
        "hit_rate": (hits / n_graded) if n_graded else 0.0,
        "top1_rate": (top1_worked / top1_graded) if top1_graded else 0.0,
        "top1_graded": top1_graded,
        "mrr": (rr_sum / rr_n) if rr_n else 0.0,
        "unverified_active": sum(1 for r in non_neg_active if not r.verified),
        "stale_active": sum(1 for r in active if _recency(r.updated, now) < _STALE_RECENCY),
        "provenance_coverage": (len(authored) / len(records)) if records else 0.0,
    }


def render(store: Store) -> str:
    m = compute(store)
    if m["records"] == 0 and m["recalls"] == 0:
        return "Store is empty — nothing to measure yet. Use recall + report_outcome first."

    lines = [
        "solmem ROI & quality",
        "",
        "recall quality (from graded history):",
        f"  graded recalls:   {m['graded_recalls']}   (applied outcomes: {m['applied_outcomes']})",
        f"  hit-rate:         {m['hit_rate']:.0%}   "
        f"(≥1 surfaced solution worked, {m['hits']}/{m['graded_recalls']})",
        f"  top-1 success:    {m['top1_rate']:.0%}   "
        f"(the #1 candidate worked, over {m['top1_graded']} graded)",
        f"  MRR:              {m['mrr']:.3f}   (1 = right answer always ranked first)",
        "",
        "store health:",
        f"  records:          {m['records']}   active {m['active']} · "
        f"superseded {m['superseded']} · negatives {m['negatives']}",
        f"  unverified:       {m['unverified_active']} active records with no confirmed success",
        f"  stale:            {m['stale_active']} active records not touched in a while",
        f"  provenance:       {m['provenance_coverage']:.0%} of records have a known author",
    ]

    per_hit = os.environ.get("SOLMEM_TOKENS_PER_HIT")
    lines += ["", "token ROI:"]
    if per_hit and per_hit.isdigit():
        saved = m["hits"] * int(per_hit)
        lines.append(f"  estimated saved:  ~{saved:,} tokens "
                     f"({m['hits']} hits × {per_hit}/hit, your assumption)")
    else:
        lines.append("  set SOLMEM_TOKENS_PER_HIT=<your measured tokens saved per reuse> "
                     "for an estimate;")
        lines.append(f"  countable now:    {m['hits']} successful reuses to multiply by it.")

    if m["graded_recalls"] < 6:
        lines += ["", "Note: too few graded recalls for stable numbers — keep using it, "
                  "then re-check. (bench_solmem.py tests ranking robustness offline.)"]
    return "\n".join(lines)
