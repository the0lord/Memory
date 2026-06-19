"""Check whether a recalled record actually applies to the current situation.

A high score means "similar", not "safe to apply blindly". Records can declare
``applies_when`` / ``does_not_apply_when`` clauses; here we match them against the
caller's context (free text and/or stack) so the agent is warned before trusting a
solution in a situation its author explicitly excluded.
"""

from __future__ import annotations

from .records import SolutionRecord, keywords_of


def _clause_hit(clause: str, ctx_tokens: set[str]) -> bool:
    """A clause matches when at least half of its meaningful keywords appear in
    the context (minimum one), e.g. 'database is Postgres' -> {database, postgres}."""
    kws = set(keywords_of(clause))
    if not kws:
        return False
    overlap = len(kws & ctx_tokens)
    return overlap >= max(1, len(kws) // 2)


def check(rec: SolutionRecord, context: str) -> dict:
    ctx_tokens = set(keywords_of(context))
    blocked = [c for c in rec.does_not_apply_when if _clause_hit(c, ctx_tokens)]
    satisfied = [c for c in rec.applies_when if _clause_hit(c, ctx_tokens)]

    safe = not blocked
    if rec.applies_when:
        confidence = len(satisfied) / len(rec.applies_when)
    else:
        confidence = 0.5  # nothing declared -> neutral
    if blocked:
        confidence = min(confidence, 0.1)

    return {
        "safe_to_try": safe,
        "confidence": round(confidence, 2),
        "satisfied": satisfied,
        "blocked_by": blocked,
        "required_checks": rec.verification,
        "risk": rec.risk,
    }


def render(rec: SolutionRecord, result: dict) -> str:
    lines = [
        f"record {rec.id}  risk={result['risk']}  "
        f"safe_to_try={result['safe_to_try']}  confidence={result['confidence']:.0%}"
    ]
    if result["blocked_by"]:
        lines.append("  ⛔ excluded because: " + "; ".join(result["blocked_by"]))
    if result["satisfied"]:
        lines.append("  ✓ matched: " + "; ".join(result["satisfied"]))
    if result["required_checks"]:
        lines.append("  required checks: " + "; ".join(result["required_checks"]))
    return "\n".join(lines)
