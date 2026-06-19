"""Turn a (query problem, past record) pair into the 5 router features, then
score it with the brain's weight vector.

The feature axes are deliberately the SAME five the selfmod engine evolves
(similarity / recency / frequency / importance / salience) so the recall brain
and the self-modification loop speak one language. Here each axis is given a
concrete, code-task meaning:

  similarity  - keyword overlap between the new problem and the stored one
  recency     - how recently the record was created or last reused
  frequency   - how often the record has been reused (log-damped)
  importance  - intrinsic weight of the solution (tags / specificity)
  salience    - the record's track record: smoothed success rate
"""

from __future__ import annotations

import math
import time

from selfmod.weights import FEATURES, WeightVector

from .records import SolutionRecord, keywords_of

# Default brain: similarity dominates, salience (does-it-actually-work) second.
# These are exactly the knobs the selfmod pipeline will later evolve from outcomes.
DEFAULT_WEIGHTS = WeightVector({
    "similarity": 0.45,
    "recency": 0.10,
    "frequency": 0.10,
    "importance": 0.10,
    "salience": 0.25,
})

_RECENCY_HALFLIFE_DAYS = 30.0
_DAY = 86_400.0


def _cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / math.sqrt(len(a) * len(b))


def features(query_keywords: set[str], rec: SolutionRecord, now: float | None = None,
             query_vec: list[float] | None = None) -> dict[str, float]:
    now = time.time() if now is None else now

    # Prefer semantic cosine when both sides carry an embedding; otherwise fall
    # back to keyword overlap. The feature layer stays pure: vectors are computed
    # by the caller (server), never via a network call from here.
    if query_vec and rec.embedding:
        from .embeddings import cosine as _embed_cosine
        similarity = _embed_cosine(query_vec, rec.embedding)
    else:
        similarity = _cosine(query_keywords, set(rec.keywords))

    age_days = max(0.0, (now - rec.updated) / _DAY)
    recency = 0.5 ** (age_days / _RECENCY_HALFLIFE_DAYS)

    frequency = math.log1p(rec.uses) / math.log1p(50)  # saturates around ~50 reuses
    frequency = min(1.0, frequency)

    # Importance: a tagged, reasonably specific solution is worth more than a
    # one-liner with no metadata. Bounded to [0, 1].
    importance = min(1.0, 0.4 + 0.1 * len(rec.tags) + min(0.3, len(rec.keywords) / 80.0))

    salience = rec.trust  # success rate, regression-penalized; in (0, 1)

    return {
        "similarity": similarity,
        "recency": recency,
        "frequency": frequency,
        "importance": importance,
        "salience": salience,
    }


def score(weights: WeightVector, feats: dict[str, float]) -> float:
    return sum(weights.values[f] * feats[f] for f in FEATURES)


def rank(
    weights: WeightVector,
    query: str,
    records: list[SolutionRecord],
    limit: int = 5,
    now: float | None = None,
    query_vec: list[float] | None = None,
) -> list[tuple[SolutionRecord, float, dict[str, float]]]:
    """Return the top-``limit`` records as (record, score, features), best first.

    ``query_vec`` (the query embedding) enables semantic similarity; when omitted,
    similarity falls back to keyword overlap.
    """
    qkw = set(keywords_of(query))
    scored = []
    for rec in records:
        feats = features(qkw, rec, now, query_vec)
        scored.append((rec, score(weights, feats), feats))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:limit]
