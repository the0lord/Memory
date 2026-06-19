"""Prove semantic recall wins where keyword recall fails — deterministically.

Stubs the embedder (no network) so a query that shares NO keywords with the right
record still matches it via embedding cosine.

Run:  py test_embeddings.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import solmem.embeddings as emb
from solmem.server import SolMem
from solmem.store import Store

# Two topics, orthogonal vectors. The query for the "lock" topic deliberately
# shares no tokens with the stored "lock" record's problem text.
_VECS = {
    "wallet withdrawal TOCTOU race condition": [1.0, 0.0, 0.0],   # record A (lock)
    "UnicodeEncodeError cp1252 windows arrows": [0.0, 1.0, 0.0],  # record B (encoding)
    "concurrent debit double spend safety":     [1.0, 0.0, 0.0],  # query -> topic lock
}


def fake_embed_one(text: str):
    return _VECS.get(text.strip())


def main() -> None:
    emb.embed_one = fake_embed_one  # monkeypatch: deterministic, offline

    tmp = Path(tempfile.mkdtemp(prefix="solmem_emb_"))
    app = SolMem(Store(home=tmp, git=False))

    a = app.record("wallet withdrawal TOCTOU race condition",
                   "use SELECT FOR UPDATE row lock", project="bank")
    b = app.record("UnicodeEncodeError cp1252 windows arrows",
                   "reconfigure stdout utf-8", project="bank")
    assert "semantic" in a and "semantic" in b, (a, b)
    print("record A ->", a)
    print("record B ->", b)

    # Query shares NO keywords with record A's problem.
    query = "concurrent debit double spend safety"
    from solmem.records import keywords_of
    overlap = set(keywords_of(query)) & set(keywords_of("wallet withdrawal TOCTOU race condition"))
    assert not overlap, f"test bug: queries share keywords {overlap}"
    print(f"\nquery: {query!r}  (keyword overlap with record A: {len(overlap)})")

    hit = app.recall(query, project="bank", limit=2)
    first = [ln for ln in hit.splitlines() if ln.startswith("#1")][0]
    # record A's solution should be the top hit, found via embedding, not keywords
    assert "SELECT FOR UPDATE" in hit, hit
    a_id = a.split()[2]
    assert f"id={a_id}" in first, (first, a_id)
    print("recall #1 ->", first.strip())
    print("\nPASS — semantic recall surfaced the right record with zero keyword overlap.")
    print(f"(store: {tmp})")


if __name__ == "__main__":
    main()
