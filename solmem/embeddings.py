"""Semantic embeddings for recall similarity, via Mistral (stdlib HTTP only).

Records are embedded once at record-time and the vector is stored on the record;
a query is embedded once per recall. Similarity then becomes the cosine of those
vectors instead of mere keyword overlap, so "can't encode arrows" matches
"UnicodeEncodeError" even with no shared tokens.

Everything degrades gracefully: no MISTRAL_API_KEY, no network, or any API error
returns None and the caller falls back to keyword cosine. The key is read only
from the environment, never written anywhere.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request

MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"
DEFAULT_MODEL = "mistral-embed"


def available() -> bool:
    return bool(os.environ.get("MISTRAL_API_KEY"))


def embed_texts(texts: list[str], *, model: str | None = None,
                timeout: float = 15.0) -> list[list[float]] | None:
    """Embed a batch of texts. Returns one vector per text, or None on any failure."""
    key = os.environ.get("MISTRAL_API_KEY")
    if not key or not texts:
        return None
    payload = {"model": model or os.environ.get("MISTRAL_EMBED_MODEL", DEFAULT_MODEL),
               "input": texts}
    req = urllib.request.Request(
        MISTRAL_EMBED_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        rows = sorted(body["data"], key=lambda d: d.get("index", 0))
        return [r["embedding"] for r in rows]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, OSError):
        return None


def embed_one(text: str) -> list[float] | None:
    vecs = embed_texts([text])
    return vecs[0] if vecs else None


def cosine(a: list[float], b: list[float]) -> float:
    """Raw cosine clamped to [0, 1]. Text embeddings almost never go negative, so
    clamping (rather than remapping [-1,1]->[0,1]) keeps the full discriminative
    spread instead of compressing everything into a narrow band near the top."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))
