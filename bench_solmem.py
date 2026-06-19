"""Interference benchmark (MINTEval-style): does recall degrade as memory grows?

We plant one correct solution per topic, then bury the store under increasing
piles of distractor records that partially share vocabulary (the interference).
For each store size we ask each topic's query and measure whether the planted
record still surfaces. A memory that "remembers everything" is useless if the
right record sinks as the store grows — this measures exactly that.

Offline & deterministic: uses keyword similarity (no API), so it reports the
floor. Semantic embeddings would only lift these numbers. Run: py bench_solmem.py
"""

from __future__ import annotations

import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from solmem.features import DEFAULT_WEIGHTS, rank
from solmem.records import SolutionRecord

VOCAB = [f"tok{i:03d}" for i in range(400)]      # shared token pool
TOPICS = 40                                      # distinct problems planted
WORDS_PER_TOPIC = 8
QUERY_KEEP = 0.6                                 # fraction of topic words the query reuses
SIZES = [TOPICS, 100, 250, 500, 1000]           # growing store (interference)


def _doc(words: list[str]) -> str:
    return "error in " + " ".join(words) + " module failing at runtime"


def build(seed: int = 11):
    rng = random.Random(seed)
    topics = []          # (planted_record, query_text)
    topic_words = []
    for t in range(TOPICS):
        words = rng.sample(VOCAB, WORDS_PER_TOPIC)
        topic_words.append(words)
        planted = SolutionRecord.create(_doc(words), f"fix for topic {t}", project="bench")
        keep = max(2, int(WORDS_PER_TOPIC * QUERY_KEEP))
        q_words = rng.sample(words, keep) + rng.sample(VOCAB, 1)  # partial + 1 noise
        topics.append((planted, _doc(q_words)))

    # The realistic interference: the store fills with *near-duplicate* problems —
    # each distractor is some topic's words with a couple swapped out. These are
    # the confusable wrong records that can bury the planted answer as N grows.
    distractors = []
    for i in range(max(SIZES)):
        w = rng.choice(topic_words)[:]
        for _ in range(rng.randint(2, 3)):       # swap 2-3 words -> confusable, but wrong
            w[rng.randrange(len(w))] = rng.choice(VOCAB)
        distractors.append(SolutionRecord.create(_doc(w), f"near-dup {i}", project="bench"))
    return topics, distractors


def evaluate(topics, pool) -> tuple[float, float, float]:
    top1 = hit5 = 0
    rr = 0.0
    for planted, query in topics:
        ranked = rank(DEFAULT_WEIGHTS, query, pool, limit=len(pool))
        order = [r.id for r, _, _ in ranked]
        pos = order.index(planted.id) if planted.id in order else len(order)
        if pos == 0:
            top1 += 1
        if pos < 5:
            hit5 += 1
        rr += 1.0 / (pos + 1)
    n = len(topics)
    return top1 / n, hit5 / n, rr / n


def main() -> None:
    topics, distractors = build()
    planted = [p for p, _ in topics]

    print("Interference benchmark — keyword similarity floor "
          f"({TOPICS} topics, query reuses {QUERY_KEEP:.0%} of words)\n")
    print(f"{'store size':>10} {'records':>8} {'top-1':>7} {'hit@5':>7} {'MRR':>7}")
    print("  " + "-" * 44)

    first = last = None
    for n in SIZES:
        pool = planted + distractors[: max(0, n - TOPICS)]
        top1, hit5, mrr = evaluate(topics, pool)
        print(f"{n:>10} {len(pool):>8} {top1:>6.0%} {hit5:>6.0%} {mrr:>7.3f}")
        if first is None:
            first = top1
        last = top1

    print()
    drop = (first - last)
    if last >= 0.6:
        verdict = (f"ROBUST — top-1 holds at {last:.0%} even at {SIZES[-1]} records "
                   f"(down {drop:.0%} from {first:.0%} at {SIZES[0]}).")
    else:
        verdict = (f"DEGRADES — top-1 fell to {last:.0%} at {SIZES[-1]} records "
                   f"(from {first:.0%}). Stronger similarity (embeddings) or scoping "
                   f"by project/stack would help.")
    print(verdict)
    print("\nThis is the keyword floor; Mistral embeddings raise it. Real token ROI "
          "comes from the `metrics` tool on live usage.")


if __name__ == "__main__":
    main()
