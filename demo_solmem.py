"""Demo: the recall brain self-modifies from real outcomes.

We seed a throwaway team store with solutions on several distinct topics, then
simulate developers recalling and reporting outcomes where the *keyword-similar*
solution is the one that actually works. The brain starts deliberately BAD
(over-trusting recency, barely using similarity). Through the same 5-stage
pipeline the selfmod engine uses, it should discover that similarity is what
predicts a working recall -- and its held-out ranking quality should climb.

Run:  py demo_solmem.py
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from selfmod.weights import WeightVector
from solmem.evolve import build_episodes, evolve
from solmem.server import SolMem
from solmem.store import Store

BOLD, DIM, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"

# Eight topics; each record's solution is the right answer when the query is about
# that topic. The only signal that generalizes across topics is keyword similarity.
TOPICS = {
    "encoding":   ("UnicodeEncodeError cp1252 windows console arrows", "reconfigure stdout encoding utf-8"),
    "imports":    ("ModuleNotFoundError no module named requests python", "pip install requests or use urllib stdlib"),
    "asyncio":    ("asyncio event loop already running jupyter", "use nest_asyncio or await directly in notebook"),
    "docker":     ("docker compose port already allocated bind", "stop conflicting container or change host port mapping"),
    "git":        ("git rebase conflict detached head recover", "git reflog then git checkout the lost commit sha"),
    "sql":        ("postgres connection pool exhausted timeout", "raise pool max size and close sessions in finally"),
    "regex":      ("python regex catastrophic backtracking slow", "avoid nested quantifiers use atomic group or rewrite"),
    "cors":       ("browser cors preflight blocked api fetch", "add access-control-allow-origin header on server"),
}


def seed_store(app: SolMem) -> dict[str, str]:
    ids = {}
    for topic, (problem, solution) in TOPICS.items():
        msg = app.record(problem, solution, project="demo", tags=[topic])
        ids[topic] = msg.split()[2]  # "Recorded solution <id> (demo). ..."
    return ids


def simulate_traffic(app: SolMem, ids: dict[str, str], episodes: int, rng: random.Random) -> None:
    topics = list(TOPICS)
    for _ in range(episodes):
        topic = rng.choice(topics)
        problem, _ = TOPICS[topic]
        # query = the topic's keywords, lightly shuffled/truncated (real queries vary)
        words = problem.split()
        rng.shuffle(words)
        query = " ".join(words[: rng.randint(4, len(words))])
        app.recall(query, project="demo", limit=5)
        # ground truth: the matching topic's solution is the one that works
        worked = rng.random() > 0.12  # 12% of the time the "right" one still fails
        app.report_outcome(ids[topic], worked)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="solmem_evo_"))
    app = SolMem(Store(home=tmp, git=False))

    # Start from a deliberately bad brain: trusts recency, ignores similarity.
    bad = WeightVector({"similarity": 0.08, "recency": 0.45, "frequency": 0.20,
                        "importance": 0.07, "salience": 0.20}).normalized()
    app.store.save_brain(bad, commit=False)

    print(f"{BOLD}══ solmem — recall brain self-modification ══{RESET}")
    print(f"start brain:  {bad.pretty()}")
    print(f"{DIM}(bad on purpose: recency-heavy, similarity-blind){RESET}\n")

    ids = seed_store(app)
    rng = random.Random(1)
    simulate_traffic(app, ids, episodes=60, rng=rng)
    print(f"simulated 60 recall+outcome episodes; "
          f"{len(build_episodes(app.store))} are usable training examples.\n")

    print(f"{BOLD}evolving:{RESET}")
    for cycle in range(1, 11):
        report = evolve(app.store, seed=cycle)
        if not report.ran:
            print(f"  cycle {cycle:2}  {DIM}{report.detail}{RESET}")
            continue
        tag = f"{GREEN}PROMOTED{RESET}" if report.promoted else f"{RED}rejected{RESET}"
        arrow = f"{report.before:.3f} → {report.after:.3f}" if report.promoted else f"{report.before:.3f}"
        print(f"  cycle {cycle:2}  {tag}  held-out NDCG@5 {arrow}")

    final = app.store.load_brain()
    print(f"\n{BOLD}result{RESET}")
    print(f"  final brain:  {final.pretty()}")
    print(f"  similarity weight: {bad.values['similarity']:+.3f} → {final.values['similarity']:+.3f}")
    print(f"{DIM}  → the brain learned that keyword similarity predicts a working "
          f"recall, exactly the hidden truth of the simulated traffic.{RESET}")
    print(f"\n{DIM}store: {tmp}{RESET}")


if __name__ == "__main__":
    main()
