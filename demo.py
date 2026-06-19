"""End-to-end demo of the self-modifying memory-routing AI.

Run:  py demo.py

Scenario A — guarded evolution
    Seed the router with a deliberately *bad* genome (over-trusts recency, barely
    uses similarity) and let it self-modify. Through budget-bounded edits that
    must clear all five pipeline stages, it discovers that similarity + importance
    are what actually drive good retrieval -- the world's hidden utility.

Scenario B — value-drift detection & correction
    Take the converged system and corrupt its genome (simulating accumulated
    drift / tampering). Watch it detect that production backslid below the charter
    floor and auto-correct, in bounded steps, back toward the golden reference.
"""

from __future__ import annotations

import sys

# Windows consoles default to cp1252; force UTF-8 so the trace symbols render.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import os
from pathlib import Path

from selfmod import WeightVector
from selfmod.dotenv import load_dotenv
from selfmod.orchestrator import SelfModifyingMemoryAI
from selfmod.pipeline import Stage

# Load a .env (project root or selfmod/) so MISTRAL_API_KEY can live in a file.
# A real shell export still wins over the file.
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env", _HERE / "selfmod" / ".env")

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def stage_line(events) -> str:
    parts = []
    for e in events:
        if e.stage is Stage.ROLLBACK:
            continue
        mark = f"{GREEN}✓{RESET}" if e.passed else f"{RED}✗{RESET}"
        parts.append(f"{e.stage.value}{mark}")
    return " → ".join(parts)


def print_cycle(log) -> None:
    promoted = log.result.promoted
    is_drift = log.result.events and "DRIFT" in log.result.events[0].detail
    tag = f"{GREEN}PROMOTED{RESET}" if promoted else f"{RED}{log.result.final_stage.value:9}{RESET}"
    note = f" {BOLD}[drift-correction]{RESET}" if is_drift else ""
    print(f"cycle {log.cycle:2}  outcome={log.production_metric:.3f}  {tag}{note}")
    print(f"        {DIM}{stage_line(log.result.events)}{RESET}")
    for e in log.result.events:
        if not e.passed and e.stage is not Stage.ROLLBACK:
            print(f"        {DIM}{e.detail}{RESET}")
            break


def scenario_a() -> SelfModifyingMemoryAI:
    bad = WeightVector({
        "similarity": 0.10, "recency": 0.45, "frequency": 0.20,
        "importance": 0.05, "salience": 0.20,
    })
    ai = SelfModifyingMemoryAI(initial=bad, seed=7)

    print(f"{BOLD}══ Scenario A — guarded evolution ══{RESET}")
    if ai.resumed:
        best = ai.golden.best()
        print(f"{GREEN}resumed from state.json{RESET}  "
              f"(cycle offset {ai.cycle_offset}, "
              f"best golden {best.metric:.3f})" if best else f"(cycle offset {ai.cycle_offset})")
    print(f"genome now:    {ai.production.pretty()}")
    print(f"outcome now:   {ai.measure():.3f} (NDCG@5)\n")

    promotions = 0
    for c in range(1, 19):
        log = ai.run_cycle(c)
        promotions += log.result.promoted
        print_cycle(log)

    print(f"\n{BOLD}Scenario A summary{RESET}")
    print(f"  promotions:    {promotions}/18")
    print(f"  final genome:  {ai.production.pretty()}")
    print(f"  final outcome: {ai.measure():.3f} (NDCG@5)")
    best = ai.golden.best()
    print(f"  best golden:   {best.metric:.3f} @ cycle {best.cycle}")
    print(f"  {ai.rejected.summary()}")
    print(f"{DIM}  → world rewards similarity (0.7) + importance (0.3); "
          f"the genome leaned on exactly those.{RESET}\n")
    return ai


def scenario_b(ai: SelfModifyingMemoryAI) -> None:
    print(f"{BOLD}══ Scenario B — value-drift detection & correction ══{RESET}")
    # Simulate accumulated drift / tampering: shove weight back onto a distractor.
    corrupted = WeightVector({
        "similarity": 0.20, "recency": 0.55, "frequency": 0.10,
        "importance": 0.10, "salience": 0.05,
    }).normalized()
    ai.production = corrupted
    print(f"{RED}injected drift:{RESET} {corrupted.pretty()}")
    print(f"degraded outcome: {ai.measure():.3f} "
          f"(charter floor ≈ best − {ai.drift.charter.regression_band:.2f})\n")

    for c in range(19, 27):
        log = ai.run_cycle(c)
        print_cycle(log)

    print(f"\n{BOLD}Scenario B summary{RESET}")
    print(f"  recovered genome:  {ai.production.pretty()}")
    print(f"  recovered outcome: {ai.measure():.3f} (NDCG@5)")
    print(f"{DIM}  → drift detected, then pulled back toward the golden reference "
          f"in budget-bounded steps.{RESET}")


def main() -> None:
    if os.environ.get("MISTRAL_API_KEY"):
        model = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
        print(f"{BOLD}proposal author:{RESET} Mistral ({model}) — live API calls per cycle\n")
    else:
        print(f"{BOLD}proposal author:{RESET} offline numeric engine "
              f"(set MISTRAL_API_KEY to use Mistral)\n")
    ai = scenario_a()
    scenario_b(ai)


if __name__ == "__main__":
    main()
