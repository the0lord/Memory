"""Persist and restore the AI's learned state across sessions.

Saves to a single JSON file: current production genome, golden trace log,
rejected-edit buffer, and cycle count. On next startup the AI resumes where
it left off instead of restarting from the initial bad genome.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .buffers import GoldenTrace, GoldenTraceLog, RejectedEdit, RejectedEditBuffer
from .weights import FEATURES, WeightVector


STATE_FILE = Path(__file__).resolve().parent.parent / "state.json"


def save(
    production: WeightVector,
    golden: GoldenTraceLog,
    rejected: RejectedEditBuffer,
    cycle: int,
    path: Path = STATE_FILE,
) -> None:
    data = {
        "saved_at": time.time(),
        "cycle": cycle,
        "production": production.values,
        "golden_traces": [
            {"weights": t.weights.values, "metric": t.metric, "cycle": t.cycle, "ts": t.ts}
            for t in golden.traces
        ],
        "rejected_edits": [
            {"direction": e.direction, "stage": e.stage, "delta_metric": e.delta_metric,
             "note": e.note, "ts": e.ts}
            for e in rejected.items
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load(
    path: Path = STATE_FILE,
) -> tuple[WeightVector, GoldenTraceLog, RejectedEditBuffer, int] | None:
    """Return (production, golden, rejected, cycle) or None if no state file."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))

    production = WeightVector({f: float(data["production"][f]) for f in FEATURES})

    golden = GoldenTraceLog()
    for t in data.get("golden_traces", []):
        golden.traces.append(GoldenTrace(
            weights=WeightVector({f: float(t["weights"][f]) for f in FEATURES}),
            metric=t["metric"], cycle=t["cycle"], ts=t.get("ts", 0.0),
        ))

    rejected = RejectedEditBuffer()
    for e in data.get("rejected_edits", []):
        rejected.items.append(RejectedEdit(
            direction=e["direction"], stage=e["stage"],
            delta_metric=e["delta_metric"], note=e.get("note", ""),
            ts=e.get("ts", 0.0),
        ))

    return production, golden, rejected, int(data.get("cycle", 0))
