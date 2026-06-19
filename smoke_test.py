"""One-call smoke test — verifies the Mistral key works before the full demo.

Uses mistral-small-latest (cheapest, ~20 output tokens) so free-tier cost is
minimal. Prints pass/fail and token usage without revealing the key.

Run:  py smoke_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
from selfmod.dotenv import load_dotenv
load_dotenv(HERE / ".env", HERE / "selfmod" / ".env")

from selfmod.llm import MistralProposer
from selfmod.router import MemoryRouter
from selfmod.weights import EditBudget, WeightVector
from selfmod.workload import World
from selfmod.buffers import RejectedEditBuffer

key = os.environ.get("MISTRAL_API_KEY")
if not key:
    print("FAIL — MISTRAL_API_KEY not set (check selfmod/.env)")
    sys.exit(1)

print(f"key:   {'*' * (len(key) - 4)}{key[-4:]}  (length {len(key)})")
print("model: mistral-small-latest  (free-tier / cheapest)\n")

router  = MemoryRouter()
budget  = EditBudget()
world   = World(seed=7)
base    = WeightVector({"similarity":0.2,"recency":0.4,"frequency":0.15,
                        "importance":0.1,"salience":0.15}).normalized()
replay  = world.batch(40, "real")
rej     = RejectedEditBuffer()

p = MistralProposer(router, budget, model="mistral-small-latest", temperature=0.1)

# monkey-patch to capture token usage from the raw response
_usage: dict = {}
_orig_call = p._call
def _call_with_usage(prompt: str) -> str:
    import json, urllib.request
    payload = {
        "model": "mistral-small-latest",
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system",  "content": p.SYSTEM_PROMPT if hasattr(p, "SYSTEM_PROMPT") else "Reply with JSON only."},
            {"role": "user",    "content": prompt},
        ],
    }
    # import the constant from the module
    from selfmod.llm import SYSTEM_PROMPT, MISTRAL_URL
    payload["messages"][0]["content"] = SYSTEM_PROMPT
    req = urllib.request.Request(
        MISTRAL_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    import urllib.error
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    _usage.update(body.get("usage", {}))
    return body["choices"][0]["message"]["content"]

p._call = _call_with_usage

print("making ONE API call...")
prop = p.propose(base, replay, rej)
fell_back = prop.rationale.startswith("[mistral fallback")

if fell_back:
    print(f"FAIL — call failed, fell back to numeric engine")
    print(f"  detail: {prop.rationale}")
    sys.exit(1)

print("PASS ✓")
print(f"\nproposal:")
print(f"  base:      {base.pretty()}")
print(f"  proposed:  {prop.proposed.pretty()}")
print(f"  distance:  {base.distance(prop.proposed):.4f}  (budget max {budget.max_l2})")
print(f"  rationale: {prop.rationale}")
if _usage:
    print(f"\ntoken usage:")
    print(f"  prompt:     {_usage.get('prompt_tokens', '?')}")
    print(f"  completion: {_usage.get('completion_tokens', '?')}")
    print(f"  total:      {_usage.get('total_tokens', '?')}")
