"""Smoke test: store round-trip + a full in-process MCP handshake.

Run:  py test_solmem.py
Uses a throwaway store dir so it never touches the real team memory.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from solmem.server import SolMem, Server
from solmem.store import Store


def rpc(server: Server, method: str, params=None, mid=1):
    msg = {"jsonrpc": "2.0", "method": method}
    if mid is not None:
        msg["id"] = mid
    if params is not None:
        msg["params"] = params
    return server.handle(msg)


def call(server: Server, tool: str, args: dict, mid=1):
    resp = rpc(server, "tools/call", {"name": tool, "arguments": args}, mid)
    return resp["result"]["content"][0]["text"]


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="solmem_test_"))
    app = SolMem(Store(home=tmp, git=False))
    server = Server(app)
    ok = True

    # 1. handshake
    init = rpc(server, "initialize", {"protocolVersion": "2025-06-18",
                                      "capabilities": {}, "clientInfo": {"name": "test"}})
    assert init["result"]["serverInfo"]["name"] == "solmem", init
    print("initialize       ->", init["result"]["serverInfo"])

    # notification: no response
    assert rpc(server, "notifications/initialized", {}, mid=None) is None
    print("initialized note -> (no response, correct)")

    # 2. tools/list
    tools = rpc(server, "tools/list")["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {"recall", "record", "record_negative", "supersede",
                     "report_outcome", "report_regression", "explain_recall",
                     "validate_applicability", "evolve", "show_brain", "rollback_brain",
                     "stats", "metrics", "readiness", "sync"}, names
    print("tools/list       ->", sorted(names))

    # 3. recall on empty store
    empty = call(server, "recall", {"problem": "anything"})
    assert "No matching memory" in empty, empty
    print("recall (empty)   -> ok")

    # 4. record two solutions
    r1 = call(server, "record", {
        "problem": "UnicodeEncodeError cp1252 when printing arrows on Windows",
        "solution": "sys.stdout.reconfigure(encoding='utf-8') at program start",
        "project": "demo", "tags": ["windows", "encoding"]})
    print("record #1        ->", r1)
    r2 = call(server, "record", {
        "problem": "ModuleNotFoundError: No module named requests",
        "solution": "pip install requests, or use urllib from stdlib",
        "project": "demo", "tags": ["python", "deps"]})
    print("record #2        ->", r2)

    # 5. recall should surface the encoding record for an encoding-ish query
    hit = call(server, "recall", {
        "problem": "printing unicode arrows throws UnicodeEncodeError on windows console"})
    assert "reconfigure" in hit, hit
    rec_id = [ln for ln in hit.splitlines() if ln.startswith("#1")][0].split("id=")[1].split()[0]
    print("recall (hit)     -> #1 id =", rec_id)

    # 6. report outcome -> brain learns (success_rate moves)
    before = app.store.get(rec_id).success_rate
    out = call(server, "report_outcome", {"id": rec_id, "worked": True})
    after = app.store.get(rec_id).success_rate
    assert after > before, (before, after)
    print("report_outcome   ->", out)

    # 7. persistence: a fresh Store on the same dir sees the records
    reopened = Store(home=tmp, git=False)
    assert len(reopened.all()) == 2, len(reopened.all())
    assert reopened.get(rec_id).successes == 1
    print("persistence      -> 2 records survive reopen, outcome stuck")

    # 8. evolve with too little data -> graceful no-op, not a crash
    evo = call(server, "evolve", {})
    assert "not enough graded recalls" in evo, evo
    print("evolve (sparse)  -> graceful:", evo.split(".")[0])

    # 9. Phase 1: secret scanning blocks credentials
    blocked = call(server, "record", {
        "problem": "configure aws client",
        "solution": "set aws_secret_access_key=AKIAIOSFODNN7EXAMPLEabcd1234567890wXyZ12 then run",
        "project": "demo"})
    assert "REJECTED" in blocked and len(app.store.all()) == 2, blocked
    print("secret scan      -> blocked credential record (store still 2)")

    # 10. Phase 1: regression sharply lowers trust
    t_before = app.store.get(rec_id).trust
    reg = call(server, "report_regression", {"id": rec_id, "severity": "high",
                                             "reason": "broke prod later"})
    t_after = app.store.get(rec_id).trust
    assert t_after < t_before and app.store.get(rec_id).regressions == 1, (t_before, t_after)
    print("report_regression->", reg.split(".")[0])

    # 11. Phase 1: brain versioning + rollback round-trip
    from selfmod.weights import WeightVector as WV
    base_w = app.store.load_brain()
    v1 = app.store.promote_brain(base_w, {"note": "v1"})
    bumped = WV({**base_w.values, "similarity": base_w.values["similarity"] + 0.05}).normalized()
    v2 = app.store.promote_brain(bumped, {"note": "v2"})
    assert app.store.brain_version() == v2, app.store.brain_version()
    msg = call(server, "rollback_brain", {"target_version": v1, "reason": "test"})
    assert app.store.load_brain().values == base_w.values, msg
    print("brain versioning ->", v1, "→", v2, "→ rollback:", msg)

    # --- Phase 2 ---
    # 12. risk auto-classification on a critical-domain record
    crit = call(server, "record", {
        "problem": "wallet withdrawal double-spend under concurrency",
        "solution": "use SELECT FOR UPDATE row lock around the debit",
        "project": "bank",
        "does_not_apply_when": ["database is SQLite", "event-sourced balances"],
        "verification": ["run concurrency test", "verify isolation level"]})
    assert "risk=critical" in crit, crit
    crit_id = crit.split()[2]
    print("risk classify    ->", crit.split(".")[0])

    # 13. recall surfaces risk policy + verification + recall_event_id; explain works
    rhit = call(server, "recall", {"problem": "concurrent withdrawal overspend", "project": "bank"})
    assert "risk=critical" in rhit and "NEVER auto-apply" in rhit, rhit
    ev_id = rhit.split("recall_event_id=")[1].split(")")[0].strip()
    expl = call(server, "explain_recall", {"recall_event_id": ev_id})
    assert "top drivers" in expl, expl
    print("recall+risk      -> critical policy shown; explain_recall OK")

    # 14. applicability: a blocked context is flagged unsafe
    va = call(server, "validate_applicability", {"id": crit_id, "context": "we use SQLite here"})
    assert "safe_to_try=False" in va, va
    print("applicability    ->", va.splitlines()[0])

    # 15. negative memory: recall warns against a known-bad approach
    neg = call(server, "record_negative", {
        "problem": "wallet withdrawal race condition check then insert",
        "bad_solution": "check balance then insert withdrawal row",
        "why_failed": "TOCTOU race remains under concurrent withdrawals",
        "better_direction": "use SELECT FOR UPDATE or a reservation table",
        "project": "bank"})
    assert "NEGATIVE" in neg, neg
    warnhit = call(server, "recall", {"problem": "withdrawal race condition balance check", "project": "bank"})
    assert "KNOWN BAD APPROACH" in warnhit and "do instead" in warnhit, warnhit
    print("negative memory  -> recall warns against the footgun")

    # 15b. supersede: a better solution replaces an old record (non-destructive);
    #      the old one is archived and stops surfacing, the new one takes its place.
    old = call(server, "record", {"problem": "slow json parsing in a hot loop",
        "solution": "parse with json.loads per line", "project": "demo"})
    old_jid = old.split()[2]
    sup = call(server, "supersede", {"old_id": old_jid,
        "solution": "use orjson.loads — ~5x faster in the hot loop", "reason": "faster"})
    assert "Superseded" in sup, sup
    new_jid = sup.split("→")[1].split(".")[0].strip()
    assert app.store.get(old_jid).superseded_by == new_jid, "old not linked"
    assert not app.store.get(old_jid).active, "old should be inactive"
    assert app.store.get(new_jid).supersedes == old_jid, "new not back-linked"
    sr = call(server, "recall", {"problem": "slow json parsing hot loop", "project": "demo"})
    assert new_jid in sr and old_jid not in sr, sr
    print("supersede        -> old archived, new surfaces in its place")

    # 15c. provenance: records carry an author and unverified ones are flagged
    assert app.store.get(new_jid).author == app.store.current_author()
    assert "UNVERIFIED" in sr, "unverified provenance not surfaced"
    print("provenance       -> author stamped; unverified flagged in recall")

    # 15d. metrics: ROI/quality harness runs and reflects the graded hit
    mtext = call(server, "metrics", {})
    assert "hit-rate" in mtext and "top-1 success" in mtext, mtext
    from solmem import metrics as _metrics
    mvals = _metrics.compute(app.store)
    assert mvals["hits"] >= 1 and mvals["records"] == len(app.store.all()), mvals
    print("metrics          -> hit-rate/top-1/MRR computed; hits =", mvals["hits"])

    # 16. stats
    print("\nstats:\n" + call(server, "stats", {}))

    # 17. unknown tool -> proper JSON-RPC error
    err = rpc(server, "tools/call", {"name": "nope", "arguments": {}})
    assert err["error"]["code"] == -32602, err
    print("\nunknown tool     -> error", err["error"]["code"], "(correct)")

    print("\nALL OK  (store dir:", tmp, ")")


if __name__ == "__main__":
    main()
