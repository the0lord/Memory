"""A minimal, dependency-free MCP server exposing the shared solution memory.

MCP's stdio transport is just newline-delimited JSON-RPC 2.0: read one JSON
object per line from stdin, write one per line to stdout. We implement only the
handful of methods Claude Code needs -- initialize, tools/list, tools/call, ping
-- and keep *all* diagnostics on stderr so stdout stays a clean protocol channel.

Run as:  py -m solmem.server      (registered via `claude mcp add`)
"""

from __future__ import annotations

import json
import sys

from selfmod.weights import FEATURES

from . import applicability, embeddings, metrics, privacy, risk
from .evolve import build_episodes, evolve as run_evolve
from .features import rank
from .records import SolutionRecord
from .store import Store

SERVER_NAME = "solmem"
SERVER_VERSION = "0.1.0"
DEFAULT_PROTOCOL = "2025-06-18"
_LOG_POOL = 12  # candidates to log per recall (>= shown) for richer training data

# Brain-growth readiness thresholds (the "when to add a 6th learned weight" gates).
# A linear ranker needs ~10-20 outcome-labeled examples per weight to be determined;
# we use the floor to judge the current weights and the midpoint to admit a new one.
_PER_WEIGHT_MIN = 10        # graded recalls/weight below which current weights are under-determined
_PER_WEIGHT_TARGET = 15     # graded recalls/weight of support a NEW weight needs before trialing
_PLATEAU_WINDOW = 3         # promotions to look back over for the NDCG slope
_PLATEAU_EPS = 0.01         # NDCG gain per promotion below this counts as "flat" (saturated)
# Fixed order to promote deterministic signals into learned weights (roadmap Phase 4).
_FUTURE_FEATURES = ["exact_error_match", "stack_match", "regression_penalty", "success_rate"]


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Tool definitions: schema (advertised to Claude) + handler.
# --------------------------------------------------------------------------- #
class SolMem:
    def __init__(self, store: Store | None = None):
        self.store = store or Store()

    # -- tool: recall ---------------------------------------------------
    def recall(self, problem: str, project: str = "", limit: int = 5) -> str:
        records = self.store.all()
        if project:
            scoped = [r for r in records if r.project == project]
            records = scoped or records  # fall back to global if project is empty
        if not records:
            return "No matching memory yet. Solve it fresh, then `record` the solution."

        weights = self.store.load_brain()
        brain_version = self.store.brain_version()
        query_vec = embeddings.embed_one(problem)  # None offline -> keyword fallback

        # Only active (non-superseded) records surface; a superseded record stays
        # on disk for audit but is replaced in recall by the version that overtook it.
        positives = [r for r in records if r.kind != "negative" and r.active]
        negatives = [r for r in records if r.kind == "negative" and r.active]

        # Rank a wider pool than we show: the extra candidates (with the exact
        # features they were scored on) become the negatives/positives the brain
        # learns to rank, even when a weak brain buries the right answer.
        ranked = rank(weights, problem, positives, limit=max(limit, _LOG_POOL),
                      query_vec=query_vec)
        rid = self.store.log_recall(problem, project,
                                    [(r.id, feats) for r, _, feats in ranked],
                                    brain_version=brain_version)

        top = [(r, s, f) for r, s, f in ranked if s > 0.0][:limit] or ranked[:1]

        lines = [f"{len(top)} candidate solution(s) from team memory "
                 f"(brain {brain_version}, recall_event_id={rid}):\n"]
        for i, (r, s, feats) in enumerate(top, 1):
            warn = f" ⚠ {r.regressions} regression(s)" if r.regressions else ""
            unverified = " ⚠ UNVERIFIED" if not r.verified else ""
            lines.append(
                f"#{i}  id={r.id}  score={s:.3f}  sim={feats['similarity']:.2f}  "
                f"trust={r.trust:.0%}  risk={r.risk}  (tried {r.uses}x)  "
                f"by {r.author or 'unknown'}{unverified}{warn}"
            )
            lines.append(f"    problem:  {r.problem}")
            lines.append(f"    solution: {r.solution}")
            if r.does_not_apply_when:
                lines.append(f"    ⛔ does NOT apply when: {'; '.join(r.does_not_apply_when)}")
            if risk.requires_checklist(r.risk):
                lines.append(f"    ⚠ {risk.policy(r.risk)}")
                if r.verification:
                    lines.append(f"    required checks: {'; '.join(r.verification)}")
            # Deterministic provenance guard: an unverified record on a risky task
            # could be unproven or poisoned (MINJA/MEXTRA, OWASP ASI06). Flag it.
            if not r.verified and risk.requires_checklist(r.risk):
                lines.append(f"    ⚠ unverified provenance (by {r.author or 'unknown'}, "
                             f"no confirmed successes) on a {r.risk}-risk record — confirm "
                             f"it's genuine before trusting; could be unproven or poisoned.")
            lines.append("")

        # Surface relevant known-bad approaches as warnings (never as solutions).
        if negatives:
            neg_ranked = rank(weights, problem, negatives, limit=2, query_vec=query_vec)
            shown = [(r, f) for r, _, f in neg_ranked if f["similarity"] >= 0.25]
            for r, f in shown:
                lines.append(f"⛔ KNOWN BAD APPROACH (id={r.id}): {r.solution}")
                if r.why_failed:
                    lines.append(f"    why it fails: {r.why_failed}")
                if r.better_direction:
                    lines.append(f"    do instead:   {r.better_direction}")
                lines.append("")

        lines.append("Apply with care: check applicability, run the required checks for "
                     "high/critical risk, then call report_outcome(id, worked).")
        lines.append(f"For a ranking breakdown: explain_recall(recall_event_id='{rid}').")
        return "\n".join(lines)

    # -- tool: record ---------------------------------------------------
    def record(self, problem: str, solution: str, project: str = "", tags=None,
               risk: str | None = None, applies_when=None, does_not_apply_when=None,
               verification=None) -> str:
        if not problem.strip() or not solution.strip():
            return "Both `problem` and `solution` are required."
        # Phase-1 safety gate: never persist secrets to shared memory.
        findings = privacy.scan(problem, solution)
        if findings:
            self.store.audit("secret_scan", {"action": "rejected",
                             "labels": [f.label for f in findings], "project": project})
            return privacy.report(findings)
        rec = SolutionRecord.create(
            problem, solution, project=project, tags=tags or [], risk=risk,
            applies_when=applies_when, does_not_apply_when=does_not_apply_when,
            verification=verification, author=self.store.current_author())
        vec = embeddings.embed_one(rec.problem)  # semantic match at recall time
        if vec:
            rec.embedding = vec
        self.store.put(rec)
        sem = "semantic" if vec else "keyword-only"
        return (f"Recorded solution {rec.id} ({rec.project or 'global'}, risk={rec.risk}, "
                f"{sem}). The team can recall it now.")

    # -- tool: record_negative -----------------------------------------
    def record_negative(self, problem: str, bad_solution: str, why_failed: str = "",
                        better_direction: str = "", project: str = "", tags=None) -> str:
        if not problem.strip() or not bad_solution.strip():
            return "Both `problem` and `bad_solution` are required."
        findings = privacy.scan(problem, f"{bad_solution} {better_direction}")
        if findings:
            self.store.audit("secret_scan", {"action": "rejected",
                             "labels": [f.label for f in findings], "project": project})
            return privacy.report(findings)
        rec = SolutionRecord.create(
            problem, bad_solution, project=project, tags=tags or [], kind="negative",
            why_failed=why_failed, better_direction=better_direction,
            author=self.store.current_author())
        vec = embeddings.embed_one(rec.problem)
        if vec:
            rec.embedding = vec
        self.store.put(rec)
        return (f"Recorded NEGATIVE memory {rec.id} ({rec.project or 'global'}). "
                f"recall will warn against this approach for similar problems.")

    # -- tool: supersede ------------------------------------------------
    def supersede(self, old_id: str, solution: str, problem: str = "",
                  reason: str = "", project: str = "", risk: str | None = None) -> str:
        """Replace a stale/worse solution with a better one, non-destructively.
        The old record is archived (kept for audit) but stops surfacing in recall;
        the new record links back to it. This is how a better fix overtakes an
        incumbent instead of competing with it forever (A-MEM / Zep / Memory-R1)."""
        old = self.store.get(old_id)
        if old is None:
            return f"No record with id={old_id} to supersede."
        if not solution.strip():
            return "`solution` (the better replacement) is required."
        problem = problem.strip() or old.problem
        findings = privacy.scan(problem, solution)
        if findings:
            self.store.audit("secret_scan", {"action": "rejected", "via": "supersede",
                             "labels": [f.label for f in findings]})
            return privacy.report(findings)
        new = SolutionRecord.create(
            problem, solution, project=project or old.project, tags=old.tags,
            risk=risk or old.risk, applies_when=old.applies_when,
            does_not_apply_when=old.does_not_apply_when, verification=old.verification,
            author=self.store.current_author())
        new.supersedes = old.id
        vec = embeddings.embed_one(new.problem)
        if vec:
            new.embedding = vec
        self.store.put(new)
        old.superseded_by = new.id
        old.superseded_at = new.created
        self.store.put(old)
        self.store.audit("supersessions", {"old": old.id, "new": new.id,
                         "reason": reason, "by": new.author})
        return (f"Superseded {old.id} → {new.id}. The old record is archived (kept for "
                f"audit) and {new.id} now surfaces in its place.")

    # -- tool: report_outcome ------------------------------------------
    def report_outcome(self, id: str, worked: bool, note: str = "") -> str:
        rec = self.store.get(id)
        if rec is None:
            return f"No record with id={id}."
        rec.uses += 1
        if worked:
            rec.successes += 1
        self.store.put(rec)
        # Attach the label to the recall episode that surfaced this record, so the
        # brain can later learn (via evolve) which ranking would have been right.
        self.store.attach_outcome(id, worked)
        verb = "reinforced" if worked else "penalized"
        return (
            f"Outcome logged: {id} {verb}. "
            f"Now success={rec.success_rate:.0%} over {rec.uses} tries."
        )

    # -- tool: report_regression ---------------------------------------
    def report_regression(self, id: str, reason: str = "", severity: str = "medium") -> str:
        rec = self.store.get(id)
        if rec is None:
            return f"No record with id={id}."
        rec.regressions += 1
        self.store.put(rec)
        self.store.audit("regressions", {"record_id": id, "severity": severity,
                                         "reason": reason, "brain_version": self.store.brain_version()})
        return (f"Regression logged for {id} (severity={severity}). "
                f"Trust dropped to {rec.trust:.0%}; it will rank far lower now.")

    # -- tool: explain_recall ------------------------------------------
    def explain_recall(self, recall_event_id: str) -> str:
        ep = self.store.get_recall(recall_event_id)
        if ep is None:
            return f"No recall episode with id={recall_event_id}."
        weights = self.store.weights_of_version(ep.get("brain_version", "brain_v0"))
        cands = ep.get("candidates", [])
        scored = sorted(
            ((c, sum(weights.values[f] * c["features"].get(f, 0.0) for f in FEATURES))
             for c in cands),
            key=lambda t: t[1], reverse=True)
        if not scored:
            return "Episode had no candidates."

        lines = [f"Ranking for recall {recall_event_id} under {ep.get('brain_version')}:",
                 f"query: {ep.get('query', '')}\n"]
        for rank_i, (c, total) in enumerate(scored[:3], 1):
            rec = self.store.get(c["id"])
            contribs = sorted(((f, weights.values[f] * c["features"].get(f, 0.0))
                               for f in FEATURES), key=lambda t: t[1], reverse=True)
            drivers = ", ".join(f"{f} {v:+.3f}" for f, v in contribs[:3])
            lines.append(f"#{rank_i} id={c['id']}  score={total:.3f}")
            lines.append(f"    top drivers: {drivers}")
            if rec:
                flags = []
                if rec.regressions:
                    flags.append(f"{rec.regressions} regression(s)")
                if c["features"].get("recency", 1.0) < 0.3:
                    flags.append("stale (old memory)")
                if risk.requires_checklist(rec.risk):
                    flags.append(f"{rec.risk} risk — verify first")
                if flags:
                    lines.append(f"    ⚠ warnings: {', '.join(flags)}")
        return "\n".join(lines)

    # -- tool: validate_applicability ----------------------------------
    def validate_applicability(self, id: str, context: str = "") -> str:
        rec = self.store.get(id)
        if rec is None:
            return f"No record with id={id}."
        return applicability.render(rec, applicability.check(rec, context))

    # -- tool: evolve ---------------------------------------------------
    def evolve(self) -> str:
        report = run_evolve(self.store)
        return report.detail

    # -- tool: rollback_brain ------------------------------------------
    def rollback_brain(self, target_version: str, reason: str = "") -> str:
        return self.store.rollback_brain(target_version, reason)

    # -- tool: show_brain ----------------------------------------------
    def show_brain(self) -> str:
        meta = self.store.brain_meta()
        w = self.store.load_brain()
        versions = [b.get("version", "?") for b in self.store.list_brains()]
        lines = [f"current: {meta.get('version', 'brain_v0')} "
                 f"(parent {meta.get('parent')})",
                 f"weights: {w.pretty()}"]
        if meta.get("promotion"):
            lines.append(f"promotion: {meta['promotion']}")
        lines.append(f"versions on disk: {', '.join(versions)}")
        return "\n".join(lines)

    # -- tool: stats ----------------------------------------------------
    def stats(self) -> str:
        records = self.store.all()
        w = self.store.load_brain()
        tried = sum(r.uses for r in records)
        won = sum(r.successes for r in records)
        top = sorted(records, key=lambda r: (r.uses, r.success_rate), reverse=True)[:5]
        lines = [
            f"records: {len(records)}   outcomes reported: {tried}   worked: {won}",
            f"brain:   {w.pretty()}",
            "most-used:",
        ]
        for r in top:
            lines.append(f"  {r.id}  {r.success_rate:.0%} over {r.uses}x  — {r.problem[:60]}")
        return "\n".join(lines) if records else "Store is empty."

    # -- tool: metrics --------------------------------------------------
    def metrics(self) -> str:
        return metrics.render(self.store)

    # -- tool: readiness ------------------------------------------------
    def readiness(self) -> str:
        """Is the brain ready to GROW (promote a deterministic signal into a new
        learned weight)? Checks the roadmap's three gates in order -- data coverage,
        NDCG saturation, headroom for one more -- and prints the numbers behind the
        verdict so the decision is a measurement, not a feeling."""
        nfeat = len(FEATURES)
        graded = len(build_episodes(self.store))
        per_weight = graded / nfeat if nfeat else 0.0

        metrics = [t.metric for t in sorted(self.store.load_golden().traces,
                                            key=lambda t: t.cycle)]
        promotions = len(metrics)
        latest = metrics[-1] if metrics else None
        if promotions >= 2:
            span = min(_PLATEAU_WINDOW, promotions - 1)
            slope = (metrics[-1] - metrics[-1 - span]) / span  # NDCG gain per promotion
        else:
            span, slope = 0, None

        lines = [
            "Brain growth readiness (roadmap: small now, grow only when data earns it)",
            f"  learned weights:        {nfeat}",
            f"  usable graded recalls:  {graded}  ({per_weight:.1f} per weight)",
            f"  brain promotions:       {promotions}"
            + (f"   latest held-out NDCG@5={latest:.3f}" if latest is not None else ""),
        ]
        if slope is not None:
            trend = "climbing" if slope > _PLATEAU_EPS else "flat"
            lines.append(f"  recent NDCG slope:      {slope:+.4f}/promotion "
                         f"({trend}, last {span})")

        # Gate 1 -- data coverage: are the weights we already have well-determined?
        if per_weight < _PER_WEIGHT_MIN:
            need = nfeat * _PER_WEIGHT_MIN - graded
            verdict = (f"NOT READY -- too little data. The current {nfeat} weights aren't "
                       f"well-determined yet (want >={_PER_WEIGHT_MIN}/weight). Collect "
                       f"~{need} more graded recalls before considering growth.")
        # Gate 2 -- saturation: a new weight only helps once the current ones plateau.
        elif slope is None:
            verdict = ("NOT READY -- the brain hasn't evolved enough to judge a plateau. "
                       "Keep using recall + report_outcome, then run evolve.")
        elif slope > _PLATEAU_EPS:
            verdict = ("NOT READY -- the current brain is still improving. Tune weights "
                       "(evolve), don't grow: a new feature would only add noise.")
        # Gate 3 -- headroom: plateaued AND enough data to support one more dimension.
        elif graded < (nfeat + 1) * _PER_WEIGHT_TARGET:
            need = (nfeat + 1) * _PER_WEIGHT_TARGET - graded
            verdict = (f"HOLD -- plateaued, but a new weight needs ~{_PER_WEIGHT_TARGET}/weight "
                       f"of support. Collect ~{need} more graded recalls, then re-check.")
        else:
            i = nfeat - 5  # how many features we've already added beyond the base 5
            nxt = _FUTURE_FEATURES[i] if 0 <= i < len(_FUTURE_FEATURES) else "(none queued)"
            verdict = (f"READY -- plateaued with enough data. Trial the next feature: "
                       f"{nxt}. Add it DETERMINISTICALLY first, log it on every recall, then "
                       f"shadow-replay; promote to a learned weight only if held-out NDCG@5 "
                       f"improves with no high-risk regression.")

        lines += ["", verdict]
        return "\n".join(lines)

    # -- tool: sync -----------------------------------------------------
    def sync(self) -> str:
        return self.store.sync()


# --------------------------------------------------------------------------- #
# JSON-RPC tool registry (name -> schema + bound handler)
# --------------------------------------------------------------------------- #
def build_tools(app: SolMem) -> dict[str, dict]:
    s = lambda **p: {"type": "object", "properties": p}
    return {
        "recall": {
            "description": "Search the team's shared memory for past solutions to a "
                           "similar problem. Call this BEFORE solving a task.",
            "inputSchema": s(
                problem={"type": "string", "description": "The problem / error / task."},
                project={"type": "string", "description": "Optional project name to scope to."},
                limit={"type": "integer", "description": "Max candidates (default 5)."},
            ) | {"required": ["problem"]},
            "handler": lambda a: app.recall(a["problem"], a.get("project", ""), int(a.get("limit", 5))),
        },
        "record": {
            "description": "Save a problem and the solution that worked, so the team "
                           "(and future you) can recall it. Call AFTER solving a task.",
            "inputSchema": s(
                problem={"type": "string"},
                solution={"type": "string"},
                project={"type": "string"},
                tags={"type": "array", "items": {"type": "string"}},
                risk={"type": "string", "description": "low|medium|high|critical (auto-inferred if omitted)"},
                applies_when={"type": "array", "items": {"type": "string"}},
                does_not_apply_when={"type": "array", "items": {"type": "string"}},
                verification={"type": "array", "items": {"type": "string"},
                              "description": "checks to run before trusting this fix"},
            ) | {"required": ["problem", "solution"]},
            "handler": lambda a: app.record(
                a["problem"], a["solution"], a.get("project", ""), a.get("tags"),
                a.get("risk"), a.get("applies_when"), a.get("does_not_apply_when"),
                a.get("verification")),
        },
        "record_negative": {
            "description": "Save a KNOWN-BAD approach so recall warns against it for "
                           "similar problems. Use for footguns (races, insecure patterns).",
            "inputSchema": s(
                problem={"type": "string"},
                bad_solution={"type": "string", "description": "the approach to avoid"},
                why_failed={"type": "string"},
                better_direction={"type": "string", "description": "what to do instead"},
                project={"type": "string"},
                tags={"type": "array", "items": {"type": "string"}},
            ) | {"required": ["problem", "bad_solution"]},
            "handler": lambda a: app.record_negative(
                a["problem"], a["bad_solution"], a.get("why_failed", ""),
                a.get("better_direction", ""), a.get("project", ""), a.get("tags")),
        },
        "supersede": {
            "description": "Replace a stale or worse solution with a better one. The old "
                           "record is archived (kept for audit) but stops surfacing; the new "
                           "one takes its place. Use when you found a better fix than a "
                           "recalled record — don't just add a competitor.",
            "inputSchema": s(
                old_id={"type": "string", "description": "id of the record being replaced"},
                solution={"type": "string", "description": "the better replacement solution"},
                problem={"type": "string", "description": "optional; defaults to the old record's problem"},
                reason={"type": "string"},
                project={"type": "string"},
                risk={"type": "string", "description": "low|medium|high|critical (else inherits the old record)"},
            ) | {"required": ["old_id", "solution"]},
            "handler": lambda a: app.supersede(
                a["old_id"], a["solution"], a.get("problem", ""), a.get("reason", ""),
                a.get("project", ""), a.get("risk")),
        },
        "report_outcome": {
            "description": "Tell the brain whether a recalled solution actually worked, "
                           "so it learns which memories to trust.",
            "inputSchema": s(
                id={"type": "string"},
                worked={"type": "boolean"},
                note={"type": "string"},
            ) | {"required": ["id", "worked"]},
            "handler": lambda a: app.report_outcome(a["id"], bool(a["worked"]), a.get("note", "")),
        },
        "report_regression": {
            "description": "Report that a recalled solution worked at first but later "
                           "caused a problem. Sharply lowers that record's future rank.",
            "inputSchema": s(
                id={"type": "string"},
                reason={"type": "string"},
                severity={"type": "string", "description": "low | medium | high | critical"},
            ) | {"required": ["id"]},
            "handler": lambda a: app.report_regression(a["id"], a.get("reason", ""), a.get("severity", "medium")),
        },
        "explain_recall": {
            "description": "Explain why a recall ranked its candidates the way it did: "
                           "per-feature score drivers + warnings (stale, regressions, risk).",
            "inputSchema": s(
                recall_event_id={"type": "string"},
            ) | {"required": ["recall_event_id"]},
            "handler": lambda a: app.explain_recall(a["recall_event_id"]),
        },
        "validate_applicability": {
            "description": "Check whether a record applies to the current context "
                           "(its applies_when / does_not_apply_when clauses) before using it.",
            "inputSchema": s(
                id={"type": "string"},
                context={"type": "string", "description": "current stack/situation, free text"},
            ) | {"required": ["id"]},
            "handler": lambda a: app.validate_applicability(a["id"], a.get("context", "")),
        },
        "evolve": {
            "description": "Self-modify the recall brain from accumulated outcomes: "
                           "propose a re-weighting and graduate it through the 5-stage "
                           "pipeline (promote only if it improves held-out ranking).",
            "inputSchema": s(),
            "handler": lambda a: app.evolve(),
        },
        "show_brain": {
            "description": "Show the current recall brain: version, parent, weights, "
                           "and the versions archived on disk.",
            "inputSchema": s(),
            "handler": lambda a: app.show_brain(),
        },
        "rollback_brain": {
            "description": "Roll the recall brain back to a previous archived version "
                           "(e.g. after a regression). Use show_brain to list versions.",
            "inputSchema": s(
                target_version={"type": "string", "description": "e.g. brain_v3"},
                reason={"type": "string"},
            ) | {"required": ["target_version"]},
            "handler": lambda a: app.rollback_brain(a["target_version"], a.get("reason", "")),
        },
        "stats": {
            "description": "Show store size, outcome tallies, the current recall brain, "
                           "and the most-used records.",
            "inputSchema": s(),
            "handler": lambda a: app.stats(),
        },
        "metrics": {
            "description": "Measure whether recall is paying off: hit-rate, top-1 success, "
                           "MRR from real graded history, plus store health and a token-ROI "
                           "estimate. Use this to decide if the memory is worth growing.",
            "inputSchema": s(),
            "handler": lambda a: app.metrics(),
        },
        "readiness": {
            "description": "Assess whether the recall brain is ready to GROW (promote a "
                           "signal into a new learned weight): checks data coverage, NDCG "
                           "saturation, and headroom, and prints the numbers behind the verdict.",
            "inputSchema": s(),
            "handler": lambda a: app.readiness(),
        },
        "sync": {
            "description": "Share local memory with the team: git pull --rebase then push.",
            "inputSchema": s(),
            "handler": lambda a: app.sync(),
        },
    }


# --------------------------------------------------------------------------- #
# JSON-RPC dispatch
# --------------------------------------------------------------------------- #
class Server:
    def __init__(self, app: SolMem | None = None):
        self.app = app or SolMem()
        self.tools = build_tools(self.app)

    def handle(self, msg: dict) -> dict | None:
        """Return a response dict for a request, or None for a notification."""
        method = msg.get("method")
        mid = msg.get("id")
        is_notification = "id" not in msg

        try:
            result = self._route(method, msg.get("params") or {})
        except _RpcError as e:
            return _err(mid, e.code, e.message)
        except Exception as e:  # never crash the loop on a bad tool call
            log("tool error:", repr(e))
            return _err(mid, -32603, f"internal error: {e}")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _route(self, method: str, params: dict):
        if method == "initialize":
            return {
                "protocolVersion": params.get("protocolVersion", DEFAULT_PROTOCOL),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [
                {"name": n, "description": t["description"], "inputSchema": t["inputSchema"]}
                for n, t in self.tools.items()
            ]}
        if method == "tools/call":
            name = params.get("name")
            tool = self.tools.get(name)
            if tool is None:
                raise _RpcError(-32602, f"unknown tool: {name}")
            text = tool["handler"](params.get("arguments") or {})
            return {"content": [{"type": "text", "text": text}], "isError": False}
        raise _RpcError(-32601, f"method not found: {method}")

    def serve(self, stdin=sys.stdin, stdout=sys.stdout) -> None:
        log(f"{SERVER_NAME} {SERVER_VERSION} ready  (home={self.app.store.home}, "
            f"git={self.app.store.git})")
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log("dropped non-JSON line")
                continue
            response = self.handle(msg)
            if response is not None:
                stdout.write(json.dumps(response) + "\n")
                stdout.flush()


class _RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _err(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    Server().serve()


if __name__ == "__main__":
    main()
