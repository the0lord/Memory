# solmem — shared, self-improving solution memory for Claude Code

A team-wide memory of problem→solution pairs, delivered as an **MCP server** plus a
**thin skill**. When you solve a task, the solution is recorded; when a similar
problem reappears, it's recalled (faster, cheaper); and the recall ranking
self-improves from real outcomes using the `selfmod` engine.

```
problem appears ──► recall similar past solution ──► apply it (fast/cheap)
       │                       │
       │                  report_outcome(worked) ──► brain learns what to trust
       ▼                       │
   solve fresh ──► record(problem, solution) ──► team can recall it next time
```

## Architecture

| Piece | What it does |
|---|---|
| `solmem/records.py` | The `SolutionRecord` (problem, solution, outcomes, keywords). |
| `solmem/features.py` | The 5 router features (similarity/recency/frequency/importance/salience) + scoring. Same axes the `selfmod` engine evolves. |
| `solmem/store.py` | Git-backed store: one JSON file per record + per recall episode (conflict-free), `brain.json` = recall weights, `golden.json` / `rejected.json` = pipeline memory. |
| `solmem/evolve.py` | Wraps graded recall history as a selfmod `World` and runs the full proposal + 5-stage pipeline to self-modify the brain. |
| `solmem/privacy.py` | Secret scanner — refuses to store keys/tokens/DB-URLs in shared memory. |
| `solmem/embeddings.py` | Mistral semantic embeddings for `similarity` (keyword fallback). |
| `solmem/risk.py` | Deterministic risk classifier (low/medium/high/critical) + recall policy. |
| `solmem/applicability.py` | `applies_when` / `does_not_apply_when` matching against context. |
| `solmem/server.py` | Stdlib MCP server. Tools: `recall`, `record`, `record_negative`, `report_outcome`, `report_regression`, `explain_recall`, `validate_applicability`, `evolve`, `show_brain`, `rollback_brain`, `stats`, `sync`. |
| `.claude/skills/solmem/SKILL.md` | Teaches Claude the recall-first / record-after habit. |

The recall brain reuses the **selfmod** self-modification engine *verbatim*: the 5
weights are the genome, each graded recall is a labeled ranking example, and the
proposal engine + 5-stage pipeline (SANDBOX→SHADOW→GATED→MONITORED→PROMOTED, else
ROLLBACK) promote a re-weighting only if it improves held-out ranking. `demo_solmem.py`
shows a deliberately bad brain recovering held-out NDCG@5 from 0.64 → 1.00.

## V2 roadmap (safe)

Following the "don't grow the learned genome faster than outcome data supports"
principle: keep **5 learned weights**, add safety as **deterministic guards**, learn
slowly through replay.

| Phase | Scope | Status |
|---|---|---|
| 1 | Event logging + `brain_version` + **secret scanning** + brain versioning/rollback + regression reporting | **done** |
| 2 | `explain_recall`, applicability checks, risk levels, verification checklist, negative memory | **done** |
| 3 | selfmod replay bridge + shadow eval + promotion reports | done (via `evolve.py`) |
| 4 | Add learned features one at a time: exact_error_match → stack_match → regression_penalty → success_rate | later |
| 5 | Multi-brain routing — only once outcome volume justifies it | later |

North star: *recall fast; learn slow, tested, reversible, auditable; never store secrets.*

## Setup

1. **Choose where the shared memory lives.** For a team, make it a git repo everyone
   clones (records as individual JSON files merge cleanly). Point `SOLMEM_HOME` at it.
   For a solo trial, omit it and it defaults to `./solmem_store`.

2. **Register the server with Claude Code:**

   ```sh
   claude mcp add solmem --env SOLMEM_HOME=C:\path\to\team-memory -- py D:\Fun\Memory\solmem_server.py
   ```

   (Add `--scope user` to make it available in every project, or `--scope project` to
   share the registration via the repo's `.mcp.json`.)

3. **Make Claude use it automatically.** The skill nudges it, but the most reliable
   trigger is a line in your project `CLAUDE.md`:

   > Before starting any coding or debugging task, call `solmem.recall`. After solving,
   > `solmem.record` the solution, and `solmem.report_outcome` for any recalled fix you tried.

4. **Verify:** `/mcp` in Claude Code should list `solmem` with 5 tools.

## Test

```sh
py test_solmem.py        # store round-trip + full in-process MCP handshake
py demo_solmem.py        # brain self-modifies from simulated outcomes (NDCG climbs)
```

## Semantic recall

`similarity` uses Mistral embeddings (`mistral-embed`, 1024-dim) when
`MISTRAL_API_KEY` is set: the problem is embedded once at `record` time (stored on
the record) and the query is embedded once per `recall`. With no key or on any
network error it falls back to keyword cosine — no behavior change. So "can't
encode arrows" matches "UnicodeEncodeError" even with no shared words. Verified by
`test_embeddings.py`.

## Notes

- Zero dependencies — MCP and the Mistral embeddings client are hand-rolled over
  stdlib (urllib + JSON-RPC over stdio).
- `recall` is read-only; `report_outcome` is what moves a record's success rate.
- `sync` does `git pull --rebase --autostash` then `push` — run at a stopping point.
- Never record secrets; the store is meant to be committed and shared.
