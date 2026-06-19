---
name: solmem
description: Use the team's shared solution memory. Invoke at the START of any coding/debugging task to recall how the team solved similar problems before, and at the END to record the solution and report whether a recalled fix worked. Keeps the team fast and consistent by reusing proven solutions instead of re-deriving them.
---

# solmem — shared, self-improving team memory

The `solmem` MCP server is the team's collective memory of problem→solution pairs.
It gets faster and cheaper over time: proven solutions are reused, and the recall
ranking self-improves from real outcomes. Use it on every non-trivial task.

## The loop — follow it every task

1. **Recall first.** Before solving, call `recall(problem, project?)` with a concise
   statement of the problem, error, or task. If a candidate clearly fits, apply it
   instead of solving from scratch — that's the whole point (faster, cheaper, consistent).

2. **Judge honestly.** Each candidate shows a `success` rate and how many times it was
   tried. A high score on a closely-matching problem is worth trusting; a low-similarity
   hit is just a hint. You don't have to use any of them.

3. **Report the outcome.** If you applied a recalled solution, call
   `report_outcome(id, worked)` once you know whether it actually fixed the problem.
   This is what teaches the brain which memories to trust. Determine `worked` by, in
   order of preference:
   - **tests / exit code** — ran the project's tests or the relevant command; pass = worked.
   - **self-assessment** — no tests available, but the change clearly resolves the issue.
   - **developer override** — the human says it did or didn't work.

4. **Record what's new.** After solving a task in a way worth remembering, call
   `record(problem, solution, project?, tags?)`. Write the `problem` the way it would
   reappear (the error text, the symptom, the task), and the `solution` as the concrete
   fix — enough that a teammate could apply it without context. Skip the trivial or the
   purely project-specific.

## When to use which tool

- `recall` — start of a task, or whenever stuck on a recognizable problem.
- `record` — after a real fix worth reusing.
- `report_outcome` — after trying a recalled solution (always, win or lose).
- `record_negative` — when you find a *known-bad* approach (a footgun). recall will
  warn against it for similar problems. Use for races, insecure patterns, etc.
- `supersede` — when you solved a problem *better* than a recalled record (faster,
  safer, more correct), call `supersede(old_id, solution)` instead of plain `record`.
  The old record is archived and the new one takes its place — this is how memory
  improves instead of getting stuck on a stale answer. recall is a prior, not a
  mandate: if you can do better, do it and supersede.
- `explain_recall` — pass the `recall_event_id` from a recall to see *why* candidates
  ranked as they did (per-feature drivers + warnings).
- `validate_applicability` — before applying a record on a risky task, check it against
  your current context (its applies_when / does_not_apply_when clauses).
- `report_regression` — if a previously-good solution later breaks something, report it;
  it sinks that record's rank fast.
- `stats` — to inspect store size, outcome tallies, and the current recall brain.
- `metrics` — to see whether recall is actually paying off: hit-rate, top-1 success,
  MRR from real graded history, store health, and a token-ROI estimate. Check this
  before deciding the memory is worth growing.
- `evolve` — occasionally (after a batch of graded recalls), to let the brain
  self-modify: it proposes a re-weighting and promotes it only if it improves
  held-out ranking. Safe to call anytime; it no-ops until there's enough signal.
- `sync` — to share local memory with the team (git pull --rebase then push). Run it
  at a natural stopping point, not mid-edit.

## Risk-aware use (important)

recall shows a `risk` level per candidate. Respect it:

- **low** — safe to suggest directly.
- **medium** — suggest, but run the listed checks.
- **high** — run the verification checklist before applying.
- **critical** (withdrawals, custody, signing keys, migrations, auth) — **never
  auto-apply**. Surface it, run every required check, and get explicit human
  confirmation first. When recording such solutions, set `risk` and fill
  `verification` and `does_not_apply_when`.

## Good hygiene

- Phrase `problem` for *future recall*: include the error class, key symbols, the
  framework — the words a teammate would search with.
- Keep `solution` actionable and self-contained.
- Always `report_outcome` after using a recall — a recall you never grade teaches nothing.
- Don't record secrets, credentials, or anything you wouldn't commit to a shared repo.
