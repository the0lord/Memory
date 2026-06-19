# selfmod — a self-modifying memory-routing AI

A decision engine that **modifies its own decision weights** based on real-world
outcomes, where every self-edit must survive a **five-stage deployment pipeline**
before it reaches production. Built as a runnable, dependency-free Python
prototype (stdlib only).

The decision domain is **memory routing** — ranking which stored memories to
retrieve for a query — which ties into the memory-architecture research in
[`BLEEDING_EDGE.md`](BLEEDING_EDGE.md). The same skeleton generalizes to any
weighted-decision engine; only [`workload.py`](selfmod/workload.py) and the
feature set are domain-specific.

```
                       ┌─────────────┐
                       │   PROPOSAL   │ ← edit budget bounds magnitude
                       └──────┬───────┘
                  ┌───────────▼───────────┐
                  │   SANDBOX             │ ← narrow synthetic workload
                  └───────────┬───────────┘
                              │ pass
                  ┌───────────▼───────────┐
                  │   SHADOW              │ ← broad real workload, no user impact
                  └───────────┬───────────┘
                              │ pass (catches overfit the sandbox misses)
                  ┌───────────▼───────────┐
                  │   GATED               │ ← stakeholder consensus for major edits
                  └───────────┬───────────┘
                              │ approved
                  ┌───────────▼───────────┐
                  │   MONITORED           │ ← live, auto-rollback trigger
                  └───────────┬───────────┘
                              │ stable
                  ┌───────────▼───────────┐
                  │   PROMOTED            │ ← golden trace recorded
                  └───────────────────────┘

  Failure at any stage → ROLLBACK  (signal → rejected-edit buffer)
```

## Run it

```bash
py demo.py      # Windows
python demo.py  # elsewhere
```

No dependencies, no API keys, no network. Output is a colored per-cycle trace of
each proposal walking the pipeline.

### Optional: let Mistral author the proposals

By default proposals are generated numerically (offline). Set a key and the
proposal author becomes a live LLM — everything else (budget, pipeline, consensus,
drift) is unchanged:

```bash
export MISTRAL_API_KEY=...                 # from a .env / shell, never committed
export MISTRAL_MODEL=mistral-large-latest  # optional, this is the default
py demo.py
```

The model is handed the current weights, the measured per-weight NDCG@5
sensitivities, and the recently-failed directions, and returns a JSON weight delta
+ rationale. Any failure (no key, network, bad JSON) falls back to the numeric
engine, so the loop never breaks. Note: with a key set, each cycle makes a live API
call.

**Scenario A — guarded evolution.** Starts from a deliberately bad genome
(over-trusts `recency`, barely uses `similarity`) and self-modifies for 18
cycles. It climbs from NDCG@5 ≈ 0.40 to ≈ 0.96 and discovers weights close to the
world's hidden utility (`0.7·similarity + 0.3·importance`), driving the three
distractor features to ~zero — without ever being told what the answer is.

**Scenario B — value-drift detection & correction.** The converged genome is
corrupted (simulating accumulated drift / tampering). Production backslides below
the charter floor; the system detects it and pulls the genome back toward the
golden reference in budget-bounded steps until it recovers, then stops correcting.

## How the spec maps to the code

| Capability (from the brief)                          | Where it lives |
|------------------------------------------------------|----------------|
| Evolve decision weights from real-world outcomes     | [`proposals.py`](selfmod/proposals.py) — outcome-gradient estimated on a replay buffer of real interactions |
| Fork shadow copies to test before applying           | [`pipeline.py`](selfmod/pipeline.py) `SHADOW` — candidate scored on live traffic in parallel, zero user impact |
| 5-stage deployment pipeline                           | [`pipeline.py`](selfmod/pipeline.py) — `SANDBOX → SHADOW → GATED → MONITORED → PROMOTED` |
| Detect value drift & auto-correct toward the goal    | [`drift.py`](selfmod/drift.py) — adaptive charter floor + correction toward golden reference |
| Stakeholder feedback + consensus for major changes   | [`feedback.py`](selfmod/feedback.py) — `GATED` requires a consensus vote for major edits |
| Preserve signal from failed experiments              | [`buffers.py`](selfmod/buffers.py) — `RejectedEditBuffer`, used to damp risky proposals |
| Bound the magnitude of any single change             | [`weights.py`](selfmod/weights.py) — `EditBudget` (L2 cap, per-feature cap, max features changed) |

## Design notes

- **What's being modified.** The genome is five per-feature weights
  ([`FEATURES`](selfmod/weights.py)) the router scores memories with. That vector
  *is* the self-modifiable surface; everything else guards changes to it.

- **Outcomes, not labels.** Learning uses no ground-truth gradient. The proposal
  engine replays recent real interactions and estimates how each weight moves the
  outcome metric (NDCG@5) by finite differences — the same signal you'd get from
  production retrieval feedback.

- **Why SHADOW exists.** `SANDBOX` uses a *narrow* synthetic distribution; `SHADOW`
  uses the *broad* real one. An edit can win on the sandbox yet regress on live
  traffic — `SHADOW` is the stage that catches it (visible in the demo trace as a
  SANDBOX-pass / SHADOW-reject).

- **Value drift = backsliding, not not-yet-arrived.** The charter floor is
  adaptive: it tracks the best outcome ever promoted minus a regression band, so
  healthy upward learning is never mistaken for drift, while regression below a
  previously-achieved level is.

- **Nothing wasted.** Every rejected edit's direction and measured harm are kept;
  future proposals that resemble a known-bad direction are damped before they're
  ever budgeted.

## Module map

```
selfmod/
  weights.py       WeightVector (genome) + EditBudget + major-change test
  workload.py      simulated memory store + synthetic/real workloads w/ ground truth
  router.py        the decision engine being modified (scoring, ranking, NDCG@5)
  proposals.py     outcome-gradient proposal engine (budget-bounded, buffer-damped)
  llm.py           Mistral proposal author (pluggable, env-keyed, safe fallback)
  pipeline.py      the 5-stage deployment pipeline + rollback
  feedback.py      stakeholders + consensus for the GATED stage
  drift.py         charter, value-drift detection, correction toward golden
  buffers.py       rejected-edit buffer + golden-trace log
  orchestrator.py  SelfModifyingMemoryAI — the per-cycle loop tying it together
demo.py            runnable two-scenario demonstration
```

## LLM-authored proposals (implemented)

The proposal author is pluggable. `selfmod.llm.MistralProposer` conforms to the
same `propose(base, replay, rejected) -> Proposal` contract as the numeric engine,
and [`orchestrator.py`](selfmod/orchestrator.py) selects it automatically when
`MISTRAL_API_KEY` is present (`select_proposer`). To use a different provider
(Claude, etc.), implement the same one-method contract and inject it:

```python
ai = SelfModifyingMemoryAI(initial=genome, proposer=MyProposer(router, budget))
```

The model only *authors* a weight delta. It cannot bypass the edit budget, the
five stages, stakeholder consensus, or drift correction — those guards are
model-agnostic and run identically whatever produced the proposal.

> **Secrets:** the key is read only from the `MISTRAL_API_KEY` environment
> variable and is never written to disk. Keep it in a gitignored `.env` or your
> shell, never in source or chat.
