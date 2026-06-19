"""The unit of shared team memory: one problem and the solution that worked.

A record is born when Claude finishes a task (``record``), recalled when a similar
problem reappears (``recall``), and graded when the recalled solution is tried
(``report_outcome``). The grade history is what lets the recall brain self-modify:
records that keep working float to the top, ones that mislead sink.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field


# Cheap, dependency-free tokenizer for offline similarity. Lowercase word/identifier
# tokens, including dotted/underscored code symbols, minus a small stopword set.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]+")
_STOP = frozenset(
    "the a an and or of to in is it for on with this that be as at by from".split()
)


def keywords_of(text: str) -> list[str]:
    """Distinct, stopworded lowercase tokens used for offline keyword similarity."""
    seen: dict[str, None] = {}
    for m in _TOKEN_RE.finditer(text.lower()):
        tok = m.group(0)
        if tok not in _STOP and len(tok) > 1:
            seen.setdefault(tok, None)
    return list(seen)


@dataclass
class SolutionRecord:
    id: str
    problem: str
    solution: str
    project: str = ""
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)  # semantic vector (optional)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    uses: int = 0          # times this record was surfaced by recall
    successes: int = 0     # times the recalled solution actually worked
    regressions: int = 0   # times it worked then later caused a problem

    # Provenance (security: who a record came from — OWASP ASI06, MINJA/MEXTRA).
    author: str = ""                     # git user / SOLMEM_AUTHOR at record time
    origin: str = "local"                # "local" | "synced"

    # Supersession (A-MEM Memory Evolution / Zep temporal validity / Memory-R1 UPDATE):
    # a better solution replaces an old one non-destructively — the old record is
    # kept for audit but stops surfacing, so a challenger overtakes an incumbent
    # instead of competing with it forever.
    superseded_by: str = ""              # id of the record that replaced this one
    supersedes: str = ""                 # id of the record this one replaced
    superseded_at: float = 0.0

    # Phase 2: deterministic guards & applicability (not learned weights)
    kind: str = "solution"               # "solution" | "negative"
    risk: str = "medium"                 # low | medium | high | critical
    applies_when: list[str] = field(default_factory=list)
    does_not_apply_when: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    better_direction: str = ""           # for negatives: what to do instead
    why_failed: str = ""                 # for negatives: why the bad approach fails

    @property
    def success_rate(self) -> float:
        """Laplace-smoothed so a single fluke neither crowns nor kills a record."""
        return (self.successes + 1.0) / (self.uses + 2.0)

    @property
    def trust(self) -> float:
        """Success rate, then halved per reported regression. A solution that
        later broke something should sink fast regardless of past wins."""
        return self.success_rate * (0.5 ** self.regressions)

    @property
    def active(self) -> bool:
        """Still in play — not replaced by a newer record. Only active records
        surface in recall; superseded ones remain on disk for audit."""
        return not self.superseded_by

    @property
    def verified(self) -> bool:
        """Has at least one confirmed success and isn't superseded. Unverified
        records (e.g. a freshly injected/poisoned one) get a provenance caveat
        at recall rather than silent trust."""
        return self.active and self.successes >= 1

    @classmethod
    def create(
        cls,
        problem: str,
        solution: str,
        project: str = "",
        tags: list[str] | None = None,
        risk: str | None = None,
        applies_when: list[str] | None = None,
        does_not_apply_when: list[str] | None = None,
        verification: list[str] | None = None,
        kind: str = "solution",
        better_direction: str = "",
        why_failed: str = "",
        author: str = "",
    ) -> "SolutionRecord":
        from .risk import classify, normalize
        now = time.time()
        # ID derives from content + birth time: stable, filename-safe, collision-proof.
        digest = hashlib.sha1(f"{problem}\x00{solution}\x00{now}".encode()).hexdigest()
        # Risk is deterministic: caller override, else inferred from the text.
        level = normalize(risk) if risk else classify(f"{problem} {solution}")
        return cls(
            id=digest[:16],
            problem=problem.strip(),
            solution=solution.strip(),
            project=project.strip(),
            tags=list(tags or []),
            keywords=keywords_of(f"{problem} {solution} {' '.join(tags or [])}"),
            created=now,
            updated=now,
            kind=kind,
            risk=level,
            applies_when=list(applies_when or []),
            does_not_apply_when=list(does_not_apply_when or []),
            verification=list(verification or []),
            better_direction=better_direction.strip(),
            why_failed=why_failed.strip(),
            author=author.strip(),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SolutionRecord":
        fields = {f for f in cls.__dataclass_fields__}  # tolerate extra/old keys
        return cls(**{k: v for k, v in d.items() if k in fields})
