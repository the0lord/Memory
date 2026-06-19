"""Secret scanning — refuse to store credentials in shared memory.

solmem records are committed to a shared git repo, and this team's solutions touch
KMS, custody, withdrawals, and databases. So `record` must never persist a private
key, token, or connection string. We scan before writing and reject (with the
finding *labels*, never the secret itself) so the developer can redact and retry.

Patterns are intentionally simple and length-bounded — no nested quantifiers — to
avoid catastrophic regex backtracking on large solution bodies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# (label, compiled pattern). Order is informational only.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private key block", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("AWS secret access key", re.compile(
        r"(?i)aws_secret_access_key\s*[:=]\s*[\"']?[A-Za-z0-9/+]{40}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("OpenAI/Mistral-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    ("DB connection string with credentials", re.compile(
        r"(?i)\b(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://"
        r"[^\s:/@]+:[^\s:/@]+@")),
    ("private key assignment", re.compile(
        r"(?i)\b(?:private[_-]?key|secret[_-]?key|seed[_-]?phrase|mnemonic)\s*[:=]\s*"
        r"[\"']?\S{12,}")),
    ("generic credential assignment", re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|access[_-]?key)\s*[:=]\s*"
        r"[\"']?[A-Za-z0-9/+_\-]{16,}")),
    ("dotenv secret line", re.compile(
        r"(?m)^[A-Z][A-Z0-9_]{2,}=(?:[A-Za-z0-9/+_\-]{16,})$")),
]


@dataclass
class Finding:
    label: str
    where: str          # "problem" | "solution"
    snippet: str        # redacted preview, safe to show/log


def _redact_snippet(match: str) -> str:
    """Keep just enough to recognize the hit; hide the secret body."""
    head = match[:6]
    return f"{head}…[redacted {max(0, len(match) - 6)} chars]"


def scan(problem: str, solution: str) -> list[Finding]:
    findings: list[Finding] = []
    for where, text in (("problem", problem), ("solution", solution)):
        for label, pat in _PATTERNS:
            m = pat.search(text)
            if m:
                findings.append(Finding(label, where, _redact_snippet(m.group(0))))
    return findings


def report(findings: list[Finding]) -> str:
    lines = ["REJECTED — possible secret(s) detected; nothing was stored.",
             "Remove or redact these, then record again:"]
    for f in findings:
        lines.append(f"  • {f.label} in {f.where}: {f.snippet}")
    lines.append("Never commit credentials to shared memory.")
    return "\n".join(lines)
