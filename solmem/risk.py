"""Deterministic risk classification + recall policy.

Risk is NOT a learned weight — it is a hard guard. A solution touching withdrawals
or signing keys must never be auto-applied no matter how well it scores. We infer a
level from the problem/solution text (overridable at record time) and attach a
policy that tells the agent how much verification is required before trusting it.
"""

from __future__ import annotations

LEVELS = ("low", "medium", "high", "critical")
_ORDER = {lvl: i for i, lvl in enumerate(LEVELS)}

# Most specific / dangerous first. Word stems, matched case-insensitively.
_CRITICAL = (
    "withdrawal", "withdraw", "custody", "private key", "seed phrase", "mnemonic",
    "kms", "hsm", "signing key", "sign transaction", "payout", "settlement",
    "real money", "aml", "sanction", "cold wallet", "key material",
)
_HIGH = (
    "payment", "auth", "authentication", "authorization", "permission", "rbac",
    "migration", "password", "credential", "token", "encryption", "decrypt",
    "balance", "ledger", "wallet", "deposit", "transfer", "oauth", "jwt", "session",
)
_LOW = (
    "format", "lint", "rename", "typo", "comment", "readme", "docs", "whitespace",
    "import order", "log message", "spelling",
)

_POLICY = {
    "low": "safe to suggest directly.",
    "medium": "suggest, but run the listed checks.",
    "high": "require the verification checklist before applying.",
    "critical": "NEVER auto-apply — require explicit human verification first.",
}


def classify(text: str) -> str:
    t = text.lower()
    if any(k in t for k in _CRITICAL):
        return "critical"
    if any(k in t for k in _HIGH):
        return "high"
    if any(k in t for k in _LOW):
        return "low"
    return "medium"


def normalize(level: str | None) -> str:
    level = (level or "").strip().lower()
    return level if level in _ORDER else "medium"


def policy(level: str) -> str:
    return _POLICY.get(normalize(level), _POLICY["medium"])


def requires_checklist(level: str) -> bool:
    return _ORDER[normalize(level)] >= _ORDER["high"]
