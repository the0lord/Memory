"""Minimal .env loader (stdlib only, no dependency).

Reads KEY=VALUE lines from the first existing file among the given paths and sets
them as environment variables. Blank lines and #comments are ignored; surrounding
quotes are stripped. Existing environment variables win unless override=True, so a
real shell export always takes precedence over the file.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(*paths: str | Path, override: bool = False) -> str | None:
    """Load the first existing .env among ``paths``. Returns the path used, or None."""
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and (override or key not in os.environ):
                os.environ[key] = val
        return str(path)
    return None
