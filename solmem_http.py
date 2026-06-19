"""Launcher for the hosted (HTTP) solmem MCP server.

    SOLMEM_TOKENS="tok=alice,tok2=bob" SOLMEM_HOME=/data py solmem_http.py

Env:
  SOLMEM_TOKENS        required — 'token=author' pairs (comma-separated)
  SOLMEM_ADMIN_TOKENS  optional — tokens allowed to evolve/rollback the brain
  SOLMEM_HOME          where the shared store lives (default ./solmem_store)
  SOLMEM_PORT          listen port (default 8730)
  SOLMEM_HTTP_HOST     bind address (default 0.0.0.0)
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from solmem.http_server import serve  # noqa: E402

if __name__ == "__main__":
    serve(os.environ.get("SOLMEM_HTTP_HOST", "0.0.0.0"),
          int(os.environ.get("SOLMEM_PORT", "8730")))
