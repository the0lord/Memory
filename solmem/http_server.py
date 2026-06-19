"""Hosted MCP transport: Streamable HTTP over the stdlib, wrapping the existing
JSON-RPC dispatch so a whole team can connect to one server (one shared brain).

Why this is small: ``Server.handle()`` already turns a JSON-RPC message into a
response with no knowledge of the transport. This module is just an HTTP front
door for it — POST a JSON-RPC message to ``/mcp``, get the response back.

Security (the shared-memory threat model — MINJA/MEXTRA, OWASP ASI06):
  * Bearer-token auth, required. The server refuses to start without tokens.
  * Each token maps to an author name, so provenance survives in hosted mode
    (a record is attributed to the developer who created it, not the host).
  * ``evolve`` / ``rollback_brain`` are admin-only — a regular user cannot
    promote or roll back the shared brain.
  * One global lock serializes writes; reads/writes never interleave.

Zero dependencies: http.server + threading, same as the rest of the project.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .server import Server, SolMem
from .store import Store

# Tools that mutate the shared brain — restricted to admin tokens when any exist.
ADMIN_TOOLS = {"evolve", "rollback_brain"}


def _parse_token_map(raw: str) -> dict[str, str]:
    """Parse 'alice=tok,bob=tok2' (or a bare 'tok') into {token: author}.

    Format is name=token — the developer's name on the left, their secret bearer
    token on the right (the natural reading). A bare entry with no '=' is treated
    as a token with author 'unknown'.
    """
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" in pair:
            author, tok = pair.split("=", 1)
            out[tok.strip()] = author.strip() or "unknown"
        else:
            out[pair] = "unknown"
    return out


def load_tokens() -> tuple[dict[str, str], set[str]]:
    """Return (token->author, {admin_tokens}). Admin tokens also authenticate."""
    tokens = _parse_token_map(os.environ.get("SOLMEM_TOKENS", ""))
    admins = _parse_token_map(os.environ.get("SOLMEM_ADMIN_TOKENS", ""))
    tokens.update(admins)  # admin tokens are valid users too
    single = os.environ.get("SOLMEM_TOKEN")
    if single:
        tokens[single.strip()] = os.environ.get("SOLMEM_AUTHOR", "unknown")
    return tokens, set(admins)


class _Handler(BaseHTTPRequestHandler):
    server_version = "solmem-http/0.1"
    protocol_version = "HTTP/1.1"

    # -- helpers --------------------------------------------------------
    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self) -> tuple[str | None, bool]:
        """Return (author, is_admin) or (None, False) if the token is unknown."""
        hdr = self.headers.get("Authorization", "")
        if hdr.startswith("Bearer "):
            tok = hdr[7:].strip()
            if tok in self.server.tokens:           # type: ignore[attr-defined]
                return self.server.tokens[tok], tok in self.server.admin_tokens  # type: ignore[attr-defined]
        return None, False

    # -- routes ---------------------------------------------------------
    def do_GET(self) -> None:
        # Unauthenticated health check for load balancers / sanity.
        if self.path.rstrip("/") in ("", "/health"):
            return self._json(200, {"status": "ok", "server": "solmem"})
        # Our tools are pure request/response; no server-initiated SSE stream.
        return self._json(405, {"error": "method not allowed"})

    def do_POST(self) -> None:
        # Drain the request body FIRST, always — otherwise an early return (401/404)
        # leaves it in the socket and the next keep-alive request desyncs into a 400.
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""

        if self.path.rstrip("/") not in ("/mcp", ""):
            return self._json(404, {"error": "not found"})

        author, is_admin = self._auth()
        if author is None:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Bearer realm="solmem"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        try:
            msg = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._json(400, {"error": "invalid json"})

        # Admin gate for brain-mutating tools.
        if (msg.get("method") == "tools/call"
                and (msg.get("params") or {}).get("name") in ADMIN_TOOLS
                and not is_admin):
            tool = (msg.get("params") or {}).get("name")
            return self._json(200, {
                "jsonrpc": "2.0", "id": msg.get("id"),
                "result": {"isError": True, "content": [{"type": "text", "text":
                    f"'{tool}' is admin-only on this hosted server. "
                    f"Ask an admin to run it."}]}})

        srv: Server = self.server.mcp           # type: ignore[attr-defined]
        with self.server.lock:                  # type: ignore[attr-defined]
            srv.app.store.request_author = author
            try:
                response = srv.handle(msg)
            finally:
                srv.app.store.request_author = None

        if response is None:                    # a notification — nothing to return
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        return self._json(200, response)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("solmem-http: " + (fmt % args) + "\n")


def make_server(host: str, port: int, tokens: dict[str, str],
                admin_tokens: set[str], app: SolMem | None = None) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.mcp = Server(app)                      # type: ignore[attr-defined]
    httpd.tokens = tokens                        # type: ignore[attr-defined]
    httpd.admin_tokens = admin_tokens            # type: ignore[attr-defined]
    httpd.lock = threading.Lock()                # type: ignore[attr-defined]
    return httpd


def serve(host: str = "0.0.0.0", port: int = 8730) -> None:
    tokens, admins = load_tokens()
    if not tokens:
        sys.stderr.write(
            "FATAL: a networked memory server must not run open. Set SOLMEM_TOKENS="
            "'tok=alice,tok2=bob' (and optionally SOLMEM_ADMIN_TOKENS) first.\n")
        raise SystemExit(2)
    httpd = make_server(host, port, tokens, admins)
    home = httpd.mcp.app.store.home              # type: ignore[attr-defined]
    sys.stderr.write(f"solmem HTTP MCP on {host}:{port}  "
                     f"({len(tokens)} token(s), {len(admins)} admin, home={home})\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
