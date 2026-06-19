"""HTTP transport test: handshake, auth, per-token provenance, admin gating.

Spins the server on an ephemeral port against a throwaway store. Run: py test_http.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from solmem.http_server import make_server
from solmem.server import SolMem
from solmem.store import Store


def post(port: int, msg: dict, token: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/mcp",
                                 data=json.dumps(msg).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, None


def call(port, tool, args, token, mid=1):
    _, resp = post(port, {"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                          "params": {"name": tool, "arguments": args}}, token)
    return resp["result"]["content"][0]["text"]


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="solmem_http_"))
    app = SolMem(Store(home=tmp, git=False))
    tokens = {"alice-tok": "alice", "admin-tok": "boss"}
    httpd = make_server("127.0.0.1", 0, tokens, {"admin-tok"}, app=app)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        # 1. health check needs no auth
        import urllib.request as u
        with u.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            assert json.loads(r.read())["status"] == "ok"
        print("health           -> ok (no auth)")

        # 2. missing/invalid token -> 401
        code, _ = post(port, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert code == 401, code
        code, _ = post(port, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, token="nope")
        assert code == 401, code
        print("auth             -> 401 without a valid Bearer token")

        # 3. handshake over HTTP
        _, init = post(port, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                              "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                       token="alice-tok")
        assert init["result"]["serverInfo"]["name"] == "solmem", init
        # notification -> 202, no body
        code, body = post(port, {"jsonrpc": "2.0", "method": "notifications/initialized"},
                          token="alice-tok")
        assert code == 202 and body is None, (code, body)
        print("handshake        -> initialize OK, notification -> 202")

        # 4. tools/list
        _, tl = post(port, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, token="alice-tok")
        assert any(t["name"] == "recall" for t in tl["result"]["tools"])
        print("tools/list       ->", len(tl["result"]["tools"]), "tools")

        # 5. provenance from token: alice's record is attributed to 'alice'
        out = call(port, "record", {"problem": "n+1 query in orders list",
                   "solution": "prefetch with a join", "project": "web"}, token="alice-tok")
        rid = out.split()[2]
        assert app.store.get(rid).author == "alice", app.store.get(rid).author
        print("provenance       -> record by alice-tok attributed to 'alice'")

        # 6. admin gating: non-admin cannot evolve; admin can
        denied = call(port, "evolve", {}, token="alice-tok")
        assert "admin-only" in denied, denied
        allowed = call(port, "evolve", {}, token="admin-tok")
        assert "not enough graded recalls" in allowed, allowed
        print("admin gate       -> alice denied evolve; boss allowed")

        print("\nALL OK  (store:", tmp, ")")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
