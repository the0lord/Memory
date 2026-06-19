"""Git-backed shared store: the team's accumulated problem->solution memory.

Layout (under SOLMEM_HOME, a directory the team keeps in git):

    SOLMEM_HOME/
        records/<id>.json   one file per solution  -> append-only, conflict-free
        brain.json          the evolving recall weights (the genome)

Each record is its own file so two teammates recording different solutions never
collide on merge. Writes are committed locally (best-effort); pulling/pushing to
share with the team is an explicit ``sync()`` so we never surprise anyone with
network calls mid-task.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from selfmod.buffers import GoldenTrace, GoldenTraceLog, RejectedEdit, RejectedEditBuffer
from selfmod.weights import FEATURES, WeightVector

from .features import DEFAULT_WEIGHTS
from .records import SolutionRecord

_RECALL_LOG_CAP = 1000  # keep the most recent N recall episodes


def default_home() -> Path:
    env = os.environ.get("SOLMEM_HOME")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "solmem_store"


class Store:
    def __init__(self, home: Path | None = None, git: bool = True):
        self.home = (home or default_home()).resolve()
        self.records_dir = self.home / "records"
        self.recalls_dir = self.home / "recalls"
        self.versions_dir = self.home / "brain_versions"
        self.audit_dir = self.home / "audit"
        self.brain_path = self.home / "brain.json"
        self.brain_prev_path = self.home / "brain.previous.json"
        self.golden_path = self.home / "golden.json"
        self.rejected_path = self.home / "rejected.json"
        for d in (self.records_dir, self.recalls_dir, self.versions_dir, self.audit_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.git = git and self._is_git_repo()
        # Set per-request by the HTTP transport so a hosted server attributes each
        # record to the calling developer (resolved from their token), not the host.
        self.request_author: str | None = None

    # -- records --------------------------------------------------------
    def _path(self, rec_id: str) -> Path:
        return self.records_dir / f"{rec_id}.json"

    def all(self) -> list[SolutionRecord]:
        out: list[SolutionRecord] = []
        for p in sorted(self.records_dir.glob("*.json")):
            try:
                out.append(SolutionRecord.from_dict(json.loads(p.read_text("utf-8"))))
            except (json.JSONDecodeError, OSError, TypeError):
                continue  # a malformed file shouldn't sink every recall
        return out

    def get(self, rec_id: str) -> SolutionRecord | None:
        p = self._path(rec_id)
        if not p.exists():
            return None
        return SolutionRecord.from_dict(json.loads(p.read_text("utf-8")))

    def put(self, rec: SolutionRecord, *, commit: bool = True) -> None:
        rec.updated = rec.updated or time.time()
        self._path(rec.id).write_text(
            json.dumps(rec.to_dict(), indent=2, ensure_ascii=False), "utf-8"
        )
        if commit:
            self._commit(f"solmem: record {rec.id} ({rec.project or 'global'})")

    # -- recall episodes (labeled ranking data for evolution) -----------
    def log_recall(self, query: str, project: str,
                   candidates: list[tuple[str, dict]], brain_version: str = "brain_v0") -> str:
        """Persist a recall episode: query + the candidates we surfaced with the
        exact features they were scored on, plus the brain version that ranked
        them. Outcomes get attached later. Each episode is its own file so team
        members never collide on merge."""
        rid = uuid.uuid4().hex[:16]
        ep = {
            "id": rid, "ts": time.time(), "query": query, "project": project,
            "brain_version": brain_version,
            "candidates": [{"id": cid, "features": feats} for cid, feats in candidates],
            "outcomes": {},
        }
        (self.recalls_dir / f"{rid}.json").write_text(
            json.dumps(ep, indent=2, ensure_ascii=False), "utf-8")
        self._prune_recalls()
        self._commit(f"solmem: recall {rid}")
        return rid

    def get_recall(self, rid: str) -> dict | None:
        p = self.recalls_dir / f"{rid}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def weights_of_version(self, version: str) -> WeightVector:
        """Weights for a given brain version (archived or current), else current."""
        if version and version == self.brain_version():
            return self.load_brain()
        p = self.versions_dir / f"{version}.json"
        if p.exists():
            try:
                vals = json.loads(p.read_text("utf-8")).get("weights", {})
                return WeightVector({f: float(vals[f]) for f in FEATURES})
            except (json.JSONDecodeError, OSError, KeyError):
                pass
        return self.load_brain()

    def all_recalls(self) -> list[dict]:
        out = []
        for p in self.recalls_dir.glob("*.json"):
            try:
                out.append(json.loads(p.read_text("utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        out.sort(key=lambda e: e.get("ts", 0.0))
        return out

    def attach_outcome(self, record_id: str, worked: bool) -> bool:
        """Attach a worked/failed label to the most recent ungraded recall that
        surfaced this record. Returns True if a recall episode was updated."""
        files = sorted(self.recalls_dir.glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
            try:
                ep = json.loads(p.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            ids = {c["id"] for c in ep.get("candidates", [])}
            if record_id in ids and record_id not in ep.get("outcomes", {}):
                ep.setdefault("outcomes", {})[record_id] = bool(worked)
                p.write_text(json.dumps(ep, indent=2, ensure_ascii=False), "utf-8")
                return True
        return False

    def _prune_recalls(self) -> None:
        files = sorted(self.recalls_dir.glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[_RECALL_LOG_CAP:]:
            try:
                p.unlink()
            except OSError:
                pass

    # -- golden traces & rejected edits (the pipeline's long-term memory) -
    def load_golden(self) -> GoldenTraceLog:
        log = GoldenTraceLog()
        if self.golden_path.exists():
            for t in json.loads(self.golden_path.read_text("utf-8")):
                log.traces.append(GoldenTrace(
                    WeightVector({f: float(t["weights"][f]) for f in FEATURES}),
                    t["metric"], t["cycle"], t.get("ts", 0.0)))
        return log

    def save_golden(self, log: GoldenTraceLog, *, commit: bool = True) -> None:
        self.golden_path.write_text(json.dumps(
            [{"weights": t.weights.values, "metric": t.metric,
              "cycle": t.cycle, "ts": t.ts} for t in log.traces], indent=2), "utf-8")
        if commit:
            self._commit("solmem: update golden traces")

    def load_rejected(self) -> RejectedEditBuffer:
        buf = RejectedEditBuffer()
        if self.rejected_path.exists():
            for e in json.loads(self.rejected_path.read_text("utf-8")):
                buf.items.append(RejectedEdit(
                    e["direction"], e["stage"], e["delta_metric"],
                    e.get("note", ""), e.get("ts", 0.0)))
        return buf

    def save_rejected(self, buf: RejectedEditBuffer, *, commit: bool = True) -> None:
        self.rejected_path.write_text(json.dumps(
            [{"direction": e.direction, "stage": e.stage,
              "delta_metric": e.delta_metric, "note": e.note, "ts": e.ts}
             for e in buf.items], indent=2), "utf-8")
        if commit:
            self._commit("solmem: update rejected-edit buffer")

    # -- brain (recall weights), versioned ------------------------------
    def brain_meta(self) -> dict:
        if not self.brain_path.exists():
            return {"version": "brain_v0", "parent": None, "weights": DEFAULT_WEIGHTS.values}
        return json.loads(self.brain_path.read_text("utf-8"))

    def load_brain(self) -> WeightVector:
        meta = self.brain_meta()
        vals = meta.get("weights", meta)  # tolerate old flat format
        return WeightVector({f: float(vals[f]) for f in FEATURES})

    def brain_version(self) -> str:
        return self.brain_meta().get("version", "brain_v0")

    def _next_version(self) -> str:
        nums = []
        for p in [self.brain_path, *self.versions_dir.glob("brain_v*.json")]:
            try:
                v = json.loads(p.read_text("utf-8")).get("version", "")
                if v.startswith("brain_v") and v[7:].isdigit():
                    nums.append(int(v[7:]))
            except (OSError, json.JSONDecodeError):
                continue
        return f"brain_v{(max(nums) + 1) if nums else 1}"

    def save_brain(self, wv: WeightVector, *, commit: bool = True) -> None:
        """Plain write (used to seed/initialize). Keeps the current version label."""
        meta = self.brain_meta()
        self._write_brain({"version": meta.get("version", "brain_v0"),
                           "parent": meta.get("parent"), "weights": wv.values,
                           "saved_at": time.time()})
        if commit:
            self._commit("solmem: set recall brain")

    def promote_brain(self, wv: WeightVector, promotion: dict, *, commit: bool = True) -> str:
        """Versioned promotion: archive the current brain, write a new version."""
        cur = self.brain_meta()
        cur_version = cur.get("version", "brain_v0")
        new_version = self._next_version()
        if self.brain_path.exists():  # archive the outgoing brain
            (self.versions_dir / f"{cur_version}.json").write_text(
                json.dumps(cur, indent=2), "utf-8")
            self.brain_prev_path.write_text(json.dumps(cur, indent=2), "utf-8")
        self._write_brain({"version": new_version, "parent": cur_version,
                           "weights": wv.values, "created_at": time.time(),
                           "promotion": promotion})
        self.audit("promotions", {"from": cur_version, "to": new_version,
                                  "promotion": promotion})
        if commit:
            self._commit(f"solmem: promote {cur_version} → {new_version}")
        return new_version

    def rollback_brain(self, target_version: str, reason: str = "", *, commit: bool = True) -> str:
        src = self.versions_dir / f"{target_version}.json"
        if not src.exists():
            return f"no archived brain '{target_version}' to roll back to"
        cur = self.brain_meta()
        target = json.loads(src.read_text("utf-8"))
        if self.brain_path.exists():
            self.brain_prev_path.write_text(json.dumps(cur, indent=2), "utf-8")
        target["rolled_back_from"] = cur.get("version")
        target["rollback_reason"] = reason
        self._write_brain(target)
        self.audit("rollbacks", {"from": cur.get("version"), "to": target_version,
                                 "reason": reason})
        if commit:
            self._commit(f"solmem: rollback to {target_version}")
        return f"rolled back to {target_version} (from {cur.get('version')})"

    def list_brains(self) -> list[dict]:
        out = [self.brain_meta()]
        for p in sorted(self.versions_dir.glob("brain_v*.json")):
            try:
                out.append(json.loads(p.read_text("utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def _write_brain(self, meta: dict) -> None:
        self.brain_path.write_text(json.dumps(meta, indent=2), "utf-8")

    # -- audit + regression events --------------------------------------
    def audit(self, kind: str, entry: dict) -> None:
        entry = {"ts": time.time(), **entry}
        path = self.audit_dir / f"{kind}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- git ------------------------------------------------------------
    def _git(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.home), *args],
            capture_output=True, text=True, check=check,
        )

    def current_author(self) -> str:
        """Best-effort identity for provenance, stamped on every record so recall
        can show who a memory came from (OWASP ASI06). A per-request author (set by
        the HTTP transport from the caller's token) wins, else SOLMEM_AUTHOR env,
        else git user.name, else 'unknown'."""
        if self.request_author:
            return self.request_author
        a = os.environ.get("SOLMEM_AUTHOR")
        if a and a.strip():
            return a.strip()
        if self.git:
            try:
                r = self._git("config", "user.name")
                if r.returncode == 0 and r.stdout.strip():
                    return r.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
        return "unknown"

    def _is_git_repo(self) -> bool:
        try:
            r = self._git("rev-parse", "--is-inside-work-tree")
            return r.returncode == 0 and r.stdout.strip() == "true"
        except (OSError, FileNotFoundError):
            return False

    def _commit(self, message: str) -> None:
        if not self.git:
            return
        try:
            self._git("add", "-A")
            # nothing staged -> commit would fail; that's fine to ignore
            self._git("commit", "-m", message)
        except (OSError, subprocess.SubprocessError):
            pass

    def sync(self) -> str:
        """Explicit team sync: pull --rebase then push. Best-effort, returns a note."""
        if not self.git:
            return "store is not a git repo; nothing to sync"
        try:
            self._commit("solmem: sync checkpoint")
            pull = self._git("pull", "--rebase", "--autostash")
            push = self._git("push")
            return f"pull: {pull.returncode==0} | push: {push.returncode==0}"
        except (OSError, subprocess.SubprocessError) as e:
            return f"sync failed: {e}"
