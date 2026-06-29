"""JSON file-based implementation of ``MemoryStore``.

Stores data in ``~/.multi_agent_coder/memory/`` with per-repo namespacing.
Thread-safe via file locking.  Designed as the first backend — the
``MemoryStore`` interface ensures SQLite, DuckDB, Chroma, LanceDB, or
PostgreSQL can be swapped in by changing one config value.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from issue_resolver.core.interfaces import MemoryStore


def _default_memory_dir() -> Path:
    """Return the default memory directory, respecting platform conventions."""
    # Try XDG / AppData first, fall back to ~/.multi_agent_coder
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "multi_agent_coder" / "memory"


def _repo_namespace(repo_path: str) -> str:
    """Deterministic short namespace from a repo path."""
    normalised = Path(repo_path).resolve().as_posix()
    digest = hashlib.sha256(normalised.encode()).hexdigest()[:12]
    slug = Path(repo_path).name or "repo"
    return f"{slug}_{digest}"


class JsonMemoryStore(MemoryStore):
    """Persistent JSON-backed memory store.

    Data layout on disk::

        <root>/
          <namespace>/
            store.json          ← main key-value store
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        namespace: str = "default",
    ) -> None:
        self._root = Path(root_dir) if root_dir else _default_memory_dir()
        self._namespace = namespace
        self._store_dir = self._root / self._namespace
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._store_path = self._store_dir / "store.json"
        self._lock = threading.Lock()
        self._cache: dict[str, Any] | None = None

    # ----- internal helpers -----

    def _load(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache
        if self._store_path.is_file():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._cache = data
                    return self._cache
            except (json.JSONDecodeError, OSError):
                pass
        self._cache = {}
        return self._cache

    def _save(self) -> None:
        data = self._cache if self._cache is not None else {}
        tmp = self._store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._store_path)

    # ----- MemoryStore interface -----

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._load().get(key)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._load()[key] = value
            self._save()

    def query(self, prefix: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load()
        results: list[dict[str, Any]] = []
        for k, v in data.items():
            if not k.startswith(prefix):
                continue
            if not isinstance(v, dict):
                continue
            if filters:
                if all(v.get(fk) == fv for fk, fv in filters.items()):
                    results.append(v)
            else:
                results.append(v)
        return results

    def list_keys(self, prefix: str = "") -> list[str]:
        with self._lock:
            data = self._load()
        return sorted(k for k in data if k.startswith(prefix))

    def delete(self, key: str) -> bool:
        with self._lock:
            data = self._load()
            if key in data:
                del data[key]
                self._save()
                return True
            return False

    # ----- convenience helpers -----

    def put_repo_summary(self, summary: Any) -> None:
        """Persist a ``RepoSummary`` (or any dict-serialisable object)."""
        data = summary.to_dict() if hasattr(summary, "to_dict") else dict(summary)
        self.put("repo_summary", data)

    def get_repo_summary(self) -> dict[str, Any] | None:
        return self.get("repo_summary")

    def add_fix_record(self, record: Any) -> None:
        """Append a ``FixRecord`` to the fix history list."""
        data = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        with self._lock:
            store = self._load()
            history = store.get("fix_history", [])
            if not isinstance(history, list):
                history = []
            history.append(data)
            store["fix_history"] = history
            self._save()

    def get_fix_history(self) -> list[dict[str, Any]]:
        with self._lock:
            store = self._load()
        history = store.get("fix_history", [])
        return history if isinstance(history, list) else []

    def put_embeddings(self, key: str, embeddings: list[list[float]]) -> None:
        """Store embedding vectors (list of lists)."""
        self.put(f"embeddings:{key}", embeddings)

    def get_embeddings(self, key: str) -> list[list[float]] | None:
        return self.get(f"embeddings:{key}")

    @classmethod
    def for_repo(cls, repo_path: str, root_dir: str | Path | None = None) -> "JsonMemoryStore":
        """Create a store namespaced to a specific repository."""
        ns = _repo_namespace(repo_path)
        return cls(root_dir=root_dir, namespace=ns)
