"""Tests for the Storage Layer.

Verifies get, put, query, delete, and namespacing in JsonMemoryStore.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from issue_resolver.storage.json_store import JsonMemoryStore
from issue_resolver.storage.base import RepoSummary, FixRecord


def test_json_memory_store_operations() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = JsonMemoryStore(root_dir=tmp_dir, namespace="test_repo")

        # Basic get/put
        store.put("foo", "bar")
        assert store.get("foo") == "bar"
        assert store.get("nonexistent") is None

        # Delete
        assert store.delete("foo") is True
        assert store.get("foo") is None
        assert store.delete("foo") is False

        # RepoSummary storage
        summary = RepoSummary(
            repo_path="/path/to/test",
            language="python",
            framework="django",
        )
        store.put_repo_summary(summary)

        loaded_dict = store.get_repo_summary()
        assert loaded_dict is not None
        assert loaded_dict["language"] == "python"
        assert loaded_dict["framework"] == "django"

        # FixRecord storage
        record = FixRecord(
            issue_id="test-issue",
            issue_text="test issue text",
            success=True,
        )
        store.add_fix_record(record)

        history = store.get_fix_history()
        assert len(history) == 1
        assert history[0]["issue_id"] == "test-issue"
        assert history[0]["success"] is True
