import os
from issue_resolver.utils.patch_engine import FuzzyPatchEngine, parse_and_apply_blocks
from issue_resolver.utils.ripgrep_search import smart_search
from issue_resolver.nodes.researcher import _manage_context_budget

def test_fuzzy_patch_engine_exact():
    content = "line 1\nline 2\nline 3\n"
    engine = FuzzyPatchEngine(content)
    res = engine.apply_block("line 2", "line 2 modified")
    assert res["success"] is True
    assert engine.file_content == "line 1\nline 2 modified\nline 3\n"

def test_fuzzy_patch_engine_fuzzy():
    content = "def hello():\n    print('world')\n    return True\n"
    engine = FuzzyPatchEngine(content)
    res = engine.apply_block("def hello():\n    print('worldd')", "def hello():\n    print('world')\n    print('again')")
    assert res["success"] is True

def test_context_budget_manager():
    snippets = [
        "# --- file: a.py ---\n" + "A" * 5000,
        "# --- file: b.py ---\n" + "B" * 5000,
        "# --- file: a.py ---\n" + "A" * 6000,
        "# --- file: c.py ---\n" + "C" * 3000
    ]
    processed = _manage_context_budget(snippets)
    assert len(processed) == 3
    b_snip = [s for s in processed if "b.py" in s][0]
    assert "[...truncated]" in b_snip
