"""Tests for the Repository Knowledge Graph.

Verifies construction, module retrieval, call graph traversing,
and dependency tracking.
"""

from __future__ import annotations

from issue_resolver.intelligence.knowledge_graph import (
    CallEdge,
    ClassNode,
    FunctionNode,
    ImportEdge,
    ModuleNode,
    RepoKnowledgeGraph,
)


def test_knowledge_graph_construction() -> None:
    graph = RepoKnowledgeGraph()

    # Add modules
    graph.modules["src/utils.py"] = ModuleNode(
        path="src/utils.py",
        language="python",
        classes=["Calculator"],
        functions=["helper_func"],
    )
    graph.modules["src/main.py"] = ModuleNode(
        path="src/main.py",
        language="python",
        functions=["main"],
        imports=["src/utils.py"],
    )

    # Add class
    graph.classes["src/utils.py::Calculator"] = ClassNode(
        name="Calculator",
        file_path="src/utils.py",
        methods=["add"],
    )

    # Add functions
    graph.functions["src/utils.py::Calculator.add"] = FunctionNode(
        name="add",
        file_path="src/utils.py",
        parent_class="Calculator",
        is_method=True,
    )
    graph.functions["src/utils.py::helper_func"] = FunctionNode(
        name="helper_func",
        file_path="src/utils.py",
    )
    graph.functions["src/main.py::main"] = FunctionNode(
        name="main",
        file_path="src/main.py",
        calls=["add"],
    )

    # Add edges
    graph.import_edges.append(ImportEdge(source="src/main.py", target="src/utils.py"))
    graph.call_edges.append(CallEdge(caller="src/main.py::main", callee="src/utils.py::Calculator.add"))

    # Assert basic structure
    assert len(graph.modules) == 2
    assert len(graph.classes) == 1
    assert len(graph.functions) == 3

    # Test module retrieval
    assert graph.get_module("src/utils.py") is not None
    assert graph.get_module("nonexistent.py") is None

    # Test imports/dependencies
    assert graph.get_dependencies("src/main.py") == ["src/utils.py"]
    assert graph.get_dependents("src/utils.py") == ["src/main.py"]

    # Test affected modules (blast radius)
    assert graph.get_affected_modules(["src/utils.py"]) == ["src/main.py", "src/utils.py"]

    # Test callers/callees
    assert "src/utils.py::Calculator.add" in graph.get_callees("main")
    assert "src/main.py::main" in graph.get_callers("add")
