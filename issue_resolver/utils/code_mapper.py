from __future__ import annotations
import ast
import os
from pathlib import Path

class CodeMapper:
    @staticmethod
    def generate_outline(file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"File {file_path} not found"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            return f"File {file_path} contains syntax errors"
        outline = [f"MAP FOR {file_path}"]
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                modules = ", ".join(alias.name for alias in node.names)
                outline.append(f"  Import from {node.module}: {modules}")
            elif isinstance(node, ast.Import):
                modules = ", ".join(alias.name for alias in node.names)
                outline.append(f"  Import: {modules}")
            elif isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                outline.append(f"  Class {node.name}({bases}):")
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        args = ast.unparse(sub_node.args)
                        outline.append(f"    Method: def {sub_node.name}({args})")
            elif isinstance(node, ast.FunctionDef):
                args = ast.unparse(node.args)
                outline.append(f"  Function: def {node.name}({args})")
        return "\n".join(outline)

    @staticmethod
    def generate_repo_map(repo_path: str) -> str:
        repo_dir = Path(repo_path).resolve()
        py_files = []
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in (".git", ".venv", "venv", "__pycache__", "build", "dist", "node_modules")]
            for file in files:
                if file.endswith(".py"):
                    py_files.append(Path(root) / file)
        outlines = []
        for py_file in py_files:
            rel = py_file.relative_to(repo_dir).as_posix()
            outline = CodeMapper.generate_outline(str(py_file))
            outline = outline.replace(f"MAP FOR {py_file}", f"MAP FOR {rel}")
            outlines.append(outline)
        return "\n\n".join(outlines)
