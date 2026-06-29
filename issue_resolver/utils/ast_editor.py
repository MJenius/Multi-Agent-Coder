"""AST-aware code editing utility.

Modifies Python files programmatically using Python's native ``ast`` module.
For non-Python languages (or parsing errors), falls back to regex-based/line-based
manipulation.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


class ASTEditor:
    """Modifies source code elements programmatically.

    Maintains source structure and minimizes formatting disruption.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.content = self.file_path.read_text(encoding="utf-8", errors="replace")
        self.lines = self.content.split("\n")
        self.language = self._detect_language()

    def _detect_language(self) -> str:
        suffix = self.file_path.suffix.lower()
        if suffix == ".py":
            return "python"
        if suffix in (".js", ".jsx", ".ts", ".tsx"):
            return "javascript"
        return "generic"

    def replace_method(
        self,
        class_name: str,
        method_name: str,
        new_source: str,
    ) -> bool:
        """Replace the body of a class method.  Returns True on success."""
        if self.language != "python":
            return self._replace_method_regex(class_name, method_name, new_source)

        try:
            tree = ast.parse(self.content)
        except SyntaxError:
            return self._replace_method_regex(class_name, method_name, new_source)

        # Locate class and method node
        class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break

        if not class_node:
            return False

        method_node = None
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                method_node = node
                break

        if not method_node:
            return False

        # Get lines to replace
        start_line = method_node.lineno - 1
        end_line = getattr(method_node, "end_lineno", method_node.lineno)

        # Determine indentation of original method
        orig_first_line = self.lines[start_line]
        indent = len(orig_first_line) - len(orig_first_line.lstrip())

        # Indent the new source to match
        indented_lines = []
        for i, line in enumerate(new_source.splitlines()):
            if i == 0:
                # Keep original signature indentation
                indented_lines.append(" " * indent + line.lstrip())
            else:
                indented_lines.append(" " * (indent + 4) + line)

        # Apply replacement
        self.lines[start_line:end_line] = indented_lines
        self.content = "\n".join(self.lines)
        return True

    def replace_function(self, name: str, new_source: str) -> bool:
        """Replace a top-level function.  Returns True on success."""
        if self.language != "python":
            return self._replace_function_regex(name, new_source)

        try:
            tree = ast.parse(self.content)
        except SyntaxError:
            return self._replace_function_regex(name, new_source)

        func_node = None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                func_node = node
                break

        if not func_node:
            return False

        start_line = func_node.lineno - 1
        end_line = getattr(func_node, "end_lineno", func_node.lineno)

        # Determine indentation (should be 0 for top-level)
        orig_first_line = self.lines[start_line]
        indent = len(orig_first_line) - len(orig_first_line.lstrip())

        indented_lines = []
        for i, line in enumerate(new_source.splitlines()):
            indented_lines.append(" " * indent + line)

        self.lines[start_line:end_line] = indented_lines
        self.content = "\n".join(self.lines)
        return True

    def add_import(self, module: str, symbol: str | None = None) -> bool:
        """Add an import statement at the top of the file."""
        if symbol:
            statement = f"from {module} import {symbol}"
        else:
            statement = f"import {module}"

        if statement in self.content:
            return True  # Already present

        # Find place to insert (after existing imports, or at line 1)
        insert_idx = 0
        if self.language == "python":
            try:
                tree = ast.parse(self.content)
                last_import_line = 0
                for node in tree.body:
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        last_import_line = getattr(node, "end_lineno", node.lineno)
                if last_import_line > 0:
                    insert_idx = last_import_line
            except SyntaxError:
                pass

        self.lines.insert(insert_idx, statement)
        self.content = "\n".join(self.lines)
        return True

    def save(self) -> None:
        """Write modified content back to disk."""
        self.file_path.write_text(self.content, encoding="utf-8", newline="")

    # ----- regex fallback -----

    def _replace_method_regex(self, class_name: str, method_name: str, new_source: str) -> bool:
        # Simple line-based search for class and method
        class_found = False
        method_idx = -1
        indent = 4

        for i, line in enumerate(self.lines):
            if not class_found:
                if re.match(r"^\s*(class|interface)\s+" + class_name + r"\b", line):
                    class_found = True
            else:
                # Find method definition inside class
                if re.search(r"\b" + method_name + r"\b\s*\(", line):
                    method_idx = i
                    indent = len(line) - len(line.lstrip())
                    break

        if method_idx == -1:
            return False

        # Find end of method body by counting braces or indentation
        end_idx = method_idx + 1
        if self.language in ("javascript", "generic"):
            # Count curly braces
            braces = 0
            started = False
            for j in range(method_idx, len(self.lines)):
                line = self.lines[j]
                braces += line.count("{")
                braces -= line.count("}")
                if "{" in line:
                    started = True
                if started and braces <= 0:
                    end_idx = j + 1
                    break
        else:
            # indent based (python fallback)
            for j in range(method_idx + 1, len(self.lines)):
                line = self.lines[j]
                if line.strip() and len(line) - len(line.lstrip()) <= indent:
                    end_idx = j
                    break

        # Apply replacement
        indented_lines = [" " * indent + line.lstrip() for line in new_source.splitlines()]
        self.lines[method_idx:end_idx] = indented_lines
        self.content = "\n".join(self.lines)
        return True

    def _replace_function_regex(self, name: str, new_source: str) -> bool:
        func_idx = -1
        for i, line in enumerate(self.lines):
            if re.search(r"\b(def|function|func)\s+" + name + r"\b", line):
                func_idx = i
                break

        if func_idx == -1:
            return False

        end_idx = func_idx + 1
        if self.language in ("javascript", "generic"):
            braces = 0
            started = False
            for j in range(func_idx, len(self.lines)):
                line = self.lines[j]
                braces += line.count("{")
                braces -= line.count("}")
                if "{" in line:
                    started = True
                if started and braces <= 0:
                    end_idx = j + 1
                    break
        else:
            for j in range(func_idx + 1, len(self.lines)):
                line = self.lines[j]
                if line.strip() and len(line) - len(line.lstrip()) == 0:
                    end_idx = j
                    break

        self.lines[func_idx:end_idx] = new_source.splitlines()
        self.content = "\n".join(self.lines)
        return True
