"""Language Server Protocol bridge for precise symbol resolution.

Wraps language-specific servers for high-precision Find Definition,
Find References, and type information.  Falls back gracefully when
servers are not installed.

Supported servers:
  - Python:    jedi-language-server (pip install jedi-language-server)
  - JS/TS:     typescript-language-server (npm install -g typescript-language-server)
  - Go:        gopls (go install golang.org/x/tools/gopls@latest)

Servers are spawned as long-lived subprocesses for the duration of
a pipeline run, then shut down cleanly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


class LSPBridge:
    """Bridge to a language server for precise symbol resolution.

    Usage::

        lsp = LSPBridge("/path/to/repo", "python")
        if lsp.is_available:
            defn = lsp.find_definition("calculate_total", "src/utils.py", 42)
            refs = lsp.find_references("calculate_total", "src/utils.py", 42)
        lsp.shutdown()
    """

    def __init__(self, repo_path: str, language: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        self.language = language.lower()
        self._server: subprocess.Popen | None = None
        self._available = False
        self._request_id = 0
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._responses: dict[int, Any] = {}
        self._response_events: dict[int, threading.Event] = {}
        self._initialized = False
        self._initialize()

    @property
    def is_available(self) -> bool:
        """True if the LSP server started successfully."""
        return self._available

    def _initialize(self) -> None:
        """Start the language server subprocess."""
        cmd = self._get_server_command()
        if not cmd:
            print(f"[LSPBridge] No server command for language: {self.language}")
            return

        try:
            self._server = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.repo_path,
            )
        except FileNotFoundError:
            print(f"[LSPBridge] Server binary not found: {cmd[0]}")
            return
        except Exception as exc:
            print(f"[LSPBridge] Failed to start server: {exc}")
            return

        # Start reader thread
        self._reader_thread = threading.Thread(target=self._read_responses, daemon=True)
        self._reader_thread.start()

        # Send initialize request
        try:
            init_result = self._send_request("initialize", {
                "processId": os.getpid(),
                "rootUri": f"file://{self.repo_path.replace(os.sep, '/')}",
                "rootPath": self.repo_path,
                "capabilities": {
                    "textDocument": {
                        "definition": {"dynamicRegistration": False},
                        "references": {"dynamicRegistration": False},
                        "hover": {"dynamicRegistration": False},
                    }
                },
            }, timeout=10.0)

            if init_result is not None:
                self._send_notification("initialized", {})
                self._available = True
                self._initialized = True
                print(f"[LSPBridge] Server initialized for {self.language}")
            else:
                print(f"[LSPBridge] Initialize request timed out for {self.language}")
                self._shutdown_server()
        except Exception as exc:
            print(f"[LSPBridge] Initialization failed: {exc}")
            self._shutdown_server()

    def _get_server_command(self) -> list[str] | None:
        """Return the command to start the LSP server for the current language."""
        commands: dict[str, list[str]] = {
            "python": [sys.executable, "-m", "jedi_language_server"],
            "javascript": ["typescript-language-server", "--stdio"],
            "typescript": ["typescript-language-server", "--stdio"],
            "go": ["gopls", "serve"],
        }
        return commands.get(self.language)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_message(self, message: dict) -> None:
        """Send a JSON-RPC message to the server."""
        if not self._server or not self._server.stdin:
            return
        body = json.dumps(message)
        header = f"Content-Length: {len(body.encode('utf-8'))}\r\n\r\n"
        try:
            self._server.stdin.write(header.encode("utf-8"))
            self._server.stdin.write(body.encode("utf-8"))
            self._server.stdin.flush()
        except (BrokenPipeError, OSError):
            self._available = False

    def _send_request(self, method: str, params: dict, timeout: float = 5.0) -> Any:
        """Send a request and wait for the response."""
        req_id = self._next_id()
        event = threading.Event()
        with self._lock:
            self._response_events[req_id] = event

        self._send_message({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        })

        if event.wait(timeout=timeout):
            with self._lock:
                result = self._responses.pop(req_id, None)
                self._response_events.pop(req_id, None)
            return result
        else:
            with self._lock:
                self._response_events.pop(req_id, None)
            return None

    def _send_notification(self, method: str, params: dict) -> None:
        """Send a notification (no response expected)."""
        self._send_message({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        })

    def _read_responses(self) -> None:
        """Background thread reading JSON-RPC responses from the server."""
        if not self._server or not self._server.stdout:
            return
        try:
            while self._server.poll() is None:
                # Read headers
                headers = {}
                while True:
                    line = self._server.stdout.readline()
                    if not line:
                        return
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        break
                    if ":" in line_str:
                        key, val = line_str.split(":", 1)
                        headers[key.strip().lower()] = val.strip()

                content_length = int(headers.get("content-length", 0))
                if content_length <= 0:
                    continue

                body = self._server.stdout.read(content_length)
                if not body:
                    return

                try:
                    msg = json.loads(body.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue

                if "id" in msg and "method" not in msg:
                    req_id = msg["id"]
                    with self._lock:
                        self._responses[req_id] = msg.get("result")
                        event = self._response_events.get(req_id)
                        if event:
                            event.set()
        except Exception:
            pass

    def _file_uri(self, file_path: str) -> str:
        """Convert a relative file path to a file:// URI."""
        abs_path = str(Path(self.repo_path, file_path).resolve())
        return f"file://{abs_path.replace(os.sep, '/')}"

    def find_definition(
        self, symbol: str, file_path: str, line: int, character: int = 0,
    ) -> dict[str, Any] | None:
        """Find where a symbol is defined using LSP textDocument/definition.

        Returns ``{file, line, end_line}`` or ``None``.
        """
        if not self._available:
            return None

        uri = self._file_uri(file_path)
        result = self._send_request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character},
        })

        if not result:
            return None

        # Handle Location or Location[]
        locations = result if isinstance(result, list) else [result]
        if not locations:
            return None

        loc = locations[0]
        target_uri = loc.get("uri", "")
        target_range = loc.get("range", {})
        start = target_range.get("start", {})
        end = target_range.get("end", {})

        # Convert URI back to relative path
        rel_path = target_uri.replace("file://", "")
        try:
            rel_path = str(Path(rel_path).relative_to(self.repo_path))
        except (ValueError, TypeError):
            pass

        return {
            "file": rel_path.replace("\\", "/"),
            "line": start.get("line", 0) + 1,
            "end_line": end.get("line", 0) + 1,
        }

    def find_references(
        self, symbol: str, file_path: str, line: int, character: int = 0,
    ) -> list[dict[str, Any]]:
        """Find all references to a symbol using LSP textDocument/references.

        Returns ``[{file, line, context}]``.
        """
        if not self._available:
            return []

        uri = self._file_uri(file_path)
        result = self._send_request("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character},
            "context": {"includeDeclaration": False},
        })

        if not result or not isinstance(result, list):
            return []

        refs: list[dict[str, Any]] = []
        for loc in result:
            target_uri = loc.get("uri", "")
            target_range = loc.get("range", {})
            start = target_range.get("start", {})

            rel_path = target_uri.replace("file://", "")
            try:
                rel_path = str(Path(rel_path).relative_to(self.repo_path))
            except (ValueError, TypeError):
                pass

            refs.append({
                "file": rel_path.replace("\\", "/"),
                "line": start.get("line", 0) + 1,
                "context": "",
            })

        return refs

    def find_hover(
        self, file_path: str, line: int, character: int,
    ) -> str | None:
        """Get type/docstring info for a position using LSP textDocument/hover."""
        if not self._available:
            return None

        uri = self._file_uri(file_path)
        result = self._send_request("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character},
        })

        if not result:
            return None

        contents = result.get("contents", "")
        if isinstance(contents, dict):
            return contents.get("value", "")
        if isinstance(contents, str):
            return contents
        if isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, dict):
                    parts.append(item.get("value", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return None

    def shutdown(self) -> None:
        """Clean shutdown of the LSP server subprocess."""
        self._shutdown_server()

    def _shutdown_server(self) -> None:
        if self._server and self._server.poll() is None:
            try:
                self._send_request("shutdown", {}, timeout=3.0)
                self._send_notification("exit", {})
                self._server.wait(timeout=5)
            except Exception:
                try:
                    self._server.kill()
                except Exception:
                    pass
        self._server = None
        self._available = False

    def __del__(self) -> None:
        self.shutdown()
