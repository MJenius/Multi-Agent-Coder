from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.config import RESEARCHER_MODEL_CANDIDATES
from issue_resolver.llm_utils import invoke_with_role_fallback
from issue_resolver.utils.ripgrep_search import smart_search, generate_search_variants
from issue_resolver.tools import (
    REPO_TOOLS,
    list_files,
    search_code,
    read_file,
    generate_repo_map,
    get_symbol_definition,
)


_TOOL_MAP = {
    "list_files": list_files,
    "search_code": search_code,
    "read_file": read_file,
    "generate_repo_map": generate_repo_map,
    "get_symbol_definition": get_symbol_definition,
}

_SYSTEM_PROMPT = """\
You are the Researcher agent. Find the source code relevant to a GitHub issue FAST.

Available tools:
  read_file(file_path)               - Read a file (truncated at 500 lines). USE THIS FIRST if you know the file.
  search_code(query, directory)      - Grep for a string across code files.
  get_symbol_definition(symbol, dir) - Find where a function/class is defined.
  generate_repo_map(directory)       - Get a tree view of the repo structure. ONLY if you don't know where to look.
  list_files(directory)              - List code files in a specific folder.

SPEED RULES:
1. If the issue mentions a SPECIFIC FILE PATH, call read_file() IMMEDIATELY.
2. If the issue has a HINT, follow the hint directly with read_file().
3. Only call generate_repo_map() if you have NO idea where the relevant code is.
4. Target searches to SPECIFIC folders, never search root '.'.
5. Read up to 3 target files. Multi-file context is often needed.
6. After reading 3 files OR hitting the line limit, STOP and summarize findings.

CONSTRAINTS:
- NEVER read more than 3 files total.
- NEVER use list_files on root directory for large repos.
- Prefer search_code with specific folder paths over broad searches.
- When done, simply state what you found.
"""

_MAX_TOOL_ROUNDS = 8
_MAX_FILES_READ = 3
_MAX_TOTAL_LINES = 500

_CODE_EXTENSIONS = r'(?:cs|py|js|ts|tsx|xaml|java|go|cpp|h|jsx|csproj|sln|rb|rs|swift|kt)'


def _extract_hints_from_issue(issue_text: str) -> list[str]:
    path_pattern = rf'(?:\./)?([A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-.]+)*\.(?:{_CODE_EXTENSIONS}))'
    all_matches = re.findall(path_pattern, issue_text)

    seen = set()
    unique_paths = []
    for match in all_matches:
        normalized = match.lstrip('./')
        if normalized not in seen:
            unique_paths.append(normalized)
            seen.add(normalized)

    final_hints = []
    for path in unique_paths:
        if '/' not in path:
            has_full_path = any(path in p and '/' in p for p in unique_paths)
            if not has_full_path:
                final_hints.append(path)
        else:
            final_hints.append(path)

    return final_hints


def _detect_language(repo_path: str) -> str:
    repo = Path(repo_path).resolve()

    marker_files = {
        "csharp": ["*.sln", "*.csproj"],
        "python": ["setup.py", "requirements.txt", "pyproject.toml", "setup.cfg"],
        "nodejs": ["package.json", "package-lock.json"],
        "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    }

    for lang, patterns in marker_files.items():
        for pattern in patterns:
            if "*" in pattern:
                if list(repo.glob(pattern)):
                    return lang
            else:
                if (repo / pattern).exists():
                    return lang

    ext_counts = {"csharp": 0, "python": 0, "nodejs": 0, "java": 0}
    ext_map = {".cs": "csharp", ".py": "python", ".js": "nodejs", ".ts": "nodejs", ".java": "java"}

    try:
        for item in repo.iterdir():
            if item.is_file() and item.suffix in ext_map:
                ext_counts[ext_map[item.suffix]] += 1
            elif item.is_dir() and item.name not in {
                'bin', 'obj', '.git', 'node_modules', '__pycache__', '.vs'
            }:
                try:
                    for child in item.iterdir():
                        if child.is_file() and child.suffix in ext_map:
                            ext_counts[ext_map[child.suffix]] += 1
                except (OSError, PermissionError):
                    continue
    except (OSError, PermissionError):
        pass

    max_lang = max(ext_counts, key=ext_counts.get)
    return max_lang if ext_counts[max_lang] > 0 else "unknown"


def _try_search_variations(
    base_query: str,
    directory: str,
    language: str = "unknown",
    max_attempts: int = 3,
) -> tuple[str, bool]:
    queries = [base_query]

    snake_case = re.sub(r'([A-Z])', r'_\1', base_query).lower().lstrip('_')
    if snake_case != base_query.lower() and snake_case not in queries:
        queries.append(snake_case)

    if base_query.lower() not in queries:
        queries.append(base_query.lower())

    tokens = re.split(r'([A-Z][a-z]+|[a-z]+|_)', base_query)
    tokens = [t for t in tokens if t and t != '_']
    for token in tokens[:max_attempts]:
        if token.lower() not in queries:
            queries.append(token.lower())

    print(f"[Researcher] Search variations for '{base_query}': {queries[:max_attempts]}")

    for attempt, query in enumerate(queries[:max_attempts], 1):
        try:
            result = search_code.invoke({"query": query, "directory": directory})
            match_count = len([
                line for line in result.split('\n')
                if line.strip() and ':' in line and not line.startswith('[')
            ])
            if match_count > 0:
                print(f"[Researcher] Found {match_count} match(es) for '{query}' (attempt {attempt})")
                return result, True
            else:
                print(f"[Researcher] No matches for '{query}' (attempt {attempt})")
        except Exception as e:
            print(f"[Researcher] Error searching for '{query}': {e}")

    return f"No matches found after {len(queries)} search variations of '{base_query}'.", False


def _extract_keywords_from_issue(issue_text: str) -> list[str]:
    _STOP_WORDS = {
        'always', 'using', 'encode', 'should', 'would', 'could', 'there',
        'title', 'issue', 'error', 'fixed', 'fails', 'build', 'tests',
        'false', 'true', 'null', 'none', 'undefined',
    }

    keywords: list[str] = []

    repro_match = re.search(
        r'(?:##\s+)?(?:To Reproduce|Steps to reproduce|Reproduction Steps|REPRO):?\n(.*?)(?:\n##|$)',
        issue_text,
        re.IGNORECASE | re.DOTALL,
    )
    if repro_match:
        repro_section = repro_match.group(1)
        code_block_pattern = r'```(?:python|javascript|java|csharp|cs|js|py)?\n(.*?)```'
        for code_block in re.findall(code_block_pattern, repro_section, re.DOTALL):
            identifiers = re.findall(
                r'\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*)\b', code_block, re.IGNORECASE
            )
            for identifier in identifiers:
                parts = identifier.split('.')
                for part in parts:
                    if 4 <= len(part) <= 20 and part.lower() not in _STOP_WORDS and part not in keywords:
                        keywords.append(part)

    code_block_pattern = r'```(?:python|javascript|java|csharp|cs|js|py)?\n(.*?)```'
    for code_block in re.findall(code_block_pattern, issue_text, re.DOTALL):
        identifiers = re.findall(
            r'\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*)\b', code_block, re.IGNORECASE
        )
        for identifier in identifiers:
            parts = identifier.split('.')
            for part in parts:
                if 4 <= len(part) <= 20 and part.lower() not in _STOP_WORDS and part not in keywords:
                    keywords.append(part)

    for m in re.findall(r'`([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?)`', issue_text):
        name = m.split('(')[0]
        if len(name) >= 4 and name.lower() not in _STOP_WORDS and name not in keywords:
            keywords.append(name)

    for m in re.findall(r'\b([a-z_][a-zA-Z0-9_]{3,})\b', issue_text):
        has_mixed_case = any(c.isupper() for c in m)
        has_underscore = '_' in m
        if (has_mixed_case or has_underscore) and m.lower() not in _STOP_WORDS and m not in keywords:
            keywords.append(m)

    title_line = issue_text.splitlines()[0].strip()
    if title_line.lower().startswith("title:"):
        title_line = title_line[6:].strip()

    if title_line:
        for m in re.findall(r'\b([A-Z]{2,})\b', title_line):
            if m.lower() not in _STOP_WORDS and m not in keywords:
                keywords.append(m)

        for m in re.findall(r'\b([A-Za-z][A-Za-z0-9]*-[A-Za-z0-9]+)\b', title_line):
            if m.lower() not in _STOP_WORDS and m not in keywords:
                keywords.append(m)

        for m in re.findall(r'\b([A-Za-z]{5,})\b', title_line):
            if m.lower() not in _STOP_WORDS and m.lower() not in {k.lower() for k in keywords}:
                keywords.append(m)

    seen: set[str] = set()
    unique: list[str] = []

    keywords_sorted = sorted(
        keywords,
        key=lambda x: (0 if '_' in x else 1, -len(x)),
    )

    for k in keywords_sorted:
        if k.lower() not in seen:
            seen.add(k.lower())
            unique.append(k)

    return unique


def _get_top_file_from_search(search_result: str) -> str | None:
    from collections import Counter
    file_counts: Counter[str] = Counter()

    for line in search_result.split('\n'):
        if ':' in line and not line.startswith('['):
            parts = line.split(':', 2)
            if len(parts) >= 2:
                file_path = parts[0].strip()
                if file_path:
                    file_counts[file_path.replace('\\', '/')] += 1

    if not file_counts:
        return None

    candidates = file_counts.most_common()

    def _score(path: str) -> int:
        name = Path(path).name.lower()
        path_lower = path.lower()
        if 'test' in path_lower or 'spec' in path_lower:
            return 3
        if 'index' in name or 'main' in name:
            return 2
        if 'src/' in path_lower or 'lib/' in path_lower:
            return 0
        return 1

    best = min(candidates, key=lambda item: (_score(item[0]), -item[1]))
    return best[0]


def researcher_node(state: AgentState) -> dict:
    print("[Researcher] Starting codebase exploration...")

    repo_path = state.get("repo_path", ".")
    issue_text = state.get("issue", "(no issue provided)")
    errors = state.get("errors", "")

    human_str = f"GitHub Issue:\n{issue_text}\n\nRepository path: {repo_path}\n\n"
    if errors:
        human_str += f"Supervisor Feedback/Errors:\n{errors}\n\n"

    issue_lower = issue_text.lower()
    if any(kw in issue_lower for kw in ["encod", "eci", "utf-8", "utf8", "charset", "character encode"]):
        human_str += "HINT: This is an ENCODING issue. Look for classes/enums with names like:\n"
        human_str += "  - Data, Generator, Encoder, Manager\n"
        human_str += "  - Methods: Encode, Compress, Prepare, SetEncoding\n"
        human_str += "  - Enums: EncodingMode, ECI, Compression, CharacterSet\n"
        human_str += "Search for these patterns first.\n\n"
    elif any(kw in issue_lower for kw in ["null", "npe", "exception", "error"]):
        human_str += "HINT: This is an ERROR handling issue. Look for:\n"
        human_str += "  - Methods that could throw exceptions\n"
        human_str += "  - Missing null checks or validations\n"
        human_str += "  - Error handling patterns\n\n"

    human_str += "Please explore the repository and find the relevant code."

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_str),
    ]

    snippets: list[str] = list(state.get("file_context", []))
    files_read = 0
    total_lines = 0

    history_additions: list[dict] = []

    history_additions.extend(
        append_to_history("Researcher", "Strategic Targeting", f"Repo: {repo_path}\nIssue: {issue_text}")
    )

    language = _detect_language(repo_path)
    print(f"[Researcher] Detected language: {language}")
    history_additions.extend(append_to_history("Researcher", "Language Detection", f"Language: {language}"))

    contribution_guidelines = ""
    contributing_path = Path(repo_path) / "CONTRIBUTING.md"
    if contributing_path.exists():
        try:
            result = read_file.invoke({"file_path": str(contributing_path.resolve())})
            if not result.startswith("Error"):
                contribution_guidelines = result
                line_count = result.count("\n")
                print(f"[Researcher] Read CONTRIBUTING.md ({line_count} lines)")
                history_additions.extend(
                    append_to_history("Researcher", "Contributing Guide", f"Read CONTRIBUTING.md ({line_count} lines)")
                )
        except Exception as e:
            print(f"[Researcher] Could not read CONTRIBUTING.md: {e}")

    hint_files = _extract_hints_from_issue(issue_text)
    if hint_files:
        print(f"[Researcher] Found {len(hint_files)} direct hint(s): {hint_files}")
        history_additions.extend(append_to_history("Researcher", "Hint Extraction", f"Hints: {hint_files}"))

        for idx, hint_file in enumerate(hint_files[:_MAX_FILES_READ], 1):
            if files_read >= _MAX_FILES_READ:
                break

            normalized_hint = hint_file.lstrip('./')
            repo_name = Path(repo_path).name
            if normalized_hint.startswith(repo_name + '/'):
                normalized_hint = normalized_hint[len(repo_name) + 1:]

            safe_path_resolved = (Path(repo_path) / normalized_hint).resolve()

            if not safe_path_resolved.is_file():
                print(f"[Researcher] Skipping '{hint_file}' (not a real file)")
                history_additions.extend(
                    append_to_history("Researcher", "Hint Skip", f"'{hint_file}' not found in repo")
                )
                continue

            try:
                result = read_file.invoke({"file_path": str(safe_path_resolved)})
                if result.startswith("Error"):
                    print(f"[Researcher] Failed to read {hint_file}: {result[:100]}")
                    history_additions.extend(
                        append_to_history("Researcher", "Hint Read", f"Failed: {hint_file} ({result[:80]})")
                    )
                else:
                    lines_in_file = result.count("\n")
                    print(f"[Researcher] Read {lines_in_file} lines from {hint_file}")
                    snippet = f"# --- [HINTED] file: {hint_file} ---\n{result}"
                    snippets.append(snippet)
                    files_read += 1
                    total_lines += lines_in_file + 1
                    history_additions.extend(
                        append_to_history("Researcher", "Hint Read", f"{hint_file} ({lines_in_file} lines)")
                    )
            except Exception as e:
                print(f"[Researcher] Exception reading {hint_file}: {e}")
                history_additions.extend(
                    append_to_history("Researcher", "Hint Read", f"Exception: {str(e)[:80]}")
                )

    if snippets and files_read >= 1:
        print(
            f"[Researcher] Hints provided {files_read} file(s), {total_lines} lines. "
            f"Skipping LLM search."
        )
        history_additions.extend(
            append_to_history(
                "Researcher",
                "Targeting Complete",
                f"Collected {len(snippets)} snippets (from hints). Read {files_read} files.",
            )
        )
        return_dict: dict = {
            "file_context": snippets,
            "history": history_additions,
        }
        if contribution_guidelines:
            return_dict["contribution_guidelines"] = contribution_guidelines
        return return_dict

    if not snippets:
        keywords = _extract_keywords_from_issue(issue_text)
        if keywords:
            print(f"[Researcher] Auto-search keywords from issue: {keywords[:3]}")
            history_additions.extend(
                append_to_history("Researcher", "Auto-Search", f"Keywords: {keywords[:3]}")
            )

            for keyword in keywords[:2]:
                if files_read >= _MAX_FILES_READ:
                    break
                try:
                    ripgrep_matches = smart_search(
                        keyword, repo_path, prefer_core_lib=True, max_results=10
                    )
                    if ripgrep_matches:
                        match_count = len(ripgrep_matches)
                        variants_found = generate_search_variants(keyword)
                        print(
                            f"[Researcher] Ripgrep '{keyword}' (variants: {variants_found}): "
                            f"{match_count} match(es)"
                        )

                        top_match = ripgrep_matches[0]
                        top_file = top_match['file']

                        if top_file:
                            top_file_normalized = top_file.replace('\\', '/')
                            file_path = str((Path(repo_path) / top_file_normalized).resolve())
                            file_result = read_file.invoke({"file_path": file_path})
                            if not file_result.startswith("Error"):
                                lines_in_file = file_result.count("\n")
                                print(f"[Researcher] Auto-read '{top_file_normalized}' ({lines_in_file} lines)")
                                snippet = f"# --- file: {top_file_normalized} ---\n{file_result}"
                                snippets.append(snippet)
                                files_read += 1
                                total_lines += lines_in_file + 1
                                history_additions.extend(
                                    append_to_history(
                                        "Researcher",
                                        "Auto-Read (Ripgrep)",
                                        f"{top_file_normalized} ({lines_in_file} lines)",
                                    )
                                )
                                continue

                    result = search_code.invoke({"query": keyword, "directory": repo_path})
                    if not result.startswith("No matches"):
                        match_count = len([
                            l for l in result.split('\n')
                            if l.strip() and ':' in l and not l.startswith('[')
                        ])
                        print(f"[Researcher] Auto-search '{keyword}': {match_count} match(es)")

                        top_file = _get_top_file_from_search(result)
                        if top_file:
                            top_file_normalized = top_file.replace('\\', '/')
                            file_path = str((Path(repo_path) / top_file_normalized).resolve())
                            file_result = read_file.invoke({"file_path": file_path})
                            if not file_result.startswith("Error"):
                                lines_in_file = file_result.count("\n")
                                print(f"[Researcher] Auto-read '{top_file_normalized}' ({lines_in_file} lines)")
                                snippet = f"# --- file: {top_file_normalized} ---\n{file_result}"
                                snippets.append(snippet)
                                files_read += 1
                                total_lines += lines_in_file + 1
                                history_additions.extend(
                                    append_to_history(
                                        "Researcher",
                                        "Auto-Read",
                                        f"{top_file_normalized} ({lines_in_file} lines)",
                                    )
                                )
                                break
                    else:
                        print(f"[Researcher] Auto-search '{keyword}': no matches in {repo_path}")
                        if repo_path != ".":
                            print("[Researcher] Retrying search at root directory (.)")
                            result = search_code.invoke({"query": keyword, "directory": "."})
                            if not result.startswith("No matches"):
                                top_file = _get_top_file_from_search(result)
                                if top_file:
                                    file_path = str(
                                        (Path(repo_path) / top_file.replace('\\', '/')).resolve()
                                    )
                                    file_result = read_file.invoke({"file_path": file_path})
                                    if not file_result.startswith("Error"):
                                        lines_in_file = file_result.count("\n")
                                        print(
                                            f"[Researcher] Auto-read '{top_file}' "
                                            f"({lines_in_file} lines)"
                                        )
                                        snippet = f"# --- file: {top_file} ---\n{file_result}"
                                        snippets.append(snippet)
                                        files_read += 1
                                        total_lines += lines_in_file + 1
                                        history_additions.extend(
                                            append_to_history(
                                                "Researcher",
                                                "Auto-Read (Root)",
                                                f"{top_file} ({lines_in_file} lines)",
                                            )
                                        )
                                        break
                except Exception as e:
                    print(f"[Researcher] Auto-search error for '{keyword}': {e}")

    max_rounds = 3

    if snippets:
        context_note = (
            f"\n\nIMPORTANT: The system already read {files_read} file(s) from hints "
            f"({total_lines} lines). These files are already in context. "
            f"Focus ONLY on finding additional relevant files. "
            f"You need at most {_MAX_FILES_READ - files_read} more file(s)."
        )
        messages[-1] = HumanMessage(content=messages[-1].content + context_note)

    for round_num in range(1, max_rounds + 1):
        print(f"[Researcher]  |-- Round {round_num}/{max_rounds}")

        try:
            response, chosen_model = invoke_with_role_fallback(
                role="Researcher",
                candidates=RESEARCHER_MODEL_CANDIDATES,
                messages=messages,
                temperature=0,
                tools=REPO_TOOLS,
            )
            if round_num == 1:
                print(f"[Researcher] Using model: {chosen_model}")
        except Exception as exc:
            print(f"[Researcher] [ERROR] LLM call failed: {exc}")
            break

        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            print("[Researcher] No more tool calls -- wrapping up.")
            break

        for tc in tool_calls:
            fn_name = tc["name"]
            fn_args = tc["args"]
            call_id = tc["id"]

            print(f"[Researcher]    --> {fn_name}({fn_args})")

            log_payload = f"Tool: {fn_name}\nArgs: {json.dumps(fn_args)}"
            history_additions.extend(
                append_to_history("Researcher", "Tool Call", log_payload, max_length=150)
            )

            if fn_name == "read_file":
                if files_read >= _MAX_FILES_READ:
                    result = (
                        f"[BLOCKED] Already read {_MAX_FILES_READ} files -- "
                        "limit reached. Please summarise with what you have."
                    )
                    messages.append(ToolMessage(content=result, tool_call_id=call_id))
                    continue
                if total_lines >= _MAX_TOTAL_LINES:
                    result = (
                        f"[BLOCKED] Already read {total_lines} lines -- "
                        "line budget exhausted."
                    )
                    messages.append(ToolMessage(content=result, tool_call_id=call_id))
                    continue

            tool_fn = _TOOL_MAP.get(fn_name)
            if tool_fn is None:
                result = f"Unknown tool '{fn_name}'."
            else:
                try:
                    result = tool_fn.invoke(fn_args)
                except Exception as exc:
                    result = f"Tool error: {exc}"

            if fn_name == "read_file" and not result.startswith(("Error", "[BLOCKED")):
                files_read += 1
                total_lines += result.count("\n") + 1
                file_label = fn_args.get("file_path", "unknown")
                snippet = f"# --- file: {file_label} ---\n{result}"
                snippets.append(snippet)

            if fn_name == "search_code" and not result.startswith(("No matches", "Error", "ERROR")):
                top_file = _get_top_file_from_search(result)
                if top_file and files_read < _MAX_FILES_READ:
                    top_file_normalized = top_file.replace('\\', '/')
                    auto_path = str((Path(repo_path) / top_file_normalized).resolve())
                    try:
                        auto_result = read_file.invoke({"file_path": auto_path})
                        if not auto_result.startswith(("Error", "[BLOCKED")):
                            lines_in_auto = auto_result.count("\n")
                            print(
                                f"[Researcher]    Auto-read from search: "
                                f"'{top_file_normalized}' ({lines_in_auto} lines)"
                            )
                            snippet = f"# --- file: {top_file_normalized} ---\n{auto_result}"
                            snippets.append(snippet)
                            files_read += 1
                            total_lines += lines_in_auto + 1
                    except Exception:
                        pass

            messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    if not snippets:
        print("[Researcher] [FALLBACK] All phases returned 0 snippets. Running last-resort scan.")
        history_additions.extend(
            append_to_history(
                "Researcher",
                "Last-Resort Scan",
                "LLM returned no tool calls. Scanning repo directly.",
            )
        )

        try:
            map_result = generate_repo_map.invoke({"directory": repo_path})
            if map_result and not map_result.startswith(("Error", "ERROR")):
                print(f"[Researcher] [FALLBACK] Repo map obtained ({len(map_result)} chars).")
        except Exception as map_exc:
            map_result = ""
            print(f"[Researcher] [FALLBACK] Repo map failed: {map_exc}")

        fallback_keywords = _extract_keywords_from_issue(issue_text)
        print(f"[Researcher] [FALLBACK] Searching for title terms: {fallback_keywords[:4]}")

        for term in fallback_keywords[:4]:
            if files_read >= _MAX_FILES_READ:
                break
            try:
                result = search_code.invoke({"query": term, "directory": repo_path})
                if result and not result.startswith(("No matches", "Error", "ERROR")):
                    top_file = _get_top_file_from_search(result)
                    if top_file and files_read < _MAX_FILES_READ:
                        top_file_normalized = top_file.replace("\\", "/")
                        auto_path = str((Path(repo_path) / top_file_normalized).resolve())
                        try:
                            file_result = read_file.invoke({"file_path": auto_path})
                            if not file_result.startswith(("Error", "[BLOCKED")):
                                lines_found = file_result.count("\n")
                                print(
                                    f"[Researcher] [FALLBACK] Read "
                                    f"'{top_file_normalized}' ({lines_found} lines)"
                                )
                                snippets.append(
                                    f"# --- file: {top_file_normalized} ---\n{file_result}"
                                )
                                files_read += 1
                                total_lines += lines_found + 1
                                history_additions.extend(
                                    append_to_history(
                                        "Researcher",
                                        "Last-Resort Read",
                                        f"{top_file_normalized} ({lines_found} lines) via '{term}'",
                                    )
                                )
                                break
                        except Exception:
                            pass
            except Exception as search_exc:
                print(f"[Researcher] [FALLBACK] Search error for '{term}': {search_exc}")

        if not snippets and map_result:
            snippets.append(f"# --- repo map ---\n{map_result[:3000]}")
            history_additions.extend(
                append_to_history(
                    "Researcher",
                    "Last-Resort Read",
                    "Using repo map as context (no source file found)",
                )
            )

    print(
        f"[Researcher] Done -- collected {len(snippets)} snippet(s), "
        f"{files_read} file(s) read, ~{total_lines} lines."
    )

    history_additions.extend(
        append_to_history(
            "Researcher",
            "Targeting Complete",
            f"Collected {len(snippets)} snippets. Read {files_read} files.",
        )
    )

    return_dict: dict = {
        "file_context": snippets,
        "history": history_additions,
    }
    if contribution_guidelines:
        return_dict["contribution_guidelines"] = contribution_guidelines
    return return_dict
