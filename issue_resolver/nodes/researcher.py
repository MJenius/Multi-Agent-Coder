"""
Researcher Node -- Phase 2: LLM-driven codebase exploration.

Uses Groq-hosted models with tool-binding to:
  1. List .py files in the target repository.
  2. Search for relevant function / class names mentioned in the issue.
  3. Read the most relevant files (max 3 files, <=500 lines each).
  4. Populate file_context with discovered code snippets.
  
Phase 1 Improvements:
  - Detects direct file hints (e.g., "🎯 HINT: QRCoder/AsciiQRCode.cs")
  - Detects repository language (C#, Python, Node.js, Java)
  - Implements fallback search strategy for 0-result queries
  - Timeouts on all tool calls to prevent hangs

Phase 3A Improvements:
  - Ripgrep-based search with case variant detection
  - CamelCase / snake_case / kebab-case variant generation
  - Core library prioritization over test files
"""

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
    get_symbol_definition
)


# Map tool names --> callables for dispatching
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
  search_code(query, directory)      - Grep for a string across code files. EXCELLENT for exact identifier searches.
  get_symbol_definition(symbol, dir) - Find where a function/class is defined.
  generate_repo_map(directory)       - Get a tree view of the repo structure. ONLY if you don't know where to look.
  list_files(directory)              - List code files in a specific folder.

SPEED RULES (CRITICAL):
──────────────────────
1. If the issue mentions a SPECIFIC FILE PATH → call read_file() IMMEDIATELY. Do NOT map first.
2. If the issue has a HINT (🎯) → follow the hint directly with read_file().
3. Only call generate_repo_map() if you have NO idea where the relevant code is.
4. Target searches to SPECIFIC folders (e.g., './QRCoder', './src'), never search root '.'.
5. For encoding/mode issues: Search for related classes (Data, Generator, Encoder, Manager, etc.)
6. Read up to 3 target files. Multi-file context is often needed for architectural issues.
7. After reading 3 files OR hitting the line limit, STOP and summarize findings.

EXACT IDENTIFIER SEARCH (Phase 3A):
──────────────────────────────────
If the issue mentions a specific attribute or identifier (e.g., "subscription_item", "calculateTotal"):
→ Use search_code(query, directory) to find where this exact identifier appears in the code.
→ Example: search_code("subscription_item", "./") will find all occurrences of that attribute.
→ This is far more reliable than manual browsing and will quickly pinpoint the issue location.

ISSUE-SPECIFIC GUIDANCE:
──────────────────────
- Encoding/ECI mode issue? → Find the Data or Encoder class, look for mode/encoding enums or configs
- Null/error handling? → Find the method + surrounding error checks
- Performance issue? → Find the hot loop/class + its dependencies
- AttributeError on optional field? → Find where the field is accessed and where it may be missing

CONSTRAINTS:
- NEVER read more than 3 files total.
- NEVER use list_files on root directory for large repos.
- Prefer search_code with specific folder paths over broad searches.
- When done, simply state what you found. No need for additional exploration.
"""


# ---------------------------------------------------------------------------
# Constants -- memory guards
# ---------------------------------------------------------------------------
_MAX_TOOL_ROUNDS = 8   # Increased to allow more search/read cycles for complex issues
_MAX_FILES_READ = 3    # cap on read_file calls
_MAX_TOTAL_LINES = 500  # soft cap across all files read


# ---------------------------------------------------------------------------
# PHASE 1 IMPROVEMENTS: Helper Functions
# ---------------------------------------------------------------------------

# Supported code file extensions for hint extraction
_CODE_EXTENSIONS = r'(?:cs|py|js|ts|tsx|xaml|java|go|cpp|h|jsx|csproj|sln|rb|rs|swift|kt)'

def _extract_hints_from_issue(issue_text: str) -> list[str]:
    """Extract ONLY file paths from issue description. Never returns sentences.
    
    Strategy: Find all substrings that look like file paths (e.g., QRCoder/AsciiQRCode.cs)
    and return just those paths. Sentences containing paths are NOT returned.
    
    Args:
        issue_text: GitHub issue body text.
    
    Returns:
        List of file paths found (relative paths like "QRCoder/AsciiQRCode.cs").
    """
    # Single robust pattern: match file paths with optional directory components
    # Matches: "QRCoder/AsciiQRCode.cs", "src/main.py", "Button.tsx", "./utils/helper.js"
    # Does NOT match full sentences or prose text
    path_pattern = rf'(?:\./)?([A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-.]+)*\.(?:{_CODE_EXTENSIONS}))'
    all_matches = re.findall(path_pattern, issue_text)
    
    # Deduplicate while preserving order, normalize leading ./
    seen = set()
    unique_paths = []
    for match in all_matches:
        normalized = match.lstrip('./')
        if normalized not in seen:
            unique_paths.append(normalized)
            seen.add(normalized)
    
    # Prefer full paths (with /) over bare filenames
    # If we have both "AsciiQRCode.cs" and "QRCoder/AsciiQRCode.cs", keep only the full path
    final_hints = []
    for path in unique_paths:
        if '/' not in path:
            # Bare filename -- only keep if no full path version exists
            has_full_path = any(path in p and '/' in p for p in unique_paths)
            if not has_full_path:
                final_hints.append(path)
        else:
            final_hints.append(path)
    
    return final_hints


def _detect_language(repo_path: str) -> str:
    """Detect primary programming language of the repository (fast, no deep scan).
    
    Checks only for known marker files and top-level/immediate child extensions.
    Never uses recursive globs (those can hang on 2500+ file repos).
    
    Args:
        repo_path: Path to repository root.
    
    Returns:
        Language identifier: "csharp", "python", "nodejs", "java", "unknown"
    """
    repo = Path(repo_path).resolve()
    
    # Fast check: known marker FILES (no recursive glob needed)
    marker_files = {
        "csharp": ["*.sln", "*.csproj"],
        "python": ["setup.py", "requirements.txt", "pyproject.toml", "setup.cfg"],
        "nodejs": ["package.json", "package-lock.json"],
        "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    }
    
    for lang, patterns in marker_files.items():
        for pattern in patterns:
            if "*" in pattern:
                # Glob only in root dir (non-recursive)
                if list(repo.glob(pattern)):
                    return lang
            else:
                if (repo / pattern).exists():
                    return lang
    
    # Quick fallback: check root + 1 level deep for common extensions
    ext_counts = {"csharp": 0, "python": 0, "nodejs": 0, "java": 0}
    ext_map = {".cs": "csharp", ".py": "python", ".js": "nodejs", ".ts": "nodejs", ".java": "java"}
    
    try:
        for item in repo.iterdir():
            if item.is_file() and item.suffix in ext_map:
                ext_counts[ext_map[item.suffix]] += 1
            elif item.is_dir() and item.name not in {'bin', 'obj', '.git', 'node_modules', '__pycache__', '.vs'}:
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
    max_attempts: int = 3
) -> tuple[str, bool]:
    """Try progressively broader searches if base query fails.
    
    Implements fallback strategy:
      1. Original query: "ASCIIQRCode"
      2. CamelCase → snake_case: "ascii_qr_code"
      3. Partial tokens: "ASCII", "QRCode", etc.
    
    Args:
        base_query: Original search term.
        directory: Directory to search in.
        language: Detected language ("csharp", "python", etc.) for smarter fallbacks.
        max_attempts: Maximum number of search variations to try.
    
    Returns:
        Tuple of (result_text, found_matches). found_matches=True if matches > 0.
    """
    queries = [base_query]
    
    # Add snake_case variant
    snake_case = re.sub(r'([A-Z])', r'_\1', base_query).lower().lstrip('_')
    if snake_case != base_query.lower() and snake_case not in queries:
        queries.append(snake_case)
    
    # Add lowercase variant
    if base_query.lower() not in queries:
        queries.append(base_query.lower())
    
    # Add partial tokens (split on CamelCase or underscores)
    tokens = re.split(r'([A-Z][a-z]+|[a-z]+|_)', base_query)
    tokens = [t for t in tokens if t and t != '_']
    for token in tokens[:max_attempts]:
        if token.lower() not in queries:
            queries.append(token.lower())
    
    print(f"[Researcher] Search variations for '{base_query}': {queries[:max_attempts]}")
    
    for attempt, query in enumerate(queries[:max_attempts], 1):
        try:
            result = search_code.invoke({"query": query, "directory": directory})
            match_count = len([line for line in result.split('\n') if line.strip() and ':' in line and not line.startswith('[')])
            
            if match_count > 0:
                print(f"[Researcher] ✅ Found {match_count} match(es) for '{query}' (attempt {attempt})")
                return result, True
            else:
                print(f"[Researcher] ❌ No matches for '{query}' (attempt {attempt})")
        except Exception as e:
            print(f"[Researcher] Error searching for '{query}': {e}")
    
    return f"No matches found after {len(queries)} search variations of '{base_query}'.", False


def _extract_keywords_from_issue(issue_text: str) -> list[str]:
    """Extract searchable identifiers and technical terms from issue text.

    Strategy (in priority order):
      1. Identifiers from code blocks: `subscription_item`, `item.subscription_item`, etc.
      2. Backtick-wrapped identifiers: `isMobilePhone`, `calculate_total()`
      3. snake_case and camelCase names in prose (includes 4+ char mixed case)
      4. All-caps abbreviations from the title: UTF, ECI, HTTP, URL
      5. Hyphenated technical terms from the title: UTF-8, ISO-8859, Base64
      6. Meaningful words (>=5 chars) from the title as last-resort search seeds
    """
    _STOP_WORDS = {
        'always', 'using', 'encode', 'should', 'would', 'could', 'there',
        'title', 'issue', 'error', 'fixed', 'fails', 'build', 'tests',
        'false', 'true', 'null', 'none', 'undefined',
    }

    keywords: list[str] = []

    # 0. Extract identifiers from code blocks (PRE-CALL: highest priority)
    # Matches: ```python ... item.subscription_item ... ```
    code_block_pattern = r'```(?:python|javascript|java|csharp|cs|js|py)?\n(.*?)```'
    for code_block in re.findall(code_block_pattern, issue_text, re.DOTALL):
        # Extract all identifiers from the code block
        # Matches: variable names, property access (obj.property), method calls
        identifiers = re.findall(r'\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*)\b', code_block, re.IGNORECASE)
        for identifier in identifiers:
            # Use the full identifier (e.g., "item.subscription_item") or just the property
            parts = identifier.split('.')
            for part in parts:
                # Filter: 4-20 chars, avoid common words
                if 4 <= len(part) <= 20 and part.lower() not in _STOP_WORDS and part not in keywords:
                    keywords.append(part)

    # 1. Backtick-wrapped identifiers: `isMobilePhone`, `calculate_total()`
    for m in re.findall(r'`([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?)`', issue_text):
        name = m.split('(')[0]
        if len(name) >= 4 and name.lower() not in _STOP_WORDS and name not in keywords:
            keywords.append(name)

    # 2. camelCase identifiers in prose (4+ chars, must have mixed case OR contain underscore)
    # This catches: subscriptionItem, subscription_item, calculateTotal, etc.
    for m in re.findall(r'\b([a-z_][a-zA-Z0-9_]{3,})\b', issue_text):
        has_mixed_case = any(c.isupper() for c in m)
        has_underscore = '_' in m
        if (has_mixed_case or has_underscore) and m.lower() not in _STOP_WORDS and m not in keywords:
            keywords.append(m)

    # 3-6: Mine the issue title for technical terms
    title_line = issue_text.splitlines()[0].strip()
    if title_line.lower().startswith("title:"):
        title_line = title_line[6:].strip()

    if title_line:
        # 3. All-caps abbreviations (UTF, ECI, URL, HTTP, JSON, QR …)
        for m in re.findall(r'\b([A-Z]{2,})\b', title_line):
            if m.lower() not in _STOP_WORDS and m not in keywords:
                keywords.append(m)

        # 4. Hyphenated technical terms (UTF-8, ISO-8859, TLS-1.2 …)
        for m in re.findall(r'\b([A-Za-z][A-Za-z0-9]*-[A-Za-z0-9]+)\b', title_line):
            if m.lower() not in _STOP_WORDS and m not in keywords:
                keywords.append(m)

        # 5. Longer meaningful words >= 5 chars that aren't stop words
        for m in re.findall(r'\b([A-Za-z]{5,})\b', title_line):
            if m.lower() not in _STOP_WORDS and m.lower() not in {k.lower() for k in keywords}:
                keywords.append(m)

    # Deduplicate preserving order (case-insensitive), prioritize shorter, more specific names
    seen: set[str] = set()
    unique: list[str] = []
    
    # Sort by: underscores first (more specific), then length (prefer shorter/simpler)
    keywords_sorted = sorted(
        keywords,
        key=lambda x: (0 if '_' in x else 1, len(x))
    )
    
    for k in keywords_sorted:
        if k.lower() not in seen:
            seen.add(k.lower())
            unique.append(k)
    
    return unique


def _get_top_file_from_search(search_result: str) -> str | None:
    """Parse search_code output and return the most relevant file path.
    
    Prefers source implementation files over test files and entry points.
    Priority: src/lib files > other source > test files > index/main files.
    """
    from collections import Counter
    file_counts: Counter[str] = Counter()
    
    for line in search_result.split('\n'):
        if ':' in line and not line.startswith('['):
            parts = line.split(':', 2)
            if len(parts) >= 2:
                file_path = parts[0].strip()
                if file_path:
                    # Normalize path separators
                    file_counts[file_path.replace('\\', '/')] += 1
    
    if not file_counts:
        return None
    
    candidates = file_counts.most_common()
    
    # Score each candidate: lower score = better priority
    def _score(path: str) -> int:
        name = Path(path).name.lower()
        path_lower = path.lower()
        if 'test' in path_lower or 'spec' in path_lower:
            return 3  # Test files: low priority
        if 'index' in name or 'main' in name:
            return 2  # Entry points: medium-low
        if 'src/' in path_lower or 'lib/' in path_lower:
            return 0  # Source/library: highest
        return 1  # Other source files
    
    # Sort by score (ascending), then by match count (descending)
    best = min(candidates, key=lambda item: (_score(item[0]), -item[1]))
    return best[0]


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------
def researcher_node(state: AgentState) -> dict:
    """Run the Researcher agent: analyse the issue, call tools, return context."""
    print("[Researcher] Starting codebase exploration...")

    repo_path = state.get("repo_path", ".")
    issue_text = state.get("issue", "(no issue provided)")
    errors = state.get("errors", "")

    human_str = f"GitHub Issue:\n{issue_text}\n\nRepository path: {repo_path}\n\n"
    if errors:
        human_str += f"Supervisor Feedback/Errors:\n{errors}\n\n"
    
    # Add smart hints based on issue keywords
    issue_lower = issue_text.lower()
    if any(kw in issue_lower for kw in ["encod", "eci", "utf-8", "utf8", "charset", "character encode"]):
        human_str += "HINT: This is an ENCODING issue. Look for classes/enums with names like:\n"
        human_str += "  - Data (e.g., QRCodeData), Generator, Encoder, Manager\n"
        human_str += "  - Methods: Encode, Compress, Prepare, SetEncoding\n"
        human_str += "  - Enums: EncodingMode, ECI, Compression, CharacterSet\n"
        human_str += "Search for these patterns first.\n\n"
    elif any(kw in issue_lower for kw in ["null", "npe", "exception", "error"]):
        human_str += "HINT: This is an ERROR handling issue. Look for:\n"
        human_str += "  - Methods that could throw exceptions\n"
        human_str += "  - Missing null checks or validations\n"
        human_str += "  - Error handling patterns\n\n"
    
    human_str += "Please explore the repository and find the relevant code."

    # Seed the conversation with system prompt + issue
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_str),
    ]

    snippets: list[str] = list(state.get("file_context", []))
    files_read = 0
    total_lines = 0

    # Initialize history for this round
    history_additions: list[dict] = []
    
    # Store initial issue into history
    history_additions.extend(append_to_history("Researcher", "Strategic Targeting", f"Repo: {repo_path}\nIssue: {issue_text}"))

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: PRE-SCAN -- Detect language and extract hints
    # ─────────────────────────────────────────────────────────────────────────
    language = _detect_language(repo_path)
    print(f"[Researcher] Detected language: {language}")
    history_additions.extend(append_to_history("Researcher", "Language Detection", f"Language: {language}"))
    
    # ── Auto-read CONTRIBUTING.md if present ───────────────────────────────
    contribution_guidelines = ""
    contributing_path = Path(repo_path) / "CONTRIBUTING.md"
    if contributing_path.exists():
        try:
            result = read_file.invoke({"file_path": str(contributing_path.resolve())})
            if not result.startswith("Error"):
                contribution_guidelines = result
                line_count = result.count("\n")
                print(f"[Researcher] ✅ Read CONTRIBUTING.md ({line_count} lines)")
                history_additions.extend(
                    append_to_history("Researcher", "Contributing Guide", f"Read CONTRIBUTING.md ({line_count} lines)")
                )
        except Exception as e:
            print(f"[Researcher] ⚠ Could not read CONTRIBUTING.md: {e}")
    
    hint_files = _extract_hints_from_issue(issue_text)
    if hint_files:
        print(f"[Researcher] Found {len(hint_files)} direct hint(s): {hint_files}")
        history_additions.extend(append_to_history("Researcher", "Hint Extraction", f"Hints: {hint_files}"))
        
        # Read hinted files immediately (BEFORE LLM tool loop)
        print(f"[Researcher] Reading {len(hint_files)} hint file(s)...")
        for idx, hint_file in enumerate(hint_files[:_MAX_FILES_READ], 1):
            if files_read >= _MAX_FILES_READ:
                print("[Researcher] Reached max file limit; skipping remaining hints.")
                break
            
            # Normalize path: strip leading repo path if hint includes it
            normalized_hint = hint_file.lstrip('./')
            repo_name = Path(repo_path).name
            if normalized_hint.startswith(repo_name + '/'):
                normalized_hint = normalized_hint[len(repo_name) + 1:]
            
            # Build absolute path and validate existence BEFORE reading
            safe_path_resolved = (Path(repo_path) / normalized_hint).resolve()
            
            if not safe_path_resolved.is_file():
                print(f"[Researcher]    ⏭ Skipping '{hint_file}' (not a real file in repo)")
                history_additions.extend(
                    append_to_history("Researcher", "Hint Skip", f"'{hint_file}' not found in repo")
                )
                continue
            
            try:
                result = read_file.invoke({"file_path": str(safe_path_resolved)})
                if result.startswith("Error"):
                    print(f"[Researcher]    ❌ Failed to read {hint_file}: {result[:100]}")
                    history_additions.extend(
                        append_to_history("Researcher", "Hint Read", f"Failed: {hint_file} ({result[:80]})")
                    )
                else:
                    lines_in_file = result.count("\n")
                    print(f"[Researcher]    ✅ Read {lines_in_file} lines from {hint_file}")
                    
                    snippet = f"# --- [HINTED] file: {hint_file} ---\n{result}"
                    snippets.append(snippet)
                    files_read += 1
                    total_lines += lines_in_file + 1
                    
                    history_additions.extend(
                        append_to_history("Researcher", "Hint Read", f"✅ {hint_file} ({lines_in_file} lines)")
                    )
            except Exception as e:
                print(f"[Researcher]    ❌ Exception reading {hint_file}: {e}")
                history_additions.extend(
                    append_to_history("Researcher", "Hint Read", f"Exception: {str(e)[:80]}")
                )
    
    # Early exit if hints provided ANY relevant content
    if snippets and files_read >= 1:
        print(f"[Researcher] ✅ Hints provided {files_read} file(s), {total_lines} lines. Skipping LLM search.")
        print(f"[Researcher] Done -- collected {len(snippets)} snippet(s), "
              f"{files_read} file(s) read, ~{total_lines} lines.")
        history_additions.extend(
            append_to_history("Researcher", "Targeting Complete", f"Collected {len(snippets)} snippets (from hints). Read {files_read} files.")
        )
        return_dict = {
            "file_context": snippets,
            "history": history_additions
        }
        if contribution_guidelines:
            return_dict["contribution_guidelines"] = contribution_guidelines
        return return_dict

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1.5: AUTO-SEARCH -- Extract keywords from issue, search & auto-read
    # 
    # Enhanced with Phase 3A: Ripgrep variant detection
    # When hints fail (e.g., "Validator.js" isn't a real file), extract function
    # names from the issue text and search for them. This doesn't depend on the
    # LLM chaining tool calls correctly.
    # ─────────────────────────────────────────────────────────────────────────
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
                    # Phase 3A: Try ripgrep smart search first (with case variants)
                    ripgrep_matches = smart_search(keyword, repo_path, prefer_core_lib=True, max_results=10)
                    if ripgrep_matches:
                        match_count = len(ripgrep_matches)
                        variants_found = generate_search_variants(keyword)
                        print(f"[Researcher] ✅ Ripgrep '{keyword}' (variants: {variants_found}): {match_count} match(es)")
                        
                        # Auto-read the most relevant file (highest priority)
                        top_match = ripgrep_matches[0]
                        top_file = top_match['file']
                        
                        if top_file:
                            # Normalize separator and build path
                            top_file_normalized = top_file.replace('\\', '/')
                            file_path = str((Path(repo_path) / top_file_normalized).resolve())
                            file_result = read_file.invoke({"file_path": file_path})
                            if not file_result.startswith("Error"):
                                lines_in_file = file_result.count("\n")
                                print(f"[Researcher] ✅ Auto-read '{top_file_normalized}' ({lines_in_file} lines)")
                                snippet = f"# --- file: {top_file_normalized} ---\n{file_result}"
                                snippets.append(snippet)
                                files_read += 1
                                total_lines += lines_in_file + 1
                                history_additions.extend(
                                    append_to_history("Researcher", "Auto-Read (Ripgrep)", f"✅ {top_file_normalized} ({lines_in_file} lines)")
                                )
                                continue  # Got what we need, try next keyword
                    
                    # Fallback: Try standard search_code if ripgrep didn't find anything
                    result = search_code.invoke({"query": keyword, "directory": repo_path})
                    if not result.startswith("No matches"):
                        match_count = len([l for l in result.split('\n') if l.strip() and ':' in l and not l.startswith('[')])
                        print(f"[Researcher] ✅ Auto-search '{keyword}' (fallback): {match_count} match(es)")
                        
                        # Auto-read the most relevant file from results
                        top_file = _get_top_file_from_search(result)
                        if top_file:
                            # Normalize separator and build path
                            top_file_normalized = top_file.replace('\\', '/')
                            file_path = str((Path(repo_path) / top_file_normalized).resolve())
                            file_result = read_file.invoke({"file_path": file_path})
                            if not file_result.startswith("Error"):
                                lines_in_file = file_result.count("\n")
                                print(f"[Researcher] ✅ Auto-read '{top_file_normalized}' ({lines_in_file} lines)")
                                snippet = f"# --- file: {top_file_normalized} ---\n{file_result}"
                                snippets.append(snippet)
                                files_read += 1
                                total_lines += lines_in_file + 1
                                history_additions.extend(
                                    append_to_history("Researcher", "Auto-Read", f"✅ {top_file_normalized} ({lines_in_file} lines)")
                                )
                                break  # Got what we need
                    else:
                        print(f"[Researcher] ❌ Auto-search '{keyword}': no matches (ripgrep + fallback)")
                except Exception as e:
                    print(f"[Researcher] ⚠ Auto-search error for '{keyword}': {e}")
    
    # NOTE: We do NOT early-exit after auto-search anymore.
    # Auto-search finds quick wins (1-2 files), but for complex issues
    # (like encoding modes), we need the LLM loop to understand relationships.
    # If auto-search found something, the LLM can refine. If not, LLM will search.

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: LLM-DRIVEN SEARCH (Fall back to this only if hints didn't work)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Reduce rounds when using LLM fallback -- we need speed
    max_rounds = 3  # Enough for: repo_map → search → read_file
    
    # Inject context about what we already know (from hints that partially worked)
    if snippets:
        context_note = (
            f"\n\nIMPORTANT: The system already read {files_read} file(s) from hints "
            f"({total_lines} lines). These files are already in context. "
            f"Focus ONLY on finding additional relevant files, do NOT re-read files you already have. "
            f"You need at most {_MAX_FILES_READ - files_read} more file(s)."
        )
        messages[-1] = HumanMessage(content=messages[-1].content + context_note)
    
    for round_num in range(1, max_rounds + 1):
        print(f"[Researcher]  |-- Round {round_num}/{max_rounds}")

        # ── Ask the LLM ────────────────────────────────────────────
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

        # ── If there are no tool calls, the agent is done ──────────
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            print("[Researcher] No more tool calls -- wrapping up.")
            break

        # ── Execute each tool call ─────────────────────────────────
        for tc in tool_calls:
            fn_name = tc["name"]
            fn_args = tc["args"]
            call_id = tc["id"]

            print(f"[Researcher]    --> {fn_name}({fn_args})")
            
            # Formulate the payload - only store filename and brief summary in state
            # but sys_logger inside log handles the printing. 
            log_payload = f"Tool: {fn_name}\nArgs: {json.dumps(fn_args)}"
            history_additions.extend(append_to_history("Researcher", "Tool Call", log_payload, max_length=150))

            # Enforce memory guards for read_file
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

            # Dispatch the tool
            tool_fn = _TOOL_MAP.get(fn_name)
            if tool_fn is None:
                result = f"Unknown tool '{fn_name}'."
            else:
                try:
                    result = tool_fn.invoke(fn_args)
                except Exception as exc:
                    result = f"Tool error: {exc}"

            # Track read_file usage
            if fn_name == "read_file" and not result.startswith(("Error", "[BLOCKED")):
                files_read += 1
                total_lines += result.count("\n") + 1
                # Store as a context snippet
                file_label = fn_args.get("file_path", "unknown")
                snippet = f"# --- file: {file_label} ---\n{result}"
                snippets.append(snippet)

            # Auto-read: if search_code found matches but LLM might not follow up
            if fn_name == "search_code" and not result.startswith(("No matches", "Error", "ERROR")):
                top_file = _get_top_file_from_search(result)
                if top_file and files_read < _MAX_FILES_READ:
                    top_file_normalized = top_file.replace('\\', '/')
                    auto_path = str((Path(repo_path) / top_file_normalized).resolve())
                    try:
                        auto_result = read_file.invoke({"file_path": auto_path})
                        if not auto_result.startswith(("Error", "[BLOCKED")):
                            lines_in_auto = auto_result.count("\n")
                            print(f"[Researcher]    🔄 Auto-read from search: '{top_file_normalized}' ({lines_in_auto} lines)")
                            snippet = f"# --- file: {top_file_normalized} ---\n{auto_result}"
                            snippets.append(snippet)
                            files_read += 1
                            total_lines += lines_in_auto + 1
                    except Exception:
                        pass

            messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2.5: LAST-RESORT SCAN -- runs only if LLM produced zero results
    # (covers cases where: LLM returned plain text with no tool calls, API
    #  failed silently, or all tool calls returned empty matches)
    # ─────────────────────────────────────────────────────────────────────────
    if not snippets:
        print("[Researcher] [FALLBACK] All LLM phases returned 0 snippets. Running last-resort scan.")
        history_additions.extend(
            append_to_history("Researcher", "Last-Resort Scan", "LLM returned no tool calls. Scanning repo directly.")
        )

        # Step 1: generate repo map to discover directory structure
        try:
            map_result = generate_repo_map.invoke({"directory": repo_path})
            if map_result and not map_result.startswith(("Error", "ERROR")):
                print(f"[Researcher] [FALLBACK] Repo map obtained ({len(map_result)} chars).")
        except Exception as map_exc:
            map_result = ""
            print(f"[Researcher] [FALLBACK] Repo map failed: {map_exc}")

        # Step 2: search for keywords derived from the issue title
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
                                print(f"[Researcher] [FALLBACK] ✅ Read '{top_file_normalized}' ({lines_found} lines)")
                                snippets.append(f"# --- file: {top_file_normalized} ---\n{file_result}")
                                files_read += 1
                                total_lines += lines_found + 1
                                history_additions.extend(
                                    append_to_history(
                                        "Researcher", "Last-Resort Read",
                                        f"✅ {top_file_normalized} ({lines_found} lines) via '{term}'"
                                    )
                                )
                                break  # one file is enough for the coder to orient
                        except Exception:
                            pass
            except Exception as search_exc:
                print(f"[Researcher] [FALLBACK] Search error for '{term}': {search_exc}")

        if not snippets and map_result:
            # Store the repo map itself as context if we still have nothing —
            # better than returning empty-handed; the coder can use it to navigate.
            snippets.append(f"# --- repo map ---\n{map_result[:3000]}")
            history_additions.extend(
                append_to_history("Researcher", "Last-Resort Read", "Using repo map as context (no source file found)")
            )

    print(f"[Researcher] Done -- collected {len(snippets)} snippet(s), "
          f"{files_read} file(s) read, ~{total_lines} lines.")

    history_additions.extend(append_to_history("Researcher", "Targeting Complete", f"Collected {len(snippets)} snippets. Read {files_read} files."))

    return_dict = {
        "file_context": snippets,
        "history": history_additions
    }
    if contribution_guidelines:
        return_dict["contribution_guidelines"] = contribution_guidelines
    return return_dict
