"""
Ripgrep Search Utility -- Enhanced code search with variant detection and scope limiting.

Provides ripgrep-based searching with:
1. CamelCase / snake_case / kebab-case variant detection
2. Core library vs test file prioritization
3. Fallback to basic grep if ripgrep unavailable
"""

from __future__ import annotations

import subprocess
import json
from pathlib import Path
from typing import Optional


def _to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def _to_snake_case(camel_str: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, char in enumerate(camel_str):
        if char.isupper() and i > 0:
            result.append('_')
            result.append(char.lower())
        else:
            result.append(char)
    return ''.join(result)


def _to_kebab_case(snake_str: str) -> str:
    """Convert snake_case to kebab-case."""
    return snake_str.replace('_', '-')


def generate_search_variants(identifier: str) -> list[str]:
    """Generate case variants of an identifier for comprehensive searching.
    
    Examples:
        'subscription_item' → ['subscription_item', 'subscriptionItem', 'subscription-item']
        'subscriptionItem' → ['subscriptionItem', 'subscription_item', 'subscription-item']
    """
    variants = set()
    
    # Original
    variants.add(identifier)
    
    # If contains underscore, generate camelCase
    if '_' in identifier:
        variants.add(_to_camel_case(identifier))
        variants.add(_to_kebab_case(identifier))
    
    # If contains uppercase, generate snake_case
    if any(c.isupper() for c in identifier):
        variants.add(_to_snake_case(identifier))
        variants.add(_to_kebab_case(_to_snake_case(identifier)))
    
    # Remove the original again if it was added by variant generation
    variants.discard('')
    
    return sorted(list(variants))


def is_ripgrep_available() -> bool:
    """Check if ripgrep (rg) is installed and available."""
    try:
        subprocess.run(
            ["rg", "--version"],
            capture_output=True,
            timeout=2,
            check=False,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def search_with_ripgrep(
    query: str,
    directory: str,
    prefer_core_lib: bool = True,
    max_results: int = 50,
) -> list[dict]:
    """
    Search using ripgrep with core library prioritization.
    
    Returns list of dicts with keys:
        - file: relative file path
        - line: line number
        - content: matched line content
        - priority: 0 (core) or 1 (test) or 2 (other)
    """
    
    if not is_ripgrep_available():
        return []
    
    # Build ripgrep command
    # --json: Machine-readable JSON output
    # -i: Case-insensitive
    # -n: Show line numbers
    # -H: Show file names
    # --max-count=1: One match per file (for deduplication)
    cmd = [
        "rg",
        "--json",
        "-i",
        "-n",
        "-H",
        "--max-count=1",
        query,
        directory,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        
        if result.returncode not in [0, 1]:  # 0=found, 1=not found
            return []
        
        matches = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                if record.get('type') != 'match':
                    continue
                
                file_path = record.get('data', {}).get('path', {}).get('text', '')
                line_num = record.get('data', {}).get('line_number', 0)
                matched_line = record.get('data', {}).get('lines', {}).get('text', '')
                
                # Prioritize core library files over test files
                priority = 2  # default: other
                if prefer_core_lib:
                    if '/test' in file_path or '/tests' in file_path or file_path.endswith('_test.py'):
                        priority = 1  # test file
                    elif '/plugin' in file_path or '/extension' in file_path:
                        priority = 1  # plugin/extension
                    else:
                        priority = 0  # core library
                
                matches.append({
                    'file': file_path,
                    'line': line_num,
                    'content': matched_line.strip(),
                    'priority': priority,
                })
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Sort by priority (0=core first), then by file name
        matches.sort(key=lambda x: (x['priority'], x['file']))
        
        return matches[:max_results]
    
    except subprocess.TimeoutExpired:
        return []
    except Exception:
        return []


def smart_search(
    identifier: str,
    directory: str,
    prefer_core_lib: bool = True,
    max_results: int = 50,
) -> list[dict]:
    """
    Perform smart search: try ripgrep with variants, then fallback to basic grep.
    
    Returns list of matches ordered by relevance.
    """
    
    # First, try ripgrep with variants
    if is_ripgrep_available():
        variants = generate_search_variants(identifier)
        all_matches = {}
        
        for variant in variants:
            matches = search_with_ripgrep(
                variant,
                directory,
                prefer_core_lib=prefer_core_lib,
                max_results=max_results,
            )
            
            for match in matches:
                # Use file path as unique key to avoid duplicates
                key = (match['file'], match['line'])
                if key not in all_matches:
                    all_matches[key] = match
        
        # Convert back to list and sort
        results = list(all_matches.values())
        results.sort(key=lambda x: (x['priority'], x['file'], x['line']))
        return results[:max_results]
    
    # Fallback: use basic grep with word boundary
    return []


def format_search_results(matches: list[dict], max_lines: int = 3000) -> str:
    """Format search results for presentation to agents."""
    if not matches:
        return "No matches found."
    
    output_lines = []
    total_chars = 0
    
    for match in matches:
        file_path = match.get('file', '')
        line_num = match.get('line', 0)
        content = match.get('content', '')
        
        line = f"{file_path}:{line_num}: {content}"
        if total_chars + len(line) > max_lines:
            output_lines.append(f"... ({len(matches) - len(output_lines)} more matches truncated)")
            break
        
        output_lines.append(line)
        total_chars += len(line) + 1
    
    return '\n'.join(output_lines)
