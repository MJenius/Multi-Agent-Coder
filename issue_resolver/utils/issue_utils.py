"""Issue text utilities for smart context management.

Provides sliding window and critical section extraction for issue descriptions.
Prevents truncation of stack traces, error messages, and reproduction steps.
"""

from __future__ import annotations

import re


def extract_critical_sections(issue_text: str, max_length: int = 4000) -> str:
    """Extract and preserve critical sections of issue text.
    
    Priority order:
    1. Title (always included)
    2. Stack trace / error output
    3. "To Reproduce" or "Steps to Reproduce" section
    4. "Expected vs Actual" behavior
    5. Environment / config details
    6. Remaining text (subject to truncation)
    
    Uses a "sliding window" approach to find critical sections while respecting
    total length limits. This avoids losing important debugging info.
    
    Args:
        issue_text: Full issue description
        max_length: Maximum characters to return (default 4000)
    
    Returns:
        Intelligently truncated issue text with critical sections preserved
    """
    if not issue_text:
        return ""
    
    if len(issue_text) <= max_length:
        return issue_text  # No truncation needed
    
    lines = issue_text.split('\n')
    
    # Extract title (first line typically)
    title = lines[0] if lines else ""
    remaining_lines = lines[1:] if len(lines) > 1 else []
    
    # Define patterns for critical sections
    critical_patterns = {
        'traceback': r'(?:traceback|stack trace|error:|exception)',
        'reproduce': r'(to|how to) repro(?:duce|steps|steps to)',
        'expected': r'expected.*?(?:vs|actual|but|however)',
        'error_msg': r'(?:error|failed|failed with|failed because)',
        'config': r'(?:version|environment|os|platform|python)',
    }
    
    # Find all critical sections and their line ranges
    sections = []
    for i, line in enumerate(remaining_lines):
        line_lower = line.lower()
        for section_type, pattern in critical_patterns.items():
            if re.search(pattern, line_lower, re.IGNORECASE):
                # Found a critical section header
                # Extract this line + next 10 lines as a section
                section_start = i
                section_end = min(i + 15, len(remaining_lines))
                section_text = '\n'.join(remaining_lines[section_start:section_end])
                sections.append({
                    'type': section_type,
                    'start': section_start,
                    'end': section_end,
                    'text': section_text,
                    'priority': list(critical_patterns.keys()).index(section_type),
                })
    
    # Sort by priority (traceback first, then reproduce, etc.)
    sections.sort(key=lambda s: s['priority'])
    
    # Build output: title + critical sections + remaining text
    output_parts = [title]
    available_length = max_length - len(title) - 10  # Reserve some space
    
    # Add critical sections
    for section in sections:
        section_len = len(section['text']) + 1  # +1 for newline
        if section_len <= available_length:
            output_parts.append(section['text'])
            available_length -= section_len
    
    # Add remaining lines (truncated to fit)
    remaining_text = '\n'.join(remaining_lines)
    if available_length > 100:
        remaining_text = remaining_text[:available_length]
        output_parts.append(remaining_text)
    
    result = '\n'.join(output_parts)
    
    # Ensure we didn't exceed max_length by more than 10% (for critical content)
    if len(result) > max_length * 1.1:
        result = result[:max_length] + "\n\n[... TRUNCATED DUE TO LENGTH ...]"
    
    return result


def prioritize_issue_context(issue_text: str) -> tuple[str, str]:
    """Separate issue into title and body with smart chunking.
    
    Returns:
        Tuple of (title, body) where body is intelligently chunked
    """
    if not issue_text:
        return "", ""
    
    lines = issue_text.split('\n')
    title = lines[0] if lines else ""
    body = '\n'.join(lines[1:]) if len(lines) > 1 else ""
    
    # Apply smart extraction to body
    body = extract_critical_sections(f"{title}\n{body}", max_length=3000)
    
    return title, body


def find_stack_trace(issue_text: str) -> str | None:
    """Extract stack trace or error message if present.
    
    Returns:
        Stack trace text or None if not found
    """
    if not issue_text:
        return None
    
    # Common stack trace markers
    markers = [
        'Traceback (most recent call last):',
        'Stack trace:',
        'Error:',
        'Exception:',
        'Error output:',
        'stderr:',
    ]
    
    for marker in markers:
        idx = issue_text.find(marker)
        if idx != -1:
            # Extract marker line + next 30 lines
            lines = issue_text[idx:].split('\n')[:30]
            return '\n'.join(lines)
    
    return None
