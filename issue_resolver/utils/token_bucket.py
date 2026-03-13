"""
TokenBucket Rate Limiter -- Prevents exceeding Groq TPM (tokens per minute) limits.

The Researcher and Planner nodes can consume large amounts of tokens due to
expanded context maps. This utility tracks tokens used and prevents excessive
consumption that would trigger Groq's rate limiting.

Rate Limits by Model (as of 2024):
  - qwen-2.5-coder-32b: 9000 TPM (tokens per minute)
  - llama-3.3-70b-versatile: 9000 TPM
  - llama-3.1-8b-instant: 9000 TPM

Conservative threshold: 7000 TPM (77% of limit) to leave headroom
"""

from __future__ import annotations

import time
from typing import Optional


class TokenBucket:
    """Track token usage and enforce rate limits."""
    
    # Conservative threshold: 77% of 9000 TPM
    DEFAULT_TPM_LIMIT = 7000
    MINUTE_WINDOW = 60.0
    
    def __init__(self, tpm_limit: int = DEFAULT_TPM_LIMIT, window_seconds: float = MINUTE_WINDOW):
        """Initialize token bucket.
        
        Args:
            tpm_limit: Tokens per minute limit
            window_seconds: Sliding window duration (default: 60 seconds = 1 minute)
        """
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds
        
        # Sliding window implementation
        self.tokens_used = []  # List of (timestamp, token_count) tuples
    
    def can_spend(self, tokens: int) -> bool:
        """Check if tokens can be spent without exceeding limit."""
        now = time.time()
        
        # Remove old entries outside the window
        self.tokens_used = [
            (ts, count) for ts, count in self.tokens_used
            if now - ts < self.window_seconds
        ]
        
        # Check if spending these tokens would exceed limit
        total_in_window = sum(count for _, count in self.tokens_used)
        return (total_in_window + tokens) <= self.tpm_limit
    
    def spend(self, tokens: int) -> bool:
        """Attempt to spend tokens. Returns True if successful, False if rate-limited."""
        if self.can_spend(tokens):
            self.tokens_used.append((time.time(), tokens))
            return True
        return False
    
    def get_token_usage(self) -> dict:
        """Get current token usage stats."""
        now = time.time()
        
        # Remove old entries
        self.tokens_used = [
            (ts, count) for ts, count in self.tokens_used
            if now - ts < self.window_seconds
        ]
        
        total_used = sum(count for _, count in self.tokens_used)
        remaining = self.tpm_limit - total_used
        percent_used = (total_used / self.tpm_limit) * 100 if self.tpm_limit > 0 else 0
        
        return {
            "tokens_used_this_minute": total_used,
            "tokens_remaining": remaining,
            "percent_used": percent_used,
            "limit": self.tpm_limit,
            "records_in_window": len(self.tokens_used),
        }
    
    def wait_if_needed(self) -> float:
        """
        If approaching rate limit, wait until oldest token(s) age out of window.
        Returns the number of seconds waited (0 if no wait).
        """
        now = time.time()
        stats = self.get_token_usage()
        
        # If under 70% consumption, proceed immediately
        if stats["percent_used"] < 70:
            return 0.0
        
        # If at/over limit, wait for oldest token to age out
        if not self.tokens_used:
            return 0.0
        
        oldest_timestamp = self.tokens_used[0][0]
        age = now - oldest_timestamp
        
        if age >= self.window_seconds:
            return 0.0
        
        wait_seconds = self.window_seconds - age + 0.1  # Add 100ms buffer
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            return wait_seconds
        
        return 0.0


# Global token bucket instance (session-level)
_global_token_bucket = TokenBucket()


def get_token_bucket() -> TokenBucket:
    """Get the global token bucket instance."""
    return _global_token_bucket


def reset_token_bucket() -> None:
    """Reset the global token bucket (useful for testing or new sessions)."""
    global _global_token_bucket
    _global_token_bucket = TokenBucket()


def check_rate_limit_before_call(estimated_tokens: int) -> bool:
    """Check if we can make an LLM call with estimated tokens."""
    return _global_token_bucket.can_spend(estimated_tokens)


def record_tokens_used(tokens: int) -> bool:
    """Record tokens used by an LLM call. Returns True if successful."""
    return _global_token_bucket.spend(tokens)


def wait_for_capacity(estimated_tokens: int) -> float:
    """
    If approaching rate limit, wait until we have capacity.
    Returns seconds waited.
    """
    bucket = get_token_bucket()
    
    # First check if we can spend
    if bucket.can_spend(estimated_tokens):
        return 0.0
    
    # Approach 1: Wait for tokens to age out
    wait_time = bucket.wait_if_needed()
    if wait_time > 0:
        return wait_time
    
    # Approach 2: Still doesn't fit, wait for oldest to clear
    if bucket.tokens_used:
        oldest_timestamp = bucket.tokens_used[0][0]
        now = time.time()
        age = now - oldest_timestamp
        
        if age < bucket.window_seconds:
            wait_time = bucket.window_seconds - age + 0.1
            time.sleep(wait_time)
            return wait_time
    
    return 0.0


def get_rate_limit_status() -> dict:
    """Get current rate limit status."""
    bucket = get_token_bucket()
    stats = bucket.get_token_usage()
    
    # Add a warning threshold
    warning = ""
    if stats["percent_used"] >= 70:
        warning = "APPROACHING_LIMIT"
    elif stats["percent_used"] >= 90:
        warning = "CRITICAL_LIMIT"
    
    return {
        **stats,
        "warning": warning,
    }
