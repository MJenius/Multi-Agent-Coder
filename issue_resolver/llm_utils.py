"""Groq helpers for role-based model selection and resilient invocation."""

from __future__ import annotations

import time
from typing import Any

try:
    from langchain_groq import ChatGroq  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - allows offline/unit testing without Groq package
    ChatGroq = None

from issue_resolver.config import (
    GROQ_API_KEY,
    GROQ_CONTEXT_WINDOWS,
    CODER_MAX_OUTPUT_RATIO,
    CODER_MIN_OUTPUT_TOKENS,
    CODER_TARGET_OUTPUT_TOKENS,
    LLM_BACKOFF_INITIAL_SECONDS,
    LLM_BACKOFF_MAX_SECONDS,
    LLM_BACKOFF_MULTIPLIER,
    LLM_MAX_ATTEMPTS,
)
from issue_resolver.utils.token_bucket import (
    check_rate_limit_before_call,
    record_tokens_used,
    wait_for_capacity,
    get_rate_limit_status,
)

_SELECTED_MODEL_BY_ROLE: dict[str, str] = {}
_DECOMMISSIONED_MODELS: set[str] = set()  # Models removed due to 400 errors (decommissioned)
_QUOTA_EXCEEDED_MODELS: set[str] = set()  # Models that hit daily TPD limits this session (temporary)


def calculate_max_tokens(
    model_name: str,
    input_tokens: int,
    ratio: float | None = None,
) -> int:
    """Calculate dynamic max_tokens based on model context window and input size.
    
    Args:
        model_name: Name of the Groq model (e.g., "qwen-2.5-coder-32b")
        input_tokens: Estimated number of input tokens (rough: 1 token per 4 characters)
        ratio: Output allocation ratio (default from config CODER_MAX_OUTPUT_RATIO)
    
    Returns:
        Recommended max_tokens for generation (between CODER_MIN_OUTPUT_TOKENS and CODER_TARGET_OUTPUT_TOKENS)
    """
    if ratio is None:
        ratio = CODER_MAX_OUTPUT_RATIO
    
    context_window = GROQ_CONTEXT_WINDOWS.get(model_name, 8192)  # Default to 8K if unknown
    available_tokens = context_window - input_tokens
    
    # Calculate allocation: ratio of available tokens
    allocated = int(available_tokens * ratio)
    
    # Clamp between min and target
    return max(CODER_MIN_OUTPUT_TOKENS, min(CODER_TARGET_OUTPUT_TOKENS, allocated))


_SELECTED_MODEL_BY_ROLE: dict[str, str] = {}


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_markers = (
        "429",
        "rate limit",
        "timed out",
        "timeout",
        "connection",
        "temporar",
        "unavailable",
        "service unavailable",
        "too many requests",
    )
    return any(marker in text for marker in transient_markers)


def _is_model_unavailable(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "model",
        "not found",
        "does not exist",
        "unsupported",
        "invalid model",
        "permission",
    )
    return "model" in text and any(marker in text for marker in markers)


def _is_model_decommissioned(exc: Exception) -> bool:
    """Detect permanent model decommissioning (400 error, invalid request).
    
    When a model is decommissioned, the API returns a 400 Bad Request error.
    This is permanent and cannot be recovered with retry/fallback to the same model.
    The model should be removed from session candidates to avoid repeated failures.
    """
    text = str(exc).lower()
    
    # Detect 400 Bad Request errors
    decommission_markers = (
        "400",
        "bad request",
        "invalid request",
        "model.*does not exist",
        "model.*unavailable",
        "model.*deprecated",
        "model.*retired",
    )
    
    is_bad_request = any(marker in text for marker in decommission_markers)
    return is_bad_request


def _is_quota_exceeded(exc: Exception) -> bool:
    """Detect daily quota limit exceeded (429 TPD error).
    
    Different from transient rate limits, TPD (Tokens Per Day) limits are permanent
    for the day and cannot be resolved by waiting. When hit, the model should be
    skipped for the rest of the session but can be retried tomorrow.
    
    TPD errors are a specific type of 429 (Too Many Requests) that mention:
    - 'tokens per day' or 'TPD'
    - 'daily' + ('quota' or 'limit')
    - 'exceeded' + ('daily' or 'quota')
    """
    text = str(exc).lower()
    
    # Detect TPD-specific 429 errors
    tpd_markers = (
        "tokens per day",
        "tokens_per_day",
        "tpd",
        "daily quota",
        "daily token",
        "daily limit",
    )
    
    # Must be a 429 error containing TPD indicators
    has_quota_indicator = any(marker in text for marker in tpd_markers)
    is_429 = "429" in text or "too many requests" in text
    
    return is_429 and has_quota_indicator


def _invoke_with_backoff(llm: Any, messages: list[Any], role: str) -> Any:
    delay = LLM_BACKOFF_INITIAL_SECONDS
    last_exc: Exception | None = None

    for attempt in range(1, max(1, LLM_MAX_ATTEMPTS) + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_transient_error(exc) or attempt >= max(1, LLM_MAX_ATTEMPTS):
                raise
            print(f"[{role}] [RETRY] transient LLM error on attempt {attempt}: {exc}")
            time.sleep(min(delay, LLM_BACKOFF_MAX_SECONDS))
            delay = min(delay * LLM_BACKOFF_MULTIPLIER, LLM_BACKOFF_MAX_SECONDS)

    if last_exc is None:
        raise RuntimeError(f"{role} failed with unknown LLM error")
    raise last_exc


def invoke_with_role_fallback(
    *,
    role: str,
    candidates: list[str],
    messages: list[Any],
    temperature: float,
    max_tokens: int | None = None,
    tools: list[Any] | None = None,
) -> tuple[Any, str]:
    """Invoke Groq model with role-level fallback and resilient handling.
    
    Features:
    - Role-level model persistence (remembers which model worked for a role)
    - Fallback to next candidate on any error
    - Adaptive downscaling: removes permanently decommissioned models (400 errors)
    - Transient error retry with exponential backoff
    - Rate limiting: tracks tokens used and waits before calls if approaching limits (Phase 4)
    """
    if ChatGroq is None:
        raise RuntimeError("langchain-groq is not installed")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    # Filter out decommissioned and quota-exceeded models
    excluded = _DECOMMISSIONED_MODELS | _QUOTA_EXCEEDED_MODELS
    available_candidates = [m for m in candidates if m not in excluded]
    if not available_candidates:
        if _DECOMMISSIONED_MODELS:
            raise RuntimeError(
                f"{role}: all model candidates have been decommissioned: {_DECOMMISSIONED_MODELS}. "
                "Please update model configuration in config.py"
            )
        if _QUOTA_EXCEEDED_MODELS:
            raise RuntimeError(
                f"{role}: all models exceeded their daily quota: {_QUOTA_EXCEEDED_MODELS}. "
                "Please try again tomorrow or use a different API key."
            )
        raise RuntimeError(f"{role}: no model candidates configured")

    ordered = list(dict.fromkeys(available_candidates))
    selected = _SELECTED_MODEL_BY_ROLE.get(role)
    if selected and selected in ordered:
        ordered = [selected] + [m for m in ordered if m != selected]

    # Pre-call rate limit check (Phase 4: TokenBucket)
    estimated_input_tokens = sum(len(str(msg)) // 4 for msg in messages)
    estimated_total_tokens = estimated_input_tokens + (max_tokens or 1024)
    
    rate_limit_status = get_rate_limit_status()
    if rate_limit_status.get("percent_used", 0) >= 70:
        print(f"[{role}] [RATE_LIMIT] Using {rate_limit_status.get('percent_used', 0):.1f}% of TPM limit. " 
              f"Waiting for capacity...")
        wait_seconds = wait_for_capacity(estimated_total_tokens)
        if wait_seconds > 0:
            print(f"[{role}] [RATE_LIMIT] Waited {wait_seconds:.1f}s for capacity")

    last_exc: Exception | None = None
    for model_name in ordered:
        try:
            llm = ChatGroq(
                model=model_name,
                api_key=GROQ_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            llm_to_call = llm.bind_tools(tools) if tools else llm
            response = _invoke_with_backoff(llm_to_call, messages, role)
            
            # Record tokens used (Phase 4: TokenBucket)
            # Estimate output tokens from response content
            output_text = getattr(response, "content", "")
            estimated_output_tokens = len(str(output_text)) // 4
            total_tokens_used = estimated_input_tokens + estimated_output_tokens
            record_tokens_used(total_tokens_used)
            
            _SELECTED_MODEL_BY_ROLE[role] = model_name
            return response, model_name
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            
            # Phase 4A: Detect permanent decommissioning (400 error)
            if _is_model_decommissioned(exc):
                print(f"[{role}] [DECOMMISSIONED] Model '{model_name}' is permanently unavailable (400 error)")
                print(f"[{role}] [DECOMMISSIONED] Removing from session candidates: {model_name}")
                _DECOMMISSIONED_MODELS.add(model_name)
                # Clear selected model cache if this model was selected
                if _SELECTED_MODEL_BY_ROLE.get(role) == model_name:
                    del _SELECTED_MODEL_BY_ROLE[role]
                continue  # Try next candidate
            
            # Phase 4B: Detect daily quota exceeded (429 TPD error) - skip for session
            if _is_quota_exceeded(exc):
                print(f"[{role}] [QUOTA_EXCEEDED] Model '{model_name}' hit daily TPD limit")
                print(f"[{role}] [QUOTA_EXCEEDED] Skipping for rest of session: {model_name}")
                _QUOTA_EXCEEDED_MODELS.add(model_name)
                # Clear selected model cache if this model was selected
                if _SELECTED_MODEL_BY_ROLE.get(role) == model_name:
                    del _SELECTED_MODEL_BY_ROLE[role]
                continue  # Try next candidate (don't retry same model)
            
            # Detect temporary unavailability or unsupported model errors
            if _is_model_unavailable(exc):
                print(f"[{role}] [FALLBACK] model '{model_name}' temporarily unavailable: {exc}")
                continue
            
            # For other errors, re-raise (transient errors will be retried in _invoke_with_backoff)
            raise

    if last_exc is None:
        raise RuntimeError(f"{role}: no model candidates configured")
    raise last_exc
