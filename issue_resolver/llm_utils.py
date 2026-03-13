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

_SELECTED_MODEL_BY_ROLE: dict[str, str] = {}


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
    """Invoke Groq model with role-level fallback and transient retry handling."""
    if ChatGroq is None:
        raise RuntimeError("langchain-groq is not installed")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    ordered = list(dict.fromkeys(candidates))
    selected = _SELECTED_MODEL_BY_ROLE.get(role)
    if selected and selected in ordered:
        ordered = [selected] + [m for m in ordered if m != selected]

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
            _SELECTED_MODEL_BY_ROLE[role] = model_name
            return response, model_name
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_model_unavailable(exc):
                print(f"[{role}] [FALLBACK] model '{model_name}' unavailable: {exc}")
                continue
            raise

    if last_exc is None:
        raise RuntimeError(f"{role}: no model candidates configured")
    raise last_exc
