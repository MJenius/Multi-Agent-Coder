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
    LLM_BACKOFF_INITIAL_SECONDS,
    LLM_BACKOFF_MAX_SECONDS,
    LLM_BACKOFF_MULTIPLIER,
    LLM_MAX_ATTEMPTS,
)

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
