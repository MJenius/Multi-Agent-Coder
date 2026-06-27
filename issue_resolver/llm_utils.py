from __future__ import annotations

import time
from typing import Any

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from issue_resolver.config import (
    NVIDIA_BASE_URL,
    NVIDIA_CONTEXT_WINDOWS,
    MODEL_API_KEY_MAP,
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
_DECOMMISSIONED_MODELS: set[str] = set()
_QUOTA_EXCEEDED_MODELS: set[str] = set()


def _resolve_api_key(model_name: str) -> str:
    key = MODEL_API_KEY_MAP.get(model_name, "")
    if not key:
        for prefix, mapped_key in MODEL_API_KEY_MAP.items():
            if model_name.startswith(prefix.split("/")[0]):
                return mapped_key
    return key


def calculate_max_tokens(
    model_name: str,
    input_tokens: int,
    ratio: float | None = None,
) -> int:
    if ratio is None:
        ratio = CODER_MAX_OUTPUT_RATIO

    context_window = NVIDIA_CONTEXT_WINDOWS.get(model_name, 131_072)
    available_tokens = context_window - input_tokens

    allocated = int(available_tokens * ratio)

    return max(CODER_MIN_OUTPUT_TOKENS, min(CODER_TARGET_OUTPUT_TOKENS, allocated))


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
        "503",
        "502",
        "internal server error",
        "overloaded",
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
    text = str(exc).lower()
    permanent_markers = (
        "model not found",
        "model_not_found",
        "does not exist",
        "model.*retired",
        "model.*deprecated",
        "model.*discontinued",
        "permission denied",
        "not authorized",
        "invalid model",
    )
    return any(marker in text for marker in permanent_markers)


def _is_quota_exceeded(exc: Exception) -> bool:
    text = str(exc).lower()
    tpd_markers = (
        "tokens per day",
        "tokens_per_day",
        "tpd",
        "daily quota",
        "daily token",
        "daily limit",
        "quota exceeded",
        "resource exhausted",
    )
    has_quota_indicator = any(marker in text for marker in tpd_markers)
    is_429 = "429" in text or "too many requests" in text
    return is_429 and has_quota_indicator


def _invoke_with_backoff(llm: Any, messages: list[Any], role: str) -> Any:
    delay = LLM_BACKOFF_INITIAL_SECONDS
    last_exc: Exception | None = None

    for attempt in range(1, max(1, LLM_MAX_ATTEMPTS) + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            last_exc = exc

            if _is_quota_exceeded(exc):
                print(f"[{role}] [NO_RETRY] Quota exhausted - skipping backoff for immediate rotation")
                raise

            if not _is_transient_error(exc) or attempt >= max(1, LLM_MAX_ATTEMPTS):
                raise
            print(f"[{role}] [RETRY] transient error on attempt {attempt}: {exc}")
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
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not installed")

    excluded = _DECOMMISSIONED_MODELS | _QUOTA_EXCEEDED_MODELS
    available_candidates = [m for m in candidates if m not in excluded]
    if not available_candidates:
        if _DECOMMISSIONED_MODELS:
            raise RuntimeError(
                f"{role}: all model candidates decommissioned: {_DECOMMISSIONED_MODELS}. "
                "Update model configuration in config.py"
            )
        if _QUOTA_EXCEEDED_MODELS:
            raise RuntimeError(
                f"{role}: all models exceeded daily quota: {_QUOTA_EXCEEDED_MODELS}. "
                "Try again later or use a different API key."
            )
        raise RuntimeError(f"{role}: no model candidates configured")

    ordered = list(dict.fromkeys(available_candidates))
    selected = _SELECTED_MODEL_BY_ROLE.get(role)
    if selected and selected in ordered:
        ordered = [selected] + [m for m in ordered if m != selected]

    estimated_input_tokens = sum(len(str(msg)) // 4 for msg in messages)
    estimated_total_tokens = estimated_input_tokens + (max_tokens or 4096)

    rate_limit_status = get_rate_limit_status()
    if rate_limit_status.get("percent_used", 0) >= 70:
        print(
            f"[{role}] [RATE_LIMIT] Using {rate_limit_status.get('percent_used', 0):.1f}% "
            f"of TPM limit. Waiting for capacity..."
        )
        wait_seconds = wait_for_capacity(estimated_total_tokens)
        if wait_seconds > 0:
            print(f"[{role}] [RATE_LIMIT] Waited {wait_seconds:.1f}s for capacity")

    last_exc: Exception | None = None
    for model_name in ordered:
        api_key = _resolve_api_key(model_name)
        if not api_key:
            print(f"[{role}] [SKIP] No API key configured for model '{model_name}'")
            continue

        try:
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=NVIDIA_BASE_URL,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            llm_to_call = llm.bind_tools(tools) if tools else llm
            response = _invoke_with_backoff(llm_to_call, messages, role)

            output_text = getattr(response, "content", "")
            estimated_output_tokens = len(str(output_text)) // 4
            total_tokens_used = estimated_input_tokens + estimated_output_tokens
            record_tokens_used(total_tokens_used)

            _SELECTED_MODEL_BY_ROLE[role] = model_name
            return response, model_name
        except Exception as exc:
            last_exc = exc

            if _is_model_decommissioned(exc):
                print(f"[{role}] [DECOMMISSIONED] Model '{model_name}' permanently unavailable")
                _DECOMMISSIONED_MODELS.add(model_name)
                if _SELECTED_MODEL_BY_ROLE.get(role) == model_name:
                    del _SELECTED_MODEL_BY_ROLE[role]
                continue

            if _is_quota_exceeded(exc):
                print(f"[{role}] [QUOTA_EXCEEDED] Model '{model_name}' hit daily limit")
                _QUOTA_EXCEEDED_MODELS.add(model_name)
                if _SELECTED_MODEL_BY_ROLE.get(role) == model_name:
                    del _SELECTED_MODEL_BY_ROLE[role]
                continue

            if _is_model_unavailable(exc):
                print(f"[{role}] [FALLBACK] model '{model_name}' temporarily unavailable: {exc}")
                continue

            raise

    if last_exc is None:
        raise RuntimeError(f"{role}: no model candidates configured")
    raise last_exc
