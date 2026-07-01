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
    LLM_TIMEOUT,
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
    context: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not installed")

    from issue_resolver.core.model_router import get_model_router
    router = get_model_router()

    # Determine ordered list of model configs to try
    configs_to_try = []

    # If the role is configured in the router, resolve it
    if role in router.list_roles():
        resolved_cfg = router.resolve(role, context)
        configs_to_try.append(resolved_cfg)
        # Also append fallbacks from the router
        role_cfg = router.get_role_config(role)
        if role_cfg:
            for fallback_model in role_cfg.fallback_models:
                configs_to_try.append(
                    router.resolve(role, {**(context or {}), "current_model": fallback_model})
                )
    else:
        # Fall back to passed candidates
        for model_name in candidates:
            api_key = _resolve_api_key(model_name)
            configs_to_try.append(
                from_router_cfg(model_name, api_key, temperature, max_tokens)
            )

    excluded = _DECOMMISSIONED_MODELS | _QUOTA_EXCEEDED_MODELS
    ordered_configs = []
    seen_models = set()
    for cfg in configs_to_try:
        if cfg.model not in excluded and cfg.model not in seen_models:
            seen_models.add(cfg.model)
            ordered_configs.append(cfg)

    if not ordered_configs:
        if _DECOMMISSIONED_MODELS:
            raise RuntimeError(
                f"{role}: all model candidates decommissioned: {_DECOMMISSIONED_MODELS}."
            )
        if _QUOTA_EXCEEDED_MODELS:
            raise RuntimeError(
                f"{role}: all models exceeded daily quota: {_QUOTA_EXCEEDED_MODELS}."
            )
        raise RuntimeError(f"{role}: no model candidates configured")

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
    for cfg in ordered_configs:
        api_key = cfg.api_key or _resolve_api_key(cfg.model)
        if not api_key:
            print(f"[{role}] [SKIP] No API key configured for model '{cfg.model}'")
            continue

        try:
            extra_args = {}
            if cfg.extra_body:
                extra_args["extra_body"] = cfg.extra_body

            llm = ChatOpenAI(
                model=cfg.model,
                api_key=api_key,
                base_url=cfg.base_url or NVIDIA_BASE_URL,
                temperature=temperature if temperature != 0.0 else cfg.temperature,
                max_tokens=max_tokens or cfg.max_tokens,
                timeout=LLM_TIMEOUT,
                **extra_args
            )
            llm_to_call = llm.bind_tools(tools) if tools else llm
            response = _invoke_with_backoff(llm_to_call, messages, role)

            output_text = getattr(response, "content", "")
            estimated_output_tokens = len(str(output_text)) // 4
            total_tokens_used = estimated_input_tokens + estimated_output_tokens
            record_tokens_used(total_tokens_used)

            # Record model selection in trace if running in trace context
            from issue_resolver.core.execution_trace import get_trace
            trace = get_trace()
            if trace:
                trace.record_model_selection(
                    agent=role,
                    role=role,
                    model=cfg.model,
                    reason=f"Resolved by ModelRouter with context {context or {}}"
                )
                trace.record(
                    "model_call",
                    role,
                    f"invoked {cfg.model}",
                    tokens_used=total_tokens_used,
                    model=cfg.model,
                )

            _SELECTED_MODEL_BY_ROLE[role] = cfg.model
            return response, cfg.model
        except Exception as exc:
            last_exc = exc
            print(f"[{role}] [WARNING] Model '{cfg.model}' failed: {exc}")
            
            try:
                from issue_resolver.core.model_router import get_model_router
                router = get_model_router()
                if router:
                    router.mark_failed(cfg.model)
            except Exception:
                pass
                
            if _SELECTED_MODEL_BY_ROLE.get(role) == cfg.model:
                del _SELECTED_MODEL_BY_ROLE[role]

            if _is_model_decommissioned(exc):
                print(f"[{role}] [DECOMMISSIONED] Model '{cfg.model}' permanently unavailable")
                _DECOMMISSIONED_MODELS.add(cfg.model)
            elif _is_quota_exceeded(exc):
                print(f"[{role}] [QUOTA_EXCEEDED] Model '{cfg.model}' hit daily limit")
                _QUOTA_EXCEEDED_MODELS.add(cfg.model)
                try:
                    from issue_resolver.core.model_router import get_model_router
                    router = get_model_router()
                    if router:
                        router.mark_quota_exceeded(cfg.model)
                except Exception:
                    pass
            else:
                print(f"[{role}] [FALLBACK] model '{cfg.model}' encountered error or timeout: {exc}")
            
            continue

    if last_exc is None:
        raise RuntimeError(f"{role}: no model candidates configured")
    raise last_exc


def from_router_cfg(model_name: str, api_key: str, temp: float, max_tok: int | None) -> Any:
    from issue_resolver.core.model_router import ModelConfig
    return ModelConfig(
        model=model_name,
        api_key_env="",
        base_url=NVIDIA_BASE_URL,
        temperature=temp,
        max_tokens=max_tok or 4096,
    )

